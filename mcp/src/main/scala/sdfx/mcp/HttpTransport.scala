package sdfx.mcp

import com.sun.net.httpserver.{HttpExchange, HttpServer}
import sdfx.engine.AppServices

import java.io.{ByteArrayOutputStream, OutputStream}
import java.net.{InetAddress, InetSocketAddress, URI}
import java.nio.charset.StandardCharsets
import java.security.{MessageDigest, SecureRandom}
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{ConcurrentHashMap, LinkedBlockingQueue, Semaphore, ThreadPoolExecutor, TimeUnit}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** MCP Streamable HTTP transport on 127.0.0.1 (endpoint `/mcp`).
  *
  *  - POST: one JSON-RPC message or batch (body <= `maxBodyBytes`). Notifications/responses only -> 202. Requests ->
  *    `application/json`, or an SSE stream (progress notifications + the response) when the client accepts
  *    `text/event-stream` and a `tools/call` carries a progressToken. Invalid messages -> 400 with the error.
  *  - `Mcp-Session-Id` is issued on a successful initialize and required afterwards (400 when missing, 404 when
  *    unknown/expired). Sessions idle longer than `idleTtlMillis` expire; when `maxSessions` is reached only idle
  *    sessions are evicted. DELETE ends the session; GET -> 405 (no server-initiated stream).
  *  - DNS rebinding protection: Origin (absent or http(s)://localhost|127.0.0.1|[::1]) and Host must be loopback.
  *    Optional bearer token (401 otherwise).
  *  - At most `maxConcurrentCalls` tool calls run at once (503 / -32000 beyond that).
  */
final class HttpTransport(
    newServer: () => McpServer,
    port: Int,
    token: Option[String],
    log: String => Unit = s => System.err.println(s),
    idleTtlMillis: Long = HttpTransport.DefaultIdleTtlMillis,
    maxSessions: Int = HttpTransport.DefaultMaxSessions,
    maxBodyBytes: Int = HttpTransport.DefaultMaxBodyBytes,
    maxConcurrentCalls: Int = HttpTransport.DefaultMaxConcurrentCalls
) extends AutoCloseable:
  import HttpTransport.*

  private final class Session(val id: String, val server: McpServer):
    @volatile var lastUsed: Long = System.currentTimeMillis()
    val inFlight = new AtomicInteger(0)
    def touch(): Unit = lastUsed = System.currentTimeMillis()

  private val sessions = new ConcurrentHashMap[String, Session]()
  private val random = new SecureRandom()
  private val threadNo = new AtomicInteger(0)
  private val calls = new Semaphore(maxConcurrentCalls)
  // fixed worker count, queued beyond that: quick requests still find a thread because tool calls are capped lower
  private val pool: ThreadPoolExecutor =
    val n = maxConcurrentCalls + 16
    val p = new ThreadPoolExecutor(n, n, 30, TimeUnit.SECONDS, new LinkedBlockingQueue[Runnable](1024), { (r: Runnable) =>
      val t = new Thread(r, s"mcp-http-${threadNo.incrementAndGet()}")
      t.setDaemon(true)
      t
    })
    p.allowCoreThreadTimeOut(true)
    p
  private val tokenBytes = token.map(_.trim).filter(_.nonEmpty).map(_.getBytes(StandardCharsets.UTF_8))

  private val http: HttpServer =
    // limit how long a client may take to send its request (does not limit long responses / SSE streams)
    if System.getProperty("sun.net.httpserver.maxReqTime") == null then System.setProperty("sun.net.httpserver.maxReqTime", "120")
    val s = HttpServer.create(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), port), 0)
    s.setExecutor(pool)
    s.createContext("/", ex => handle(ex))
    s.start()
    s

  def boundPort: Int = http.getAddress.getPort
  def url: String = s"http://127.0.0.1:$boundPort/mcp"
  def sessionCount: Int = sessions.size

  def close(): Unit =
    try http.stop(0)
    catch case NonFatal(_) => ()
    sessions.clear()
    pool.shutdownNow()
    ()

  // ---- request handling -------------------------------------------------------------------------------------

  private def header(ex: HttpExchange, name: String): Option[String] = Option(ex.getRequestHeaders.getFirst(name)).map(_.trim)

  private def respond(ex: HttpExchange, code: Int, body: Option[ujson.Value] = None, extra: Seq[(String, String)] = Nil, text: Option[String] = None): Unit =
    try
      val h = ex.getResponseHeaders
      extra.foreach((k, v) => h.add(k, v))
      val bytes = body.map(b => JsonRpc.render(b).getBytes(StandardCharsets.UTF_8))
        .orElse(text.map(_.getBytes(StandardCharsets.UTF_8)))
      bytes match
        case Some(b) =>
          h.set("Content-Type", if body.isDefined then "application/json" else "text/plain; charset=utf-8")
          ex.sendResponseHeaders(code, b.length.toLong)
          ex.getResponseBody.write(b)
        case None => ex.sendResponseHeaders(code, -1)
    finally ex.close()

  private def errorBody(code: Int, message: String): ujson.Value = JsonRpc.error(ujson.Null, code, message)

  private def handle(ex: HttpExchange): Unit =
    try
      sweep()
      val path = ex.getRequestURI.getPath
      if path != "/mcp" && path != "/mcp/" then respond(ex, 404, text = Some("not found (MCP endpoint is /mcp)"))
      else if !originAllowed(header(ex, "Origin")) then respond(ex, 403, Some(errorBody(JsonRpc.Codes.InvalidRequest, "Forbidden origin")))
      else if !hostAllowed(header(ex, "Host")) then respond(ex, 403, Some(errorBody(JsonRpc.Codes.InvalidRequest, "Forbidden host")))
      else if !authorized(header(ex, "Authorization")) then
        respond(ex, 401, Some(errorBody(JsonRpc.Codes.InvalidRequest, "Unauthorized")), Seq("WWW-Authenticate" -> "Bearer"))
      else
        ex.getRequestMethod.toUpperCase match
          case "POST"   => post(ex)
          case "DELETE" => delete(ex)
          case _        => respond(ex, 405, extra = Seq("Allow" -> "POST, DELETE"), text = Some("method not allowed"))
    catch
      case e: Throwable =>
        log(s"[mcp-http] request failed: $e")
        try respond(ex, 500, Some(errorBody(JsonRpc.Codes.InternalError, "internal error")))
        catch case NonFatal(_) => ()

  private def authorized(h: Option[String]): Boolean = tokenBytes match
    case None => true
    case Some(expected) =>
      h.filter(_.regionMatches(true, 0, "Bearer ", 0, 7)).map(_.substring(7).trim.getBytes(StandardCharsets.UTF_8))
        .exists(got => MessageDigest.isEqual(got, expected))

  private def checkProtocolHeader(ex: HttpExchange, s: Option[Session]): Option[String] =
    header(ex, ProtocolHeader).flatMap { v =>
      if !McpServer.SupportedVersions.contains(v) then Some(s"unsupported $ProtocolHeader: $v")
      else s.flatMap(_.server.protocolVersion).filter(_ != v).map(n => s"$ProtocolHeader $v does not match the negotiated version $n")
    }

  private def delete(ex: HttpExchange): Unit =
    header(ex, SessionHeader) match
      case None => respond(ex, 400, Some(errorBody(JsonRpc.Codes.InvalidRequest, s"missing $SessionHeader header")))
      case Some(id) =>
        Option(sessions.get(id)) match
          case None => respond(ex, 404, Some(errorBody(JsonRpc.Codes.InvalidRequest, "unknown session")))
          case Some(s) =>
            checkProtocolHeader(ex, Some(s)) match
              case Some(err) => respond(ex, 400, Some(errorBody(JsonRpc.Codes.InvalidRequest, err)))
              case None =>
                sessions.remove(id)
                respond(ex, 200)

  private def isInitialize(v: ujson.Value): Boolean = v match
    case o: ujson.Obj => o.value.get("method").contains(ujson.Str("initialize")) && o.value.contains("id")
    case _            => false

  private def messages(v: ujson.Value): Seq[ujson.Value] = v match
    case ujson.Arr(a) => a.toSeq
    case other        => Seq(other)

  private def newSessionId(): String =
    val b = new Array[Byte](16)
    random.nextBytes(b)
    b.map(x => f"${x & 0xff}%02x").mkString

  /** Reads the body; Left(()) if it exceeds `maxBodyBytes` (never buffers more than that). */
  private def readBody(ex: HttpExchange): Either[Unit, String] =
    val declared = header(ex, "Content-Length").flatMap(_.toLongOption).getOrElse(-1L)
    if declared > maxBodyBytes then Left(())
    else
      val in = ex.getRequestBody
      val bos = new ByteArrayOutputStream(math.max(256, math.min(declared, 1L << 20).toInt))
      val buf = new Array[Byte](1 << 16)
      var total = 0L
      var n = in.read(buf)
      while n >= 0 && total <= maxBodyBytes do
        total += n
        if total <= maxBodyBytes then
          bos.write(buf, 0, n)
          n = in.read(buf)
      if total > maxBodyBytes then Left(()) else Right(bos.toString(StandardCharsets.UTF_8))

  private def post(ex: HttpExchange): Unit =
    readBody(ex) match
      case Left(_) => respond(ex, 413, Some(errorBody(JsonRpc.Codes.InvalidRequest, s"request body larger than $maxBodyBytes bytes")))
      case Right(body) =>
        JsonRpc.parseText(body) match
          case Left(err) => respond(ex, 400, Some(JsonRpc.parseError(err)))
          case Right(msg) =>
            val session = header(ex, SessionHeader).flatMap(id => Option(sessions.get(id)))
            checkProtocolHeader(ex, session) match
              case Some(err) => respond(ex, 400, Some(errorBody(JsonRpc.Codes.InvalidRequest, err)))
              case None =>
                if isInitialize(msg) then initialize(ex, msg)
                else if messages(msg).exists(isInitialize) then
                  respond(ex, 400, Some(errorBody(JsonRpc.Codes.InvalidRequest, "initialize must not be part of a batch")))
                else if header(ex, SessionHeader).isEmpty then
                  respond(ex, 400, Some(errorBody(JsonRpc.Codes.InvalidRequest, s"missing $SessionHeader header (send initialize first)")))
                else
                  session match
                    case None => respond(ex, 404, Some(errorBody(JsonRpc.Codes.InvalidRequest, "unknown or expired session; initialize again")))
                    case Some(s) =>
                      s.inFlight.incrementAndGet()
                      s.touch()
                      try dispatch(ex, s, msg)
                      finally
                        s.touch()
                        s.inFlight.decrementAndGet()

  private def initialize(ex: HttpExchange, msg: ujson.Value): Unit =
    val s = new Session(newSessionId(), newServer())
    val r = s.server.handle(msg, _ => ())
    if !r.exists(_.obj.contains("result")) then respond(ex, 400, r)
    else if !makeRoom() then respond(ex, 503, Some(errorBody(StdioTransport.ServerBusy, "too many active MCP sessions")))
    else
      sessions.put(s.id, s)
      respond(ex, 200, r, Seq(SessionHeader -> s.id))

  private def hasRequest(msg: ujson.Value): Boolean = messages(msg).exists {
    case o: ujson.Obj => o.value.contains("method") && o.value.contains("id")
    case _            => false
  }

  private def wantsStream(ex: HttpExchange, msg: ujson.Value): Boolean =
    header(ex, "Accept").exists(_.toLowerCase.contains("text/event-stream")) && messages(msg).exists {
      case o: ujson.Obj =>
        o.value.get("method").contains(ujson.Str("tools/call")) &&
        o.value.get("params").flatMap(_.objOpt).flatMap(_.get("_meta")).flatMap(_.objOpt).exists(_.contains("progressToken"))
      case _ => false
    }

  private def busyReply(s: Session, msg: ujson.Value): ujson.Value =
    s.server.abandon(msg)
    JsonRpc.error(s.server.requestIds(msg).headOption.getOrElse(ujson.Null), StdioTransport.ServerBusy,
      s"server busy: at most $maxConcurrentCalls concurrent tool calls")

  private def dispatch(ex: HttpExchange, s: Session, msg: ujson.Value): Unit =
    if !hasRequest(msg) then
      // notifications / responses; anything invalid comes back as an error -> 400
      s.server.handle(msg, _ => ()) match
        case Some(err) => respond(ex, 400, Some(err))
        case None      => respond(ex, 202)
    else
      val sink = Option.when(wantsStream(ex, msg))(new SseSink(ex))
      def finish(code: Int, v: ujson.Value): Unit = sink match
        case Some(sse) => sse.finish(v)
        case None      => respond(ex, code, Some(v), if code == 503 then Seq("Retry-After" -> "5") else Nil)
      s.server.accept(msg, sink.map(_.event).getOrElse(_ => ())) match
        case Left(r) => finish(200, r.getOrElse(cancelledReply(msg)))
        case Right(work) =>
          if !calls.tryAcquire() then finish(503, busyReply(s, msg))
          else
            try finish(200, work().getOrElse(cancelledReply(msg)))
            finally calls.release()

  /** A cancelled request still needs an HTTP reply; the client ignores it. */
  private def cancelledReply(msg: ujson.Value): ujson.Value =
    val id = messages(msg).collectFirst { case o: ujson.Obj if o.value.contains("id") => o("id") }.getOrElse(ujson.Null)
    JsonRpc.error(id, JsonRpc.Codes.RequestCancelled, "request cancelled")

  /** SSE response; headers are sent with the first event. */
  private final class SseSink(ex: HttpExchange):
    private val lock = new Object
    private var out: OutputStream = null
    private var broken = false

    val event: ujson.Value => Unit = v => lock.synchronized {
      if !broken then
        try
          if out == null then
            val h = ex.getResponseHeaders
            h.set("Content-Type", "text/event-stream")
            h.set("Cache-Control", "no-cache")
            ex.sendResponseHeaders(200, 0)
            out = ex.getResponseBody
          out.write(s"event: message\ndata: ${JsonRpc.render(v)}\n\n".getBytes(StandardCharsets.UTF_8))
          out.flush()
        catch case NonFatal(_) => broken = true
    }

    def finish(v: ujson.Value): Unit =
      try event(v)
      finally
        lock.synchronized {
          try if out != null then out.close()
          catch case NonFatal(_) => ()
        }
        ex.close()

  // ---- sessions ------------------------------------------------------------------------------------------------

  private def sweep(): Unit =
    val now = System.currentTimeMillis()
    sessions.values().asScala.toSeq.filter(s => s.inFlight.get() == 0 && now - s.lastUsed > idleTtlMillis).foreach { s =>
      if sessions.remove(s.id, s) then log(s"[mcp-http] session ${s.id.take(8)}... expired")
    }

  /** Ensures a free slot by evicting the longest idle session (never one with a call in flight). */
  private def makeRoom(): Boolean = synchronized {
    if sessions.size < maxSessions then true
    else
      sessions.values().asScala.toSeq.filter(_.inFlight.get() == 0).sortBy(_.lastUsed).headOption match
        case Some(victim) =>
          sessions.remove(victim.id, victim)
          true
        case None => false
  }

object HttpTransport:
  val SessionHeader = "Mcp-Session-Id"
  val ProtocolHeader = "MCP-Protocol-Version"
  val DefaultMaxSessions = 64
  val DefaultIdleTtlMillis: Long = 30L * 60 * 1000
  val DefaultMaxBodyBytes: Int = 64 * 1024 * 1024
  val DefaultMaxConcurrentCalls = 32

  private val LoopbackHosts = Set("localhost", "127.0.0.1", "[::1]")

  private def hostPart(hostHeader: String): String =
    val h = hostHeader.trim.toLowerCase
    if h.startsWith("[") then h.takeWhile(_ != ']') + "]"
    else h.takeWhile(_ != ':')

  /** No Origin (non-browser clients) is fine; otherwise exactly http(s)://localhost|127.0.0.1|[::1] with any port. */
  def originAllowed(origin: Option[String]): Boolean = origin match
    case None => true
    case Some(o) =>
      try
        val u = new URI(o)
        val scheme = Option(u.getScheme).map(_.toLowerCase).getOrElse("")
        val host = Option(u.getHost).map(_.toLowerCase).getOrElse("")
        val clean = Option(u.getRawPath).forall(p => p.isEmpty || p == "/") && u.getRawUserInfo == null && u.getRawQuery == null
        (scheme == "http" || scheme == "https") && LoopbackHosts.contains(host) && clean
      catch case NonFatal(_) => false

  def hostAllowed(host: Option[String]): Boolean = host.forall(h => LoopbackHosts.contains(hostPart(h)))

/** Entry point for the GUI: serves MCP over HTTP on 127.0.0.1:`port`, jobs go through the GUI's queue.
  * Image path inputs are limited to the outputs folder plus `SDFX_MCP_ALLOWED_DIRS`.
  */
object McpHttp:
  def start(services: AppServices, port: Int, token: Option[String]): AutoCloseable =
    val tools = new McpTools(services).all
    val t = new HttpTransport(() => new McpServer(tools), port, token)
    services.log(s"[mcp] HTTP server listening on ${t.url}${if token.exists(_.trim.nonEmpty) then " (bearer token required)" else " (no token: any local process can use it)"}")
    t
