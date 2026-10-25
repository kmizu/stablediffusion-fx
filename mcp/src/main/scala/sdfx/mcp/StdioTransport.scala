package sdfx.mcp

import java.io.{BufferedInputStream, ByteArrayOutputStream, InputStream, OutputStream}
import java.nio.charset.StandardCharsets
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{RejectedExecutionException, SynchronousQueue, ThreadPoolExecutor, TimeUnit}
import scala.util.control.NonFatal

/** MCP over stdio: newline-delimited JSON-RPC.
  *
  * Messages are accepted on the reader thread in arrival order (so `notifications/cancelled` always sees the call it
  * targets); only blocking `tools/call` work runs on a bounded worker pool. Writes are serialized, one flushed line
  * each. Lines longer than `maxLineBytes` are rejected without being buffered.
  */
final class StdioTransport(
    in: InputStream,
    out: OutputStream,
    server: McpServer,
    log: String => Unit = s => System.err.println(s),
    maxWorkers: Int = StdioTransport.DefaultMaxWorkers,
    maxLineBytes: Int = StdioTransport.DefaultMaxLineBytes
):
  private val lock = new Object
  private val threadNo = new AtomicInteger(0)
  private val pool = new ThreadPoolExecutor(0, maxWorkers, 30, TimeUnit.SECONDS, new SynchronousQueue[Runnable](), { (r: Runnable) =>
    val t = new Thread(r, s"mcp-stdio-${threadNo.incrementAndGet()}")
    t.setDaemon(true)
    t
  })
  @volatile private var closed = false

  def send(msg: ujson.Value): Unit = lock.synchronized {
    if !closed then
      try
        out.write((JsonRpc.render(msg) + "\n").getBytes(StandardCharsets.UTF_8))
        out.flush()
      catch case NonFatal(e) => log(s"[mcp] cannot write to stdout: $e")
  }

  /** Next line (without the terminator), Left(()) for an oversized line (skipped), None at EOF. */
  private def readLine(bin: InputStream): Option[Either[Unit, String]] =
    val buf = new ByteArrayOutputStream(1024)
    var tooLong = false
    var c = bin.read()
    if c < 0 then None
    else
      while c >= 0 && c != '\n' do
        if !tooLong then
          if buf.size() >= maxLineBytes then
            tooLong = true
            buf.reset()
          else buf.write(c)
        c = bin.read()
      if tooLong then Some(Left(()))
      else
        val s = buf.toString(StandardCharsets.UTF_8)
        Some(Right(if s.endsWith("\r") then s.dropRight(1) else s))

  private def reply(r: Option[ujson.Value]): Unit = r.foreach(send)

  private def onMessage(msg: ujson.Value): Unit =
    server.accept(msg, send) match
      case Left(r) => reply(r)
      case Right(work) =>
        try
          pool.execute { () =>
            try reply(work())
            catch
              case e: Throwable =>
                log(s"[mcp] message handling failed: $e")
                server.requestIds(msg).foreach(id => send(JsonRpc.error(id, JsonRpc.Codes.InternalError, "internal error")))
          }
        catch
          case _: RejectedExecutionException =>
            server.abandon(msg)
            server.requestIds(msg).foreach(id => send(JsonRpc.error(id, StdioTransport.ServerBusy, s"server busy: at most $maxWorkers concurrent tool calls")))

  /** Reads until EOF; then waits up to `drainMillis` for in-flight calls before returning. */
  def run(drainMillis: Long = 2000): Unit =
    val bin = new BufferedInputStream(in, 1 << 16)
    try
      var next = readLine(bin)
      while next.isDefined do
        next.get match
          case Left(_) =>
            send(JsonRpc.error(ujson.Null, JsonRpc.Codes.InvalidRequest, s"message larger than $maxLineBytes bytes"))
          case Right(line) =>
            val l = line.trim
            if l.nonEmpty then
              JsonRpc.parseText(l) match
                case Left(err) => send(JsonRpc.parseError(err))
                case Right(msg) =>
                  try onMessage(msg)
                  catch case e: Throwable if !e.isInstanceOf[InterruptedException] => log(s"[mcp] message dropped: $e")
        next = readLine(bin)
    catch case NonFatal(e) => log(s"[mcp] stdin closed: $e")
    finally
      pool.shutdown()
      pool.awaitTermination(drainMillis, TimeUnit.MILLISECONDS)
      lock.synchronized { closed = true }
      pool.shutdownNow()

object StdioTransport:
  val DefaultMaxWorkers = 32
  val DefaultMaxLineBytes: Int = 64 * 1024 * 1024
  /** JSON-RPC implementation-defined server error. */
  val ServerBusy = -32000
