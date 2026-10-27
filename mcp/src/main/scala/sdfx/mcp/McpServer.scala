package sdfx.mcp

import sdfx.core.Version
import sdfx.mcp.JsonRpc.*

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{ConcurrentHashMap, CopyOnWriteArrayList}
import scala.util.control.NonFatal

/** Raised by tools for bad arguments or failed operations; reported as a tool result with `isError: true`. */
final class ToolError(message: String) extends Exception(message)

/** What a tool produced. `content` is MCP content blocks; `structured` becomes `structuredContent`. */
final case class ToolResult(content: Seq[ujson.Value], structured: Option[ujson.Value] = None, isError: Boolean = false):
  def toJson: ujson.Value =
    val o = ujson.Obj("content" -> ujson.Arr.from(content), "isError" -> isError)
    structured.foreach(s => o("structuredContent") = s)
    o

object ToolResult:
  def text(s: String): ToolResult = ToolResult(Seq(Content.text(s)))
  def error(s: String): ToolResult = ToolResult(Seq(Content.text(s)), isError = true)

  /** Structured result. Per spec 2025-06-18 the serialized JSON is also returned as a text block (after the
    * optional human readable summary) for clients that ignore `structuredContent`.
    */
  def json(v: ujson.Value, summary: Option[String] = None): ToolResult =
    ToolResult(summary.map(Content.text).toSeq :+ Content.text(ujson.write(v)), Some(v))

object Content:
  def text(s: String): ujson.Value = ujson.Obj("type" -> "text", "text" -> s)
  def image(b64: String, mime: String): ujson.Value = ujson.Obj("type" -> "image", "data" -> b64, "mimeType" -> mime)

/** Per-call context: progress reporting and cancellation hooks. */
trait ToolContext:
  /** Reports progress (sent only when the client gave a progressToken; non-increasing values and anything after
    * the call finished are dropped).
    */
  def progress(progress: Double, total: Option[Double], message: String): Unit
  /** Registers a cleanup run when the client cancels this call (e.g. cancel the queued job). */
  def onCancel(f: () => Unit): Unit
  def isCancelled: Boolean

object ToolContext:
  val none: ToolContext = new ToolContext:
    def progress(progress: Double, total: Option[Double], message: String): Unit = ()
    def onCancel(f: () => Unit): Unit = ()
    def isCancelled: Boolean = false

trait McpTool:
  def name: String
  def title: String = name
  def description: String
  def inputSchema: ujson.Obj
  def outputSchema: Option[ujson.Obj] = None
  def readOnly: Boolean = false
  def destructive: Boolean = false
  def idempotent: Boolean = false
  def call(args: ujson.Obj, ctx: ToolContext): ToolResult

  def descriptor: ujson.Value =
    val o = ujson.Obj("name" -> name, "title" -> title, "description" -> description, "inputSchema" -> inputSchema)
    outputSchema.foreach(s => o("outputSchema") = s)
    o("annotations") = ujson.Obj("title" -> title, "readOnlyHint" -> readOnly, "destructiveHint" -> destructive,
      "idempotentHint" -> idempotent, "openWorldHint" -> false)
    o

final case class ServerInfo(name: String = "stablediffusion-nui", version: String = Version.app, instructions: String = McpServer.DefaultInstructions)

/** Transport-agnostic MCP server (one instance per client session).
  *
  * Transports call [[accept]] on their reader thread, in arrival order: notifications (cancel, initialized) and
  * cheap requests are handled there, `tools/call` is registered (so a later cancel always finds it) and returned as
  * work to run on a worker thread. [[handle]] does both steps synchronously.
  */
final class McpServer(tools: Seq[McpTool], val info: ServerInfo = ServerInfo()):
  import McpServer.*

  private val byName = tools.map(t => t.name -> t).toMap
  @volatile private var _protocolVersion: Option[String] = None
  @volatile private var _initialized = false

  private final class Active:
    val cancelled = new AtomicBoolean(false)
    @volatile var finished = false
    val hooks = new CopyOnWriteArrayList[() => Unit]()
    def cancel(): Unit =
      if cancelled.compareAndSet(false, true) then
        hooks.forEach(h =>
          try h()
          catch case NonFatal(_) => ()
        )

  private val active = new ConcurrentHashMap[String, Active]()
  /** Recently cancelled ids that were not (yet) active: a call registered later with that id starts cancelled. */
  private val cancelledIds = new ConcurrentHashMap[String, java.lang.Long]()

  def protocolVersion: Option[String] = _protocolVersion
  def initialized: Boolean = _initialized
  def activeCalls: Int = active.size

  /** Left = finished (the reply, if any); Right = blocking work to run on a worker thread. */
  type Accepted = Either[Option[ujson.Value], () => Option[ujson.Value]]

  /** Handles one message or a batch synchronously. None for notifications, responses and cancelled calls. */
  def handle(msg: ujson.Value, emit: ujson.Value => Unit): Option[ujson.Value] = accept(msg, emit) match
    case Left(r)     => r
    case Right(work) => work()

  def accept(msg: ujson.Value, emit: ujson.Value => Unit): Accepted = msg match
    case ujson.Arr(items) =>
      if items.isEmpty then Left(Some(error(ujson.Null, Codes.InvalidRequest, "empty batch")))
      else
        val parts = items.toSeq.map(m => acceptOne(m, emit))
        val done = parts.collect { case Left(Some(r)) => r }
        val works = parts.collect { case Right(w) => w }
        if works.isEmpty then Left(Option.when(done.nonEmpty)(ujson.Arr.from(done)))
        else
          Right { () =>
            val rs = done ++ works.flatMap(w => w())
            Option.when(rs.nonEmpty)(ujson.Arr.from(rs))
          }
    case other => acceptOne(other, emit)

  /** Ids of requests in `msg` (for error replies when the work cannot be scheduled). */
  def requestIds(msg: ujson.Value): Seq[ujson.Value] =
    val items = msg match
      case ujson.Arr(a) => a.toSeq
      case o            => Seq(o)
    items.flatMap(_.objOpt).filter(o => o.contains("method")).flatMap(_.get("id")).filterNot(_ == ujson.Null)

  /** Releases registrations made by [[accept]] for work that will never run (e.g. the pool rejected it). */
  def abandon(msg: ujson.Value): Unit = requestIds(msg).foreach(id => active.remove(idKey(id)))

  private def acceptOne(msg: ujson.Value, emit: ujson.Value => Unit): Accepted =
    parse(msg) match
      case Left(err) => Left(Some(err))
      case Right(Notification(method, params)) =>
        try onNotification(method, params)
        catch case NonFatal(e) => System.err.println(s"[mcp] notification $method failed: $e")
        Left(None)
      case Right(_: Response) => Left(None) // this server never sends requests to the client
      case Right(Request(id, "tools/call", params)) =>
        try
          val (tool, args, token) = prepareCall(obj(params))
          val k = idKey(id)
          val act = new Active
          if active.putIfAbsent(k, act) != null then
            Left(Some(error(id, Codes.InvalidRequest, s"request id ${ujson.write(id)} is already in use")))
          else
            val ts = cancelledIds.remove(k)
            if ts != null && System.currentTimeMillis() - ts <= CancelMemoryMillis then act.cancelled.set(true)
            Right(() => runCall(id, k, act, tool, args, token, emit))
        catch case e: RpcError => Left(Some(error(id, e.code, e.message, e.data)))
      case Right(Request(id, method, params)) =>
        Left(
          try Some(result(id, dispatch(method, params)))
          catch
            case e: RpcError => Some(error(id, e.code, e.message, e.data))
            case e: Throwable if !e.isInstanceOf[InterruptedException] =>
              System.err.println(s"[mcp] $method failed: $e")
              Some(error(id, Codes.InternalError, "internal error"))
        )

  private def onNotification(method: String, params: ujson.Value): Unit = method match
    case "notifications/initialized" => _initialized = true
    case "notifications/cancelled" =>
      params.objOpt.flatMap(_.get("requestId")).foreach { rid =>
        val k = idKey(rid)
        Option(active.get(k)) match
          case Some(a) => a.cancel()
          case None =>
            val now = System.currentTimeMillis()
            cancelledIds.entrySet().removeIf(e => now - e.getValue > CancelMemoryMillis)
            if cancelledIds.size < 1000 then cancelledIds.put(k, now)
      }
    case _ => () // unknown notifications are ignored

  private def obj(params: ujson.Value): ujson.Obj = params match
    case o: ujson.Obj => o
    case _            => throw RpcError(Codes.InvalidParams, "params must be an object")

  private def dispatch(method: String, params: ujson.Value): ujson.Value =
    method match
      case "initialize" => initialize(obj(params))
      case "ping"       => ujson.Obj()
      case "tools/list" => ujson.Obj("tools" -> ujson.Arr.from(tools.map(_.descriptor)))
      case _            => throw RpcError(Codes.MethodNotFound, s"Method not found: $method")

  private def initialize(p: ujson.Obj): ujson.Value =
    val requested = p.value.get("protocolVersion").flatMap(_.strOpt)
    val v = requested.filter(SupportedVersions.contains).getOrElse(LatestVersion)
    _protocolVersion = Some(v)
    ujson.Obj(
      "protocolVersion" -> v,
      "capabilities" -> ujson.Obj("tools" -> ujson.Obj("listChanged" -> false)),
      "serverInfo" -> ujson.Obj("name" -> info.name, "title" -> "Stable Diffusion NUI", "version" -> info.version),
      "instructions" -> info.instructions
    )

  private def prepareCall(p: ujson.Obj): (McpTool, ujson.Obj, Option[ujson.Value]) =
    val name = p.value.get("name").flatMap(_.strOpt).getOrElse(throw RpcError(Codes.InvalidParams, "tools/call requires a tool name"))
    val tool = byName.getOrElse(name, throw RpcError(Codes.InvalidParams, s"Unknown tool: $name"))
    val args = p.value.get("arguments") match
      case None | Some(ujson.Null) => ujson.Obj()
      case Some(o: ujson.Obj)      => o
      case Some(_)                 => throw RpcError(Codes.InvalidParams, "arguments must be an object")
    val token = p.value.get("_meta").flatMap(_.objOpt).flatMap(_.get("progressToken")).filter {
      case _: ujson.Str | _: ujson.Num => true
      case _                           => false
    }
    (tool, args, token)

  private def runCall(id: ujson.Value, k: String, act: Active, tool: McpTool, args: ujson.Obj, token: Option[ujson.Value],
      emit: ujson.Value => Unit): Option[ujson.Value] =
    val ctx = new ToolContext:
      private var last = Double.NegativeInfinity
      def progress(progress: Double, total: Option[Double], message: String): Unit =
        token.foreach { t =>
          // check and emit atomically: events arrive from several threads (submitter, job worker, engine reader)
          act.synchronized {
            if progress > last && !act.cancelled.get() && !act.finished then
              last = progress
              val o = ujson.Obj("progressToken" -> t, "progress" -> progress)
              total.foreach(x => o("total") = x)
              if message.nonEmpty then o("message") = message
              try emit(notification("notifications/progress", o))
              catch case NonFatal(_) => ()
          }
        }
      def onCancel(f: () => Unit): Unit =
        act.hooks.add(f)
        if act.cancelled.get() then
          try f()
          catch case NonFatal(_) => ()
      def isCancelled: Boolean = act.cancelled.get()
    try
      val r =
        try tool.call(args, ctx).toJson
        catch
          case e: ToolError => ToolResult.error(e.getMessage).toJson
          case e: RpcError  => throw e
          case NonFatal(e) =>
            System.err.println(s"[mcp] tool ${tool.name} failed: $e")
            ToolResult.error(s"${tool.name} failed: ${Option(e.getMessage).getOrElse(e.toString)}").toJson
      if act.cancelled.get() then None else Some(result(id, r))
    catch
      case e: RpcError => Some(error(id, e.code, e.message, e.data))
      case e: Throwable if !e.isInstanceOf[InterruptedException] =>
        // OutOfMemoryError / StackOverflowError: still answer so the client does not hang
        System.err.println(s"[mcp] tool ${tool.name} failed fatally: $e")
        Some(error(id, Codes.InternalError, s"internal error (${e.getClass.getSimpleName})"))
    finally
      act.synchronized { act.finished = true }
      active.remove(k, act)

object McpServer:
  val SupportedVersions: Seq[String] = Seq("2025-06-18", "2025-03-26", "2024-11-05")
  val LatestVersion: String = SupportedVersions.head
  val CancelMemoryMillis: Long = 60_000

  val DefaultInstructions: String =
    """Stable Diffusion NUI: local image generation (SD1.5 / SDXL checkpoints, LoRA, hires fix, img2img / inpaint, X/Y/Z grids).
      |Typical flow: get_status -> list_models (set_model if needed) -> txt2img with a descriptive prompt.
      |Images are saved as PNG with A1111-style "parameters" infotext under the app's outputs directory. Results contain
      |downscaled previews (max_image_side) plus the full-size file path, seed and infotext; reuse a seed to reproduce or vary.
      |LoRAs: pass loras=[{name, weight}] (see list_loras) or <lora:name:weight> in the prompt.
      |img2img takes init_image as base64 / a data URI, or a file path inside the outputs folder (or another allowed folder).
      |If tool calls time out on long jobs, use txt2img_async / img2img_async and poll get_job(job_id, wait_seconds).
      |Jobs share one queue with the desktop GUI; interrupt stops the running job. list_recent / get_image browse outputs.
      |Scripts (X/Y/Z plot, prompt matrix, loopback, SD upscale...): list_scripts, then run_script.""".stripMargin
