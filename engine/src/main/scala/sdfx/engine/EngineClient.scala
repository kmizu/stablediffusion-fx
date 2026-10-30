package sdfx.engine

import sdfx.core.json.SnakePickle
import sdfx.core.protocol.*
import sdfx.core.payload.RequestPayload

import scala.concurrent.{ExecutionContext, Future}
import scala.util.control.NonFatal

/** Frozen native protocol params, not an envelope or evidence that the worker executed the request. */
private[engine] final case class GenerationBody(requestType: String, bodyJson: String, capturedAtMillis: Long)

/** A running txt2img / img2img request. */
final class GenerationHandle(val result: Future[GenerationResult], interruptFn: () => Unit):
  def interrupt(): Unit = interruptFn()

/** One Temporal Lab operation and its scoped cancellation. */
final class TemporalLabHandle(val result: Future[TemporalLabResult], interruptFn: () => Future[InterruptionDecision]):
  def interrupt(): Future[InterruptionDecision] = interruptFn()

/** One VAE Tools operation and its scoped cancellation. */
final class VaeToolsHandle(val result: Future[ujson.Value], interruptFn: () => Future[InterruptionDecision]):
  def interrupt(): Future[InterruptionDecision] = interruptFn()

object GenerationMode:
  val Txt2Img: String = Protocol.Types.Txt2Img
  val Img2Img: String = Protocol.Types.Img2Img

/** Typed facade over [[EngineSupervisor]]. All calls are asynchronous. */
final class EngineClient(val supervisor: EngineSupervisor):
  private given ExecutionContext = ExecutionContext.parasitic

  private def call[T: SnakePickle.Reader](tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => (),
                                         owner: Option[String] = None,
                                         imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): Future[T] =
    supervisor.requestOwned(tpe, params, onEvent, owner, imageSecret).map(v => Json.fromJs[T](v))

  def raw(tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    supervisor.request(tpe, params, onEvent)

  private[engine] def rawOwned(tpe: String, params: ujson.Value, owner: Option[String],
                               onEvent: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    supervisor.requestOwned(tpe, params, onEvent, owner)

  def ping(): Future[ujson.Value] = raw(Protocol.Types.Ping)

  def listModels(dirs: Seq[String]): Future[Seq[ModelEntry]] =
    raw(Protocol.Types.ListModels, ujson.Obj("dirs" -> ujson.Arr.from(dirs.map(ujson.Str(_)))))
      .map(v => Json.fromJs[Seq[ModelEntry]](v("models")))

  /** `kind`: vae, lora, controlnet, embeddings, upscalers. */
  def listFiles(kind: String, dirs: Seq[String]): Future[Seq[FileEntry]] =
    raw(Protocol.Types.ListFiles, ujson.Obj("kind" -> kind, "dirs" -> ujson.Arr.from(dirs.map(ujson.Str(_)))))
      .map(v => Json.fromJs[Seq[FileEntry]](v("files")))

  def listSamplers(): Future[SamplerList] = call[SamplerList](Protocol.Types.ListSamplers)

  def loadModel(req: LoadModelRequest): Future[LoadedModel] = call[LoadedModel](Protocol.Types.LoadModel, Json.toJs(req))
  private[engine] def loadModelOwned(req: LoadModelRequest, owner: Option[String]): Future[LoadedModel] =
    call[LoadedModel](Protocol.Types.LoadModel, Json.toJs(req), owner = owner)

  /** `path = None` restores the checkpoint's own VAE. */
  def setVae(path: Option[String]): Future[LoadedModel] =
    call[LoadedModel](Protocol.Types.SetVae, ujson.Obj("path" -> path.fold(ujson.Null: ujson.Value)(ujson.Str(_))))
  private[engine] def setVaeOwned(path: Option[String], owner: Option[String]): Future[LoadedModel] =
    call[LoadedModel](Protocol.Types.SetVae, ujson.Obj("path" -> path.fold(ujson.Null: ujson.Value)(ujson.Str(_))), owner = owner)

  def generate(req: GenerationRequest, mode: String, listener: EngineEvent => Unit = _ => ()): GenerationHandle =
    generateOwned(req, mode, listener, None)

  private[engine] def generateOwned(req: GenerationRequest, mode: String, listener: EngineEvent => Unit,
                                    owner: Option[String], observer: GenerationBody => Unit = _ => (),
                                    beforeDispatch: () => Unit = () => (),
                                    imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): GenerationHandle =
    require(mode == GenerationMode.Txt2Img || mode == GenerationMode.Img2Img, s"bad mode $mode")
    val requestOwner = owner.orElse(Some(java.util.UUID.randomUUID().toString))
    // Serialize the actual typed request exactly once. Both freezing and private transport parsing are
    // required conversion work and retain ordinary failure semantics; only observation is optional.
    val bodyJson = RequestPayload.freeze(Json.toJs(req))
    val transportBody = ujson.read(bodyJson)
    val body = GenerationBody(mode, bodyJson, System.currentTimeMillis())
    try observer(body)
    catch
      // This is optional diagnostic work, not the job's owner/listener cancellation signal.
      case error: InterruptedException =>
        System.err.println(s"[sdfx] Generation payload observer failed: ${error.getClass.getSimpleName}")
      case NonFatal(error) =>
        System.err.println(s"[sdfx] Generation payload observer failed: ${error.getClass.getSimpleName}")
    // Required admission check: optional diagnostic work may have taken long enough for the
    // actual job's cancellation flag to change. Its failure must escape the observer isolation.
    beforeDispatch()
    new GenerationHandle(call[GenerationResult](mode, transportBody, listener, requestOwner, imageSecret),
      () => supervisor.interruptOwned(requestOwner))

  def upscale(req: UpscaleRequest): Future[ImagePathResult] = call[ImagePathResult](Protocol.Types.Upscale, Json.toJs(req))

  /** A unique owner prevents stale controls from interrupting another generation or window. */
  def temporalLab(req: TemporalLabRequest, listener: EngineEvent => Unit = _ => (),
      imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): TemporalLabHandle =
    val owner = java.util.UUID.randomUUID().toString
    val gate = new TemporalLabCancelGate(owner, id => supervisor.captureInterrupt(Some(id)))
    val result = supervisor.requestTemporal(req.params, listener, owner,
      () => gate.beforeDispatch(), () => gate.dispatched(), imageSecret)
      .map(Json.fromJs[TemporalLabResult](_))
    result.onComplete(_ => gate.completed())(using ExecutionContext.parasitic)
    new TemporalLabHandle(result, () => gate.cancel())

  def preprocess(req: PreprocessRequest): Future[ImagePathResult] = call[ImagePathResult](Protocol.Types.Preprocess, Json.toJs(req))

  def memory(): Future[MemoryInfo] = call[MemoryInfo](Protocol.Types.Memory)

  def systemStatus(): Future[ujson.Value] = raw(Protocol.Types.SystemStatus)
  def vaeTools(params: ujson.Obj, listener: EngineEvent => Unit = _ => (),
      imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): Future[ujson.Value] =
    vaeToolsOwned(params, listener, imageSecret).result
  def vaeToolsOwned(params: ujson.Obj, listener: EngineEvent => Unit = _ => (),
      imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): VaeToolsHandle =
    val owner = java.util.UUID.randomUUID().toString
    val gate = new TemporalLabCancelGate(owner, id => supervisor.captureInterrupt(Some(id), restartOnTimeout = false))
    val result = supervisor.requestExtras(Protocol.Types.VaeTools, params, listener, owner,
      () => gate.beforeDispatch(), () => gate.dispatched(), imageSecret = imageSecret)
    result.onComplete(_ => gate.completed())(using ExecutionContext.parasitic)
    new VaeToolsHandle(result, () => gate.cancel())
  def systemInfo(dirs: ujson.Obj = ujson.Obj(), hashes: Boolean = false): Future[ujson.Value] =
    raw(Protocol.Types.SystemInfo, ujson.Obj("dirs" -> dirs, "hashes" -> hashes))
  def systemBenchmark(params: ujson.Obj, listener: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    raw(Protocol.Types.SystemBenchmark, params, listener)

  /** Literal CLIP BPE inspection on the currently loaded model; SDXL returns both tokenizers. */
  def tokenize(text: String, inputMode: String = "text"): Future[TokenizeResult] =
    call[TokenizeResult](Protocol.Types.Tokenize, ujson.Obj("text" -> text, "input_mode" -> inputMode))

  /** A single Extras batch owns its requests and never interrupts other jobs. */
  def extrasBatch(listener: EngineEvent => Unit = _ => ()): ExtrasBatchHandle =
    val owner = java.util.UUID.randomUUID().toString
    val control = new ExtrasBatchControl(() => supervisor.interruptAfterStartupOwned(owner), () => interruptionProtection())
    extrasBatchOwned(owner, control, listener)

  private[engine] def extrasBatchOwned(owner: String, control: ExtrasBatchControl,
                                      listener: EngineEvent => Unit = _ => (),
                                      onProcess: EngineProcess => Unit = _ => (),
                                      restartOnTimeout: Boolean = true,
                                      imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): ExtrasBatchHandle =
    new ExtrasBatchHandle(control, (kind, params) =>
      val gate = new TemporalLabCancelGate(owner, id => supervisor.captureInterrupt(Some(id), restartOnTimeout))
      val result = supervisor.requestExtras(kind, params, listener, owner,
        () => control.attach(gate, finalStage = kind == "batch_interrogate_execute"), () => gate.dispatched(), onProcess, () => !control.cancellationAccepted,
        if kind == "batch_interrogate_execute" then imageSecret else None)
      result.onComplete { _ => gate.completed(); control.detach(gate) }(using ExecutionContext.parasitic)
      result)

  /** Internal WD v3 partial job: no model download, one owner across both worker requests. */
  def wdBatch(prepare: WdBatchPrepareRequest, listener: EngineEvent => Unit = _ => ()): WdBatchHandle =
    val owner = java.util.UUID.randomUUID().toString
    val control = new ExtrasBatchControl(() => supervisor.interruptAfterStartupOwned(owner, restartOnTimeout = false), () => interruptionProtection())
    wdBatchOwned(prepare, owner, control, listener)

  private[engine] def wdBatchOwned(prepare: WdBatchPrepareRequest, owner: String, control: ExtrasBatchControl,
                                   listener: EngineEvent => Unit = _ => (),
                                   imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): WdBatchHandle =
    val process = new java.util.concurrent.atomic.AtomicReference[EngineProcess]()
    val transport = extrasBatchOwned(owner, control, listener, p => process.set(p), restartOnTimeout = false, imageSecret = imageSecret)
    new WdBatchHandle(WdBatchFlow.run(transport, Json.toJs(prepare)).map(Json.fromJs[WdBatchResult](_)), transport, process)

  def interruptionProtection(enabled: Option[Boolean] = None): Future[InterruptionProtection] =
    supervisor.readerControl(Protocol.Types.InterruptionProtection,
      enabled.fold(ujson.Obj())(v => ujson.Obj("enabled" -> v))).map(Json.fromJs[InterruptionProtection](_))

  def requestInterrupt(): Future[InterruptionDecision] = supervisor.requestInterrupt()

  def interrupt(): Unit = supervisor.interrupt()

  def unload(): Future[Unit] = raw(Protocol.Types.Unload).map(_ => ())
