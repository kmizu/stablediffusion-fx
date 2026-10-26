package sdfx.engine

import sdfx.core.json.SnakePickle
import sdfx.core.protocol.*

import scala.concurrent.{ExecutionContext, Future}

/** A running txt2img / img2img request. */
final class GenerationHandle(val result: Future[GenerationResult], interruptFn: () => Unit):
  def interrupt(): Unit = interruptFn()

object GenerationMode:
  val Txt2Img: String = Protocol.Types.Txt2Img
  val Img2Img: String = Protocol.Types.Img2Img

/** Typed facade over [[EngineSupervisor]]. All calls are asynchronous. */
final class EngineClient(val supervisor: EngineSupervisor):
  private given ExecutionContext = ExecutionContext.parasitic

  private def call[T: SnakePickle.Reader](tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => (),
                                         owner: Option[String] = None): Future[T] =
    supervisor.requestOwned(tpe, params, onEvent, owner).map(v => Json.fromJs[T](v))

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
                                    owner: Option[String]): GenerationHandle =
    require(mode == GenerationMode.Txt2Img || mode == GenerationMode.Img2Img, s"bad mode $mode")
    val requestOwner = owner.orElse(Some(java.util.UUID.randomUUID().toString))
    new GenerationHandle(call[GenerationResult](mode, Json.toJs(req), listener, requestOwner),
      () => supervisor.interruptOwned(requestOwner))

  def upscale(req: UpscaleRequest): Future[ImagePathResult] = call[ImagePathResult](Protocol.Types.Upscale, Json.toJs(req))

  def preprocess(req: PreprocessRequest): Future[ImagePathResult] = call[ImagePathResult](Protocol.Types.Preprocess, Json.toJs(req))

  def memory(): Future[MemoryInfo] = call[MemoryInfo](Protocol.Types.Memory)

  def systemStatus(): Future[ujson.Value] = raw(Protocol.Types.SystemStatus)
  def vaeTools(params: ujson.Obj, listener: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    raw(Protocol.Types.VaeTools, params, listener)
  def systemInfo(dirs: ujson.Obj = ujson.Obj(), hashes: Boolean = false): Future[ujson.Value] =
    raw(Protocol.Types.SystemInfo, ujson.Obj("dirs" -> dirs, "hashes" -> hashes))
  def systemBenchmark(params: ujson.Obj, listener: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    raw(Protocol.Types.SystemBenchmark, params, listener)

  /** Literal CLIP BPE inspection on the currently loaded model; SDXL returns both tokenizers. */
  def tokenize(text: String, inputMode: String = "text"): Future[TokenizeResult] =
    call[TokenizeResult](Protocol.Types.Tokenize, ujson.Obj("text" -> text, "input_mode" -> inputMode))

  def interruptionProtection(enabled: Option[Boolean] = None): Future[InterruptionProtection] =
    supervisor.readerControl(Protocol.Types.InterruptionProtection,
      enabled.fold(ujson.Obj())(v => ujson.Obj("enabled" -> v))).map(Json.fromJs[InterruptionProtection](_))

  def requestInterrupt(): Future[InterruptionDecision] = supervisor.requestInterrupt()

  def interrupt(): Unit = supervisor.interrupt()

  def unload(): Future[Unit] = raw(Protocol.Types.Unload).map(_ => ())
