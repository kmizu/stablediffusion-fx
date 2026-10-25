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

  private def call[T: SnakePickle.Reader](tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => ()): Future[T] =
    supervisor.request(tpe, params, onEvent).map(v => Json.fromJs[T](v))

  def raw(tpe: String, params: ujson.Value = ujson.Obj(), onEvent: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    supervisor.request(tpe, params, onEvent)

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

  /** `path = None` restores the checkpoint's own VAE. */
  def setVae(path: Option[String]): Future[LoadedModel] =
    call[LoadedModel](Protocol.Types.SetVae, ujson.Obj("path" -> path.fold(ujson.Null: ujson.Value)(ujson.Str(_))))

  def generate(req: GenerationRequest, mode: String, listener: EngineEvent => Unit = _ => ()): GenerationHandle =
    require(mode == GenerationMode.Txt2Img || mode == GenerationMode.Img2Img, s"bad mode $mode")
    new GenerationHandle(call[GenerationResult](mode, Json.toJs(req), listener), () => supervisor.interrupt())

  def upscale(req: UpscaleRequest): Future[ImagePathResult] = call[ImagePathResult](Protocol.Types.Upscale, Json.toJs(req))

  def preprocess(req: PreprocessRequest): Future[ImagePathResult] = call[ImagePathResult](Protocol.Types.Preprocess, Json.toJs(req))

  def memory(): Future[MemoryInfo] = call[MemoryInfo](Protocol.Types.Memory)

  def interrupt(): Unit = supervisor.interrupt()

  def unload(): Future[Unit] = raw(Protocol.Types.Unload).map(_ => ())
