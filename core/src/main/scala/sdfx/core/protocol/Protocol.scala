package sdfx.core.protocol

import sdfx.core.json.SnakePickle
import sdfx.core.json.SnakePickle.{ReadWriter => RW, *}
import sdfx.core.model.GenerationParams

/** Engine (Python) <-> app JSON Lines protocol, version [[sdfx.core.Version.protocol]].
  *
  * Requests:  {"id": "r1", "type": "txt2img", "params": {...}}
  * Result:    {"id": "r1", "type": "result", "result": {...}}
  * Error:     {"id": "r1", "type": "error", "code": "...", "message": "...", "detail": "..."}
  * Event:     {"id": "r1", "type": "event", "event": "progress" | "preview" | "log", ...}
  * Hello:     {"id": null, "type": "hello", ...}   first line printed by the engine
  */
object Protocol:
  object Types:
    val Ping = "ping"
    val ListModels = "list_models"
    val LoadModel = "load_model"
    val SetVae = "set_vae"
    val ListFiles = "list_files"
    val ListSamplers = "list_samplers"
    val Txt2Img = "txt2img"
    val Img2Img = "img2img"
    val Preprocess = "preprocess"
    val Upscale = "upscale"
    val Interrupt = "interrupt"
    val Unload = "unload"
    val Memory = "memory"
    val Tokenize = "tokenize"
    val SystemInfo = "system_info"
    val VaeTools = "vae_tools"
    val SystemStatus = "system_status"
    val SystemBenchmark = "system_benchmark"
    val Shutdown = "shutdown"

  object ErrorCodes:
    val InvalidParams = "invalid_params"
    val ModelLoadFailed = "model_load_failed"
    val NoModel = "no_model"
    val Oom = "oom"
    val Interrupted = "interrupted"
    val Unsupported = "unsupported"
    val Internal = "internal"

case class EngineHello(
    protocol: Int,
    engineVersion: String,
    python: String = "",
    torch: Option[String] = None,
    diffusers: Option[String] = None,
    cuda: Boolean = false,
    device: String = "cpu",
    vramMb: Int = 0,
    dryRun: Boolean = false,
    features: Seq[String] = Nil
) derives RW

case class ModelEntry(name: String, path: String, kind: String, sizeMb: Double = 0) derives RW
case class FileEntry(name: String, path: String, sizeMb: Double = 0) derives RW

case class LoadModelRequest(path: String, dtype: String = "fp16", vramMode: String = "gpu", vaePath: Option[String] = None, vaeTiling: Boolean = false) derives RW
case class LoadedModel(name: String, path: String, kind: String, hash: String, loadedMs: Long = 0, vaeName: Option[String] = None, vaeHash: Option[String] = None) derives RW

case class SamplerInfo(name: String, supported: Boolean = true, defaultScheduler: String = "Uniform") derives RW
case class SamplerList(samplers: Seq[SamplerInfo], schedulers: Seq[String], upscalers: Seq[String] = Nil, preprocessors: Seq[String] = Nil) derives RW

/** Generation request sent to the engine. Names in `params.loras` / `controlnet.model` are resolved to paths here. */
case class GenerationRequest(
    params: GenerationParams,
    outputDir: String,
    loraPaths: Map[String, String] = Map.empty,
    embeddingPaths: Seq[String] = Nil,
    controlnetPaths: Map[String, String] = Map.empty,
    upscalerPaths: Map[String, String] = Map.empty,
    previewEvery: Int = 2,
    previewMethod: String = "approx",
    lbwPresets: Map[String, String] = Map.empty,
    systemInfoFields: Seq[String] = Nil,
    vaeRecoveryDir: String = ""
) derives RW

case class GeneratedImage(path: String, seed: Long, subseed: Long = -1L, width: Int = 0, height: Int = 0) derives RW

/** Persistent auxiliary output; not an engine temp file or a final-image gallery member. */
case class IntermediateArtifact(path: String, `type`: String = "image") derives RW

/** Persistent effective-weight JSON; diagnostic plot is carried in images like upstream. */
case class ComposableLoraTrace(path: String) derives RW

case class GenerationResult(
    images: Seq[GeneratedImage],
    timingS: Double = 0,
    modelName: Option[String] = None,
    modelHash: Option[String] = None,
    vaeName: Option[String] = None,
    vaeHash: Option[String] = None,
    loraHashes: Map[String, String] = Map.empty,
    scheduler: Option[String] = None,
    warnings: Seq[String] = Nil,
    intermediateArtifacts: Seq[IntermediateArtifact] = Nil,
    composableLoraTrace: Option[ComposableLoraTrace] = None,
    systemInfo: Option[ujson.Value] = None,
    vaeRecovery: Seq[ujson.Value] = Nil
) derives RW

case class UpscaleRequest(imagePath: String, upscaler: String, scale: Double, outputDir: String, upscalerPaths: Map[String, String] = Map.empty) derives RW
case class PreprocessRequest(preprocessor: String, imagePath: String, outputDir: String, args: Map[String, Double] = Map.empty) derives RW
case class ImagePathResult(path: String) derives RW
case class MemoryInfo(allocatedMb: Double = 0, reservedMb: Double = 0, totalMb: Double = 0, ramUsedMb: Double = 0) derives RW

case class TokenPiece(index: Int, id: Int, piece: String, text: String, bytesHex: String, special: Boolean) derives RW
case class TokenGroup(ids: Seq[Int], text: String) derives RW
case class TokenStream(name: String, source: String, count: Int, ids: Seq[Int], contentCapacity: Int,
                       bosId: Int, eosId: Int, padId: Int, decoded: String,
                       tokens: Seq[TokenPiece], groups: Seq[TokenGroup]) derives RW
case class TokenizeResult(inputMode: String, streams: Seq[TokenStream]) derives RW

sealed trait EngineEvent
object EngineEvent:
  case class Progress(step: Int, total: Int, pass: String = "base", etaS: Double = 0, image: Int = 0, images: Int = 1) extends EngineEvent
  case class Preview(jpegB64: String, step: Int) extends EngineEvent
  case class Log(level: String, message: String) extends EngineEvent

  /** Lenient: non-objects, unknown events and wrongly typed / null fields never throw. */
  def parse(js: ujson.Value): Option[EngineEvent] = js match
    case o: ujson.Obj =>
      def num(k: String) = o.value.get(k).collect { case ujson.Num(n) => n }
      def str(k: String) = o.value.get(k).collect { case ujson.Str(s) => s }
      str("event") match
        case Some("progress") =>
          Some(Progress(num("step").fold(0)(_.toInt), num("total").fold(0)(_.toInt), str("pass").getOrElse("base"),
            num("eta_s").getOrElse(0.0), num("image").fold(0)(_.toInt), num("images").fold(1)(_.toInt)))
        case Some("preview") => str("jpeg_b64").map(v => Preview(v, num("step").fold(0)(_.toInt)))
        case Some("log")     => Some(Log(str("level").getOrElse("info"), str("message").getOrElse("")))
        case _               => None
    case _ => None

final case class EngineError(code: String, message: String, detail: String = "") extends Exception(s"[$code] $message"):
  def isInterrupted: Boolean = code == Protocol.ErrorCodes.Interrupted

object Json:
  def toJs[T: SnakePickle.Writer](t: T): ujson.Value = SnakePickle.writeJs(t)
  def fromJs[T: SnakePickle.Reader](v: ujson.Value): T = SnakePickle.read[T](v)
