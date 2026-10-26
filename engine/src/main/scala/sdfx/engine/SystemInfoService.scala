package sdfx.engine

import sdfx.core.config.AtomicFiles
import java.nio.file.{Files, Path}
import scala.concurrent.{ExecutionContext, Future}

/** Native runtime/catalog context, plus local benchmark history. No export or transmission on refresh. */
final class SystemInfoService(services: AppServices):
  private given ExecutionContext = ExecutionContext.parasitic
  val historyFile: Path = services.home.configDir.resolve("system-benchmarks.json")
  private def dirs: ujson.Obj =
    val d = services.settings.modelDirs
    def paths(xs: Seq[String]) = ujson.Arr.from(xs.map(p => services.home.resolve(p).toString))
    ujson.Obj("checkpoints" -> paths(d.checkpoints), "lora" -> paths(d.lora), "embeddings" -> paths(d.embeddings),
      "vae" -> paths(d.vae), "hypernetworks" -> ujson.Arr(services.home.resolve("models/hypernetworks").toString))

  def full(hashes: Boolean = false): Future[ujson.Value] = services.client.systemInfo(dirs, hashes).map(enrich)
  def status(): Future[ujson.Value] = services.client.systemStatus().map(enrich)

  private def enrich(data: ujson.Value): ujson.Value =
    val javafx = Option(System.getProperty("javafx.runtime.version")).orElse(Option(System.getProperty("javafx.version")))
      .getOrElse("not initialized in this process")
    data("native") = ujson.Obj("java" -> System.getProperty("java.version"), "javafx" -> javafx,
      "engine_state" -> services.supervisor.state.toString,
      "plugins" -> ujson.Arr.from(services.registry.all.map(p => ujson.Obj("id" -> p.id, "name" -> p.name,
        "enabled" -> services.registry.isEnabled(p.id)))),
      "generation_metadata_fields" -> ujson.Arr.from(services.settings.systemInfoMetadataFields))
    data

  def history(): Seq[ujson.Value] = SystemInfoService.load(historyFile)
  def remember(record: ujson.Value): Seq[ujson.Value] = synchronized {
    val merged = SystemInfoService.merge(history(), record)
    AtomicFiles.writeString(historyFile, ujson.Arr.from(merged).render(indent = 2))
    merged
  }

object SystemInfoService:
  /** Replaces the matching hardware/runtime/model fingerprint, preserving other system records. */
  def merge(existing: Seq[ujson.Value], record: ujson.Value): Seq[ujson.Value] =
    val key = record.obj.get("fingerprint").collect { case ujson.Str(s) if s.nonEmpty => s }
      .getOrElse(throw new IllegalArgumentException("benchmark fingerprint is missing"))
    val index = existing.indexWhere(_.objOpt.exists(_.get("fingerprint").contains(ujson.Str(key))))
    if index >= 0 then existing.updated(index, record) else existing :+ record

  def load(path: Path): Seq[ujson.Value] =
    if !Files.exists(path) then Nil
    else
      val data = ujson.read(Files.readString(path))
      require(data.arr.forall(_.objOpt.exists(_.get("fingerprint").exists(_.isInstanceOf[ujson.Str]))), "Invalid benchmark history")
      data.arr.toSeq

  val metadataFields: Seq[String] = Seq("version.app", "version.engine", "version.hash", "platform.arch", "platform.cpu",
    "platform.system", "platform.release", "platform.python", "gpu.device", "gpu.cuda", "gpu.cudnn", "gpu.hip",
    "runtime.attention", "runtime.backend", "runtime.pipeline", "runtime.dtype", "runtime.vram_mode",
    "libraries.diffusers", "libraries.transformers", "libraries.torch", "version.tag", "version.tags", "version.updated", "version.url",
    "gpu.ipex", "gpu.openvino", "torch.version", "crossattention", "backend", "pipeline", "flags",
    "libs.xformers", "libs.diffusers", "libs.transformers")
