package sdfx.mcp

import sdfx.engine.AppServices
import sdfx.core.modelpresets.*
import java.nio.file.{Files, Path}

/** Data/library API only: receiving or importing a preset never starts generation. */
final class ModelPresetsTool(services: AppServices, service: ModelPresetService) extends McpTool:
  def this(services: AppServices) = this(services, services.modelPresets)
  def name = "model_presets"
  override def title = "Model presets"
  def description = "Read, revision-save, share, resolve, fetch, or set a checkpoint's model preset thumbnail. Fetch sends the full checkpoint SHA256 to Civitai (or uses the stored Civitai URL), updates URL/triggers and preserves presets unless overwrite=true explicitly imports remote sharing presets. Its first thumbnail is saved only if absent. No generation starts."
  def inputSchema = Schema.obj(Seq(
    "action" -> Schema.str("get, save, share, resolve, fetch, or thumbnail"),
    "checkpoint" -> Schema.str("Catalog checkpoint name or configured checkpoint path; default loaded checkpoint"),
    "expected_revision" -> ujson.Obj("type" -> ujson.Arr("string", "null"), "description" -> "Required for save/fetch: revision from get, or null for a new library"),
    "document" -> ujson.Obj("type" -> "object", "description" -> "Complete upstream exchange document for save"),
    "sharing_text" -> Schema.str("Alternative JSON or ###ModelPresets### sharing text for save"),
    "preset" -> Schema.str("Preset name for resolve; defaults to the stored default preset"),
    "image_base64" -> Schema.str("Thumbnail action: base64 encoded image, at most 16 MiB decoded; saved as a readable PNG beside the checkpoint"),
    "replace" -> Schema.bool("Thumbnail action: explicitly replace an existing thumbnail (default false)"),
    "overwrite" -> Schema.bool("Fetch only: explicitly replace the entire library with remote sharing presets if present (default false)")
  ), Seq("action"))
  private def path(name: Option[String]): Path =
    val value = name.orElse(services.supervisor.loadedModel.map(_.path)).getOrElse(throw ToolError("Select a checkpoint"))
    services.catalog.findCheckpoint(value).map(c => Path.of(c.path)).getOrElse {
      val candidate = services.home.resolve(value)
      if !Files.isRegularFile(candidate) || !services.generator.withinDirs(candidate, services.settings.modelDirs.checkpoints) then
        throw ToolError("Checkpoint is not in the configured catalog folders")
      candidate
    }
  def call(input: ujson.Obj, ctx: ToolContext): ToolResult =
    val args = Args(input, Seq("action", "checkpoint", "expected_revision", "document", "sharing_text", "preset", "image_base64", "replace", "overwrite"))
    def check(): Unit = if ctx.isCancelled then throw new java.util.concurrent.CancellationException("Operation cancelled")
    val action = args.reqStr("action")
    require(Set("get", "save", "share", "resolve", "fetch", "thumbnail")(action), "Unknown model preset action")
    ctx.progress(0, None, "Hashing checkpoint identity")
    val model = service.identify(path(args.str("checkpoint")), () => check())
    val previous = service.load(model)
    def expected: Option[String] =
      require(input.obj.contains("expected_revision"), "Supply expected_revision from get (or null for a new library)")
      args.str("expected_revision")
    var thumbnailWarning: Option[String] = None
    val loaded = action match
      case "save" =>
        val document = args.objOpt("document").map(v => ModelPresetDocument.parse(v.render()))
          .orElse(args.str("sharing_text").map(ModelPresetDocument.parseShared))
          .getOrElse(throw ToolError("Supply document or sharing_text"))
        val revision = expected
        check(); service.save(model, document, revision)
      case "fetch" =>
        val revision = expected
        require(previous.revision == revision, "Presets changed on disk. Reload before fetching.")
        ctx.progress(1, None, "Fetching Civitai model information")
        val fetched = service.civitai.fetch(model.sha256, previous.document, () => check())
        check()
        val saved = service.save(model, fetched.selectDocument(args.bool("overwrite").getOrElse(false)), revision)
        try fetched.thumbnail.foreach(bytes => service.saveThumbnail(model, bytes, false, () => check()))
        catch case scala.util.control.NonFatal(error) =>
          thumbnailWarning = Some("Presets saved; thumbnail not saved: " + Option(error.getMessage).getOrElse(error.toString))
        saved
      case "thumbnail" =>
        val encoded = args.reqStr("image_base64")
        require(encoded.length <= ((16 * 1024 * 1024 + 2) / 3) * 4, "Thumbnail input is too large")
        val bytes = java.util.Base64.getDecoder.decode(encoded)
        service.saveThumbnail(model, bytes, args.bool("replace").getOrElse(false), () => check())
        previous
      case _ => previous
    val result = ujson.Obj("model_sha256" -> model.sha256, "checkpoint" -> model.path.toString,
      "document" -> ujson.read(loaded.document.json), "revision" -> loaded.revision.map(ujson.Str(_)).getOrElse(ujson.Null),
      "file" -> service.store.path(model.sha256).toString)
    val thumbnail = service.thumbnailPath(model)
    if Files.isRegularFile(thumbnail) then result("thumbnail") = thumbnail.toString
    thumbnailWarning.foreach(value => result("warning") = value)
    if action == "share" then result("sharing_text") = loaded.document.share
    if action == "resolve" then
      val selected = args.str("preset").getOrElse(loaded.document.defaultPreset)
      val text = loaded.document.presets.getOrElse(selected, throw ToolError("Preset not found"))
      result("preset") = ujson.Str(selected)
      result("infotext") = ujson.Str(text)
    ToolResult.json(result)
