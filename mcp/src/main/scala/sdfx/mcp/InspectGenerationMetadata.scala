package sdfx.mcp

import java.nio.file.{Files, Path}
import java.util.Base64
import sdfx.core.infotext.Infotext
import sdfx.core.plugins.api.{Control, Plugin, Values}
import sdfx.core.png.PngMetadata
import sdfx.plugins.{ConditioningHighresFixPlugin, Img2ImgHiresFixPlugin, StandardPlugins, TokenDownsamplingPlugin}
import scala.util.control.NonFatal

/** Inspect only: no model load, plugin execution, script execution or generation. */
final class InspectGenerationMetadataTool(policy: PathPolicy, base: Path) extends McpTool:
  def name = "inspect_generation_metadata"
  override def title = "Inspect generation metadata"
  def description = "Read PNG parameters and return typed txt2img arguments and explicit AlwaysOn replay candidates. Never starts generation; missing inputs, unsupported settings and unknown metadata require review."
  def inputSchema = Schema.obj(Seq("image" -> Schema.str("PNG path in allowed folders, base64 PNG, or PNG data URI.")), Seq("image"))
  def call(input: ujson.Obj, ctx: ToolContext): ToolResult =
    val args = Args(input, Seq("image"))
    if ctx.isCancelled then throw ToolError("cancelled")
    if !input.value.get("image").exists(_.isInstanceOf[ujson.Str]) then throw ToolError("image must be a string")
    ToolResult.json(InspectGenerationMetadata.inspect(args.reqStr("image"), policy, base))

object InspectGenerationMetadata:
  private val MaxBytes = 16 * 1024 * 1024
  private val signature = Array[Byte](-119, 80, 78, 71, 13, 10, 26, 10)

  private def read(spec: String, policy: PathPolicy, base: Path): Array[Byte] =
    val s = spec.trim
    if s.isEmpty then throw ToolError("image is empty")
    if PathPolicy.isNetworkPath(s) || Images.looksLikePath(s) && !s.startsWith("data:") then
      val path = policy.resolve(s, "image", base)
      val in = Files.newInputStream(path, java.nio.file.LinkOption.NOFOLLOW_LINKS)
      val bytes = try in.readNBytes(MaxBytes + 1) finally in.close()
      if bytes.length > MaxBytes then throw ToolError("image exceeds 16 MiB")
      bytes
    else
      val data = if s.startsWith("data:") then
        if !s.startsWith("data:image/png;base64,") then throw ToolError("image must be a PNG base64 data URI")
        s.substring("data:image/png;base64,".length)
      else s
      if data.length.toLong > ((MaxBytes.toLong + 2) / 3) * 4 then throw ToolError("image exceeds 16 MiB")
      val bytes = try Base64.getDecoder.decode(data) catch case NonFatal(_) => throw ToolError("invalid base64 PNG")
      if bytes.length > MaxBytes then throw ToolError("image exceeds 16 MiB")
      bytes

  // Bound ImageIO metadata expansion before it sees the PNG, including compressed text.
  private def validateChunks(bytes: Array[Byte]): Unit =
    if bytes.length < 20 || !java.util.Arrays.equals(bytes.take(8), signature) then throw ToolError("image is not PNG")
    var offset = 8
    var ended = false
    var textBudget = 1024 * 1024
    def nul(start: Int, end: Int): Int =
      (start until end).find(i => bytes(i) == 0).getOrElse(throw ToolError("invalid PNG text"))
    def account(start: Int, end: Int, compressed: Boolean): Unit =
      val n = if compressed then
        val stream = new java.util.zip.InflaterInputStream(new java.io.ByteArrayInputStream(bytes, start, end - start))
        try stream.readNBytes(textBudget + 1).length
        catch case NonFatal(_) => throw ToolError("invalid compressed PNG text")
        finally stream.close()
      else end - start
      if n > textBudget then throw ToolError("PNG text exceeds 1 MiB")
      textBudget -= n
    while offset <= bytes.length - 12 && !ended do
      val n = java.nio.ByteBuffer.wrap(bytes, offset, 4).getInt.toLong & 0xffffffffL
      if n > bytes.length.toLong - offset - 12 then throw ToolError("invalid PNG chunk")
      val kind = new String(bytes, offset + 4, 4, java.nio.charset.StandardCharsets.US_ASCII)
      val start = offset + 8
      val end = start + n.toInt
      if kind == "tEXt" then
        val keyEnd = nul(start, end)
        account(keyEnd + 1, end, false)
      if kind == "zTXt" then
        val keyEnd = nul(start, end)
        if keyEnd + 2 > end || bytes(keyEnd + 1) != 0 then throw ToolError("invalid compressed PNG text")
        account(keyEnd + 2, end, true)
      if kind == "iCCP" then
        val nameEnd = nul(start, end)
        if nameEnd + 2 > end || bytes(nameEnd + 1) != 0 then throw ToolError("invalid PNG color profile")
        account(nameEnd + 2, end, true)
      if kind == "iTXt" then
        val keyEnd = nul(start, end)
        if keyEnd + 3 >= end || (bytes(keyEnd + 1) != 0 && bytes(keyEnd + 1) != 1) || bytes(keyEnd + 2) != 0 then throw ToolError("invalid PNG text")
        val languageEnd = nul(keyEnd + 3, end)
        val translationEnd = nul(languageEnd + 1, end)
        account(translationEnd + 1, end, bytes(keyEnd + 1) == 1)
      offset += 12 + n.toInt
      if kind == "IEND" then ended = true
    if !ended then throw ToolError("truncated PNG")

  private def typed(plugin: Plugin, values: Values): ujson.Obj =
    val controls = plugin.controls.map(c => c.key -> c).toMap
    ujson.Obj.from(values.map.toSeq.flatMap { (key, value) => controls.get(key).map { c =>
      val v: ujson.Value = c match
        case _: Control.Check => ujson.Bool(value.equalsIgnoreCase("true"))
        case _: Control.IntField => ujson.Num(value.toInt)
        case _: Control.Slider => ujson.Num(value.toDouble)
        case _ => ujson.Str(value)
      key -> v
    }})

  def inspect(spec: String, policy: PathPolicy, base: Path): ujson.Obj =
    val bytes = read(spec, policy, base)
    validateChunks(bytes)
    val temp = Files.createTempFile("sdfx-mcp-infotext-", ".png")
    val text = try
      Files.write(temp, bytes)
      Images.rejectProtectedPng(temp, "image")
      PngMetadata.readParameters(temp).getOrElse(throw ToolError("PNG has no parameters infotext"))
    finally Files.deleteIfExists(temp)
    val parsed = Infotext.parse(text)
    val p = try Infotext.toParams(parsed) catch case NonFatal(_) => throw ToolError("invalid generation parameters")
    val workErrors = try
      McpTools.checkWork(p)
      Nil
    catch case e: ToolError => List(e.getMessage)
    val errors = p.validate ++ Option.when(!p.cfgScale.isFinite || !p.subseedStrength.isFinite || !p.hires.scale.isFinite || !p.hires.denoisingStrength.isFinite)("non-finite numeric parameter") ++
      Option.when(p.hires.enabled && (p.hires.scale < 1 || p.hires.scale > 4 || p.hires.secondPassSteps < 0 || p.hires.secondPassSteps > 150 || p.hires.resizeX < 0 || p.hires.resizeX > 4096 || p.hires.resizeY < 0 || p.hires.resizeY > 4096))("hires parameters outside MCP limits") ++
      Option.when(p.seed < -1L || p.seed > 4294967295L)("seed outside MCP range") ++
      Option.when(p.subseed < -1L || p.subseed > 4294967295L)("subseed outside MCP range") ++
      Option.when(p.eta.exists(v => !v.isFinite || v < 0 || v > 10))("eta outside MCP range") ++
      Option.when(p.hires.cfgScale.exists(v => !v.isFinite || v < 0 || v > 30))("hires CFG outside MCP range") ++
      Option.when(p.loras.size > 32 || p.loras.exists(l => !l.weight.isFinite || math.abs(l.weight) > 10 || l.unet.exists(v => !v.isFinite || math.abs(v) > 10) || l.lbw.exists(_.length > 16384) || l.dyn.exists(v => v < 1 || v > 65536) || l.start.exists(v => !v.isFinite || v < 0 || v > 1) || l.stop.exists(v => !v.isFinite || v < 0 || v > 1) || l.start.zip(l.stop).exists((a, b) => a > b)))("manual LoRA outside MCP range") ++
      Option.when(p.batchSize.toLong * p.nIter > McpTools.MaxImages)("batch exceeds MCP limit") ++ workErrors
    if errors.nonEmpty then throw ToolError("parameters cannot be replayed: " + errors.mkString("; "))
    val args = ujson.Obj("prompt" -> p.prompt, "negative_prompt" -> p.negativePrompt, "seed" -> p.seed.toDouble,
      "steps" -> p.steps, "cfg_scale" -> p.cfgScale, "width" -> p.width, "height" -> p.height,
      "batch_size" -> p.batchSize, "n_iter" -> p.nIter, "sampler_name" -> p.samplerName,
      "scheduler" -> p.scheduler, "clip_skip" -> p.clipSkip)
    args("subseed") = ujson.Num(p.subseed.toDouble)
    args("subseed_strength") = ujson.Num(p.subseedStrength)
    p.eta.foreach(v => args("eta") = ujson.Num(v))
    if p.loras.nonEmpty then args("manual_loras") = ujson.Arr.from(p.loras.map { l =>
      val o = ujson.Obj("name" -> l.name, "weight" -> l.weight)
      l.unet.foreach(v => o("unet") = ujson.Num(v)); l.lbw.foreach(v => o("lbw") = ujson.Str(v))
      l.start.foreach(v => o("start") = ujson.Num(v)); l.stop.foreach(v => o("stop") = ujson.Num(v)); l.dyn.foreach(v => o("dyn") = ujson.Num(v))
      o
    })
    if p.hires.enabled then args("hires") = ujson.Obj("enabled" -> true, "scale" -> p.hires.scale,
      "upscaler" -> p.hires.upscaler, "steps" -> p.hires.secondPassSteps,
      "denoising_strength" -> p.hires.denoisingStrength, "resize_x" -> p.hires.resizeX, "resize_y" -> p.hires.resizeY,
      "prompt" -> p.hires.prompt, "negative_prompt" -> p.hires.negativePrompt)
    if p.hires.enabled then
      val h = args("hires")
      p.hires.samplerName.foreach(v => h("sampler_name") = ujson.Str(v))
      p.hires.scheduler.foreach(v => h("scheduler") = ujson.Str(v))
      p.hires.cfgScale.foreach(v => h("cfg_scale") = ujson.Num(v))
    val f = parsed.fields
    val candidates = Seq(
      "conditioning_highres_fix" -> ConditioningHighresFixPlugin.replay(f),
      "img2img_hires_fix" -> Img2ImgHiresFixPlugin.replay(f),
      "token_downsampling" -> TokenDownsamplingPlugin.replay(f),
      "model_keyword" -> Option.when(f.contains("Model keyword settings") || f.get("Model keyword replay").exists(_.equalsIgnoreCase("true")))(Values(Map("enabled" -> "false")))
    )
    val plugins = StandardPlugins.alwaysOn.map(p => p.id -> p).toMap
    val on = ujson.Obj.from(candidates.flatMap { (id, values) => values.flatMap(v => plugins.get(id).map(p => id -> typed(p, v))) })
    PluginMetadataCandidates.freeu(f).foreach(v => on("freeu") = v)
    PluginMetadataCandidates.hypertile(f).foreach(v => on("hypertile") = v)
    PluginMetadataCandidates.pag(f).foreach(v => on("pag") = v)
    p.hooks.get("composable_lora").collect { case o: ujson.Obj => o }.foreach { hook =>
      val values = Values(hook.value.toSeq.collect { case (k, ujson.Bool(v)) => k -> v.toString }.toMap.updated("enabled", "true"))
      plugins.get("composable_lora").foreach(plugin => on("composable_lora") = typed(plugin, values))
    }
    p.hooks.get("hires_fix_tweaks").collect { case o: ujson.Obj => o }.foreach { hook =>
      val values = Values(hook.value.toSeq.collect {
        case (k, ujson.Str(v)) => k -> v
        case (k, ujson.Bool(v)) => k -> v.toString
        case (k, ujson.Num(v)) => k -> Values.fmt(v)
        case ("styles", a: ujson.Arr) => "styles" -> a.render()
      }.toMap)
      plugins.get("hires_fix_tweaks").foreach(plugin => on("hires_fix_tweaks") = typed(plugin, values))
    }
    def flag(key: String): Option[Boolean] = f.get(key).flatMap { value =>
      value.trim.toLowerCase(java.util.Locale.ROOT) match
        case "true" => Some(true)
        case "false" => Some(false)
        case _ => None
    }
    val numbers = flag(sdfx.core.image.GridNumbers.NumbersKey)
    val xyz = flag(sdfx.core.image.GridNumbers.XyzKey)
    if numbers.isDefined || xyz.isDefined then
      on("grid_numbers") = ujson.Obj("enabled" -> true, "image_numbers" -> numbers.getOrElse(true), "xyz_info" -> xyz.getOrElse(false))
    val represented = Set("Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size", "Batch size", "Batch count", "Clip skip",
      "Variation seed", "Variation seed strength", "Eta", "Hires upscale", "Hires resize", "Hires steps", "Hires upscaler", "Hires prompt", "Hires negative prompt", "Hires sampler", "Hires schedule type", "Hires CFG Scale", "Hires CFG scale") ++
      Option.when(p.loras.nonEmpty)(sdfx.core.infotext.ManualLoraMetadata.key) ++ (if p.hires.enabled then Set("Denoising strength") else Set.empty[String]) ++
      (if f.get("Conditioning Highres Fix").exists(ConditioningHighresFixPlugin.modes.contains) then Set("Conditioning Highres Fix") else Set.empty[String]) ++
      (if f.get(sdfx.core.infotext.Img2ImgHiresMetadata.key).flatMap(sdfx.core.infotext.Img2ImgHiresMetadata.parse).isDefined then Set(sdfx.core.infotext.Img2ImgHiresMetadata.key) else Set.empty[String]) ++
      (if on.value.contains("token_downsampling") then Set("ToDo factor", "ToDo factor2", "ToDo disable after") else Set.empty[String]) ++
      (if on.value.contains("model_keyword") then Set("Model keyword settings", "Model keyword replay") else Set.empty[String]) ++
      (if on.value.contains("composable_lora") then Set("Composable LoRA") ++ Seq("with_step", "uc_text_model_encoder", "uc_diffusion_model", "plot_weights", "single_no_uc", "hires_global_step").map("Composable LoRA " + _) else Set.empty[String]) ++
      (if on.value.contains("hires_fix_tweaks") then Set("Hires seed") else Set.empty[String]) ++
      (if numbers.isDefined then Set(sdfx.core.image.GridNumbers.NumbersKey) else Set.empty[String]) ++
      (if xyz.isDefined then Set(sdfx.core.image.GridNumbers.XyzKey) else Set.empty[String]) ++
      (if on.value.contains("freeu") then Set("FreeU Version", "FreeU Stages") else Set.empty[String]) ++
      (if on.value.contains("hypertile") then Set("Hypertile tile size", "Hypertile swap size", "Hypertile max depth", "Hypertile scale depth") else Set.empty[String]) ++
      (if on.value.contains("pag") then Set("PAG scale", "PAG adaptive scale", "PAG applied layers") else Set.empty[String])
    val unconsumed = f.filterNot((key, _) => represented.contains(key))
    ujson.Obj("infotext" -> text, "generation_args" -> args, "always_on" -> on,
      "unapplied_fields" -> ujson.Obj.from(unconsumed.map((k, v) => k -> ujson.Str(v))),
      "warnings" -> ujson.Arr("Review candidates and missing model, LoRA, image, mask and unsupported plugin settings before replay; nothing was executed."))
