package sdfx.core.presets

import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.mutable
import scala.util.control.NonFatal

final case class CompatibilityResult(record: PresetRecord, issues: Vector[PresetIssue])

/** Source interchange is a sparse form-value codec. Unknown source entries never become executable controls. */
object PresetCompatibility:
  private val limits = PresetLimits()
  private enum Conversion:
    case Plain, ResizeMode, MaskedContent, MaskMode, InpaintArea, SameSampler, Script, Tiling
  private final case class Route(target: FieldKey, conversion: Conversion = Conversion.Plain)
  private def key(parts: String*): FieldKey = FieldKey(parts.toVector)
  private def generation(name: String, conversion: Conversion = Conversion.Plain): Route =
    Route(key("generation", name), conversion)
  private def plugin(id: String, control: String): Route = Route(key("plugin", id, control))
  private def error(code: String, path: String, message: String): PresetError =
    PresetError(Vector(PresetIssue(code, path, message)))
  private def report(code: String, path: String, message: String): PresetIssue =
    PresetIssue(code, path, message)

  // Metadata includes aliases with no native destination. Preserving a synonym is not
  // evidence that its third-party extension is installed or that its controls match ours.
  val configSynonyms: Vector[(String, String)] = Vector(
    "txt2img_hires_steps" -> "txt2img_steps_alt",
    "txt2img_enable_hr" -> "txt2img_show_second_pass",
    "controlnet_control_mod_radio" -> "controlnet_control_mode_radio",
    "txt2img_enable_hr" -> "txt2img_hr-checkbox",
    "script_txt2img_adetailer_ad_inpaint_full_res" -> "script_txt2img_adetailer_ad_inpaint_only_masked",
    "script_txt2img_adetailer_ad_inpaint_full_res_padding" -> "script_txt2img_adetailer_ad_inpaint_only_masked_padding",
    "script_txt2img_adetailer_ad_inpaint_full_res_2nd" -> "script_txt2img_adetailer_ad_inpaint_only_masked_2nd",
    "script_txt2img_adetailer_ad_inpaint_full_res_padding_2nd" -> "script_txt2img_adetailer_ad_inpaint_only_masked_padding_2nd",
    "tiledvae-t2i-enable" -> "MD-t2i-enable",
    "tiledvae-t2i-vae2gpu" -> "MD-t2i-vae2gpu",
    "tiledvae-t2i-enc-size" -> "MD-t2i-enc-size",
    "tiledvae-t2i-dec-size" -> "MD-t2i-dec-size",
    "tiledvae-t2i-fastenc" -> "MD-t2i-fastenc",
    "tiledvae-t2i-fastenc-colorfix" -> "MD-t2i-fastenc-colorfix",
    "tiledvae-t2i-fastdec" -> "MD-t2i-fastdec",
    "tiledvae-i2i-enable" -> "MD-i2i-enable",
    "tiledvae-i2i-vae2gpu" -> "MD-i2i-vae2gpu",
    "tiledvae-i2i-enc-size" -> "MD-i2i-enc-size",
    "tiledvae-i2i-dec-size" -> "MD-i2i-dec-size",
    "tiledvae-i2i-fastenc" -> "MD-i2i-fastenc",
    "tiledvae-i2i-fastenc-colorfix" -> "MD-i2i-fastenc-colorfix",
    "tiledvae-i2i-fastdec" -> "MD-i2i-fastdec",
    "MD-t2i-enabled" -> "MD-t2i-enabled-checkbox",
    "MD-i2i-enabled" -> "MD-i2i-enabled-checkbox",
    "MDV-t2i-enabled" -> "MDV-t2i-enabled-checkbox",
    "MDV-i2i-enabled" -> "MDV-i2i-enabled-checkbox",
    "script_txt2img_adetailer_ad_enable" -> "script_txt2img_adetailer_ad_main_accordion-checkbox",
    "script_img2img_adetailer_ad_enable" -> "script_img2img_adetailer_ad_main_accordion-checkbox"
  )

  private val commonConfig: Map[String, Route] = Map(
    "sampling" -> generation("sampler"),
    "scheduler" -> generation("scheduler"),
    "steps" -> generation("steps"),
    "width" -> generation("width"),
    "height" -> generation("height"),
    "batch_count" -> generation("batchCount"),
    "batch_size" -> generation("batchSize"),
    "cfg_scale" -> generation("cfgScale"),
    "restore_faces" -> plugin("face_restore", "enabled"),
    "prompt" -> generation("prompt"),
    "neg_prompt" -> generation("negativePrompt"),
    "seed" -> generation("seed"),
    "subseed" -> generation("subseed"),
    "subseed_strength" -> generation("subseedStrength"),
    "subseed_show" -> generation("variationEnabled"),
    "tiling" -> Route(key("plugin", "asymmetric_tiling", "enabled"), Conversion.Tiling)
  )
  private val txtConfig: Map[String, Route] = Map(
    "enable_hr" -> generation("hiresEnabled"),
    "hr-checkbox" -> generation("hiresEnabled"),
    "show_second_pass" -> generation("hiresEnabled"),
    "hr_scale" -> generation("hiresScale"),
    "hr_upscaler" -> generation("hiresUpscaler"),
    "sampling_alt" -> generation("hiresUpscaler"),
    "hires_steps" -> generation("hiresSteps"),
    "steps_alt" -> generation("hiresSteps"),
    "denoising_strength" -> generation("denoisingStrength"),
    "hr_resize_x" -> generation("hiresResizeX"),
    "hr_resize_y" -> generation("hiresResizeY")
  )
  private val imgConfig: Map[String, Route] = Map(
    "denoising_strength" -> generation("denoisingStrength"),
    "mask_blur" -> generation("maskBlur"),
    "inpainting_fill" -> generation("inpaintingFill", Conversion.MaskedContent),
    "mask_mode" -> generation("maskInvert", Conversion.MaskMode),
    "inpaint_full_res" -> generation("inpaintFullRes", Conversion.InpaintArea),
    "inpaint_full_res_padding" -> generation("inpaintPadding")
  )
  private val utility: Map[String, Route] = Map(
    "Prompt" -> generation("prompt"),
    "Negative prompt" -> generation("negativePrompt"),
    "Sampling steps" -> generation("steps"),
    "Sampling Steps" -> generation("steps"),
    "Sampling method" -> generation("sampler"),
    "Width" -> generation("width"),
    "Height" -> generation("height"),
    "Restore faces" -> plugin("face_restore", "enabled"),
    "Tiling" -> Route(key("plugin", "asymmetric_tiling", "enabled"), Conversion.Tiling),
    "Hires. fix" -> generation("hiresEnabled"),
    "Hires fix" -> generation("hiresEnabled"),
    "Highres. fix" -> generation("hiresEnabled"),
    "Upscaler" -> generation("hiresUpscaler"),
    "Upscale by" -> generation("hiresScale"),
    "Hires. steps" -> generation("hiresSteps"),
    "Hires steps" -> generation("hiresSteps"),
    "Resize width to" -> generation("hiresResizeX"),
    "Resize height to" -> generation("hiresResizeY"),
    "Hires sampling method" -> generation("hiresSampler", Conversion.SameSampler),
    "Seed" -> generation("seed"),
    "Extra" -> generation("variationEnabled"),
    "Variation seed" -> generation("subseed"),
    "Variation strength" -> generation("subseedStrength"),
    "Denoising strength" -> generation("denoisingStrength"),
    "Batch count" -> generation("batchCount"),
    "Batch size" -> generation("batchSize"),
    "CFG Scale" -> generation("cfgScale"),
    "Script" -> Route(key("script", "selected"), Conversion.Script),
    "Resize mode" -> generation("resizeMode", Conversion.ResizeMode),
    "Mask blur" -> generation("maskBlur"),
    "Mask mode" -> generation("maskInvert", Conversion.MaskMode),
    "Masked content" -> generation("inpaintingFill", Conversion.MaskedContent),
    "Inpaint area" -> generation("inpaintFullRes", Conversion.InpaintArea),
    "Only masked padding, pixels" -> generation("inpaintPadding")
  )
  private val unsupportedUtility: Set[String] = Set(
    "Firstpass width", "Firstpass height", "Resize seed from width", "Resize seed from height",
    "Image CFG Scale", "Input directory", "Output directory",
    "Inpaint batch mask directory (required for inpaint batch processing only)",
    "Scale", "Mask transparency"
  )
  private val xyzLabels = Map(
    "X Type" -> "x_type", "Y Type" -> "y_type", "Z Type" -> "z_type",
    "X type" -> "x_type", "Y type" -> "y_type", "Z type" -> "z_type",
    "X Values" -> "x_values", "Y Values" -> "y_values", "Z Values" -> "z_values",
    "X values" -> "x_values", "Y values" -> "y_values", "Z values" -> "z_values",
    "Draw legend" -> "draw_legend", "Keep -1 for seeds" -> "keep_random_seed",
    "Include Sub Images" -> "include_sub_images", "Include Sub Grids" -> "include_sub_grids"
  )
  private val xyzCanonical = Vector(
    "X Type" -> "x_type", "Y Type" -> "y_type", "Z Type" -> "z_type",
    "X Values" -> "x_values", "Y Values" -> "y_values", "Z Values" -> "z_values",
    "Draw legend" -> "draw_legend", "Keep -1 for seeds" -> "keep_random_seed",
    "Include Sub Images" -> "include_sub_images", "Include Sub Grids" -> "include_sub_grids"
  )
  private val controlNetKeys = Map(
    "enable_checkbox" -> "enabled",
    "preprocessor_dropdown" -> "preprocessor",
    "model_dropdown" -> "model",
    "control_weight_slider" -> "weight",
    "start_control_step_slider" -> "guidanceStart",
    "ending_control_step_slider" -> "guidanceEnd"
  )
  private val controlNetPattern =
    raw"^(txt2img|img2img)_controlnet_ControlNet-([0-9]+)_controlnet_(.+)$$".r
  private val xyzPattern = raw"^script_(txt2img|img2img)_xyz_plot_([xyz])_(type|values)$$".r
  private val tilingControls = Vector("enabled", "tile_x", "tile_y", "start_step", "stop_step")
  private val tilingKeys = tilingControls.map(c => key("plugin", "asymmetric_tiling", c)).toSet
  private val tilingTrueValues: Map[FieldKey, PresetAtom] = Map(
    key("plugin", "asymmetric_tiling", "enabled") -> PresetAtom.Flag(true),
    key("plugin", "asymmetric_tiling", "tile_x") -> PresetAtom.Flag(true),
    key("plugin", "asymmetric_tiling", "tile_y") -> PresetAtom.Flag(true),
    key("plugin", "asymmetric_tiling", "start_step") -> PresetAtom.Integer(0),
    key("plugin", "asymmetric_tiling", "stop_step") -> PresetAtom.Integer(-1))

  private def sourceMode(mode: PresetMode): String =
    if mode == PresetMode.Txt2Img then "txt2img" else "img2img"

  private def configRoute(id: String, mode: PresetMode): Option[Route] =
    val prefix = sourceMode(mode) + "_"
    if id == "hr_sampler" && mode == PresetMode.Txt2Img then
      Some(generation("hiresSampler", Conversion.SameSampler))
    else if id == "resize_mode" && mode != PresetMode.Txt2Img then
      Some(generation("resizeMode", Conversion.ResizeMode))
    else if id == "hires_prompt" && mode == PresetMode.Txt2Img then Some(generation("hiresPrompt"))
    else if id == "hires_neg_prompt" && mode == PresetMode.Txt2Img then Some(generation("hiresNegative"))
    else if id == "script_list" then Some(Route(key("script", "selected"), Conversion.Script))
    else id match
      case xyzPattern(origin, axis, part) if origin == sourceMode(mode) =>
        Some(plugin("xyz_grid", axis + "_" + part))
      case controlNetPattern(origin, slot, control) if origin == sourceMode(mode) =>
        slot.toIntOption.filter(n => n >= 0 && n < 3).flatMap(_ => controlNetKeys.get(control)).map { property =>
          Route(key("controlnet", slot, property))
        }
      case _ if id.startsWith(prefix) =>
        val suffix = id.substring(prefix.length)
        commonConfig.get(suffix).orElse {
          if mode == PresetMode.Txt2Img then txtConfig.get(suffix) else imgConfig.get(suffix)
        }
      case _ => None

  private def route(id: String, format: String, mode: Option[PresetMode]): Option[Route] =
    if format == "config-presets" then mode.flatMap(configRoute(id, _))
    else if format == "preset-utilities" then
      utility.get(id).orElse(xyzLabels.get(id).map(plugin("xyz_grid", _)))
    else None

  private def available(schemas: Map[PresetMode, PresetSchema], format: String,
      mode: Option[PresetMode]): Map[FieldKey, PresetFieldSpec] =
    val candidates =
      if format == "preset-utilities" then schemas.values.toVector
      else mode match
        case Some(PresetMode.Txt2Img) => schemas.get(PresetMode.Txt2Img).toVector
        case Some(_) => schemas.get(PresetMode.Img2Img).toVector ++ schemas.get(PresetMode.Inpaint).toVector
        case None => Vector.empty
    candidates.flatMap(_.fields).groupBy(_.key).view.mapValues(_.head).toMap

  private def sourceText(raw: RawJson): String =
    PresetJson.scalarAtom(raw, limits).toOption.flatten match
      case Some(PresetAtom.Text(value)) => value
      case _ => ""
  private def decimal(atom: PresetAtom): Option[BigDecimal] = atom match
    case PresetAtom.Integer(value) => Some(BigDecimal(value))
    case PresetAtom.Decimal(value) => Some(value)
    case _ => None
  private def exactLong(atom: PresetAtom): Option[Long] =
    decimal(atom).flatMap { value =>
      try Some(value.bigDecimal.longValueExact())
      catch case NonFatal(_) => None
    }
  private def inBounds(value: PresetAtom, spec: PresetFieldSpec): Boolean =
    decimal(value) match
      case Some(number) => spec.min.forall(number >= _) && spec.max.forall(number <= _)
      case None => true

  private val resizeLabels = Vector("Just resize", "Crop and resize", "Resize and fill")
  private val contentLabels = Vector("fill", "original", "latent noise", "latent nothing")
  private def fixedChoice(atom: PresetAtom, labels: Vector[String], spec: PresetFieldSpec): Option[PresetAtom] =
    val position = atom match
      case PresetAtom.Text(value) => labels.indexOf(value)
      case value => exactLong(value).filter(n => n >= 0 && n < labels.size).map(_.toInt).getOrElse(-1)
    if position < 0 then None
    else
      val destination = spec.kind match
        case PresetFieldKind.Integer => Some(PresetAtom.Integer(position.toLong))
        case PresetFieldKind.Text => Some(PresetAtom.Text(labels(position)))
        case PresetFieldKind.Choice => spec.defaultValue match
          case _: PresetAtom.Integer => Some(PresetAtom.Integer(position.toLong))
          case _: PresetAtom.Text => Some(PresetAtom.Text(labels(position)))
          case _ => None
        case _ => None
      destination.filter(value => spec.kind != PresetFieldKind.Choice || spec.choices.contains(value))
  private def fixedBoolean(atom: PresetAtom, labels: Vector[String]): Option[PresetAtom] =
    atom match
      case PresetAtom.Flag(value) => Some(PresetAtom.Flag(value))
      case PresetAtom.Text(value) =>
        val index = labels.indexOf(value)
        Option.when(index >= 0)(PresetAtom.Flag(index == 1))
      case other => exactLong(other).filter(n => n == 0 || n == 1).map(n => PresetAtom.Flag(n == 1))

  private def normalize(atom: PresetAtom, spec: PresetFieldSpec, conversion: Conversion): Option[PresetAtom] =
    val converted: Option[PresetAtom] = conversion match
      case Conversion.ResizeMode => fixedChoice(atom, resizeLabels, spec)
      case Conversion.MaskedContent => fixedChoice(atom, contentLabels, spec)
      case Conversion.MaskMode => fixedBoolean(atom, Vector("Inpaint masked", "Inpaint not masked"))
      case Conversion.InpaintArea => fixedBoolean(atom, Vector("Whole picture", "Only masked"))
      case Conversion.SameSampler => atom match
        case PresetAtom.Text(value) => Some(PresetAtom.Text(if value == "Use same sampler" then "" else value))
        case _ => None
      case Conversion.Script => atom match
        case PresetAtom.Text("None") => Some(PresetAtom.Text(""))
        case PresetAtom.Text("X/Y/Z plot") => Some(PresetAtom.Text("xyz_grid"))
        case PresetAtom.Text("Loopback") => Some(PresetAtom.Text("loopback"))
        case PresetAtom.Text("SD upscale") => Some(PresetAtom.Text("sd_upscale"))
        case _ => None
      case _ => spec.kind match
        case PresetFieldKind.Integer => exactLong(atom).map(PresetAtom.Integer.apply)
        case PresetFieldKind.Decimal => decimal(atom).map(PresetAtom.Decimal.apply)
        case PresetFieldKind.Flag => atom match
          case value: PresetAtom.Flag => Some(value)
          case _ => None
        case PresetFieldKind.Text | PresetFieldKind.PathText | PresetFieldKind.Choice => atom match
          case value: PresetAtom.Text => Some(value)
          case _ => None
    converted.filter(inBounds(_, spec))

  private def tiling(atom: PresetAtom, specs: Map[FieldKey, PresetFieldSpec]): Option[Map[FieldKey, PresetAtom]] =
    atom match
      case PresetAtom.Flag(false) =>
        val enabled = key("plugin", "asymmetric_tiling", "enabled")
        Option.when(specs.contains(enabled))(Map(enabled -> PresetAtom.Flag(false)))
      case PresetAtom.Flag(true) =>
        Option.when(tilingTrueValues.keys.forall(specs.contains))(tilingTrueValues)
      case _ => None

  private def synonymsOf(id: String): Set[String] =
    val visited = mutable.HashSet(id)
    var changed = true
    while changed do
      changed = false
      configSynonyms.foreach { (left, right) =>
        if visited(left) && visited.add(right) then changed = true
        if visited(right) && visited.add(left) then changed = true
      }
    visited.toSet

  private def bindingRoute(id: String, format: String,
      bindings: Vector[SourceBinding]): Either[PresetIssue, Option[Route]] =
    val candidates = if format == "config-presets" then synonymsOf(id) else Set(id)
    val matches = bindings.filter(b => b.format == format && candidates(b.sourceKey)).map(_.target).distinct
    if matches.size > 1 then Left(report("alias-conflict", id, "multiple explicit source bindings"))
    else Right(matches.headOption.map(Route(_)))

  private def resolvedRoute(id: String, format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding]): Either[PresetIssue, Option[Route]] =
    val standard = route(id, format, mode)
    bindingRoute(id, format, bindings).flatMap {
      case Some(bound) if standard.exists(_.target != bound.target) =>
        Left(report("alias-conflict", id, "explicit binding conflicts with standard field"))
      case Some(bound) => Right(Some(standard.getOrElse(bound)))
      case None => Right(standard)
    }

  private def metadataContext(format: String, mode: Option[PresetMode]): Either[PresetIssue, Unit] =
    if format != "config-presets" && format != "preset-utilities" then
      Left(report("format", "$", "unsupported source format"))
    else if format == "config-presets" &&
        !mode.exists(m => m == PresetMode.Txt2Img || m == PresetMode.Img2Img) then
      Left(report("mode", "$", "Config Presets requires explicit txt2img or img2img mode"))
    else if format == "preset-utilities" && mode.nonEmpty then
      Left(report("mode", "$", "Preset Utilities uses one shared source collection"))
    else Right(())

  private def metadataBindings(bindings: Vector[SourceBinding]): Either[PresetIssue, Unit] =
    if bindings == null then Left(report("binding", "$", "null binding list"))
    else if bindings.size > limits.maxBindings then Left(report("limit", "$", "too many bindings"))
    else bindings.find(b => b == null || metadataSourceKey(b.format).isLeft ||
        metadataSourceKey(b.sourceKey).isLeft || !validFieldKey(b.target)) match
      case Some(_) => Left(report("binding", "$", "invalid source binding"))
      case None => Right(())

  private def metadataField(field: FieldKey): Either[PresetIssue, Unit] =
    if validFieldKey(field) then Right(())
    else Left(report("field-key", "$", "invalid native field key"))

  private def metadataSourceKey(id: String): Either[PresetIssue, Unit] =
    if id == null || id.isEmpty || id.length > limits.maxKeyBytes then
      return Left(report("source-key", "$", "invalid or oversized source ID"))
    var at = 0
    var bytes = 0
    while at < id.length do
      val unit = id.charAt(at)
      val width =
        if Character.isHighSurrogate(unit) then
          if at + 1 >= id.length || !Character.isLowSurrogate(id.charAt(at + 1)) then
            return Left(report("source-key", "$", "invalid source ID Unicode"))
          at += 2
          4
        else if Character.isLowSurrogate(unit) then
          return Left(report("source-key", "$", "invalid source ID Unicode"))
        else
          at += 1
          if unit < 0x80 then 1 else if unit < 0x800 then 2 else 3
      bytes += width
      if bytes > limits.maxKeyBytes then
        return Left(report("source-key", "$", "invalid or oversized source ID"))
    Right(())

  private def sourceKeyMatchesMode(id: String, format: String,
      mode: Option[PresetMode]): Boolean =
    format != "config-presets" ||
      ((!id.startsWith("txt2img_") || mode.contains(PresetMode.Txt2Img)) &&
       (!id.startsWith("img2img_") || mode.contains(PresetMode.Img2Img)))

  /** Original decoded source IDs in source order, without converting any source values. */
  def sourceFieldIds(record: PresetRecord): Either[PresetError, Vector[String]] =
    if record == null || record.source == null then
      Left(error("source", "$", "missing preset record or source"))
    else record.source match
      case None => Right(Vector.empty)
      case Some(source) if source == null || source.originalFields == null =>
        Left(error("source", "$", "missing original source fields"))
      case Some(source) =>
        PresetJson.objectMembers(source.originalFields, limits).flatMap { members =>
          if members.size > limits.maxFieldsPerRecord then
            Left(error("limit", "$", "too many original source fields"))
          else Right(members.map(_.name))
        }

  /** Resolve identity without filtering out a known field absent from today's native schema. */
  def resolveSourceId(id: String, format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding],
      schemas: Map[PresetMode, PresetSchema]): Either[PresetIssue, Vector[FieldKey]] =
    for
      _ <- metadataContext(format, mode)
      _ <- metadataSourceKey(id)
      _ <- metadataBindings(bindings)
      _ <- {
        if schemas == null || schemas.values.exists(s => s == null || s.fields == null ||
            s.fields.size > limits.maxRegisteredFieldsPerMode) then
          Left(report("limit", "$", "invalid or oversized schema"))
        else Right(())
      }
      _ <- {
        if !sourceKeyMatchesMode(id, format, mode) then
          Left(report("mode", id, "source ID belongs to another mode"))
        else Right(())
      }
      selected <- resolvedRoute(id, format, mode, bindings)
      keys <- selected match
        case Some(Route(_, Conversion.Tiling)) => Right(tilingControls.map(c => key("plugin", "asymmetric_tiling", c)))
        case Some(found) => Right(Vector(found.target))
        case None => Left(report("unsupported", id, "no verified native binding"))
    yield keys

  def importRecords(document: SourceDocument, schemas: Map[PresetMode, PresetSchema],
      bindings: Vector[SourceBinding]): Either[PresetError, Vector[CompatibilityResult]] =
    if document == null || document.raw == null then return Left(error("source", "$", "missing source document"))
    if !Set("config-presets", "preset-utilities")(document.format) then
      return Left(error("format", "$", "unsupported source format"))
    if document.format == "config-presets" &&
        !document.mode.exists(m => m == PresetMode.Txt2Img || m == PresetMode.Img2Img) then
      return Left(error("mode", "$", "Config Presets requires explicit txt2img or img2img mode"))
    if document.format == "preset-utilities" && document.mode.nonEmpty then
      return Left(error("mode", "$", "Preset Utilities uses one shared source collection"))
    if schemas == null || bindings == null || bindings.size > limits.maxBindings then
      return Left(error("schema", "$", "invalid schema or binding list"))
    if schemas.values.exists(_.fields.size > limits.maxRegisteredFieldsPerMode) then
      return Left(error("limit", "$", "too many registered fields"))
    val specs = available(schemas, document.format, document.mode)
    PresetJson.objectMembers(document.raw, limits).flatMap { presets =>
      if presets.size > limits.maxRecords then Left(error("limit", "$", "too many source records"))
      else
        val results = Vector.newBuilder[CompatibilityResult]
        var failure: Option[PresetError] = None
        presets.foreach { preset =>
          if failure.isEmpty then
            val name = preset.name
            if name.isEmpty || name.codePointCount(0, name.length) > limits.maxNameCodePoints ||
                name.codePoints().toArray.exists(cp => Character.isISOControl(cp)) ||
                !name.codePoints().toArray.exists(cp => !Character.isWhitespace(cp)) then
              failure = Some(error("name", name, "invalid preset name"))
            else
              PresetJson.objectMembers(preset.value, limits) match
                case Left(e) => failure = Some(e)
                case Right(members) =>
                  if members.size > limits.maxFieldsPerRecord then
                    failure = Some(error("limit", name, "too many source fields"))
                  else
                    val issues = Vector.newBuilder[PresetIssue]
                    val opaque = Vector.newBuilder[(String, RawJson)]
                    val values = mutable.LinkedHashMap.empty[FieldKey, PresetAtom]
                    val owners = mutable.Map.empty[FieldKey, String]
                    val conflicted = mutable.Set.empty[FieldKey]
                    members.foreach { member =>
                      val id = member.name
                      val path = s"$name.$id"
                      val resolution = resolvedRoute(id, document.format, document.mode, bindings)
                      val selected = resolution match
                        case Left(issue) =>
                          issues += issue.copy(path = path)
                          None
                        case Right(value) => value
                      val parsed = PresetJson.scalarAtom(member.value, limits)
                      (selected, parsed) match
                        case (Some(destination), Right(Some(atom))) =>
                          val mapped =
                            if destination.conversion == Conversion.Tiling then tiling(atom, specs)
                            else specs.get(destination.target).flatMap { spec =>
                              normalize(atom, spec, destination.conversion).map(v => Map(destination.target -> v))
                            }
                          mapped match
                            case None =>
                              opaque += id -> member.value
                              val reason =
                                if unsupportedUtility(id) then "historical field has no native equivalent"
                                else if !specs.contains(destination.target) then "destination field unavailable"
                                else "source value is incompatible with declared destination"
                              issues += report("unavailable", path, reason)
                            case Some(changes) =>
                              changes.foreach { (field, value) =>
                                if !conflicted(field) then
                                  owners.get(field) match
                                    case Some(previous) if values(field) != value =>
                                      values.remove(field)
                                      conflicted += field
                                      issues += report("alias-conflict", path,
                                        s"$previous and $id disagree for ${field.render}")
                                      opaque += id -> member.value
                                    case _ =>
                                      values(field) = value
                                      owners.getOrElseUpdate(field, id)
                              }
                        case _ =>
                          opaque += id -> member.value
                          issues += report("unavailable", path,
                            if resolution == Right(None) then "no verified native binding"
                            else "source value is not a supported scalar")
                    }
                    val collection = if document.format == "preset-utilities" then "utilities"
                      else if document.mode.contains(PresetMode.Txt2Img) then "config-txt2img" else "config-img2img"
                    val record = PresetRecord(
                      PresetRecord.initialId(collection, name), name, values.toMap,
                      Some(PresetSource(document.format, name, document.mode, preset.value)),
                      Set.empty, opaque.result(), Vector.empty)
                    results += CompatibilityResult(record, issues.result())
        }
        failure match
          case Some(e) => Left(e)
          case None => Right(results.result())
    }

  private def reverseKey(format: String, mode: Option[PresetMode], field: FieldKey,
      original: Vector[RawMember], bindings: Vector[SourceBinding]): Option[String] =
    val explicit = bindings.filter(b => b.format == format && b.target == field).map(_.sourceKey)
    val oldVerified = original.map(_.name).find { name =>
      resolvedRoute(name, format, mode, bindings).toOption.flatten.exists(_.target == field)
    }
    oldVerified.orElse(explicit.headOption).orElse {
      val sourceModeName = mode.map(sourceMode).getOrElse("txt2img")
      val suffix = field.segments match
        case Vector("generation", "prompt") => Some("prompt")
        case Vector("generation", "negativePrompt") => Some("neg_prompt")
        case Vector("generation", "sampler") => Some("sampling")
        case Vector("generation", "scheduler") => Some("scheduler")
        case Vector("generation", "steps") => Some("steps")
        case Vector("generation", "width") => Some("width")
        case Vector("generation", "height") => Some("height")
        case Vector("generation", "batchCount") => Some("batch_count")
        case Vector("generation", "batchSize") => Some("batch_size")
        case Vector("generation", "cfgScale") => Some("cfg_scale")
        case Vector("generation", "seed") => Some("seed")
        case Vector("generation", "subseed") => Some("subseed")
        case Vector("generation", "subseedStrength") => Some("subseed_strength")
        case Vector("generation", "variationEnabled") => Some("subseed_show")
        case Vector("generation", "hiresEnabled") => Some("enable_hr")
        case Vector("generation", "hiresScale") => Some("hr_scale")
        case Vector("generation", "hiresUpscaler") => Some("hr_upscaler")
        case Vector("generation", "hiresSteps") => Some("hires_steps")
        case Vector("generation", "hiresResizeX") => Some("hr_resize_x")
        case Vector("generation", "hiresResizeY") => Some("hr_resize_y")
        case Vector("generation", "denoisingStrength") => Some("denoising_strength")
        case Vector("generation", "maskBlur") => Some("mask_blur")
        case Vector("generation", "inpaintingFill") => Some("inpainting_fill")
        case Vector("generation", "maskInvert") => Some("mask_mode")
        case Vector("generation", "inpaintFullRes") => Some("inpaint_full_res")
        case Vector("generation", "inpaintPadding") => Some("inpaint_full_res_padding")
        case Vector("generation", "resizeMode") => Some("resize_mode")
        case Vector("plugin", "face_restore", "enabled") => Some("restore_faces")
        case Vector("plugin", "asymmetric_tiling", "enabled") => Some("tiling")
        case _ => None
      if format == "preset-utilities" then
        val preferred = Vector("Prompt", "Negative prompt", "Sampling steps", "Sampling method",
          "Width", "Height", "Restore faces", "Tiling", "Hires. fix", "Upscaler",
          "Upscale by", "Hires steps", "Resize width to", "Resize height to",
          "Hires sampling method", "Seed", "Extra", "Variation seed", "Variation strength",
          "Denoising strength", "Batch count", "Batch size", "CFG Scale", "Script",
          "Resize mode", "Mask blur", "Mask mode", "Masked content", "Inpaint area",
          "Only masked padding, pixels")
        preferred.find(label => utility.get(label).exists(_.target == field))
          .orElse(xyzCanonical.collectFirst {
            case (label, control) if field == key("plugin", "xyz_grid", control) => label
          })
      else field.segments match
        case Vector("generation", "hiresPrompt") => Some("hires_prompt")
        case Vector("generation", "hiresNegative") => Some("hires_neg_prompt")
        case Vector("generation", "hiresSampler") => Some("hr_sampler")
        case Vector("generation", "resizeMode") if mode.contains(PresetMode.Img2Img) => Some("resize_mode")
        case Vector("script", "selected") => Some("script_list")
        case Vector("plugin", "xyz_grid", control) if
            Set("x_type", "y_type", "z_type", "x_values", "y_values", "z_values")(control) =>
          Some(s"script_${sourceModeName}_xyz_plot_${control}")
        case Vector("controlnet", slot, control) if slot.toIntOption.exists(n => n >= 0 && n < 3) =>
          controlNetKeys.find(_._2 == control).map { (source, _) =>
            s"${sourceModeName}_controlnet_ControlNet-${slot}_controlnet_${source}"
          }
        case _ => suffix.map(s => s"${sourceModeName}_${s}").filter { candidate =>
          mode.exists(m => configRoute(candidate, m).exists(_.target == field))
        }
    }

  private def standardMetadataIds(format: String, mode: Option[PresetMode]): Vector[String] =
    if format == "preset-utilities" then (utility.keysIterator ++ xyzLabels.keysIterator).toVector.sorted
    else
      val m = mode.get
      val prefix = sourceMode(m) + "_"
      val suffixes = commonConfig.keysIterator ++
        (if m == PresetMode.Txt2Img then txtConfig.keysIterator else imgConfig.keysIterator)
      val extras =
        if m == PresetMode.Txt2Img then Vector("hr_sampler", "hires_prompt", "hires_neg_prompt")
        else Vector("resize_mode")
      val xyz = for axis <- Vector("x", "y", "z"); part <- Vector("type", "values")
        yield s"script_${sourceMode(m)}_xyz_plot_${axis}_${part}"
      val control = for slot <- 0 until 3; name <- controlNetKeys.keys.toVector.sorted
        yield s"${sourceMode(m)}_controlnet_ControlNet-${slot}_controlnet_${name}"
      (suffixes.map(prefix + _) ++ extras ++ Vector("script_list") ++ xyz ++ control).toVector.sorted

  /** Preferred, verified source IDs for field inspection. A compound ID is only metadata. */
  def sourceIdsForField(field: FieldKey, format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding]): Either[PresetIssue, Vector[String]] =
    for
      _ <- metadataContext(format, mode)
      _ <- metadataField(field)
      _ <- metadataBindings(bindings)
      ids <- {
        val explicit = bindings.iterator.filter(b => b.format == format && b.target == field)
          .map(_.sourceKey).toVector.distinct.sorted
        val canonical = reverseKey(format, mode, field, Vector.empty, Vector.empty).toVector
        val synonyms =
          if format == "config-presets" then
            (explicit ++ canonical).flatMap(synonymsOf).distinct.sorted
          else Vector.empty[String]
        val candidates = (explicit ++ canonical ++ synonyms ++ standardMetadataIds(format, mode)).distinct
        val found = Vector.newBuilder[String]
        var conflict: Option[PresetIssue] = None
        candidates.foreach { id =>
          val standardForField = route(id, format, mode).exists(_.target == field)
          val relevant = explicit.contains(id) || canonical.contains(id) ||
            synonyms.contains(id) || standardForField
          if relevant && conflict.isEmpty && sourceKeyMatchesMode(id, format, mode) then
            resolvedRoute(id, format, mode, bindings) match
              case Left(issue) => conflict = Some(issue)
              case Right(Some(r)) if r.target == field => found += id
              case _ => ()
        }
        conflict match
          case Some(issue) => Left(issue)
          case None => Right(found.result())
      }
    yield ids

  /** Convert a complete tracked set to source IDs; this does not write the text file. */
  def exportTrackedIds(tracked: Set[FieldKey], format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding], supportedOnly: Boolean): Either[PresetError, Vector[String]] =
    def asError(issue: PresetIssue): PresetError = PresetError(Vector(issue))
    metadataContext(format, mode).left.map(asError).flatMap { _ =>
      metadataBindings(bindings).left.map(asError).flatMap { _ =>
        if tracked == null || tracked.size > limits.maxRegisteredFieldsPerMode then
          Left(error("limit", "$", "invalid or oversized tracked set"))
        else if tracked.exists(f => !validFieldKey(f)) then
          Left(error("field-key", "$", "invalid native field key"))
        else
          val ids = mutable.LinkedHashMap.empty[String, FieldKey]
          val issues = Vector.newBuilder[PresetIssue]
          val completeTiling = tilingKeys.subsetOf(tracked)
          val compound =
            if format == "preset-utilities" then "Tiling"
            else s"${sourceMode(mode.get)}_tiling"
          val verifiedCompound = resolvedRoute(compound, format, mode, bindings) match
            case Right(Some(r)) => r.conversion == Conversion.Tiling &&
              r.target == key("plugin", "asymmetric_tiling", "enabled")
            case _ => false
          val grouped = completeTiling && verifiedCompound
          if grouped then ids(compound) = key("plugin", "asymmetric_tiling", "enabled")
          tracked.toVector.sortBy(_.render).foreach { field =>
            if !(grouped && tilingKeys(field)) then
              sourceIdsForField(field, format, mode, bindings) match
                case Left(issue) => issues += issue.copy(path = field.render)
                case Right(candidates) =>
                  val plain = candidates.find { candidate =>
                    resolvedRoute(candidate, format, mode, bindings) match
                      case Right(Some(r)) => r.target == field && r.conversion != Conversion.Tiling
                      case _ => false
                  }
                  plain match
                    case None => issues += report("unsupported", field.render,
                      "field has no independent source ID")
                    case Some(id) if ids.contains(id) && ids(id) != field =>
                      issues += report("alias-conflict", field.render,
                        "two fields export to one source ID")
                    case Some(id) => ids(id) = field
          }
          val omissions = issues.result()
          if omissions.exists(_.code != "unsupported") ||
              (omissions.nonEmpty && !supportedOnly) then Left(PresetError(omissions))
          else if ids.size > limits.maxBindings then
            Left(error("limit", "$", "too many tracked source IDs"))
          else Right(ids.keys.toVector.sorted)
      }
    }

  private def sourceOption(value: PresetAtom, labels: Vector[String]): Option[Int] = value match
    case PresetAtom.Text(text) => Option.when(labels.contains(text))(labels.indexOf(text))
    case other => exactLong(other).filter(n => n >= 0 && n < labels.size).map(_.toInt)

  private def encodedValue(value: PresetAtom, route: Route, format: String,
      sourceKey: String): Either[PresetError, SourceValue] =
    val sourceValue: Option[PresetAtom] = route.conversion match
      case Conversion.Plain => Some(value)
      case Conversion.SameSampler => value match
        case PresetAtom.Text(text) =>
          Some(PresetAtom.Text(if text.isEmpty then "Use same sampler" else text))
        case _ => None
      case Conversion.Script => value match
        case PresetAtom.Text("") => Some(PresetAtom.Text("None"))
        case PresetAtom.Text("xyz_grid") => Some(PresetAtom.Text("X/Y/Z plot"))
        case PresetAtom.Text("loopback") => Some(PresetAtom.Text("Loopback"))
        case PresetAtom.Text("sd_upscale") => Some(PresetAtom.Text("SD upscale"))
        case _ => None
      case Conversion.ResizeMode | Conversion.MaskedContent =>
        val labels = if route.conversion == Conversion.ResizeMode then resizeLabels else contentLabels
        sourceOption(value, labels).map { index =>
          if format == "preset-utilities" then PresetAtom.Text(labels(index))
          else PresetAtom.Integer(index.toLong)
        }
      case Conversion.MaskMode | Conversion.InpaintArea =>
        val labels = if route.conversion == Conversion.MaskMode then
          Vector("Inpaint masked", "Inpaint not masked")
        else Vector("Whole picture", "Only masked")
        val index = value match
          case PresetAtom.Flag(flag) => Some(if flag then 1 else 0)
          case other => sourceOption(other, labels)
        index.map { n =>
          if format == "preset-utilities" then PresetAtom.Text(labels(n))
          else PresetAtom.Integer(n.toLong)
        }
      case Conversion.Tiling => None // a five-field compound, handled before scalar fields
    sourceValue match
      case Some(atom) => Right(SourceValue.Atom(atom))
      case None => Left(error("unsupported", sourceKey, "native value has no equivalent source value"))

  private def validName(value: String): Boolean =
    value != null && value.codePointCount(0, value.length) <= limits.maxNameCodePoints &&
      value.codePoints().toArray.exists(cp => !Character.isWhitespace(cp)) &&
      !value.codePoints().toArray.exists(cp => Character.isISOControl(cp))

  private def validSourceKey(value: String): Boolean =
    value != null && value.nonEmpty && value.length <= limits.maxKeyBytes &&
      value.getBytes(UTF_8).length <= limits.maxKeyBytes

  private def validFieldKey(value: FieldKey): Boolean =
    if value == null || value.segments == null || value.segments.isEmpty ||
        value.segments.size > limits.maxKeyBytes ||
        value.segments.exists(s => s == null || s.isEmpty || s.length > limits.maxKeyBytes) then false
    else
      var bytes = value.segments.size - 1
      value.segments.foreach { segment => bytes += segment.getBytes(UTF_8).length }
      if bytes > limits.maxKeyBytes then false
      else
        val encoded = value.render // segments and aggregate bytes were capped before encoding
        encoded.length <= limits.maxKeyBytes &&
          encoded.getBytes(UTF_8).length <= limits.maxKeyBytes &&
          FieldKey.parse(encoded).toOption.contains(value)

  private def exportPreflight(records: Vector[PresetRecord],
      bindings: Vector[SourceBinding]): Either[PresetError, Unit] =
    if records.size > limits.maxRecords || bindings.size > limits.maxBindings then
      return Left(error("limit", "$", "too many source records or bindings"))
    val names = mutable.HashSet.empty[String]
    var failure: Option[PresetError] = None
    var at = 0
    while at < records.size && failure.isEmpty do
      val record = records(at)
      if record == null || !validName(record.name) then
        failure = Some(error("name", "$", "invalid preset name"))
      else if !names.add(record.name) then
        failure = Some(error("duplicate-name", record.name, "duplicate preset name"))
      else if record.fields == null || record.opaque == null || record.dirty == null ||
          record.fields.size + record.opaque.size > limits.maxFieldsPerRecord ||
          record.dirty.size > limits.maxFieldsPerRecord then
        failure = Some(error("limit", record.name, "too many source fields"))
      else if record.fields.exists((field, atom) => !validFieldKey(field) || atom == null) ||
          record.dirty.exists(field => !validFieldKey(field)) then
        failure = Some(error("field-key", record.name, "invalid native field key or value"))
      else if record.opaque.exists((name, raw) => !validSourceKey(name) || raw == null) then
        failure = Some(error("source-key", record.name, "invalid opaque source key or value"))
      at += 1
    var bindingAt = 0
    while bindingAt < bindings.size && failure.isEmpty do
      val binding = bindings(bindingAt)
      if binding == null || !validSourceKey(binding.format) ||
          !validSourceKey(binding.sourceKey) || !validFieldKey(binding.target) then
        failure = Some(error("binding", "$", "invalid source binding"))
      bindingAt += 1
    failure match
      case Some(e) => Left(e)
      case None => Right(())

  private final class ExportIssueCollector:
    private val issues = Vector.newBuilder[PresetIssue]
    private var count = 0
    def result: Vector[PresetIssue] = issues.result()
    def add(recordName: String, issue: PresetIssue): Option[PresetError] =
      // A valid native/source document has at most maxNodes lexical nodes. Every
      // omission needs at least one field node, so this does not lower valid-library
      // limits. Check before append; never expose a partial vector on overflow.
      if count >= limits.maxNodes then Some(error("limit", recordName, "too many export diagnostics"))
      else
        count += 1
        issues += issue.copy(path = s"$recordName: ${issue.path}")
        None

  private def exportOne(record: PresetRecord, format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding],
      supportedOnly: Boolean,
      assessment: Option[ExportIssueCollector] = None): Either[PresetError, (String, Vector[(String, SourceValue)])] =
    if record.source.exists(_.format != format) && !supportedOnly then
      return Left(error("format", record.name, "record provenance belongs to another source format"))
    val originals = record.source.filter(_.format == format).map { source =>
      PresetJson.objectMembers(source.originalFields, limits)
    }.getOrElse(Right(Vector.empty))
    originals.flatMap { rawMembers =>
      if rawMembers.size > limits.maxFieldsPerRecord then
        Left(error("limit", record.name, "too many original source fields"))
      else
        val entries = mutable.LinkedHashMap.empty[String, SourceValue]
        rawMembers.foreach(m => entries(m.name) = SourceValue.Raw(m.value))
        record.opaque.foreach { (name, raw) =>
          if !entries.contains(name) then entries(name) = SourceValue.Raw(raw)
        }
        val fromSource = record.source.exists(_.format == format)
        val initialTouched = if fromSource then record.dirty else record.fields.keySet ++ record.dirty
        val enabled = key("plugin", "asymmetric_tiling", "enabled")
        val sourceTiling = if format == "preset-utilities" then "Tiling"
          else s"${mode.map(sourceMode).getOrElse("txt2img")}_tiling"
        val originalTiling = rawMembers.filter(m =>
          resolvedRoute(m.name, format, mode, bindings).toOption.flatten.exists(r =>
            r.target == enabled && r.conversion == Conversion.Tiling))
        val selectedEnabled = reverseKey(format, mode, enabled, rawMembers, bindings)
        val compoundSelected = selectedEnabled.exists(sourceKey =>
          resolvedRoute(sourceKey, format, mode, bindings).toOption.flatten.exists(r =>
            r.target == enabled && r.conversion == Conversion.Tiling))
        val extraDrafts = record.fields.keySet.intersect(tilingKeys - enabled)
        val disabledOrRemoved = record.fields.get(enabled) match
          case Some(PresetAtom.Flag(false)) | None => true
          case _ => false
        val touched = if disabledOrRemoved &&
            (initialTouched.contains(enabled) ||
              (originalTiling.nonEmpty && initialTouched.exists(tilingKeys))) then
          initialTouched ++ extraDrafts
          else initialTouched
        var covered = Set.empty[FieldKey]
        var failure: Option[PresetError] = None
        def refuse(e: PresetError): Unit = assessment match
          case None => if !supportedOnly then failure = Some(e)
          case Some(collector) =>
            if e.issues.exists(_.code != "unsupported") then failure = Some(e)
            else e.issues.foreach { issue =>
              if failure.isEmpty then failure = collector.add(record.name, issue)
            }
        if compoundSelected && (touched.contains(enabled) ||
            (originalTiling.nonEmpty && touched.exists(tilingKeys))) then
          originalTiling.foreach(m => entries.remove(m.name))
          val candidate = selectedEnabled.getOrElse(sourceTiling)
          val validTilingRoute = resolvedRoute(candidate, format, mode, bindings) match
            case Right(Some(r)) => r.target == enabled && r.conversion == Conversion.Tiling
            case _ => false
          if !validTilingRoute then
            refuse(error("alias-conflict", candidate, "Tiling source key has conflicting binding"))
            covered = Set(enabled)
          else record.fields.get(enabled) match
            case Some(PresetAtom.Flag(true)) if tilingTrueValues.forall((k, v) =>
                record.fields.get(k).contains(v)) =>
              entries(candidate) = SourceValue.Atom(PresetAtom.Flag(true))
              covered = tilingKeys
            case Some(PresetAtom.Flag(true)) =>
              refuse(error("unsupported", candidate, "Tiling=true needs X/Y and full-time defaults"))
              covered = Set(enabled)
            case Some(PresetAtom.Flag(false)) =>
              entries(candidate) = SourceValue.Atom(PresetAtom.Flag(false))
              covered = Set(enabled)
            case None =>
              // An explicitly deselected source field removes the old compound.
              covered = tilingKeys -- record.fields.keySet
            case _ =>
              refuse(error("unsupported", candidate, "Tiling requires a boolean enabled field"))
              covered = Set(enabled)
        touched.diff(covered).toVector.sortBy(_.render).foreach { field =>
          if failure.isEmpty then
            rawMembers.foreach { member =>
              if resolvedRoute(member.name, format, mode, bindings).toOption.flatten.exists(_.target == field) then
                entries.remove(member.name)
            }
            reverseKey(format, mode, field, rawMembers, bindings) match
              case None => refuse(error("unsupported", field.render, "field has no source export binding"))
              case Some(sourceKey) =>
                resolvedRoute(sourceKey, format, mode, bindings) match
                  case Right(Some(sourceRoute)) if sourceRoute.target == field &&
                      sourceRoute.conversion != Conversion.Tiling =>
                    record.fields.get(field).foreach { value =>
                      if entries.contains(sourceKey) then
                        refuse(error("alias-conflict", sourceKey, "two fields export to one source key"))
                      else
                        encodedValue(value, sourceRoute, format, sourceKey) match
                          case Left(e) => refuse(e)
                          case Right(encoded) => entries(sourceKey) = encoded
                    }
                  case _ => refuse(error("alias-conflict", sourceKey,
                    "reverse binding would not import to the same native field"))
        }
        failure match
          case Some(e) => Left(e)
          case None if entries.size > limits.maxFieldsPerRecord =>
            Left(error("limit", record.name, "too many exported source fields"))
          case None => Right(record.name -> entries.toVector)
    }

  /** Complete read-only omission review using exactly the record export routes, dirty
    * fields and compound rules. Invalid input, alias conflicts and budget exhaustion
    * are hard errors; no partial review or encoded output is returned in those cases.
    */
  def exportRecordIssues(records: Vector[PresetRecord], format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding]): Either[PresetError, Vector[PresetIssue]] =
    if records == null || bindings == null then return Left(error("source", "$", "null export data"))
    if mode == null then return Left(error("mode", "$", "null source mode"))
    metadataContext(format, mode).left.map(issue => PresetError(Vector(issue))).flatMap { _ =>
      exportPreflight(records, bindings).flatMap { _ =>
        validateAssessmentBindings(bindings, format, mode).flatMap { _ =>
          val collector = new ExportIssueCollector
          // Stream one assessed record at a time through the existing source limits.
          // No whole-native-envelope budget: raw provenance and typed atoms may overlap.
          val assessed = records.iterator.map { record =>
            validateAssessmentRecord(record).flatMap { _ =>
              exportOne(record, format, mode, bindings, supportedOnly = false, assessment = Some(collector))
            }
          }
          PresetJson.validateSourceRecords(assessed, limits).map(_ => collector.result)
        }
      }
    }

  private def validateAssessmentBindings(bindings: Vector[SourceBinding], format: String,
      mode: Option[PresetMode]): Either[PresetError, Unit] =
    metadataBindings(bindings) match
      case Left(issue) => return Left(PresetError(Vector(issue)))
      case _ => ()
    val active = bindings.iterator.filter(_.format == format)
    while active.hasNext do
      resolvedRoute(active.next().sourceKey, format, mode, bindings) match
        case Left(issue) => return Left(PresetError(Vector(issue)))
        case _ => ()
    Right(())

  private def validateAssessmentRecord(record: PresetRecord): Either[PresetError, Unit] =
    if record.id == null || record.source == null || record.extra == null then
      return Left(error("source", record.name, "invalid preset record"))
    record.source match
      case Some(source) if source == null || source.originalFields == null || source.mode == null =>
        return Left(error("source", record.name, "invalid source provenance"))
      case Some(source) =>
        if !validName(source.name) then return Left(error("name", record.name, "invalid source preset name"))
        metadataContext(source.format, source.mode) match
          case Left(issue) => return Left(PresetError(Vector(issue.copy(path = record.name))))
          case _ => ()
      case None => ()
    val atoms = record.fields.valuesIterator
    while atoms.hasNext do
      PresetJson.validateSourceAtom(atoms.next(), limits) match
        case Left(e) => return Left(e)
        case _ => ()
    Right(())

  def exportRecords(records: Vector[PresetRecord], format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding], supportedOnly: Boolean): Either[PresetError, Array[Byte]] =
    if records == null || bindings == null then return Left(error("source", "$", "null export data"))
    if !Set("config-presets", "preset-utilities")(format) then
      return Left(error("format", "$", "unsupported source format"))
    if format == "config-presets" &&
        !mode.exists(m => m == PresetMode.Txt2Img || m == PresetMode.Img2Img) then
      return Left(error("mode", "$", "Config Presets export requires explicit mode"))
    if format == "preset-utilities" && mode.nonEmpty then
      return Left(error("mode", "$", "Preset Utilities export is shared"))
    exportPreflight(records, bindings).flatMap { _ =>
      PresetJson.writeSourceRecords(
        records.iterator.map(r => exportOne(r, format, mode, bindings, supportedOnly)), limits)
    }

  private[presets] def configDefaultKeys(mode: PresetMode,
      schemas: Map[PresetMode, PresetSchema]): Set[FieldKey] =
    val catalog = available(schemas, "config-presets", Some(mode)).keySet
    val common = Vector("sampling", "scheduler", "steps", "width", "height", "batch_count",
      "batch_size", "cfg_scale", "restore_faces")
    val hires = Vector("enable_hr", "hr_scale", "hr_upscaler", "hires_steps", "denoising_strength")
    val all = common ++ (if mode == PresetMode.Txt2Img then hires else Vector("denoising_strength"))
    all.flatMap(s => configRoute(s"${sourceMode(mode)}_${s}", mode).map(_.target)).filter(catalog).toSet

  private[presets] def utilityDefaultKeys(schemas: Map[PresetMode, PresetSchema]): Set[FieldKey] =
    val catalog = available(schemas, "preset-utilities", None).keySet
    val direct = utility.values.map(_.target).filter(catalog).toSet
    if tilingKeys.subsetOf(catalog) then direct ++ tilingKeys else direct
