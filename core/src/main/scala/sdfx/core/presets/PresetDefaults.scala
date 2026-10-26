package sdfx.core.presets

import java.nio.charset.StandardCharsets.UTF_8

/** The Config entries follow the pinned MIT source's names, order and scalar values.
  * The Utilities reset is authored from this app's declared component defaults.
  */
object PresetDefaults:
  private type Values = Vector[(String, String)]
  private def v(entries: (String, String)*): Values = entries.toVector
  private def quoted(value: String): String = ujson.write(ujson.Str(value))

  private val txtDefaults: Vector[(String, Values)] = Vector(
    "None" -> v(),
    "SD1.5 - 512x512" -> v("txt2img_width" -> "512", "txt2img_height" -> "512"),
    "SD2.1 - 768x768" -> v("txt2img_width" -> "768", "txt2img_height" -> "768"),
    "SDXL --- 1024x1024" -> v("txt2img_width" -> "1024", "txt2img_height" -> "1024"),
    "SDXL --- 1024x1024 with Refiner" ->
      v("txt2img_width" -> "1024", "txt2img_height" -> "1024", "txt2img_enable-checkbox" -> "true"),
    "Flux.1 Dev - 256x256 to 1920x1080 (0.1 to 2.0 megapixels), 20 steps, CFG 1, Distilled CFG 3.5, Euler Simple" ->
      v("txt2img_width" -> "1024", "txt2img_height" -> "1024", "txt2img_sampling" -> quoted("Euler"),
        "txt2img_scheduler" -> quoted("Simple"), "txt2img_distilled_cfg_scale" -> "3.5",
        "txt2img_cfg_scale" -> "1", "txt2img_steps" -> "20"),
    "Flux.1 Schnell - 256x256 to 1920x1080 (0.1 to 2.0 megapixels), 4 steps, CFG 1, Distilled CFG 3.5, Euler Simple" ->
      v("txt2img_width" -> "1024", "txt2img_height" -> "1024", "txt2img_sampling" -> quoted("Euler"),
        "txt2img_scheduler" -> quoted("Simple"), "txt2img_distilled_cfg_scale" -> "3.5",
        "txt2img_cfg_scale" -> "1", "txt2img_steps" -> "4"),
    "High res ----------- [Hires fix - Upscale by: 2, Denoising: 0.3, Hires steps: 10]" ->
      v("txt2img_enable_hr" -> "true", "txt2img_hr_scale" -> "2", "txt2img_hires_steps" -> "10",
        "txt2img_denoising_strength" -> "0.3", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "1"),
    "SD1.5 Low quality ------ steps: 10, batch size: 4, DPM++ 2M" ->
      v("txt2img_sampling" -> quoted("DPM++ 2M"), "txt2img_steps" -> "10",
        "txt2img_enable_hr" -> "false", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "4"),
    "SD1.5 Medium quality - steps: 15, batch size: 4, DPM++ 2M" ->
      v("txt2img_sampling" -> quoted("DPM++ 2M"), "txt2img_steps" -> "15",
        "txt2img_enable_hr" -> "false", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "4"),
    "SD1.5 High quality ------ steps: 30, batch size: 4, DPM++ 2S a" ->
      v("txt2img_sampling" -> quoted("DPM++ 2S a"), "txt2img_steps" -> "30",
        "txt2img_enable_hr" -> "false", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "4"),
    "SD1.5 - 1080p --- 432x768 -> 1920x1080, [Hires fix - Upscale by: 2.5, Denoising: 0.4, Hires steps: 10]" ->
      v("txt2img_width" -> "768", "txt2img_height" -> "432", "txt2img_enable_hr" -> "true",
        "txt2img_hr_scale" -> "2.5", "txt2img_hires_steps" -> "10",
        "txt2img_denoising_strength" -> "0.4", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "1"),
    "SD1.5 - 1440p --- 432x768 -> 2560x1440, [Hires fix - Upscale by: 3.3334, Denoising: 0.35, Hires steps: 10]" ->
      v("txt2img_width" -> "768", "txt2img_height" -> "432", "txt2img_enable_hr" -> "true",
        "txt2img_hr_scale" -> "3.3334", "txt2img_hires_steps" -> "10",
        "txt2img_denoising_strength" -> "0.35", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "1"),
    "SD1.5 - 4k -------- 432x768 -> 3840x2160, [Upscale by: 5, Denoising: 0.3, Hires steps: 15]" ->
      v("txt2img_width" -> "768", "txt2img_height" -> "432", "txt2img_enable_hr" -> "true",
        "txt2img_hr_scale" -> "5", "txt2img_hires_steps" -> "15",
        "txt2img_denoising_strength" -> "0.3", "txt2img_batch_count" -> "1", "txt2img_batch_size" -> "1")
  )
  private val imgDefaults: Vector[(String, Values)] = Vector(
    "None" -> v(),
    "Low denoising ------- denoising: 0.25, steps: 20, DPM++ 2M" ->
      v("img2img_sampling" -> quoted("DPM++ 2M"), "img2img_steps" -> "20",
        "img2img_denoising_strength" -> "0.25"),
    "Medium denoising -- denoising: 0.40, steps: 20, DPM++ 2M" ->
      v("img2img_sampling" -> quoted("DPM++ 2M"), "img2img_steps" -> "20",
        "img2img_denoising_strength" -> "0.4"),
    "High denoising ------- denoising: 0.75, steps: 30, DPM++ 2M" ->
      v("img2img_sampling" -> quoted("DPM++ 2M"), "img2img_steps" -> "30",
        "img2img_denoising_strength" -> "0.75")
  )

  private def imported(defaults: Vector[(String, Values)], mode: PresetMode,
      schemas: Map[PresetMode, PresetSchema]): Vector[PresetRecord] =
    val source = defaults.map { (name, entries) =>
      val fields = entries.map { (id, value) => quoted(id) + ":" + value }.mkString(",")
      quoted(name) + ":{" + fields + "}"
    }.mkString("{", ",", "}")
    val raw = PresetJson.readSource(source.getBytes(UTF_8), PresetLimits())
      .fold(e => throw new IllegalStateException(e.issues.mkString("; ")), identity)
    PresetCompatibility.importRecords(SourceDocument("config-presets", Some(mode), raw, ""),
      schemas, Vector.empty)
      .fold(e => throw new IllegalStateException(e.issues.mkString("; ")), _.map(_.record))

  def library(schemas: Map[PresetMode, PresetSchema]): PresetLibrary =
    val txt = PresetCollection("config-txt2img",
      PresetCompatibility.configDefaultKeys(PresetMode.Txt2Img, schemas),
      Vector.empty, imported(txtDefaults, PresetMode.Txt2Img, schemas), Vector.empty)
    val img = PresetCollection("config-img2img",
      PresetCompatibility.configDefaultKeys(PresetMode.Img2Img, schemas),
      Vector.empty, imported(imgDefaults, PresetMode.Img2Img, schemas), Vector.empty)
    val all = Vector(PresetMode.Txt2Img, PresetMode.Img2Img, PresetMode.Inpaint)
    val common = all.flatMap(schemas.get).map(_.fields.map(f => f.key -> f).toMap)
    val defaults =
      if common.size != all.size then Map.empty[FieldKey, PresetAtom]
      else
        val shared = common.map(_.keySet).reduce(_ intersect _)
        shared.filter(PresetCompatibility.utilityDefaultKeys(schemas)).toVector.flatMap { field =>
          val values = common.map(_(field).defaultValue)
          Option.when(values.distinct.size == 1)(field -> values.head)
        }.toMap
    val reset = PresetRecord(PresetRecord.initialId("utilities", "Reset common controls"),
      "Reset common controls", defaults, None, Set.empty, Vector.empty, Vector.empty)
    val utilities = PresetCollection("utilities", PresetCompatibility.utilityDefaultKeys(schemas),
      Vector.empty, Vector(reset), Vector.empty)
    PresetLibrary(1, Vector(txt, img, utilities), Vector.empty)
