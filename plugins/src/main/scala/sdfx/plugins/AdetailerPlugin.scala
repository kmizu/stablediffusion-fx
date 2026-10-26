package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** ADetailer (Bing-su/adetailer), re-implemented: detects faces / hands / people after generation and inpaints each
  * detection at full resolution. Two units ("1st", "2nd") like the extension's default tabs.
  * Engine side: python/sdfx/adetailer.py + Engine._adetailer.
  */
class AdetailerPlugin extends HookPlugin:
  import AdetailerPlugin.*
  def id = "adetailer"
  def name = "ADetailer"
  def hook = "adetailer"
  override def description =
    "Finds faces, hands or people in each result and redraws them at full resolution. YOLO models download from " +
      "Bingsu/adetailer on first use; opencv_face needs no download."

  override def controls: Seq[Control] = Check("enabled", "Enabled") +: Units.flatMap(unitControls)

  override def enabledBy: Map[String, String] = Units.flatMap { u =>
    val k = u.key
    Seq(s"${k}_inpaint_width" -> s"${k}_use_inpaint_width_height", s"${k}_inpaint_height" -> s"${k}_use_inpaint_width_height",
      s"${k}_steps" -> s"${k}_use_steps", s"${k}_cfg_scale" -> s"${k}_use_cfg_scale", s"${k}_sampler" -> s"${k}_use_sampler",
      s"${k}_inpaint_padding" -> s"${k}_inpaint_only_masked")
  }.toMap

  private def unitControls(u: AdUnit): Seq[Control] =
    val k = u.key
    Seq(
      Section(s"${k}_main", s"${u.label} unit", expanded = u.index == 0),
      Choice(s"${k}_model", "Detection model", Models, if u.index == 0 then "face_yolov8n.pt" else "None"),
      Text(s"${k}_prompt", "Prompt", multiline = true, placeholder = "Empty = main prompt. [PROMPT] inserts it, [SEP] splits per detection"),
      Text(s"${k}_negative_prompt", "Negative prompt", multiline = true, placeholder = "Empty = main negative prompt"),
      Slider(s"${k}_confidence", "Detection confidence", 0, 1, 0.01, 0.3),
      Slider(s"${k}_denoising_strength", "Inpaint denoising strength", 0, 1, 0.01, 0.4),
      Section(s"${k}_advanced", s"${u.label} unit: mask and inpainting", expanded = false),
      Text(s"${k}_custom_model", "Custom model (path or Bingsu/adetailer file name, overrides the list)"),
      IntField(s"${k}_mask_k_largest", "Mask only the top k largest (0 = all)", 0, 10, 0),
      Slider(s"${k}_mask_min_ratio", "Mask min area ratio", 0, 1, 0.001, 0.0),
      Slider(s"${k}_mask_max_ratio", "Mask max area ratio", 0, 1, 0.001, 1.0),
      IntField(s"${k}_x_offset", "Mask x offset", -200, 200, 0),
      IntField(s"${k}_y_offset", "Mask y offset", -200, 200, 0),
      IntField(s"${k}_dilate_erode", "Mask erosion (-) / dilation (+)", -128, 128, 4),
      Choice(s"${k}_mask_merge_invert", "Mask merge mode", Seq("None", "Merge", "Merge and Invert"), "None"),
      IntField(s"${k}_mask_blur", "Inpaint mask blur", 0, 64, 4),
      Check(s"${k}_inpaint_only_masked", "Inpaint only masked", true),
      IntField(s"${k}_inpaint_padding", "Inpaint only masked padding", 0, 256, 32),
      Check(s"${k}_color_correction", "Match colour and contrast of the redrawn area to the original", true),
      Check(s"${k}_use_inpaint_width_height", "Use separate width/height"),
      IntField(s"${k}_inpaint_width", "Inpaint width", 64, 2048, 512),
      IntField(s"${k}_inpaint_height", "Inpaint height", 64, 2048, 512),
      Check(s"${k}_use_steps", "Use separate steps"),
      IntField(s"${k}_steps", "Steps", 1, 150, 28),
      Check(s"${k}_use_cfg_scale", "Use separate CFG scale"),
      Slider(s"${k}_cfg_scale", "CFG scale", 0, 30, 0.5, 7.0),
      Check(s"${k}_use_sampler", "Use separate sampler"),
      DynamicChoice(s"${k}_sampler", "Sampler", ChoiceSource.Samplers, "DPM++ 2M", allowNone = false)
    )

  def hookFields(v: Values): ujson.Obj =
    val units = Units.flatMap { u =>
      val k = u.key
      val custom = v.str(s"${k}_custom_model").trim
      val model = if custom.nonEmpty then custom else v.str(s"${k}_model", "None")
      if model == "None" || model.isEmpty then None
      else
        Some(ujson.Obj(
          "model" -> model,
          "prompt" -> v.str(s"${k}_prompt"), "negative_prompt" -> v.str(s"${k}_negative_prompt"),
          "confidence" -> v.double(s"${k}_confidence", 0.3),
          "mask_k_largest" -> v.int(s"${k}_mask_k_largest", 0),
          "mask_min_ratio" -> v.double(s"${k}_mask_min_ratio", 0.0), "mask_max_ratio" -> v.double(s"${k}_mask_max_ratio", 1.0),
          "x_offset" -> v.int(s"${k}_x_offset", 0), "y_offset" -> v.int(s"${k}_y_offset", 0),
          "dilate_erode" -> v.int(s"${k}_dilate_erode", 4), "mask_merge_invert" -> v.str(s"${k}_mask_merge_invert", "None"),
          "mask_blur" -> v.int(s"${k}_mask_blur", 4), "denoising_strength" -> v.double(s"${k}_denoising_strength", 0.4),
          "inpaint_only_masked" -> v.bool(s"${k}_inpaint_only_masked", true),
          "inpaint_padding" -> v.int(s"${k}_inpaint_padding", 32),
          "color_correction" -> v.bool(s"${k}_color_correction", true),
          "use_inpaint_width_height" -> v.bool(s"${k}_use_inpaint_width_height"),
          "inpaint_width" -> v.int(s"${k}_inpaint_width", 512), "inpaint_height" -> v.int(s"${k}_inpaint_height", 512),
          "use_steps" -> v.bool(s"${k}_use_steps"), "steps" -> v.int(s"${k}_steps", 28),
          "use_cfg_scale" -> v.bool(s"${k}_use_cfg_scale"), "cfg_scale" -> v.double(s"${k}_cfg_scale", 7.0),
          "use_sampler" -> v.bool(s"${k}_use_sampler"), "sampler" -> v.str(s"${k}_sampler", "DPM++ 2M")))
    }
    ujson.Obj("units" -> ujson.Arr.from(units))

  override def isActive(v: Values): Boolean = v.bool("enabled") && hookFields(v)("units").arr.nonEmpty

  /** ADetailer's own infotext keys ("ADetailer model", "ADetailer model 2nd", ...). XYZ overrides apply to the 1st. */
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    val units = o.value.get("units").map(_.arr.toSeq).getOrElse(Nil)
    val top = o.value.filter((k, _) => k != "units")
    units.zipWithIndex.flatMap { (u, i) =>
      val m = if i == 0 then u.obj ++ top else u.obj
      val suffix = if i == 0 then "" else s" ${Units(i).label}"
      def num(k: String) = m.get(k).map(Hooks.num).getOrElse("")
      Seq(
        s"ADetailer model$suffix" -> num("model"),
        s"ADetailer confidence$suffix" -> num("confidence"),
        s"ADetailer dilate erode$suffix" -> num("dilate_erode"),
        s"ADetailer mask blur$suffix" -> num("mask_blur"),
        s"ADetailer denoising strength$suffix" -> num("denoising_strength"),
        s"ADetailer inpaint only masked$suffix" -> num("inpaint_only_masked"),
        s"ADetailer inpaint padding$suffix" -> num("inpaint_padding")
      ) ++ Seq("prompt" -> "prompt", "negative_prompt" -> "negative prompt").flatMap { (k, label) =>
        m.get(k).map(_.str).filter(_.nonEmpty).map(t => s"ADetailer $label$suffix" -> t)
      }
    } ++ (if units.nonEmpty then Seq("ADetailer version" -> "sdfx") else Nil)

  override def xyzAxes: Seq[XyzAxis] = Seq(
    numAxis("ADetailer denoising strength", "denoising_strength"),
    numAxis("ADetailer confidence", "confidence"),
    numAxis("ADetailer mask blur", "mask_blur"),
    numAxis("ADetailer dilate erode", "dilate_erode"),
    XyzAxis("ADetailer model")((p, s) => Hooks.setOverride(p, hook, "model", ujson.Str(s.trim)))
  )

object AdetailerPlugin:
  final case class AdUnit(index: Int, key: String, label: String)
  val Units: Seq[AdUnit] = Seq(AdUnit(0, "u1", "1st"), AdUnit(1, "u2", "2nd"))
  val Models: Seq[String] = Seq("None", "face_yolov8n.pt", "face_yolov8s.pt", "face_yolov8m.pt", "face_yolov9c.pt",
    "hand_yolov8n.pt", "hand_yolov8s.pt", "hand_yolov9c.pt", "person_yolov8n-seg.pt", "person_yolov8s-seg.pt",
    "person_yolov8m-seg.pt", "deepfashion2_yolov8s-seg.pt", "opencv_face")
