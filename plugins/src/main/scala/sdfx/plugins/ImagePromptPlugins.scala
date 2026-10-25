package sdfx.plugins


import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

import java.nio.file.{Files, Paths}

/** Image-conditioning AlwaysOn plugins. Image controls hold absolute file paths; model choices are names that the
  * host resolves to paths (see Generator.resolveHookModels).
  */
private[plugins] object ImageHooks:
  def image(v: Values, key: String = "image"): String = v.str(key).trim

  def images(v: Values, key: String): Seq[String] = v.str(key).split("[\r\n]+").map(_.trim).filter(_.nonEmpty).toSeq

  def exists(path: String): Boolean = path.nonEmpty && Files.isRegularFile(Paths.get(path))

  def fileName(path: String): String = if path.isEmpty then "" else Paths.get(path).getFileName.toString

/** Base for hook plugins that need an image: inactive (instead of failing) while no image is chosen. */
abstract class ImageHookPlugin extends HookPlugin:
  def needsImage: Boolean = true
  override def isActive(v: Values): Boolean = v.bool("enabled") && (!needsImage || ImageHooks.exists(ImageHooks.image(v)))

/** Reference-only (forge_preprocessor_reference). */
class ReferencePlugin extends ImageHookPlugin:
  def id = "reference"
  def name = "Reference only"
  def hook = "reference"
  override def description = "Self-attention also attends to a reference image: keeps its style / subject without a model."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    ImageFile("image", "Reference image"),
    Slider("style_fidelity", "Style fidelity", 0, 1, 0.01, 0.5),
    Slider("attention_auto_machine_weight", "Reference weight", 0, 2, 0.01, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("image" -> ImageHooks.image(v),
    "style_fidelity" -> v.double("style_fidelity", 0.5), "attention_auto_machine_weight" -> v.double("attention_auto_machine_weight", 1.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Reference image" -> ImageHooks.fileName(field(o, "image")),
    "Reference style fidelity" -> field(o, "style_fidelity"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Reference style fidelity", "style_fidelity"))

/** Revision (forge_preprocessor_revision): SDXL clip-vision image prompt. */
class RevisionPlugin extends ImageHookPlugin:
  def id = "revision"
  def name = "Revision (SDXL image prompt)"
  def hook = "revision"
  override def description = "SDXL: adds the CLIP-G embedding of an image to the pooled prompt (downloads the encoder once)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    ImageFile("image", "Image"),
    Slider("strength", "Strength", 0, 2, 0.01, 1.0),
    Check("ignore_prompt", "Ignore prompt (image only)")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("image" -> ImageHooks.image(v), "strength" -> v.double("strength", 1.0),
    "ignore_prompt" -> v.bool("ignore_prompt"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Revision image" -> ImageHooks.fileName(field(o, "image")),
    "Revision strength" -> field(o, "strength"), "Revision ignore prompt" -> field(o, "ignore_prompt"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Revision strength", "strength"))

/** IP-Adapter (sd_forge_ipadapter). Models live in the ControlNet folders like in forge. */
class IpAdapterPlugin extends ImageHookPlugin:
  def id = "ip_adapter"
  def name = "IP-Adapter"
  def hook = "ip_adapter"
  override def description = "Image prompt adapter (ip-adapter*.safetensors/.bin from the ControlNet folders)."
  override def isActive(v: Values): Boolean = super.isActive(v) && v.str("model").nonEmpty && v.str("model") != "None"
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    DynamicChoice("model", "IP-Adapter model", ChoiceSource.ControlNets, "", allowNone = true),
    ImageFile("image", "Image"),
    Slider("scale", "Weight", 0, 2, 0.01, 0.7),
    Text("image_encoder", "Image encoder (HF repo or folder; empty = h94/IP-Adapter default)")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("model" -> v.str("model"), "image" -> ImageHooks.image(v),
    "scale" -> v.double("scale", 0.7), "image_encoder" -> v.str("image_encoder").trim)
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("IP-Adapter" -> field(o, "model"),
    "IP-Adapter weight" -> field(o, "scale"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("IP-Adapter weight", "scale"))

/** ControlNet-LLLite (sd_forge_controlllite). */
class ControlLLLitePlugin extends HookPlugin:
  def id = "controllllite"
  def name = "ControlNet-LLLite"
  def hook = "controllllite"
  override def description = "kohya's lightweight SDXL ControlNet (model from the ControlNet folders). No image = init image."
  override def isActive(v: Values): Boolean = v.bool("enabled") && v.str("model").nonEmpty && v.str("model") != "None"
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    DynamicChoice("model", "LLLite model", ChoiceSource.ControlNets, "", allowNone = true),
    ImageFile("image", "Control image (already preprocessed)"),
    Slider("strength", "Strength", 0, 10, 0.01, 1.0),
    Slider("start_percent", "Start", 0, 1, 0.01, 0.0),
    Slider("end_percent", "End", 0, 1, 0.01, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("model" -> v.str("model"), "image" -> ImageHooks.image(v),
    "strength" -> v.double("strength", 1.0), "start_percent" -> v.double("start_percent", 0.0), "end_percent" -> v.double("end_percent", 1.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("ControlNet-LLLite" -> field(o, "model"),
    "ControlNet-LLLite strength" -> field(o, "strength"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("LLLite strength", "strength"))

/** Fooocus inpaint patch (sd_forge_fooocus_inpaint). */
class FooocusInpaintPlugin extends HookPlugin:
  def id = "fooocus_inpaint"
  def name = "Fooocus Inpaint"
  def hook = "fooocus_inpaint"
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "SDXL inpainting with Fooocus' patch (inpaint_v26.fooocus.patch + fooocus_inpaint_head.pth in a ControlNet folder)."
  override def isActive(v: Values): Boolean = v.bool("enabled") && v.str("model").nonEmpty && v.str("model") != "None"
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    DynamicChoice("model", "Patch (inpaint_v2x.fooocus.patch)", ChoiceSource.ControlNets, "", allowNone = true),
    Text("head", "Inpaint head file (empty = fooocus_inpaint_head.pth next to the patch)"),
    Slider("start_percent", "Start", 0, 1, 0.01, 0.0),
    Slider("end_percent", "End", 0, 1, 0.01, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("model" -> v.str("model"), "head" -> v.str("head").trim,
    "start_percent" -> v.double("start_percent", 0.0), "end_percent" -> v.double("end_percent", 1.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Fooocus inpaint" -> field(o, "model"))

/** PhotoMaker (sd_forge_photomaker). */
class PhotoMakerPlugin extends HookPlugin:
  def id = "photomaker"
  def name = "PhotoMaker"
  def hook = "photomaker"
  override def tabs: Set[Tab] = Set(Tab.Txt2Img)
  override def description = "SDXL identity from photos. Put the trigger word (default 'img') after the class word, e.g. 'a man img'."
  override def isActive(v: Values): Boolean = v.bool("enabled") && v.str("model").nonEmpty && ImageHooks.images(v, "images").nonEmpty
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    DynamicChoice("model", "PhotoMaker model", ChoiceSource.PhotoMakerModels, "", allowNone = true),
    ImageFile("images", "ID images", multiple = true),
    Text("trigger_word", "Trigger word", "img"),
    IntField("start_merge_step", "Start merging at step", 0, 100, 10)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("model" -> v.str("model"),
    "images" -> ujson.Arr.from(ImageHooks.images(v, "images").map(ujson.Str(_))),
    "trigger_word" -> v.str("trigger_word", "img"), "start_merge_step" -> v.int("start_merge_step", 10))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("PhotoMaker" -> field(o, "model"))

/** Restore faces after generation (A1111 "Restore faces" with GFPGAN / CodeFormer). */
class FaceRestorePlugin extends HookPlugin:
  def id = "face_restore"
  def name = "Restore faces"
  def hook = "face_restore"
  override def description = "Runs GFPGAN or CodeFormer (models/face_restore, models/GFPGAN, models/Codeformer) on detected faces."
  override def isActive(v: Values): Boolean = v.bool("enabled") && v.str("model").nonEmpty && v.str("model") != "None"
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    DynamicChoice("model", "Model", ChoiceSource.FaceRestorers, "", allowNone = true),
    Slider("visibility", "Visibility", 0, 1, 0.01, 1.0),
    Slider("codeformer_weight", "CodeFormer weight (0 = max effect)", 0, 1, 0.01, 0.5)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("model" -> v.str("model"), "visibility" -> v.double("visibility", 1.0),
    "codeformer_weight" -> v.double("codeformer_weight", 0.5))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Face restoration" -> field(o, "model"))

