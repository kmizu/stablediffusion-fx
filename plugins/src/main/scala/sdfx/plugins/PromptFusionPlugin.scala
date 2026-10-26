package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Prompt Fusion language and geometry from ljleb/prompt-fusion-extension (MIT). */
class PromptFusionPlugin extends HookPlugin:
  def id = "prompt_fusion"
  def name = "Prompt Fusion"
  def hook = "prompt_fusion"
  override def description = "Interpolate prompt embeddings during each image's sampling, using curves, attention ramps and reusable variables."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("curves", "In either prompt: [night:forest: , 15] blends embeddings from the first step to step 16. " +
      "Add :bezier or :catmull before ] for curved interpolation. Blank endpoints span the sampling pass; " +
      "decimal positions such as .5 use fractions of the pass."),
    Note("language", "[red:blue::mean] averages embeddings; [red|blue:0.5] interpolates a repeating alternation. " +
      "(red:1,2) ramps attention within its surrounding prompt-editing interval. " +
      "Define $color = red on one line, then use $color on a later line. " +
      "Functions work too: $subject($x) = portrait of $x, followed by $subject(a fox)."),
    Section("geometry", "Interpolation geometry"),
    Slider("slerp_scale", "Slerp scale", -2, 2, .05, 0),
    Check("negative_origin", "Negative prompt origin", true),
    Slider("slerp_epsilon", "Slerp epsilon", 0, 1, .0001, .0001),
    Note("geometry_help", "Scale 0 uses linear geometry; 1 uses spherical geometry. Other values blend or extrapolate. " +
      "The positive branch can use the current negative embedding as origin; the negative branch uses the empty prompt. " +
      "Epsilon selects linear fallback for nearly parallel or opposite embeddings."),
    Note("cost", "Curves encode each endpoint once. Attention ramps may encode a prompt for every sampling step. " +
      "Hires and img2img restart interpolation using the actual sampling pass length."),
    Note("compatibility", "Supports SD1.x, SD2.x and SDXL, including pooled embeddings, prompt editing and Weight Gradient. " +
      "Use Neutral Prompt operators outside Fusion expressions. PhotoMaker, NegPiP negative terms and " +
      "img2img Alternative inversion using Fusion syntax report an error.")
  )
  override def enabledBy = Map("slerp_scale" -> "enabled", "negative_origin" -> "enabled", "slerp_epsilon" -> "enabled")
  def hookFields(v: Values): ujson.Obj =
    val scale = v.double("slerp_scale")
    val epsilon = v.double("slerp_epsilon")
    require(!scale.isNaN && !scale.isInfinite, "Slerp scale must be finite")
    require(!epsilon.isNaN && !epsilon.isInfinite && epsilon >= 0 && epsilon <= 1, "Slerp epsilon must be between 0 and 1")
    ujson.Obj("slerp_scale" -> scale, "negative_origin" -> v.bool("negative_origin"), "slerp_epsilon" -> epsilon)
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "Prompt Fusion" -> "True", "Prompt Fusion Slerp Scale" -> Hooks.num(o("slerp_scale")),
    "Prompt Fusion Negative Origin" -> o("negative_origin").bool.toString,
    "Prompt Fusion Slerp Epsilon" -> Hooks.num(o("slerp_epsilon"))
  )
