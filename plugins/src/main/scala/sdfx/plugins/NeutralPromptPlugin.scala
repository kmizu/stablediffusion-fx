package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Neutral Prompt, adapted from ljleb/sd-webui-neutral-prompt (MIT). */
class NeutralPromptPlugin extends HookPlugin:
  def id = "neutral_prompt"
  def name = "Neutral Prompt"
  def hook = "neutral_prompt"
  override def description = "Combine prompts by their image predictions, with nested groups and signed weights."
  private val keywords = Seq("AND", "AND_PERP", "AND_SALT", "AND_TOPK")
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("cfg_rescale", "CFG rescale", 0, 1, 0.01, 0),
    Note("syntax", "Write a portrait AND_TOPK red scarf :0.7 in the main prompt. " +
      "AND adds, AND_PERP adds only the perpendicular difference, AND_SALT selects salient regions, " +
      "and AND_TOPK keeps the strongest 5% of differences. Negative weights reverse the effect."),
    Note("groups", "Nest groups with brackets: a forest AND_PERP [a city AND_SALT lights :0.5] :-0.7. " +
      "Each additional prompt takes another model evaluation per step. " +
      "Reference also evaluates the reference image for each prompt."),
    Section("formatter", "Add a prompt"),
    Check("format_enabled", "Add extra prompt"),
    Text("extra_prompt", "Extra prompt", multiline = true, placeholder = "red scarf, or [a city AND_SALT lights]"),
    Choice("prompt_type", "Combine with", keywords, "AND_PERP"),
    Slider("prompt_weight", "Prompt weight", -3, 3, 0.05, 1),
    Note("formatter_help", "Adds this text to the positive prompt when generating. " +
      "The saved image includes the full combined prompt; repeating the same settings does not append it twice."),
    Section("compatibility", "Compatibility", expanded = false),
    Note("pending", "Composed prompts currently report errors with PhotoMaker, SAG, APG, Mahiro, " +
      "Latent Modifier, Dynamic Thresholding or Detail Daemon."),
    Note("supported", "CADS, CFG Schedule, Reference, ControlLLLite (SDXL), IP-Adapter (including Plus), " +
      "ControlNet and PAG are supported. " +
      "Prompt editing, Weight Gradient, NegPiP, hires, img2img and " +
      "img2img Alternative are supported.")
  )
  override def enabledBy = Map("cfg_rescale" -> "enabled", "format_enabled" -> "enabled",
    "extra_prompt" -> "format_enabled", "prompt_type" -> "format_enabled", "prompt_weight" -> "format_enabled")
  def hookFields(v: Values): ujson.Obj =
    val phi = v.double("cfg_rescale")
    require(!phi.isNaN && !phi.isInfinite && phi >= 0 && phi <= 1, "CFG rescale must be between 0 and 1")
    ujson.Obj("cfg_rescale" -> phi)
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("Neutral Prompt" -> "True", "CFG Rescale phi" -> Hooks.num(o("cfg_rescale")))
  override def process(p: GenerationParams, v: Values): GenerationParams =
    val text = v.str("extra_prompt").trim
    val prompt =
      if v.bool("format_enabled") && text.nonEmpty then
        val keyword = v.str("prompt_type", "AND_PERP")
        val weight = v.double("prompt_weight", 1)
        require(keywords.contains(keyword), "Unknown Neutral Prompt combination")
        require(!weight.isNaN && !weight.isInfinite, "Prompt weight must be finite")
        val suffix = s"\n$keyword $text :${Values.fmt(weight)}"
        if p.prompt.stripTrailing.endsWith(suffix) then p.prompt else p.prompt + suffix
      else p.prompt
    super.process(p.copy(prompt = prompt), v)
