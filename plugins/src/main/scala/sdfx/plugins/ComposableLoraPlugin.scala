package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Composable LoRA/LyCORIS with steps, a2569875 and opparco (AGPL-3.0). */
class ComposableLoraPlugin extends HookPlugin:
  def id = "composable_lora"
  def name = "Composable LoRA / LyCORIS"
  def hook = "composable_lora"
  override def description = "Keep each adapter with its own AND prompt; control negative conditioning and per-step weights."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("branches", "Example: a red fox <lora:Fox:0.8> AND a white owl <lora:Owl:0.7>. " +
      "Each branch uses only its own adapters; a shared adapter may have a different weight in each branch. " +
      "AND combines image predictions and costs one model evaluation per branch. It does not assign spatial regions."),
    Section("negative", "Negative conditioning"),
    Check("uc_text_model_encoder", "Use LoRA in negative text encoder"),
    Check("uc_diffusion_model", "Use LoRA in negative denoiser"),
    Check("single_no_uc", "Respect negative switches for one prompt"),
    Note("single_help", "Upstream behavior: a single prompt normally keeps LoRA in both negative paths. " +
      "Select the single-prompt exclusion to make the two switches above control it too."),
    Section("steps", "Weight through sampling"),
    Check("with_step", "Composable LoRA with steps"),
    Check("hires_global_step", "Treat hires steps as global"),
    Check("plot_weights", "Plot effective LoRA weights"),
    Note("schedule", "[<lora:A:1>::10] ends after step 10; [<lora:A:1>:10] starts at step 10. " +
      "[[<lora:A:1>::25]:10] limits both ends. Fractions such as .5 refer to the sampling pass. " +
      "[<lora:A:1>|<lora:B:1>] alternates adapters. Endpoints are inclusive."),
    Note("controllers", "[<lora:A:1> #increment:10] ramps up after step 10; #decrease ramps down. " +
      "#cmd\\(sin\\(life\\)\\) controls the lifecycle with a math expression. \\u0023 can replace #. " +
      "Plots appear as an additional diagnostic image in the gallery."),
    Section("commands", "Custom commands", expanded = false),
    Note("cmd_help", "Math expressions support weight, life, step, steps, warmup/cooldown, normalized sin/cos, " +
      "radian sinr/cosr and the documented Python math helpers. Return a multiplier of the tag weight. " +
      "Python functions/statements and native object inspection use Settings > Allow custom code."),
    Note("context", "Advanced command context: lora, lora_module, lora_type, lora_name, lora_count, block_lora_count, " +
      "is_negative, layer_name, current_prompt and sd_processing. Objects and layer names refer to this standalone engine. " +
      "Text encoding uses step=-1; set enable_prepare_step=True to apply a command there."),
    Section("compatibility", "Formats and compatibility", expanded = false),
    Note("formats", "Supports native LoRA, spatial/CP LoCon, LoHa, LoKr, full-rank deltas and IA3. " +
      "SDXL encodes both text encoders and pooled conditioning. Hires can restart or continue the step clock."),
    Note("limits", "Neutral Prompt and Prompt Fusion keep their own syntax. Native LoRA Block Weight/start/stop " +
      "fields, PhotoMaker, Alternative inversion and NegPiP signed terms have explicit compatibility errors. " +
      "Use these step controllers or cmd layer_name for adapter timing and per-layer weights. " +
      "Composed Neutral limits also apply: SAG, APG, Mahiro, Latent Modifier, Dynamic Thresholding and Detail Daemon.")
  )
  private val fields = Seq("with_step", "uc_text_model_encoder", "uc_diffusion_model", "plot_weights", "single_no_uc", "hires_global_step")
  override def enabledBy = fields.map(_ -> "enabled").toMap
  def hookFields(v: Values): ujson.Obj = ujson.Obj.from(fields.map(k => k -> ujson.Bool(v.bool(k))))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Composable LoRA" -> "True") ++
    fields.map(k => ("Composable LoRA " + k) -> o(k).bool.toString)
