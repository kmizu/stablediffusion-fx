package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

import scala.util.Random

/** Shared helpers for the simple hook plugins below. */
private[plugins] object HookFields:
  def keys(prefix: String, o: ujson.Obj, names: Seq[(String, String)]): Seq[(String, String)] =
    names.flatMap((f, label) => o.value.get(f).map(v => s"$prefix $label" -> Hooks.num(v)))

/** APG is your CFG (reForge-APGIsYourCFG): adaptive projected guidance replaces CFG. */
class ApgPlugin extends HookPlugin:
  def id = "apg"
  def name = "APG is now your CFG"
  def hook = "apg"
  override def description = "Adaptive Projected Guidance: reduces oversaturation at high CFG (replaces CFG)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("momentum", "Momentum", -1.5, 1.0, 0.01, 0.5),
    Slider("adaptive_momentum", "Adaptive momentum", 0, 1, 0.001, 0.18),
    Slider("norm_threshold", "Normalization threshold", 0, 50, 0.05, 15.0),
    Slider("eta", "Eta (parallel component)", 0, 1, 0.1, 1.0),
    Check("guidance_limiter", "Guidance limiter (plain CFG outside the sigma range)"),
    Slider("guidance_sigma_start", "Limiter sigma start", -1, 100, 0.01, 5.42),
    Slider("guidance_sigma_end", "Limiter sigma end", -1, 100, 0.01, 0.28)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "momentum" -> v.double("momentum", 0.5), "adaptive_momentum" -> v.double("adaptive_momentum", 0.18),
    "norm_threshold" -> v.double("norm_threshold", 15.0), "eta" -> v.double("eta", 1.0),
    "guidance_limiter" -> v.bool("guidance_limiter"),
    "guidance_sigma_start" -> v.double("guidance_sigma_start", 5.42), "guidance_sigma_end" -> v.double("guidance_sigma_end", 0.28))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("apgisyourcfg_enabled" -> "True") ++ HookFields.keys("apgisyourcfg", o,
      Seq("momentum" -> "momentum", "adaptive_momentum" -> "adaptive_momentum", "norm_threshold" -> "norm_thr", "eta" -> "eta"))
      .map((k, v) => k.replace(" ", "_") -> v)
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("APG momentum", "momentum"), numAxis("APG eta", "eta"),
    numAxis("APG norm threshold", "norm_threshold"))

/** Mahiro CFG (mahiro_reforge). */
class MahiroPlugin extends HookPlugin:
  def id = "mahiro"
  def name = "Mahiro CFG"
  def hook = "mahiro"
  override def description = "Post-CFG function that leans towards the positive prompt where it agrees with CFG."
  override def controls: Seq[Control] = Seq(Check("enabled", "Enabled"))
  def hookFields(v: Values): ujson.Obj = ujson.Obj()
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("mahiro_cfg_enabled" -> "True")

/** RescaleCFG (reForge-RescaleCFG). */
class RescaleCfgPlugin extends HookPlugin:
  def id = "rescale_cfg"
  def name = "RescaleCFG"
  def hook = "rescale_cfg"
  override def description = "Rescales the guided prediction's std to the positive one (for v-prediction / ZTSNR models)."
  override def controls: Seq[Control] = Seq(Check("enabled", "Enabled"), Slider("multiplier", "Multiplier", 0, 1, 0.01, 0.7))
  def hookFields(v: Values): ujson.Obj = ujson.Obj("multiplier" -> v.double("multiplier", 0.7))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("RescaleCFG multiplier" -> field(o, "multiplier"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("RescaleCFG multiplier", "multiplier"))

/** Advanced model sampling (reForge-advanced_model_sampling): prediction type / zero-terminal-SNR overrides. */
class ModelSamplingPlugin extends HookPlugin:
  def id = "model_sampling"
  def name = "Advanced Model Sampling"
  def hook = "model_sampling"
  override def description = "Overrides the model's prediction type (eps / v-prediction) and zero terminal SNR."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Choice("sampling", "Sampling type", Seq("default", "eps", "v_prediction", "v_prediction_zsnr"), "default"),
    Check("zsnr", "Zero terminal SNR (rescale betas)")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("sampling" -> v.str("sampling", "default"), "zsnr" -> v.bool("zsnr"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("Model sampling" -> field(o, "sampling"), "Model sampling ZSNR" -> field(o, "zsnr"))

/** Kohya HRFix / Deep Shrink (sd_forge_kohya_hrfix). */
class KohyaHrFixPlugin extends HookPlugin:
  def id = "kohya_hrfix"
  def name = "Kohya HRFix (Deep Shrink)"
  def hook = "kohya_hrfix"
  override def description = "Shrinks a deep U-Net block in early steps so high resolutions stay coherent."
  private val methods = Seq("bicubic", "nearest-exact", "bilinear", "area", "bislerp")
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    IntField("block_number", "Block number", 1, 32, 3),
    Slider("downscale_factor", "Downscale factor", 0.1, 9, 0.001, 2.0),
    Slider("start_percent", "Start percent", 0, 1, 0.001, 0.0),
    Slider("end_percent", "End percent", 0, 1, 0.001, 0.35),
    Check("downscale_after_skip", "Downscale after skip", true),
    Choice("downscale_method", "Downscale method", methods, "bicubic"),
    Choice("upscale_method", "Upscale method", methods, "bicubic")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "block_number" -> v.int("block_number", 3), "downscale_factor" -> v.double("downscale_factor", 2.0),
    "start_percent" -> v.double("start_percent", 0.0), "end_percent" -> v.double("end_percent", 0.35),
    "downscale_after_skip" -> v.bool("downscale_after_skip", true),
    "downscale_method" -> v.str("downscale_method", "bicubic"), "upscale_method" -> v.str("upscale_method", "bicubic"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    HookFields.keys("kohya_hrfix", o, Seq("block_number" -> "block_number", "downscale_factor" -> "downscale_factor",
      "start_percent" -> "start_percent", "end_percent" -> "end_percent", "downscale_after_skip" -> "downscale_after_skip",
      "downscale_method" -> "downscale_method", "upscale_method" -> "upscale_method")).map((k, v) => k.replace(" ", "_") -> v)
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Deep Shrink factor", "downscale_factor"), numAxis("Deep Shrink end", "end_percent"))

/** Tiled diffusion (sd_forge_multidiffusion). */
class MultiDiffusionPlugin extends HookPlugin:
  def id = "multidiffusion"
  def name = "MultiDiffusion (tiled diffusion)"
  def hook = "multidiffusion"
  override def description = "Denoises overlapping latent tiles and blends them, for very large images."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Choice("method", "Method", Seq("MultiDiffusion", "Mixture of Diffusers"), "MultiDiffusion"),
    Slider("tile_width", "Tile width", 256, 2048, 64, 768),
    Slider("tile_height", "Tile height", 256, 2048, 64, 768),
    Slider("tile_overlap", "Tile overlap", 0, 1024, 32, 64),
    IntField("tile_batch_size", "Tile batch size", 1, 64, 4)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "method" -> v.str("method", "MultiDiffusion"), "tile_width" -> v.int("tile_width", 768),
    "tile_height" -> v.int("tile_height", 768), "tile_overlap" -> v.int("tile_overlap", 64),
    "tile_batch_size" -> v.int("tile_batch_size", 4))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("MultiDiffusion" -> s"${field(o, "method")} ${field(o, "tile_width")}x${field(o, "tile_height")} overlap ${field(o, "tile_overlap")}")

/** Latent modifier (sd_forge_latent_modifier), the commonly used subset. */
class LatentModifierPlugin extends HookPlugin:
  def id = "latent_modifier"
  def name = "Latent Modifier"
  def hook = "latent_modifier"
  override def description = "Sharpness, tonemapping, contrast, CFG-drift correction, noise and spectral tweaks."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("sharpness_multiplier", "Sharpness multiplier", -100, 100, 0.01, 0.0),
    Choice("sharpness_method", "Sharpness method", Seq("anisotropic", "gaussian"), "anisotropic"),
    Slider("tonemap_multiplier", "Tonemap multiplier", 0, 100, 0.01, 0.0),
    Choice("tonemap_method", "Tonemap method", Seq("reinhard", "arctan", "clamp"), "reinhard"),
    Slider("tonemap_percentile", "Tonemap percentile", 0, 100, 0.005, 100.0),
    Slider("contrast_multiplier", "Contrast multiplier", -100, 100, 0.01, 0.0),
    Choice("combat_method", "Combat CFG drift method", Seq("subtract", "median"), "subtract"),
    Slider("combat_cfg_drift", "Combat CFG drift", -10, 10, 0.01, 0.0),
    Choice("extra_noise_type", "Extra noise type", Seq("gaussian", "uniform"), "gaussian"),
    Slider("extra_noise_multiplier", "Extra noise multiplier", 0, 100, 0.01, 0.0),
    IntField("extra_noise_lowpass", "Extra noise lowpass (100 = off)", 1, 100, 100),
    IntField("divisive_norm_size", "Divisive norm size", 1, 255, 127),
    Slider("divisive_norm_multiplier", "Divisive norm multiplier", 0, 1, 0.01, 0.0),
    Choice("spectral_mod_mode", "Spectral mod mode", Seq("hard_clamp", "soft_clamp"), "hard_clamp"),
    Slider("spectral_mod_percentile", "Spectral mod percentile", 0, 50, 0.01, 5.0),
    Slider("spectral_mod_multiplier", "Spectral mod multiplier", -15, 15, 0.01, 0.0)
  )
  private val numeric = Seq("sharpness_multiplier" -> 0.0, "tonemap_multiplier" -> 0.0, "tonemap_percentile" -> 100.0,
    "contrast_multiplier" -> 0.0, "combat_cfg_drift" -> 0.0, "extra_noise_multiplier" -> 0.0,
    "divisive_norm_multiplier" -> 0.0, "spectral_mod_percentile" -> 5.0, "spectral_mod_multiplier" -> 0.0)
  def hookFields(v: Values): ujson.Obj =
    val o = ujson.Obj()
    numeric.foreach((k, d) => o(k) = v.double(k, d))
    o("extra_noise_lowpass") = v.int("extra_noise_lowpass", 100)
    o("divisive_norm_size") = v.int("divisive_norm_size", 127)
    Seq("sharpness_method" -> "anisotropic", "tonemap_method" -> "reinhard", "combat_method" -> "subtract",
      "extra_noise_type" -> "gaussian", "spectral_mod_mode" -> "hard_clamp").foreach((k, d) => o(k) = v.str(k, d))
    o
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    numeric.map(_._1).filter(k => o.value.get(k).exists { case ujson.Num(d) => d != 0 && !k.endsWith("percentile"); case _ => false })
      .map(k => s"latent_modifier_$k" -> field(o, k))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Latent sharpness", "sharpness_multiplier"),
    numAxis("Latent combat CFG drift", "combat_cfg_drift"), numAxis("Latent tonemap", "tonemap_multiplier"))

/** Soft inpainting (A1111 soft-inpainting extension). */
class SoftInpaintingPlugin extends HookPlugin:
  def id = "soft_inpainting"
  def name = "Soft Inpainting"
  def hook = "soft_inpainting"
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Blends original and generated content smoothly along a soft (blurred) mask."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("note", "Uses the inpaint mask; raise Mask blur for wider transitions."),
    Slider("mask_blend_power", "Schedule bias", 0, 8, 0.1, 1.0),
    Slider("mask_blend_scale", "Preservation strength", 0, 8, 0.05, 0.5),
    Slider("inpaint_detail_preservation", "Transition contrast boost", 1, 32, 0.5, 4.0),
    Slider("composite_mask_influence", "Mask influence", 0, 1, 0.01, 0.0),
    Slider("composite_difference_threshold", "Difference threshold", 0, 8, 0.25, 0.5),
    Slider("composite_difference_contrast", "Difference contrast", 0, 8, 0.25, 2.0)
  )
  private val ks = Seq("mask_blend_power" -> 1.0, "mask_blend_scale" -> 0.5, "inpaint_detail_preservation" -> 4.0,
    "composite_mask_influence" -> 0.0, "composite_difference_threshold" -> 0.5, "composite_difference_contrast" -> 2.0)
  def hookFields(v: Values): ujson.Obj = ujson.Obj.from(ks.map((k, d) => k -> ujson.Num(v.double(k, d))))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("Soft inpainting enabled" -> "True") ++ Seq("mask_blend_power" -> "schedule bias", "mask_blend_scale" -> "preservation strength",
      "inpaint_detail_preservation" -> "transition contrast boost").map((k, l) => s"Soft inpainting $l" -> field(o, k))

/** Self-Attention Guidance (sd_forge_sag). */
class SagPlugin extends HookPlugin:
  def id = "sag"
  def name = "Self Attention Guidance"
  def hook = "sag"
  override def tabs: Set[Tab] = Set(Tab.Txt2Img)
  override def description = "SD1.x txt2img: guides with blurred self-attention regions (needs DDIM/UniPC/DPM++ samplers)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("scale", "Scale", -2, 5, 0.01, 0.5),
    Slider("blur_sigma", "Blur sigma", 0, 10, 0.01, 2.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("scale" -> v.double("scale", 0.5), "blur_sigma" -> v.double("blur_sigma", 2.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("sag_enabled" -> "True", "sag_scale" -> field(o, "scale"),
    "sag_blur_sigma" -> field(o, "blur_sigma"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("SAG scale", "scale"))

/** Never OOM (sd_forge_neveroom). */
class NeverOomPlugin extends HookPlugin:
  def id = "neveroom"
  def name = "Never OOM"
  def hook = "neveroom"
  override def description = "Lowers peak VRAM for this job: sliced attention / VAE and tiled VAE."
  override def isActive(v: Values): Boolean = v.bool("unet_offload") || v.bool("vae_tiling")
  override def controls: Seq[Control] = Seq(
    Check("unet_offload", "Enabled for UNet (always maximize offload)"),
    Check("vae_tiling", "Enabled for VAE (always tiled)")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("unet_offload" -> v.bool("unet_offload"), "vae_tiling" -> v.bool("vae_tiling"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Nil

/** Random resolutions (sd_webui_random_resolutions): picks one size per generation from a list. */
class RandomResolutionsPlugin extends Plugin:
  def id = "random_resolutions"
  def name = "Random Resolutions"
  def kind = PluginKind.AlwaysOn
  override def tabs: Set[Tab] = Set(Tab.Txt2Img)
  override def description = "Each generation uses a random size from the list (or from a pixel budget and aspect ratios)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Text("resolutions", "Resolutions (WxH, one per line or comma separated)",
      "1024x1024, 1152x896, 896x1152, 1216x832, 832x1216, 1344x768, 768x1344", multiline = true),
    Check("use_seed", "Pick with the generation seed (reproducible)", true)
  )
  private val Re = """(\d+)\s*[xX×]\s*(\d+)""".r

  def parse(text: String): Seq[(Int, Int)] =
    Re.findAllMatchIn(text).map(m => (m.group(1).toInt, m.group(2).toInt))
      .map((w, h) => (math.max(64, w / 8 * 8), math.max(64, h / 8 * 8))).toSeq

  override def process(p: GenerationParams, v: Values): GenerationParams =
    val sizes = parse(v.str("resolutions"))
    if sizes.isEmpty then p
    else
      val rnd = if v.bool("use_seed", true) && p.seed >= 0 then new Random(p.seed) else new Random()
      val (w, h) = sizes(rnd.nextInt(sizes.size))
      p.copy(width = w, height = h, extra = p.extra + ("Random resolution" -> s"${w}x$h"))
