package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Sampling-time extensions from the A1111 extension index (engine side: python/sdfx/sampling_ext.py). */
private[plugins] object ExtKeys:
  /** `Name: "k1: v1, k2: v2"` style infotext value from the merged hook object. */
  def packed(o: ujson.Obj, keys: Seq[String]): String =
    keys.flatMap(k => o.value.get(k).map(v => s"$k: ${Hooks.num(v)}")).mkString(", ")

/** Detail Daemon (muerrilla/sd-webui-detail-daemon). */
class DetailDaemonPlugin extends HookPlugin:
  def id = "detail_daemon"
  def name = "Detail Daemon"
  def hook = "detail_daemon"
  override def description = "Tells the model there is less (or more) noise than there is, over a schedule: adds or removes detail."
  private val keys = Seq("mode", "amount", "start", "end", "bias", "exponent", "start_offset", "end_offset", "fade", "smooth")
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Choice("mode", "Mode", Seq("both", "cond", "uncond"), "both"),
    Slider("amount", "Detail amount", -1, 1, 0.01, 0.1),
    Slider("start", "Start", 0, 1, 0.01, 0.2),
    Slider("end", "End", 0, 1, 0.01, 0.8),
    Slider("bias", "Bias (peak position)", 0, 1, 0.01, 0.5),
    Slider("exponent", "Exponent", 0, 10, 0.05, 1.0),
    Slider("start_offset", "Start offset", -1, 1, 0.01, 0.0),
    Slider("end_offset", "End offset", -1, 1, 0.01, 0.0),
    Slider("fade", "Fade", 0, 1, 0.05, 0.0),
    Check("smooth", "Smooth", true)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "mode" -> v.str("mode", "both"), "amount" -> v.double("amount", 0.1), "start" -> v.double("start", 0.2),
    "end" -> v.double("end", 0.8), "bias" -> v.double("bias", 0.5), "exponent" -> v.double("exponent", 1.0),
    "start_offset" -> v.double("start_offset", 0.0), "end_offset" -> v.double("end_offset", 0.0),
    "fade" -> v.double("fade", 0.0), "smooth" -> v.bool("smooth", true))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Detail Daemon" -> ExtKeys.packed(o, keys))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Detail Daemon amount", "amount"), numAxis("Detail Daemon start", "start"),
    numAxis("Detail Daemon end", "end"))

/** Vectorscope CC (Haoming02/sd-webui-vectorscope-cc). */
class VectorscopeCcPlugin extends HookPlugin:
  def id = "vectorscope_cc"
  def name = "Vectorscope CC"
  def hook = "vectorscope_cc"
  override def description = "Brightness, contrast, saturation and RGB offsets applied to the latent noise while sampling."
  private val keys = Seq("brightness", "contrast", "saturation", "r", "g", "b", "method", "scaling", "strength")
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("brightness", "Brightness", -5, 5, 0.05, 0.0),
    Slider("contrast", "Contrast", -5, 5, 0.05, 0.0),
    Slider("saturation", "Saturation", -5, 5, 0.05, 0.0),
    Slider("r", "Red", -5, 5, 0.05, 0.0),
    Slider("g", "Green", -5, 5, 0.05, 0.0),
    Slider("b", "Blue", -5, 5, 0.05, 0.0),
    Choice("method", "Noise method", Seq("Straight", "Cross", "N.Random", "U.Random"), "Straight"),
    Choice("scaling", "Scaling", Seq("Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"), "Flat"),
    Slider("strength", "Overall strength", 0, 5, 0.05, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj.from(
    Seq("brightness", "contrast", "saturation", "r", "g", "b").map(k => k -> ujson.Num(v.double(k, 0.0))) ++
      Seq("method" -> ujson.Str(v.str("method", "Straight")), "scaling" -> ujson.Str(v.str("scaling", "Flat")),
        "strength" -> ujson.Num(v.double("strength", 1.0))))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Vectorscope CC" -> ExtKeys.packed(o, keys))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Vectorscope brightness", "brightness"),
    numAxis("Vectorscope contrast", "contrast"), numAxis("Vectorscope saturation", "saturation"))

/** Latent Mirroring (dfaker/SD-latent-mirroring). */
class LatentMirroringPlugin extends HookPlugin:
  def id = "latent_mirroring"
  def name = "Latent Mirroring"
  def hook = "latent_mirroring"
  override def description = "Mirrors or rotates the latent during the first steps: balanced compositions up to perfect reflections."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Choice("mode", "Mirror application mode", Seq("Alternate Steps", "Blend Average"), "Alternate Steps"),
    Choice("style", "Mirror style", Seq("Vertical Mirroring", "Horizontal Mirroring", "Horizontal+Vertical Mirroring",
      "90 Degree Rotation", "180 Degree Rotation"), "Vertical Mirroring"),
    Slider("max_step", "Maximum steps fraction to mirror at", 0, 1, 0.01, 0.25),
    Slider("strength", "Mirror strength", 0, 1, 0.01, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "mode" -> v.str("mode", "Alternate Steps"), "style" -> v.str("style", "Vertical Mirroring"),
    "max_step" -> v.double("max_step", 0.25), "strength" -> v.double("strength", 1.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "Mirror Mode" -> field(o, "mode"), "Mirror Style" -> field(o, "style"),
    "Mirror Max Step" -> field(o, "max_step"), "Mirror Strength" -> field(o, "strength"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Mirror max step", "max_step"))

/** Asymmetric tiling (tjm35/asymmetric-tiling-sd-webui). */
class AsymmetricTilingPlugin extends HookPlugin:
  def id = "asymmetric_tiling"
  def name = "Asymmetric Tiling"
  def hook = "tiling"
  override def description = "Seamless tiling, configured independently for the X and Y axes (U-Net and VAE)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Check("tile_x", "Tile X", true),
    Check("tile_y", "Tile Y", true),
    IntField("start_step", "Start tiling from step", 0, 1000, 0),
    IntField("stop_step", "Stop tiling after step (-1 = never)", -1, 1000, -1)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "tile_x" -> v.bool("tile_x", true), "tile_y" -> v.bool("tile_y", true),
    "start_step" -> v.int("start_step", 0), "stop_step" -> v.int("stop_step", -1))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "Tile X" -> field(o, "tile_x"), "Tile Y" -> field(o, "tile_y"),
    "Start Tiling From Step" -> field(o, "start_step"), "Stop Tiling After Step" -> field(o, "stop_step"))

/** anti-burn (klimaleksus/stable-diffusion-webui-anti-burn). */
class AntiBurnPlugin extends HookPlugin:
  def id = "anti_burn"
  def name = "Anti-burn"
  def hook = "anti_burn"
  override def description = "Skips the last few steps and averages the latents of the steps before them (smoother results)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    IntField("skip_last", "Skip last steps", 0, 50, 2),
    IntField("average", "Average this many steps", 1, 50, 3)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("skip_last" -> v.int("skip_last", 2), "average" -> v.int("average", 3))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("Anti-burn" -> s"skip ${field(o, "skip_last")}, average ${field(o, "average")}")
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("Anti-burn skip", "skip_last"), numAxis("Anti-burn average", "average"))

/** ReSharpen (Haoming02/sd-webui-resharpen). */
class ReSharpenPlugin extends HookPlugin:
  def id = "resharpen"
  def name = "ReSharpen"
  def hook = "resharpen"
  override def description = "Pushes every step further along its change (more detail) or back (smoother)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("sharpness", "Sharpness", -4, 4, 0.05, 0.5),
    Choice("scaling", "Scaling", Seq("Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"), "Flat")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj("sharpness" -> v.double("sharpness", 0.5), "scaling" -> v.str("scaling", "Flat"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("Resharpen" -> s"${field(o, "sharpness")}, ${field(o, "scaling")}")
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("ReSharpen sharpness", "sharpness"))

/** Diffusion Color Grading (Haoming02/sd-webui-diffusion-cg). */
class ColorGradingPlugin extends HookPlugin:
  def id = "color_grading"
  def name = "Diffusion Color Grading"
  def hook = "color_grading"
  override def description = "Recenters the latent colour channels (removes casts or tints towards a colour) and stretches the range."
  override def enabledBy: Map[String, String] =
    Map("recenter_strength" -> "recenter", "r" -> "recenter", "g" -> "recenter", "b" -> "recenter",
      "normalize_strength" -> "normalize")
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Check("recenter", "Recenter", true),
    Slider("recenter_strength", "Recenter strength", 0, 2, 0.05, 1.0),
    Slider("r", "Target red", -1, 1, 0.05, 0.0),
    Slider("g", "Target green", -1, 1, 0.05, 0.0),
    Slider("b", "Target blue", -1, 1, 0.05, 0.0),
    Check("normalize", "Normalize (stretch the dynamic range)"),
    Slider("normalize_strength", "Normalize strength", 0, 1, 0.05, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "recenter" -> v.bool("recenter", true), "recenter_strength" -> v.double("recenter_strength", 1.0),
    "r" -> v.double("r", 0.0), "g" -> v.double("g", 0.0), "b" -> v.double("b", 0.0),
    "normalize" -> v.bool("normalize"), "normalize_strength" -> v.double("normalize_strength", 1.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Diffusion CG" ->
    ExtKeys.packed(o, Seq("recenter", "recenter_strength", "r", "g", "b", "normalize", "normalize_strength")))

/** CFG Schedule (guzuligo/CFG-Schedule-for-Automatic1111-SD). */
class CfgSchedulePlugin extends HookPlugin:
  def id = "cfg_schedule"
  def name = "CFG Schedule"
  def hook = "cfg_schedule"
  override def description = "CFG scale that moves from a start value to an end value over the steps (needs CFG > 1)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("start_cfg", "CFG at start", 0, 30, 0.5, 7.0),
    Slider("end_cfg", "CFG at end", 0, 30, 0.5, 3.0),
    Choice("curve", "Curve", Seq("linear", "cosine", "square", "sqrt"), "linear"),
    Slider("start", "Begin changing at", 0, 1, 0.01, 0.0),
    Slider("end", "Finish changing at", 0, 1, 0.01, 1.0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "start_cfg" -> v.double("start_cfg", 7.0), "end_cfg" -> v.double("end_cfg", 3.0), "curve" -> v.str("curve", "linear"),
    "start" -> v.double("start", 0.0), "end" -> v.double("end", 1.0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("CFG schedule" -> ExtKeys.packed(o, Seq("start_cfg", "end_cfg", "curve", "start", "end")))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("CFG schedule start", "start_cfg"), numAxis("CFG schedule end", "end_cfg"))

/** CADS (v0xie/sd-webui-cads). */
class CadsPlugin extends HookPlugin:
  def id = "cads"
  def name = "CADS"
  def hook = "cads"
  override def description = "Condition-annealed sampling: scheduled noise on the prompt conditioning for more diverse results."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("tau1", "Tau 1", 0, 1, 0.01, 0.6),
    Slider("tau2", "Tau 2", 0, 1, 0.01, 0.9),
    Slider("noise_scale", "Noise scale", 0, 1, 0.01, 0.25),
    Slider("mixing_factor", "Mixing factor (psi)", 0, 1, 0.01, 1.0),
    Check("rescale", "Rescale", true),
    Check("apply_to_uncond", "Apply to the negative prompt too")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "tau1" -> v.double("tau1", 0.6), "tau2" -> v.double("tau2", 0.9), "noise_scale" -> v.double("noise_scale", 0.25),
    "mixing_factor" -> v.double("mixing_factor", 1.0), "rescale" -> v.bool("rescale", true),
    "apply_to_uncond" -> v.bool("apply_to_uncond"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "CADS Tau 1" -> field(o, "tau1"), "CADS Tau 2" -> field(o, "tau2"), "CADS Noise Scale" -> field(o, "noise_scale"),
    "CADS Mixing Factor" -> field(o, "mixing_factor"), "CADS Rescale" -> field(o, "rescale"))
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("CADS noise scale", "noise_scale"), numAxis("CADS tau 1", "tau1"),
    numAxis("CADS tau 2", "tau2"))
