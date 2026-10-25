package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Common shape of hook-based AlwaysOn plugins: controls -> hook JSON -> extra infotext keys. */
abstract class HookPlugin extends Plugin:
  def hook: String
  def kind: PluginKind = PluginKind.AlwaysOn
  override def requiresEngineFeatures: Seq[String] = Seq(hook)

  /** Hook fields from control values (in a stable order). */
  def hookFields(v: Values): ujson.Obj

  /** Infotext keys from the final (merged) hook object. */
  def infotextKeys(o: ujson.Obj): Seq[(String, String)]

  override def process(p: GenerationParams, v: Values): GenerationParams =
    val (p2, merged) = Hooks(p, hook, hookFields(v))
    p2.copy(extra = p2.extra ++ infotextKeys(merged))

  protected def field(o: ujson.Obj, k: String): String = o.value.get(k).map(Hooks.num).getOrElse("")

  protected def numAxis(axisName: String, f: String): XyzAxis =
    XyzAxis(axisName, isNumeric = true)((p, s) => Hooks.setOverride(p, hook, f, Hooks.numValue(s)))

/** FreeU v2 (forge "FreeU Integrated"). */
class FreeUPlugin extends HookPlugin:
  def id = "freeu"
  def name = "FreeU"
  def hook = "freeu"
  override def description = "Re-weights U-Net backbone and skip features (FreeU v2)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("b1", "B1", 0, 2, 0.01, 1.01),
    Slider("b2", "B2", 0, 2, 0.01, 1.02),
    Slider("s1", "S1", 0, 4, 0.01, 0.99),
    Slider("s2", "S2", 0, 4, 0.01, 0.95)
  )
  def hookFields(v: Values): ujson.Obj =
    ujson.Obj("b1" -> v.double("b1", 1.01), "b2" -> v.double("b2", 1.02), "s1" -> v.double("s1", 0.99), "s2" -> v.double("s2", 0.95))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    val stages = ujson.Arr(
      ujson.Obj("backbone_factor" -> o("b1"), "skip_factor" -> o("s1")),
      ujson.Obj("backbone_factor" -> o("b2"), "skip_factor" -> o("s2"))
    )
    Seq("FreeU Stages" -> stages.render(), "FreeU Version" -> "2")
  override def xyzAxes: Seq[XyzAxis] =
    Seq("b1" -> "FreeU B1", "b2" -> "FreeU B2", "s1" -> "FreeU S1", "s2" -> "FreeU S2").map((f, n) => numAxis(n, f))

object DynamicThresholdingPlugin:
  val Modes: Seq[String] = Seq(
    "Constant", "Linear Down", "Cosine Down", "Half Cosine Down", "Linear Up", "Cosine Up", "Half Cosine Up",
    "Power Up", "Power Down", "Linear Repeating", "Cosine Repeating", "Sawtooth"
  )

/** Dynamic thresholding (CFG-Fix), forge "DynamicThresholding (CFG-Fix) Integrated". */
class DynamicThresholdingPlugin extends HookPlugin:
  import DynamicThresholdingPlugin.Modes
  def id = "dynamic_thresholding"
  def name = "Dynamic Thresholding (CFG-Fix)"
  def hook = "dynamic_thresholding"
  override def description = "Allows high CFG scales by clamping the guidance to mimic a lower scale."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Check("simple_mode", "Simple mode (only mimic scale and threshold percentile)"),
    Slider("mimic_scale", "Mimic Scale", 0, 100, 0.5, 7.0),
    Slider("threshold_percentile", "Threshold Percentile", 0, 1, 0.01, 1.0),
    Choice("mimic_mode", "Mimic Mode", Modes, "Constant"),
    Slider("mimic_scale_min", "Mimic Scale Min", 0, 100, 0.5, 0.0),
    Choice("cfg_mode", "CFG Mode", Modes, "Constant"),
    Slider("cfg_scale_min", "CFG Scale Min", 0, 100, 0.5, 0.0),
    Slider("sched_val", "Schedule Value", 0, 100, 0.01, 1.0),
    Check("separate_feature_channels", "Separate Feature Channels", true),
    Choice("scaling_startpoint", "Scaling Startpoint", Seq("MEAN", "ZERO"), "MEAN"),
    Choice("variability_measure", "Variability Measure", Seq("AD", "STD"), "AD"),
    Slider("interpolate_phi", "Interpolate Phi", 0, 1, 0.01, 1.0)
  )
  def hookFields(v: Values): ujson.Obj =
    val simple = v.bool("simple_mode")
    def s(k: String, d: String) = if simple then d else v.str(k, d)
    def d(k: String, dv: Double) = if simple then dv else v.double(k, dv)
    ujson.Obj(
      "mimic_scale" -> v.double("mimic_scale", 7.0),
      "threshold_percentile" -> v.double("threshold_percentile", 1.0),
      "mimic_mode" -> s("mimic_mode", "Constant"),
      "mimic_scale_min" -> d("mimic_scale_min", 0.0),
      "cfg_mode" -> s("cfg_mode", "Constant"),
      "cfg_scale_min" -> d("cfg_scale_min", 0.0),
      "sched_val" -> d("sched_val", 1.0),
      "separate_feature_channels" -> ujson.Bool(
        simple || Set("true", "enable", "enabled", "1", "yes").contains(v.str("separate_feature_channels", "true").trim.toLowerCase)
      ),
      "scaling_startpoint" -> s("scaling_startpoint", "MEAN"),
      "variability_measure" -> s("variability_measure", "AD"),
      "interpolate_phi" -> d("interpolate_phi", 1.0)
    )
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "Dynamic thresholding enabled" -> "True",
    "Dynamic thresholding mimic scale" -> field(o, "mimic_scale"),
    "Dynamic thresholding threshold percentile" -> field(o, "threshold_percentile"),
    "Dynamic thresholding mimic mode" -> field(o, "mimic_mode"),
    "Dynamic thresholding mimic scale min" -> field(o, "mimic_scale_min"),
    "Dynamic thresholding CFG mode" -> field(o, "cfg_mode"),
    "Dynamic thresholding CFG scale min" -> field(o, "cfg_scale_min"),
    "Dynamic thresholding sched val" -> field(o, "sched_val"),
    "Dynamic thresholding separate feature channels" -> field(o, "separate_feature_channels"),
    "Dynamic thresholding scaling startpoint" -> field(o, "scaling_startpoint"),
    "Dynamic thresholding variability measure" -> field(o, "variability_measure"),
    "Dynamic thresholding interpolate phi" -> field(o, "interpolate_phi")
  )
  override def xyzAxes: Seq[XyzAxis] = Seq(
    numAxis("DynThres Mimic Scale", "mimic_scale"),
    numAxis("DynThres Threshold Percentile", "threshold_percentile")
  )

/** HyperTile (forge "HyperTile Integrated"). */
class HyperTilePlugin extends HookPlugin:
  def id = "hypertile"
  def name = "HyperTile"
  def hook = "hypertile"
  override def description = "Speeds up large images by tiling self-attention."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    IntField("tile_size", "Tile Size", 1, 2048, 256),
    IntField("swap_size", "Swap Size", 1, 128, 2),
    IntField("max_depth", "Max Depth", 0, 10, 0),
    Check("scale_depth", "Scale Depth")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "tile_size" -> v.int("tile_size", 256),
    "swap_size" -> v.int("swap_size", 2),
    "max_depth" -> v.int("max_depth", 0),
    "scale_depth" -> v.bool("scale_depth")
  )
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "Hypertile tile size" -> field(o, "tile_size"),
    "Hypertile swap size" -> field(o, "swap_size"),
    "Hypertile max depth" -> field(o, "max_depth"),
    "Hypertile scale depth" -> field(o, "scale_depth")
  )
  override def xyzAxes: Seq[XyzAxis] = Seq(
    XyzAxis("Hypertile tile size", isNumeric = true)((p, s) =>
      Hooks.setOverride(p, hook, "tile_size", ujson.Num(s.trim.toDouble.toInt.toDouble))
    )
  )

object PagPlugin:
  /** UI words to diffusers layer ids: "middle"/"mid" -> mid; "down"/"up" kept; free text split on commas/spaces. */
  def parseLayers(text: String): Seq[String] =
    val ids = text.split("[,\\s]+").toSeq.map(_.trim).filter(_.nonEmpty).map {
      case s if s.equalsIgnoreCase("middle") || s.equalsIgnoreCase("mid") => "mid"
      case s if s.equalsIgnoreCase("down")                                   => "down"
      case s if s.equalsIgnoreCase("up")                                     => "up"
      case s                                                                 => s
    }
    if ids.isEmpty then Seq("mid") else ids.distinct

/** Perturbed-Attention Guidance. */
class PagPlugin extends HookPlugin:
  def id = "pag"
  def name = "Perturbed-Attention Guidance"
  def hook = "pag"
  override def description = "Adds guidance from a perturbed self-attention pass (PAG)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("scale", "PAG Scale", 0, 30, 0.1, 3.0),
    Slider("adaptive_scale", "Adaptive Scale", 0, 1, 0.01, 0.0),
    Text("layers", "Applied layers", "mid", placeholder = "diffusers layer ids, e.g. mid, up.block_0, down.block_2")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "scale" -> v.double("scale", 3.0),
    "adaptive_scale" -> v.double("adaptive_scale", 0.0),
    "layers" -> ujson.Arr.from(PagPlugin.parseLayers(v.str("layers", "mid")).map(ujson.Str(_)))
  )
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "PAG scale" -> field(o, "scale"),
    "PAG adaptive scale" -> field(o, "adaptive_scale"),
    "PAG applied layers" -> o.value.get("layers").collect { case a: ujson.Arr => a.value.map(Hooks.num).mkString(", ") }.getOrElse("")
  )
  override def xyzAxes: Seq[XyzAxis] = Seq(numAxis("PAG scale", "scale"))
