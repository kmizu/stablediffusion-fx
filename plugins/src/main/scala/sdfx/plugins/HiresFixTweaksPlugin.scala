package sdfx.plugins

import sdfx.core.model.{GenerationParams, HiresPromptModes}
import sdfx.core.infotext.HiresSeedMetadata
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

class HiresFixTweaksPlugin extends HookPlugin:
  def id = "hires_fix_tweaks"
  def name = "Hires Fix Tweaks"
  def hook = id
  override def description = "Refine one first-pass result with independent seeds, prompt modes and styles."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enable Hires tweaks"),
    Section("batch_seed", "Hires batches and seed"),
    IntField("batch_count", "Hires batches", 1, 16, 1),
    Check("seed_enabled", "Use a separate Hires seed"),
    Text("seed", "Hires seed (0 = inherit, -1 = random)", "0"),
    Check("extras", "Hires seed extras"),
    Text("subseed", "Variation seed (0 = inherit, -1 = random)", "0"),
    Slider("strength", "Variation strength", 0, 1, .01, 0),
    IntField("resize_width", "Seed resize width", 0, 16384, 0),
    IntField("resize_height", "Seed resize height", 0, 16384, 0),
    Section("prompt_modes", "Hires prompt modes"),
    Choice("prompt_mode", "Positive prompt", HiresPromptModes.modes, "Default"),
    Choice("negative_mode", "Negative prompt", HiresPromptModes.modes, "Default"),
    Text("append_separator", "Append separator", "{newline}"),
    Text("prepend_separator", "Prepend separator", "{newline}"),
    Text("marker", "Search / replace marker", "@"),
    Check("remove_networks", "Remove first-pass network tags from Hires"),
    Check("empty_prompt", "Keep blank Hires positive prompt empty (do not inherit)"),
    Check("empty_negative_prompt", "Keep blank Hires negative prompt empty (do not inherit)"),
    Section("hr_styles", "Hires styles"),
    Check("styles_enabled", "Use separate Hires styles"),
    Text("styles", "Selected Hires styles", "[]"))

  def hookFields(v: Values): ujson.Obj =
    def seed(key: String): Long =
      val n = v.str(key, "0").trim.toLongOption.getOrElse(throw new IllegalArgumentException(s"Invalid Hires $key"))
      require(n >= -1, s"Hires $key must be -1, 0 or positive")
      n
    val styles = ujson.read(v.str("styles", "[]")).arr.toSeq.map(_.str)
    val count = v.int("batch_count", 1)
    require(count >= 1 && count <= 16, "Hires batch count must be 1..16")
    val strength = v.double("strength", 0)
    require(strength.isFinite && strength >= 0 && strength <= 1, "Hires variation strength must be 0..1")
    val width = v.int("resize_width", 0)
    val height = v.int("resize_height", 0)
    require(Seq(width, height).forall(n => n >= 0 && n <= 16384 && n % 8 == 0), "Seed resize dimensions must be multiples of 8 in 0..16384")
    ujson.Obj("batch_count" -> count, "seed_enabled" -> v.bool("seed_enabled"), "seed" -> HiresSeedMetadata.encode(seed("seed")),
      "extras" -> v.bool("extras"), "subseed" -> HiresSeedMetadata.encode(seed("subseed")), "strength" -> strength,
      "resize_width" -> width, "resize_height" -> height, "prompt_mode" -> v.str("prompt_mode", "Default"),
      "negative_mode" -> v.str("negative_mode", "Default"),
      "append_separator" -> v.str("append_separator", v.str("separator", "{newline}")),
      "prepend_separator" -> v.str("prepend_separator", v.str("separator", "{newline}")),
      "marker" -> v.str("marker", "@"), "remove_networks" -> v.bool("remove_networks"),
      "styles_enabled" -> v.bool("styles_enabled"), "styles" -> ujson.Arr.from(styles),
      "empty_prompt" -> v.bool("empty_prompt"), "empty_negative_prompt" -> v.bool("empty_negative_prompt"))

  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Nil // source-compatible templates are recorded during preparation
  override def xyzAxes: Seq[XyzAxis] = Seq(XyzAxis("[Hires fix tweaks] Hires CFG Scale", isNumeric = true) { (p, text) =>
    val cfg = text.trim.toDouble
    require(cfg.isFinite && cfg >= 0 && cfg <= 30, "Hires CFG must be 0..30")
    p.copy(hires = p.hires.copy(cfgScale = Option.when(cfg != 0)(cfg)))
  })
