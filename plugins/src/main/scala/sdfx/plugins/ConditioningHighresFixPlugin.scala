package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object ConditioningHighresFixPlugin:
  val modes: Seq[String] = Seq("empty", "lowest", "low", "medium", "high", "highest", "full")
  def replay(fields: Map[String, String]): Option[Values] =
    fields.get("Conditioning Highres Fix").filter(modes.contains) match
      case Some(mode) => Some(Values(Map("enabled" -> "true", "mode" -> mode)))
      case None => Some(Values(Map("enabled" -> "false")))

class ConditioningHighresFixPlugin extends HookPlugin:
  def id = "conditioning_highres_fix"
  def name = "Conditioning Highres Fix"
  def hook = id
  override def description = "Adjusts 9-channel inpainting image conditioning relative to denoising strength."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Choice("mode", "Fix strength", ConditioningHighresFixPlugin.modes, "high"),
    Note("meaning", "Only 9-channel inpainting models use this conditioning. It also applies during masked img2img and hires; disable it when not wanted. " +
      "Upstream recommends high and pixel upscaling for Hires rather than latent upscaling. " +
      "For denoise d: empty = 1, lowest = 1-d/4, low = 1-d/2, medium = 0.5, high = d/2, highest = d/4, full = 0."))
  def hookFields(v: Values): ujson.Obj = ujson.Obj("mode" -> v.str("mode", "high"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    Seq("Conditioning Highres Fix" -> o("mode").str)
