package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import java.awt.image.BufferedImage

/** Dedicated source-equivalent panel, alongside the nine ordinary upscaler choices. */
class Waifu2xPlugin extends ExtrasPlugin:
  def id = "waifu2x"
  def name = "Waifu2x upscale"
  override def description = "Neural 2x/4x/8x/16x upscale with Anime/Photo denoising or CARN V2. " +
    "Pinned model weights download on first use (about 10 MB each, CARN 4 MB). CARN ignores the noise choice. RGB output."
  private val noises = Seq("None", "Low", "Medium", "High")
  override def controls: Seq[Control] = Seq(
    Choice("style", "Style", Seq("Anime", "Photo", "CarnV2"), "Anime"),
    Choice("noise", "Noise reduction", noises, "Medium"),
    Choice("scale", "Scale", Seq("2x", "4x", "8x", "16x"), "2x")
  )
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    val style = v.str("style", "Anime")
    val noise = noises.indexOf(v.str("noise", "Medium"))
    val scale = v.str("scale", "2x")
    require(Seq("Anime", "Photo", "CarnV2").contains(style), "Invalid Waifu2x style")
    require(noise >= 0 && Seq("2x", "4x", "8x", "16x").contains(scale), "Invalid Waifu2x noise/scale")
    ImageTools.images(ctx, "waifu2x", img, None,
      Seq("Waifu2x style" -> style, "Waifu2x noise" -> (if style == "CarnV2" then "not applicable" else noise.toString), "Waifu2x scale" -> scale),
      "style" -> style, "noise" -> noise, "scale" -> scale.stripSuffix("x").toInt)
