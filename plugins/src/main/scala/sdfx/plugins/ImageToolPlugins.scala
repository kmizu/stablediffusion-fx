package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

import java.awt.image.BufferedImage
import java.nio.file.{Path, Paths}

/** Engine-side image tools (python/sdfx/image_tools.py) shared by the plugins below. */
private[plugins] object ImageTools:
  final case class Result(image: BufferedImage, outputs: Map[String, Path], info: Map[String, String])

  def run(ctx: PluginContext, tool: String, img: BufferedImage, params: (String, ujson.Value)*): Result =
    val src = Support.writeTemp(ctx, img)
    try
      val r = ctx.engineCall("image_tool", ujson.Obj.from(Seq("tool" -> ujson.Str(tool), "image_path" -> ujson.Str(src.toString)) ++ params))
      val main = Extras.jsonPath(r, "path").getOrElse(throw new IllegalStateException(s"$tool returned no image"))
      val outs = r.obj.get("outputs").collect { case o: ujson.Obj => o.value.collect { case (k, ujson.Str(p)) => k -> Paths.get(p) }.toMap }
        .getOrElse(Map.empty)
      val info = r.obj.get("info").collect { case o: ujson.Obj => o.value.map((k, v) => k -> Hooks.num(v)).toMap }.getOrElse(Map.empty)
      try Result(Support.readImage(main), outs, info)
      finally Support.deleteQuietly(Paths.get(main))
    finally Support.deleteQuietly(src)

  /** Runs a tool and returns its image plus (optionally) one of its extra outputs, for ExtrasPlugin.transform. */
  def images(ctx: PluginContext, tool: String, img: BufferedImage, extra: Option[String], post: Seq[(String, String)],
             params: (String, ujson.Value)*): Seq[(BufferedImage, Seq[(String, String)])] =
    val r = run(ctx, tool, img, params*)
    try
      val more = extra.flatMap(r.outputs.get).map(p => (Support.readImage(p), post :+ ("Output" -> extra.get))).toSeq
      ((r.image, post ++ r.info.toSeq.map((k, v) => s"$tool $k" -> v)) +: more)
    finally Support.deleteQuietly(r.outputs.values.toSeq*)

/** Pixel art (sd-pixel, sd-webui-pixelart, stable-diffusion-webui-pixelization; algorithmic re-implementation). */
class PixelArtPlugin extends ExtrasPlugin:
  def id = "pixel_art"
  def name = "Pixel art"
  override def description = "Turns the image into pixel art: cell-average downscale, a k-means colour palette, optional " +
    "ordered dithering and outlines."
  override def controls: Seq[Control] = Seq(
    IntField("pixel_size", "Pixel size", 1, 64, 8),
    IntField("colors", "Palette colours", 2, 256, 16),
    Slider("dither", "Ordered dithering", 0, 1, 0.05, 0.0),
    Check("outline", "Dark outlines"),
    Check("upscale", "Scale back up to the original size", true)
  )
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    ImageTools.images(ctx, "pixel_art", img, None, Seq("Pixel art" -> s"${v.int("pixel_size", 8)}px, ${v.int("colors", 16)} colours"),
      "pixel_size" -> v.int("pixel_size", 8), "colors" -> v.int("colors", 16), "dither" -> v.double("dither", 0.0),
      "outline" -> v.bool("outline"), "upscale" -> v.bool("upscale", true), "seed" -> math.max(0L, p.seed).toDouble)

/** Color Enhance (sd-webui-color-enhance, the GEGL filter). */
class ColorEnhancePlugin extends ExtrasPlugin:
  def id = "color_enhance"
  def name = "Color enhance"
  override def description = "Stretches the colourfulness (chroma) to the full range while keeping lightness and hue (GEGL Color Enhance)."
  override def controls: Seq[Control] = Seq(Slider("strength", "Strength", 0, 1, 0.05, 1.0),
    Slider("percentile", "Chroma percentile treated as the maximum", 90, 100, 0.1, 99.5))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    ImageTools.images(ctx, "color_enhance", img, None, Seq("Color enhance" -> Values.fmt(v.double("strength", 1.0))),
      "strength" -> v.double("strength", 1.0), "percentile" -> v.double("percentile", 99.5))

/** AMD FSR 1.0 integration: Lanczos upscale + FidelityFX CAS sharpening (RCAS-style). */
class FsrUpscalePlugin extends ExtrasPlugin:
  def id = "fsr_upscale"
  def name = "FSR-style upscale (CAS)"
  override def description = "Fast non-AI upscale: Lanczos resampling followed by AMD contrast adaptive sharpening."
  override def controls: Seq[Control] = Seq(Slider("scale", "Scale", 1, 4, 0.05, 2.0), Slider("sharpness", "Sharpness", 0, 1, 0.05, 0.5))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    ImageTools.images(ctx, "cas_upscale", img, None, Seq("FSR upscale" -> s"x${Values.fmt(v.double("scale", 2.0))}"),
      "scale" -> v.double("scale", 2.0), "sharpness" -> v.double("sharpness", 0.5))

/** A Person Mask Generator: person segmentation mask. */
class PersonMaskPlugin extends ExtrasPlugin:
  def id = "person_mask"
  def name = "Person mask"
  override def description = "Saves a black-and-white mask of the people in the image (YOLO person segmentation, " +
    "downloads person_yolov8n-seg.pt from Bingsu/adetailer on first use)."
  override def controls: Seq[Control] = Seq(Slider("confidence", "Detection confidence", 0, 1, 0.01, 0.3),
    IntField("dilate", "Grow mask (px)", 0, 64, 4), Check("invert", "Invert (mask the background)"))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    ImageTools.images(ctx, "person_mask", img, None, Seq("Person mask" -> (if v.bool("invert") then "background" else "people")),
      "confidence" -> v.double("confidence", 0.3), "dilate" -> v.int("dilate", 4), "invert" -> v.bool("invert"))

/** Background removal (stable-diffusion-webui-rembg, ABG, PBRemTools, sd-webui-bg-mask, GeekyRemB). */
class RemoveBackgroundPlugin extends ExtrasPlugin:
  import RemoveBackgroundPlugin.*
  def id = "remove_background"
  def name = "Remove background"
  override def description = "Cuts out the subject: rembg models (pip install rembg; isnet-anime for illustrations) or " +
    "the built-in person segmentation. Saves a transparent PNG and, optionally, the mask."
  override def controls: Seq[Control] = Seq(
    Choice("model", "Model", Models, "isnet-anime"),
    Choice("background", "Background", Seq("transparent", "white", "black", "green"), "transparent"),
    Check("alpha_matting", "Alpha matting (softer edges, slower; rembg models)"),
    Check("save_mask", "Also save the mask")
  )
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    ImageTools.images(ctx, "remove_background", img, Option.when(v.bool("save_mask"))("mask"),
      Seq("Background removal" -> v.str("model", "isnet-anime")),
      "model" -> v.str("model", "isnet-anime"), "background" -> v.str("background", "transparent"),
      "alpha_matting" -> v.bool("alpha_matting"))

object RemoveBackgroundPlugin:
  val Models: Seq[String] = Seq("isnet-anime", "isnet-general-use", "u2net", "u2netp", "u2net_human_seg", "silueta",
    "person_segmentation")

/** Shared by the mask-driven img2img scripts: inpaint the init image with a mask made by an image tool. */
private[plugins] abstract class MaskedImg2ImgScript extends Plugin:
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  def makeMask(img: BufferedImage, v: Values, ctx: PluginContext): BufferedImage
  def infotext(v: Values): Seq[(String, String)]

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val i0 = p0.img2img.get
    val img = Support.readImage(i0.initImage)
    ctx.status(s"$name: making the mask", 0, 2)
    val mask = makeMask(img, v, ctx)
    val maskPath = Support.writeTemp(ctx, mask)
    try
      val p = p0.copy(img2img = Some(i0.copy(mask = Some(maskPath.toString), maskBlur = v.int("mask_blur", 4),
        inpaintingMaskInvert = false)), extra = p0.extra ++ infotext(v) + ("Script" -> name))
      ctx.status(s"$name: inpainting", 1, 2)
      val outs = Support.step(ctx)(ctx.generate(p)).getOrElse(Nil)
      val maskOut = Option.when(v.bool("save_mask"))(ctx.saveImage(mask, "img2img", p0, Support.infotext(p0, p0.seed)))
      JobOutput(outs ++ maskOut)
    finally Support.deleteQuietly(maskPath)

/** depthmap2mask: inpaint by depth (near or far parts of the image). */
class DepthMaskPlugin extends MaskedImg2ImgScript:
  def id = "depthmap2mask"
  def name = "Depth aware img2img mask"
  override def description = "Estimates depth (DPT) and inpaints only the parts between the near/far thresholds, e.g. " +
    "keep the background and redraw the subject."
  override def controls: Seq[Control] = Seq(
    Slider("near", "From (1 = nearest)", 0, 1, 0.01, 0.5),
    Slider("far", "To", 0, 1, 0.01, 1.0),
    Check("invert", "Invert (redraw everything else)"),
    IntField("blur", "Mask softness", 0, 32, 4),
    IntField("mask_blur", "Inpaint mask blur", 0, 64, 4),
    Check("save_mask", "Also save the mask")
  )
  def makeMask(img: BufferedImage, v: Values, ctx: PluginContext): BufferedImage =
    val r = ImageTools.run(ctx, "depth_mask", img, "near" -> v.double("near", 0.5), "far" -> v.double("far", 1.0),
      "invert" -> v.bool("invert"), "blur" -> v.int("blur", 4))
    Support.deleteQuietly(r.outputs.values.toSeq*)
    r.image
  def infotext(v: Values) = Seq("Depth mask" -> s"${Values.fmt(v.double("near", 0.5))}-${Values.fmt(v.double("far", 1.0))}${if v.bool("invert") then " inverted" else ""}")

/** Inpaint background / Mask2Background: keep the people, redraw the background from the prompt. */
class InpaintBackgroundPlugin extends MaskedImg2ImgScript:
  def id = "inpaint_background"
  def name = "Inpaint background"
  override def description = "Masks the people (YOLO person segmentation) and redraws only the background from the prompt."
  override def controls: Seq[Control] = Seq(
    Slider("confidence", "Person detection confidence", 0, 1, 0.01, 0.3),
    IntField("dilate", "Protect a margin around people (px)", 0, 64, 8),
    IntField("mask_blur", "Inpaint mask blur", 0, 64, 4),
    Check("save_mask", "Also save the mask")
  )
  def makeMask(img: BufferedImage, v: Values, ctx: PluginContext): BufferedImage =
    ImageTools.run(ctx, "person_mask", img, "confidence" -> v.double("confidence", 0.3), "dilate" -> v.int("dilate", 8),
      "invert" -> true).image
  def infotext(v: Values) = Seq("Inpaint background" -> "people kept")
