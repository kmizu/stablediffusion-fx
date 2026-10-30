package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.plugins.imageops.ImageOps

import java.awt.image.BufferedImage
import java.awt.{Color, GradientPaint}
import java.nio.file.{Path, Paths}

/** Ultimate SD upscale (Coyote-A/ultimate-upscale-for-automatic1111), re-implemented.
  *
  * Upscales, then redraws the image tile by tile as "only masked" inpainting of the running image (each tile sees
  * the tiles drawn before it), in linear or chess order; optional seam-fix passes over the tile borders.
  */
class UltimateSdUpscalePlugin extends Plugin:
  import UltimateSdUpscalePlugin.*
  def id = "ultimate_sd_upscale"
  def name = "Ultimate SD upscale"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Upscale, then redraw tile by tile with masked inpainting; optional seam fixing."
  override def controls: Seq[Control] = Seq(
    Choice("target", "Target size type", Seq("Scale from image size", "Custom size"), "Scale from image size"),
    Slider("scale", "Scale", 1, 16, 0.05, 2.0),
    IntField("custom_width", "Custom width", 64, 8192, 2048),
    IntField("custom_height", "Custom height", 64, 8192, 2048),
    DynamicChoice("upscaler", "Upscaler", ChoiceSource.Upscalers, "None", allowNone = true),
    Choice("redraw_mode", "Redraw type", RedrawModes, "Linear"),
    IntField("tile_width", "Tile width", 0, 2048, 512),
    IntField("tile_height", "Tile height (0 = tile width)", 0, 2048, 0),
    IntField("mask_blur", "Mask blur", 0, 64, 8),
    IntField("padding", "Padding", 0, 512, 32),
    Choice("seams_fix", "Seams fix type", SeamsModes, "None"),
    Slider("seams_denoise", "Seams fix denoise", 0, 1, 0.01, 0.35),
    IntField("seams_width", "Seams fix width", 0, 128, 64),
    IntField("seams_mask_blur", "Seams fix mask blur", 0, 64, 4),
    IntField("seams_padding", "Seams fix padding", 0, 128, 16)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val i0 = p0.img2img.get
    val src = Support.readImage(ctx, i0.initImage)
    val (tw0, th0) = (v.int("tile_width", 512), v.int("tile_height", 0))
    val tw = if tw0 > 0 then tw0 else if th0 > 0 then th0 else p0.width
    val th = if th0 > 0 then th0 else tw
    val (outW, outH) =
      if v.str("target", "") == "Custom size" then (v.int("custom_width", 2048), v.int("custom_height", 2048))
      else
        val s = v.double("scale", 2.0)
        (math.round(src.getWidth * s).toInt, math.round(src.getHeight * s).toInt)
    val upscaler = v.str("upscaler", "None")
    val factor = math.max(outW.toDouble / src.getWidth, outH.toDouble / src.getHeight)
    val upPath = Support.step(ctx)(ctx.upscale(Paths.get(i0.initImage), upscaler, factor)) match
      case Some(p) => p
      case None    => return JobOutput(Nil)
    var cur =
      try
        val up = Support.readImage(ctx, upPath)
        if up.getWidth == outW && up.getHeight == outH then up else ImageOps.resize(up, outW, outH)
      finally if upPath.toString != i0.initImage then Support.deleteQuietly(upPath)

    val mode = v.str("redraw_mode", "Linear")
    val seams = v.str("seams_fix", "None")
    val seed = Support.fixSeed(p0.seed)
    val tiles = if mode == "None" then Nil else redrawOrder(outW, outH, tw, th, mode == "Chess")
    val seamJobs = seamPasses(outW, outH, tw, th, seams, v.int("seams_width", 64))
    val total = tiles.size + seamJobs.size
    ctx.log(s"Ultimate SD upscale: ${outW}x$outH, ${tiles.size} tiles ($mode), ${seamJobs.size} seam passes ($seams)")
    val extra = p0.extra ++ Seq("Ultimate SD upscale upscaler" -> upscaler,
      "Ultimate SD upscale tile_width" -> tw.toString, "Ultimate SD upscale tile_height" -> th.toString,
      "Ultimate SD upscale mask_blur" -> v.int("mask_blur", 8).toString,
      "Ultimate SD upscale padding" -> v.int("padding", 32).toString,
      "Ultimate SD upscale redraw_mode" -> mode, "Ultimate SD upscale seams_fix" -> seams)
    var done = 0

    def redraw(mask: BufferedImage, blur: Int, padding: Int, denoise: Double): Boolean =
      val imgPath = Support.writeTemp(ctx, cur)
      val maskPath = Support.writeTemp(ctx, mask)
      val p = p0.copy(
        img2img = Some(i0.copy(initImage = imgPath.toString, mask = Some(maskPath.toString), maskBlur = blur,
          inpaintFullRes = true, inpaintFullResPadding = padding, inpaintingFill = 1, resizeMode = 0,
          inpaintingMaskInvert = false, denoisingStrength = denoise)),
        width = tw, height = th, seed = seed, batchSize = 1, nIter = 1, extra = extra)
      ctx.status(s"Ultimate SD upscale ${done + 1}/$total", done, total)
      try
        Support.step(ctx)(ctx.generateTransient(p)) match
          case Some(outs) =>
            outs.headOption.foreach(o => cur = Support.readImage(ctx, o.path))
            Support.deleteQuietly(outs.map(_.path)*)
            done += 1
            true
          case None => false
      finally Support.deleteQuietly(imgPath, maskPath)

    val blur = v.int("mask_blur", 8)
    val pad = v.int("padding", 32)
    val ok = tiles.forall(r => redraw(rectMask(outW, outH, r), blur, pad, i0.denoisingStrength)) &&
      seamJobs.forall(j => redraw(j(outW, outH), v.int("seams_mask_blur", 4), v.int("seams_padding", 16),
        v.double("seams_denoise", 0.35)))
    if !ok then return JobOutput(Nil)
    val gp = p0.copy(width = outW, height = outH, seed = seed, extra = extra + ("Script" -> name))
    JobOutput(Seq(ctx.saveImage(cur, "img2img", gp, Support.infotext(gp, seed))))

object UltimateSdUpscalePlugin:
  val RedrawModes: Seq[String] = Seq("Linear", "Chess", "None")
  val SeamsModes: Seq[String] = Seq("None", "Band pass", "Half tile offset pass", "Half tile offset pass + intersections")

  final case class Rect(x: Int, y: Int, w: Int, h: Int)

  /** Tile rectangles in redraw order: row-major, or chess (all "white" squares first, then the "black" ones). */
  def redrawOrder(w: Int, h: Int, tw: Int, th: Int, chess: Boolean): Seq[Rect] =
    val cols = math.max(1, math.ceil(w.toDouble / tw).toInt)
    val rows = math.max(1, math.ceil(h.toDouble / th).toInt)
    val all = for r <- 0 until rows; c <- 0 until cols yield (r, c, Rect(c * tw, r * th, math.min(tw, w - c * tw), math.min(th, h - r * th)))
    if chess then all.filter((r, c, _) => (r + c) % 2 == 0).map(_._3) ++ all.filter((r, c, _) => (r + c) % 2 == 1).map(_._3)
    else all.map(_._3)

  def rectMask(w: Int, h: Int, r: Rect): BufferedImage =
    val m = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val g = m.createGraphics()
    try
      g.setColor(Color.WHITE)
      g.fillRect(r.x, r.y, r.w, r.h)
    finally g.dispose()
    m

  /** Masks of the seam-fix passes; each builds the mask for the final image size. */
  def seamPasses(w: Int, h: Int, tw: Int, th: Int, mode: String, width: Int): Seq[(Int, Int) => BufferedImage] =
    val cols = math.max(1, math.ceil(w.toDouble / tw).toInt)
    val rows = math.max(1, math.ceil(h.toDouble / th).toInt)
    mode match
      case "Band pass" =>
        val vertical = (1 until cols).map(c => Rect(c * tw - width / 2, 0, width, h))
        val horizontal = (1 until rows).map(r => Rect(0, r * th - width / 2, w, width))
        (vertical ++ horizontal).map(r => (ww: Int, hh: Int) => rectMask(ww, hh, r))
      case m if m.startsWith("Half tile offset pass") =>
        // tiles straddling the vertical seams, then the horizontal ones, each with a gradient fading to the tile edge
        val vs = for r <- 0 until rows; c <- 1 until cols yield Rect(c * tw - tw / 2, r * th, tw, math.min(th, h - r * th))
        val hs = for r <- 1 until rows; c <- 0 until cols yield Rect(c * tw, r * th - th / 2, math.min(tw, w - c * tw), th)
        val inter =
          if m.endsWith("intersections") then
            for r <- 1 until rows; c <- 1 until cols yield Rect(c * tw - tw / 2, r * th - th / 2, tw, th)
          else Nil
        vs.map(r => (ww: Int, hh: Int) => gradientMask(ww, hh, r, vertical = false)) ++
          hs.map(r => (ww: Int, hh: Int) => gradientMask(ww, hh, r, vertical = true)) ++
          inter.map(r => (ww: Int, hh: Int) => radialMask(ww, hh, r))
      case _ => Nil

  /** White in the middle of `r`, fading to black at the two edges across the seam direction. */
  def gradientMask(w: Int, h: Int, r: Rect, vertical: Boolean): BufferedImage =
    val m = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val g = m.createGraphics()
    try
      val (x0, y0, x1, y1) = if vertical then (0f, r.y.toFloat, 0f, (r.y + r.h / 2).toFloat) else (r.x.toFloat, 0f, (r.x + r.w / 2).toFloat, 0f)
      g.setPaint(new GradientPaint(x0, y0, Color.BLACK, x1, y1, Color.WHITE, true))
      g.fillRect(r.x, r.y, r.w, r.h)
    finally g.dispose()
    m

  def radialMask(w: Int, h: Int, r: Rect): BufferedImage =
    val m = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val cx = r.x + r.w / 2.0
    val cy = r.y + r.h / 2.0
    val rad = math.min(r.w, r.h) / 2.0
    for y <- math.max(0, r.y) until math.min(h, r.y + r.h); x <- math.max(0, r.x) until math.min(w, r.x + r.w) do
      val d = math.hypot(x - cx, y - cy) / rad
      val v = math.max(0, math.min(255, ((1 - d) * 255).round.toInt))
      m.getRaster.setSample(x, y, 0, v)
    m
