package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.plugins.imageops.ImageOps
import java.awt.Color
import java.awt.image.BufferedImage

/** Reimplementation of brick2face/seamless-tile-inpainting's documented offset-and-inpaint workflow. */
class SeamlessTilePlugin extends Plugin:
  import SeamlessTilePlugin.*
  def id = "seamless_tile"
  def name = "Seamless tile inpainting"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Moves the tile edges to the centre, then inpaints only their seams. " +
    "Use a texture prompt and an inpainting model for best results. Both directions may retain small seams."
  override def controls: Seq[Control] = Seq(
    Choice("direction", "Tiling direction", Directions, "Both"),
    IntField("mask_width", "Masked area width (px)", 1, 1024, 64),
    IntField("mask_blur", "Mask blur (px)", 0, 64, 4),
    Check("restore", "Split sides again after inpainting", true)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val direction = v.str("direction", "Both")
    require(Directions.contains(direction), s"unknown tiling direction: $direction")
    val i0 = p0.img2img.get
    val original = Support.readImage(i0.initImage)
    val src = if original.getWidth == p0.width && original.getHeight == p0.height then original
      else ImageOps.resize(original, p0.width, p0.height)
    val dx = if direction == "Vertical" then 0 else src.getWidth / 2
    val dy = if direction == "Horizontal" then 0 else src.getHeight / 2
    val width = math.max(1, v.int("mask_width", 64))
    val restore = v.bool("restore", true)
    val seed = Support.fixSeed(p0.seed)
    val input = Support.writeTemp(ctx, offset(src, dx, dy))
    try
      val maskPath = Support.writeTemp(ctx, mask(src.getWidth, src.getHeight, direction, width))
      try
        val p = p0.copy(seed = seed, img2img = Some(i0.copy(initImage = input.toString,
          mask = Some(maskPath.toString), maskBlur = v.int("mask_blur", 4), inpaintFullRes = false,
          inpaintingMaskInvert = false, inpaintingFill = 1, resizeMode = 0)), extra = p0.extra ++ Seq(
          "Script" -> name, "Seamless tile direction" -> direction, "Seamless tile mask width" -> width.toString,
          "Seamless tile restore" -> restore.toString))
        ctx.status("Inpainting tile seams", 0, 1)
        Support.step(ctx)(ctx.generateTransient(p)) match
          case None => JobOutput(Nil)
          case Some(outputs) =>
            try
              JobOutput(outputs.map { o =>
                val raw = Support.readImage(o.path)
                val img = if restore then offset(raw, -dx, -dy) else raw
                val metadata = o.params.copy(seed = o.seed, extra = o.params.extra ++ p.extra)
                ctx.saveImage(img, "img2img", metadata, o.infotext)
              })
            finally Support.deleteQuietly(outputs.map(_.path)*)
      finally Support.deleteQuietly(maskPath)
    finally Support.deleteQuietly(input)

object SeamlessTilePlugin:
  val Directions: Seq[String] = Seq("Horizontal", "Vertical", "Both")

  /** Positive offsets move pixels right/down; the negative offset is the exact inverse for odd sizes too. */
  def offset(src: BufferedImage, dx: Int, dy: Int): BufferedImage =
    val w = src.getWidth
    val h = src.getHeight
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until h; x <- 0 until w do
      out.setRGB(x, y, src.getRGB(Math.floorMod(x - dx, w), Math.floorMod(y - dy, h)))
    out

  def mask(w: Int, h: Int, direction: String, width: Int): BufferedImage =
    require(Directions.contains(direction), s"unknown tiling direction: $direction")
    val out = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY)
    val g = out.createGraphics()
    try
      g.setColor(Color.WHITE)
      if direction != "Vertical" then
        val band = math.max(1, math.min(w, width))
        g.fillRect((w - band) / 2, 0, band, h)
      if direction != "Horizontal" then
        val band = math.max(1, math.min(h, width))
        g.fillRect(0, (h - band) / 2, w, band)
    finally g.dispose()
    out
