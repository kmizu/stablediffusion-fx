package sdfx.plugins.imageops

import java.awt.image.BufferedImage
import java.awt.{Color, RenderingHints}

/** A tile rectangle inside a larger image. */
case class Tile(x: Int, y: Int, w: Int, h: Int)

/** Pure Java2D helpers used by the standard plugins (tiling, feather blending, outpaint canvas/mask). */
object ImageOps:

  // ---------------------------------------------------------------- tiling (SD upscale)

  /** Same placement as A1111 `images.split_grid`: tiles of `tileW x tileH` evenly spread so that neighbours overlap
    * by at least `overlap` and the last tile touches the right/bottom edge. Tiles are clamped to the image size.
    */
  def splitTiles(imgW: Int, imgH: Int, tileW: Int, tileH: Int, overlap: Int): Seq[Tile] =
    val tw = math.max(1, math.min(tileW, imgW))
    val th = math.max(1, math.min(tileH, imgH))
    def axis(size: Int, tile: Int): Seq[Int] =
      val ov = math.max(0, math.min(overlap, tile - 1))
      val step = tile - ov
      val n = if size <= tile then 1 else math.max(1, math.ceil((size - ov).toDouble / step).toInt)
      val d = if n > 1 then (size - tile).toDouble / (n - 1) else 0.0
      (0 until n).map(i => math.min((i * d).toInt, size - tile)).distinct
    for
      y <- axis(imgH, th)
      x <- axis(imgW, tw)
    yield Tile(x, y, tw, th)

  def crop(img: BufferedImage, t: Tile): BufferedImage =
    val out = new BufferedImage(t.w, t.h, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.drawImage(img, -t.x, -t.y, null)
    g.dispose()
    out

  /** Linear feather weight of pixel (px,py) inside a tile; edges touching the image border are not feathered. */
  private def weight(t: Tile, px: Int, py: Int, imgW: Int, imgH: Int, feather: Int): Double =
    if feather <= 0 then 1.0
    else
      def ramp(d: Int): Double = math.min(1.0, (d + 1).toDouble / (feather + 1))
      var w = 1.0
      if t.x > 0 then w = math.min(w, ramp(px))
      if t.x + t.w < imgW then w = math.min(w, ramp(t.w - 1 - px))
      if t.y > 0 then w = math.min(w, ramp(py))
      if t.y + t.h < imgH then w = math.min(w, ramp(t.h - 1 - py))
      w

  /** Reassembles processed tiles into a `w x h` image, blending overlaps with a linear feather of `feather` px. Tiles
    * of a different size than their rectangle are rescaled first.
    */
  def mergeTiles(w: Int, h: Int, tiles: Seq[(Tile, BufferedImage)], feather: Int): BufferedImage =
    val acc = Array.ofDim[Double](w * h * 3)
    val wsum = Array.ofDim[Double](w * h)
    tiles.foreach { (t, raw) =>
      val img = if raw.getWidth == t.w && raw.getHeight == t.h then raw else resize(raw, t.w, t.h)
      var py = 0
      while py < t.h do
        var px = 0
        while px < t.w do
          val gx = t.x + px
          val gy = t.y + py
          if gx >= 0 && gy >= 0 && gx < w && gy < h then
            val wt = weight(t, px, py, w, h, feather)
            val rgb = img.getRGB(px, py)
            val i = gy * w + gx
            acc(i * 3) += ((rgb >> 16) & 0xff) * wt
            acc(i * 3 + 1) += ((rgb >> 8) & 0xff) * wt
            acc(i * 3 + 2) += (rgb & 0xff) * wt
            wsum(i) += wt
          px += 1
        py += 1
    }
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    var i = 0
    while i < w * h do
      val s = wsum(i)
      if s > 0 then
        def c(k: Int) = math.max(0, math.min(255, math.round(acc(i * 3 + k) / s).toInt))
        out.setRGB(i % w, i / w, (c(0) << 16) | (c(1) << 8) | c(2))
      i += 1
    out

  def resize(img: BufferedImage, w: Int, h: Int): BufferedImage =
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
    g.drawImage(img, 0, 0, w, h, null)
    g.dispose()
    out

  // ---------------------------------------------------------------- outpainting

  /** Pixels added on each side of the original image; `canvasW/H` are multiples of 8. */
  case class Expansion(left: Int, right: Int, top: Int, bottom: Int, canvasW: Int, canvasH: Int):
    /** The rectangle kept from the original (black in the mask), inset by `blur` on expanded sides. */
    def keepRect(blur: Int): (Int, Int, Int, Int) =
      val x0 = if left > 0 then left + blur else 0
      val y0 = if top > 0 then top + blur else 0
      val x1 = if right > 0 then canvasW - right - blur else canvasW
      val y1 = if bottom > 0 then canvasH - bottom - blur else canvasH
      (x0, y0, math.max(x0, x1), math.max(y0, y1))

  private def ceil8(v: Int): Int = (v + 7) / 8 * 8

  /** Computes the expansion; rounding to a multiple of 8 goes to the right/bottom (or left/top when only that side
    * expands).
    */
  def expansion(w: Int, h: Int, pixels: Int, left: Boolean, right: Boolean, up: Boolean, down: Boolean): Expansion =
    def sides(size: Int, a: Boolean, b: Boolean): (Int, Int) =
      val l0 = if a then pixels else 0
      val r0 = if b then pixels else 0
      val total = ceil8(size + l0 + r0)
      val extra = total - (size + l0 + r0)
      if b || !a then (l0, r0 + extra) else (l0 + extra, r0)
    val (l, r) = sides(w, left, right)
    val (t, b) = sides(h, up, down)
    Expansion(l, r, t, b, w + l + r, h + t + b)

  /** The expanded canvas: original at (left, top), new areas filled by edge replication. */
  def expandCanvas(img: BufferedImage, e: Expansion): BufferedImage =
    val out = new BufferedImage(e.canvasW, e.canvasH, BufferedImage.TYPE_INT_RGB)
    val w = img.getWidth
    val h = img.getHeight
    var y = 0
    while y < e.canvasH do
      val sy = math.max(0, math.min(h - 1, y - e.top))
      var x = 0
      while x < e.canvasW do
        val sx = math.max(0, math.min(w - 1, x - e.left))
        out.setRGB(x, y, img.getRGB(sx, sy))
        x += 1
      y += 1
    out

  /** Inpaint mask: white = repaint, black = keep. */
  def outpaintMask(e: Expansion, blur: Int): BufferedImage =
    val out = new BufferedImage(e.canvasW, e.canvasH, BufferedImage.TYPE_BYTE_GRAY)
    val g = out.createGraphics()
    g.setColor(Color.WHITE)
    g.fillRect(0, 0, e.canvasW, e.canvasH)
    val (x0, y0, x1, y1) = e.keepRect(blur)
    g.setColor(Color.BLACK)
    g.fillRect(x0, y0, x1 - x0, y1 - y0)
    g.dispose()
    out

  // ---------------------------------------------------------------- misc

  /** Adds a white margin of `margin/2` px on every side (A1111 grid margin approximation). */
  def withMargin(img: BufferedImage, margin: Int): BufferedImage =
    if img == null || margin <= 0 then img
    else
      val m = math.max(1, margin / 2)
      val out = new BufferedImage(img.getWidth + 2 * m, img.getHeight + 2 * m, BufferedImage.TYPE_INT_RGB)
      val g = out.createGraphics()
      g.setColor(Color.WHITE)
      g.fillRect(0, 0, out.getWidth, out.getHeight)
      g.drawImage(img, m, m, null)
      g.dispose()
      out

  /** Concatenates images horizontally (top aligned, white background). */
  def hconcat(images: Seq[BufferedImage], gap: Int = 0): BufferedImage =
    val w = images.map(_.getWidth).sum + gap * math.max(0, images.size - 1)
    val h = images.map(_.getHeight).maxOption.getOrElse(1)
    val out = new BufferedImage(math.max(1, w), h, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.setColor(Color.WHITE)
    g.fillRect(0, 0, out.getWidth, h)
    var x = 0
    images.foreach { i =>
      g.drawImage(i, x, 0, null)
      x += i.getWidth + gap
    }
    g.dispose()
    out
