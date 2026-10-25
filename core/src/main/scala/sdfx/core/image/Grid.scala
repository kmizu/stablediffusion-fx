package sdfx.core.image

import java.awt.image.BufferedImage
import java.awt.{Color, Font, RenderingHints}

/** Image grids with optional row/column labels (A1111 style). Pure Java2D, works headless. */
object Grid:
  /** Columns that make the grid close to square, like A1111's `opts.n_rows = -1`. */
  def autoColumns(n: Int): Int = if n <= 0 then 1 else math.ceil(math.sqrt(n.toDouble)).toInt

  def layout(n: Int, cols: Int): (Int, Int) =
    val c = math.max(1, math.min(cols, n))
    (c, math.max(1, math.ceil(n.toDouble / c).toInt))

  def plain(images: Seq[BufferedImage], cols: Int = 0): BufferedImage =
    labeled(images, if cols > 0 then cols else autoColumns(images.size), Nil, Nil)

  /** Images are laid out row-major. Missing cells (null) are left empty. */
  def labeled(images: Seq[BufferedImage], cols: Int, colLabels: Seq[String], rowLabels: Seq[String], title: Option[String] = None): BufferedImage =
    require(images.nonEmpty, "grid needs at least one image")
    val (c, r) = layout(images.size, cols)
    val cw = images.filter(_ != null).map(_.getWidth).maxOption.getOrElse(64)
    val ch = images.filter(_ != null).map(_.getHeight).maxOption.getOrElse(64)
    val fontSize = math.max(14, math.min(cw, ch) / 18)
    val font = new Font(Font.SANS_SERIF, Font.PLAIN, fontSize)
    val lineH = (fontSize * 1.35).toInt
    def lines(s: String, maxW: Int): Seq[String] = wrap(s, font, maxW)
    val topH = if colLabels.nonEmpty then colLabels.map(l => lines(l, cw - 8).size).maxOption.getOrElse(1) * lineH + 12 else 0
    val leftW = if rowLabels.nonEmpty then math.min(cw, math.max(fontSize * 6, rowLabels.map(_.length).maxOption.getOrElse(0) * fontSize * 6 / 10 + 16)) else 0
    val titleH = title.fold(0)(t => lines(t, c * cw + leftW - 8).size * lineH + 12)
    val out = new BufferedImage(leftW + c * cw, titleH + topH + r * ch, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    try
      g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
      g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      g.setColor(Color.WHITE)
      g.fillRect(0, 0, out.getWidth, out.getHeight)
      g.setColor(Color.BLACK)
      g.setFont(font)
      val fm = g.getFontMetrics
      def drawCentered(text: String, x: Int, y: Int, w: Int): Unit =
        lines(text, w - 8).zipWithIndex.foreach { (ln, i) =>
          g.drawString(ln, x + (w - fm.stringWidth(ln)) / 2, y + fm.getAscent + i * lineH)
        }
      title.foreach(t => drawCentered(t, 0, 6, out.getWidth))
      colLabels.take(c).zipWithIndex.foreach((l, i) => drawCentered(l, leftW + i * cw, titleH + 6, cw))
      rowLabels.take(r).zipWithIndex.foreach { (l, i) =>
        val ls = lines(l, leftW - 8)
        val y0 = titleH + topH + i * ch + (ch - ls.size * lineH) / 2
        ls.zipWithIndex.foreach((ln, j) => g.drawString(ln, (leftW - fm.stringWidth(ln)) / 2, y0 + fm.getAscent + j * lineH))
      }
      images.zipWithIndex.foreach { (img, i) =>
        if img != null then
          val x = leftW + (i % c) * cw
          val y = titleH + topH + (i / c) * ch
          g.drawImage(img, x + (cw - img.getWidth) / 2, y + (ch - img.getHeight) / 2, null)
      }
    finally g.dispose()
    out

  private def wrap(s: String, font: Font, maxW: Int): Seq[String] =
    val img = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB)
    val g = img.createGraphics()
    try
      val fm = g.getFontMetrics(font)
      val words = s.split(" ").toSeq
      val out = scala.collection.mutable.ArrayBuffer[String]()
      var cur = ""
      words.foreach { w =>
        val cand = if cur.isEmpty then w else s"$cur $w"
        if fm.stringWidth(cand) <= math.max(maxW, 10) || cur.isEmpty then cur = cand
        else
          out += cur
          cur = w
      }
      if cur.nonEmpty || out.isEmpty then out += cur
      out.toSeq
    finally g.dispose()

  def scaleToFit(img: BufferedImage, maxSide: Int): BufferedImage =
    val s = maxSide.toDouble / math.max(img.getWidth, img.getHeight)
    if s >= 1 then img
    else
      val w = math.max(1, (img.getWidth * s).toInt)
      val h = math.max(1, (img.getHeight * s).toInt)
      val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      val g = out.createGraphics()
      g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      g.drawImage(img, 0, 0, w, h, null)
      g.dispose()
      out
