package sdfx.core.image

import java.awt.{Color, Font, Rectangle, RenderingHints}
import java.awt.image.BufferedImage

/** Client-side tile annotations from AlUlkesh/sd_grid_add_image_number (Unlicense).
  * Plans and renders on copies; no global label cache and no mutation of saved source images.
  */
object GridNumbers:
  val NumbersKey = "Grid image numbers"
  val XyzKey = "Grid XYZ info"
  val SourcesKey = "Grid source images"
  val settingKeys: Set[String] = Set(NumbersKey, XyzKey)
  final case class Label(corner: String, text: String, box: Rectangle, fontSize: Int)

  private def on(extra: Map[String, String], key: String): Boolean = extra.get(key).exists(_.equalsIgnoreCase("true"))
  def enabled(extra: Map[String, String]): Boolean = on(extra, NumbersKey) || on(extra, XyzKey)

  private def isGrid(filename: String): Boolean =
    val stem = filename.takeWhile(_ != '.').toLowerCase(java.util.Locale.ROOT)
    stem.startsWith("grid") || stem.endsWith("-grid")

  def number(filename: String): Option[String] =
    val name = filename.replace('\\', '/').split('/').lastOption.getOrElse("")
    val end = name.indexOf('-')
    val prefix = if end > 0 then name.take(end) else ""
    Option.when(!isGrid(name) && prefix.nonEmpty && prefix.forall(c => c >= '0' && c <= '9'))(prefix)

  /** Upstream corner geometry, measured using the native logical font. Unfit text shrinks, retries without its
    * label prefix, then is omitted. In particular tiny number-only cells never attempt a nonexistent ':' split.
    */
  def labels(width: Int, height: Int, filename: String, extra: Map[String, String], axes: Map[String, String] = Map.empty): Seq[Label] =
    val name = filename.replace('\\', '/').split('/').lastOption.getOrElse("")
    if !enabled(extra) || isGrid(name) then return Nil
    val items = scala.collection.mutable.ArrayBuffer[(String, String)]()
    if on(extra, NumbersKey) then number(name).foreach(n => items += "bottom_left" -> n)
    if on(extra, XyzKey) then
      Seq("X" -> "top_left", "Y" -> "top_right", "Z" -> "bottom_right").foreach { (axis, corner) =>
        axes.get(axis).orElse(extra.get(s"Grid $axis info")).filter(t => t.nonEmpty && t != "Nothing")
          .foreach(t => items += corner -> t)
      }
    val measure = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB).createGraphics()
    try
      items.toSeq.flatMap { (corner, raw) =>
        val candidates = Seq(raw) ++ Option.when(raw.contains(": "))(raw.split(": ", 2)(1)).toSeq
        candidates.iterator.flatMap { text =>
          (24 to 9 by -1).iterator.flatMap { size =>
            val metrics = measure.getFontMetrics(new Font(Font.SANS_SERIF, Font.PLAIN, size))
            val tw = metrics.stringWidth(text)
            val th = metrics.getHeight
            val bw = tw + 20
            val bh = th + 20
            val (x, y) = corner match
              case "bottom_left" => (10, height - 10 - bh)
              case "top_left" => (10, 10)
              case "top_right" => (width - tw - 30, th + 20)
              case "bottom_right" => (width - tw - 30, height - th * 2 - 40)
            Option.when(x >= 10 && y >= 10 && x + bw <= width - 10 && y + bh <= height - 10)(
              Label(corner, text, new Rectangle(x, y, bw, bh), size))
          }.take(1)
        }.take(1).toSeq
      }
    finally measure.dispose()

  def render(source: BufferedImage, filename: String, extra: Map[String, String], axes: Map[String, String] = Map.empty): BufferedImage =
    val planned = labels(source.getWidth, source.getHeight, filename, extra, axes)
    if planned.isEmpty then return source
    val out = new BufferedImage(source.getWidth, source.getHeight, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    try
      g.drawImage(source, 0, 0, null)
      g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
      planned.foreach { label =>
        val b = label.box
        g.setColor(new Color(0, 0, 0, 127))
        g.fillRect(b.x, b.y, b.width, b.height)
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, label.fontSize))
        g.setColor(Color.WHITE)
        g.drawString(label.text, b.x + 10, b.y + 10 + g.getFontMetrics.getAscent)
      }
    finally g.dispose()
    out
