package sdfx.core.comparison

import java.awt.{AlphaComposite, Color, RenderingHints}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import sdfx.core.infotext.Infotext
import sdfx.core.png.{ImageScrambleSecret, ScrambledPng}

/** Pure comparison geometry and pixels. Input images and their metadata are never changed. */
object ImageComparison:
  case class Rect(x: Double, y: Double, width: Double, height: Double)
  case class View(horizontal: Boolean = true, split: Double = .5, opacity: Double = 1,
      zoom: Double = 1, panX: Double = 0, panY: Double = 0):
    def zoomBy(delta: Double): View = copy(zoom = (zoom + delta).max(.5).min(10))
    def panBy(dx: Double, dy: Double): View = copy(panX = panX + dx, panY = panY + dy)
    def reset: View = copy(zoom = 1, panX = 0, panY = 0)
    def forNewPair: View = copy(split = .5, opacity = 1)
  def clamp(value: Double): Double = value.max(0).min(1)
  def plane(width: Double, height: Double, v: View): Rect =
    Rect(width * (1 - v.zoom) / 2 + v.panX, height * (1 - v.zoom) / 2 + v.panY, width * v.zoom, height * v.zoom)
  def fit(width: Double, height: Double, r: Rect): Rect =
    require(width > 0 && height > 0 && r.width > 0 && r.height > 0)
    val scale = math.min(r.width / width, r.height / height)
    Rect(r.x + (r.width - width * scale) / 2, r.y + (r.height - height * scale) / 2, width * scale, height * scale)
  def revealAt(x: Double, y: Double, width: Double, height: Double, v: View): Double =
    val r = plane(width, height, v)
    clamp(if v.horizontal then (x - r.x) / r.width else (y - r.y) / r.height)
  def reveal(width: Double, height: Double, v: View): Rect =
    val r = plane(width, height, v)
    if v.horizontal then Rect(r.x + r.width * clamp(v.split), r.y, r.width * (1 - clamp(v.split)), r.height)
    else Rect(r.x, r.y + r.height * clamp(v.split), r.width, r.height * (1 - clamp(v.split)))

  /** Flat opaque dark background matches the native comparison surface; no source metadata is attributed. */
  def render(a: BufferedImage, b: BufferedImage, width: Int, height: Int, v: View): BufferedImage =
    require(width > 0 && height > 0)
    val output = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g = output.createGraphics()
    try
      g.setColor(new Color(23, 33, 43))
      g.fillRect(0, 0, width, height)
      g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
      def draw(im: BufferedImage): Unit =
        val r = fit(im.getWidth, im.getHeight, plane(width, height, v))
        g.drawImage(im, r.x.round.toInt, r.y.round.toInt, r.width.round.toInt, r.height.round.toInt, null)
      draw(a)
      val r = reveal(width, height, v)
      g.clip(new java.awt.geom.Rectangle2D.Double(r.x, r.y, r.width, r.height))
      g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, clamp(v.opacity).toFloat))
      draw(b)
    finally g.dispose()
    output

  /** Explicit export only; also rejects hardlink/symlink aliases to either original. */
  def exportPng(image: BufferedImage, destination: Path, originals: Seq[Path],
      secret: Option[ImageScrambleSecret] = None): Unit =
    val dst = destination.toAbsolutePath.normalize()
    require(!originals.exists { source =>
      source.toAbsolutePath.normalize() == dst ||
        (Files.exists(dst) && Files.exists(source) && Files.isSameFile(dst, source))
    }, "Choose a new file: comparison export must not overwrite either original")
    secret match
      case Some(value) => value.use(pass => ScrambledPng.write(image, dst, Nil, Some(pass)))
      case None => require(ImageIO.write(image, "png", dst.toFile), "PNG writer unavailable")

  case class Term(text: String, changed: Boolean)
  case class Parameter(key: String, a: Option[String], b: Option[String]):
    def changed: Boolean = a != b
  case class Difference(positiveA: Seq[Term], positiveB: Seq[Term], negativeA: Seq[Term], negativeB: Seq[Term], parameters: Seq[Parameter])
  def differences(a: Option[String], b: Option[String]): Difference =
    val pa = Infotext.parse(a.getOrElse(""))
    val pb = Infotext.parse(b.getOrElse(""))
    def tags(s: String): Seq[String] = s.split(",").toSeq.map(_.trim).filter(_.nonEmpty)
    def terms(x: String, other: String): Seq[Term] =
      val shared = tags(other).toSet
      tags(x).map(t => Term(t, !shared(t)))
    Difference(terms(pa.prompt, pb.prompt), terms(pb.prompt, pa.prompt), terms(pa.negativePrompt, pb.negativePrompt),
      terms(pb.negativePrompt, pa.negativePrompt), (pa.fields.keys.toSeq ++ pb.fields.keys.filterNot(pa.fields.contains)).map(k =>
        Parameter(k, pa.fields.get(k), pb.fields.get(k))))
