package sdfx.ui.vm

import java.awt.image.{BufferedImage, DataBufferByte}
import java.awt.{BasicStroke, Color, RenderingHints}
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO

/** Mapping between a displayed (fit-to-box) image and its pixel grid. */
object MaskGeometry:
  final case class Fit(scale: Double, offsetX: Double, offsetY: Double, width: Double, height: Double):
    def toImage(x: Double, y: Double): (Double, Double) = ((x - offsetX) / scale, (y - offsetY) / scale)
    def toView(x: Double, y: Double): (Double, Double) = (x * scale + offsetX, y * scale + offsetY)

  /** Scale an `imgW x imgH` image to fit `boxW x boxH` keeping its aspect ratio, centered. */
  def fit(imgW: Double, imgH: Double, boxW: Double, boxH: Double): Fit =
    if imgW <= 0 || imgH <= 0 || boxW <= 0 || boxH <= 0 then Fit(1, 0, 0, imgW.max(0), imgH.max(0))
    else
      val s = math.min(boxW / imgW, boxH / imgH)
      val w = imgW * s
      val h = imgH * s
      Fit(s, (boxW - w) / 2, (boxH - h) / 2, w, h)

  /** Brush radius in image pixels for a brush size given in screen pixels. */
  def brushRadius(screenSize: Double, scale: Double): Double = if scale <= 0 then screenSize / 2 else screenSize / 2 / scale

  /** Bounding box (x, y, w, h) of a stroke segment clipped to the image. */
  def segmentBounds(x0: Double, y0: Double, x1: Double, y1: Double, r: Double, w: Int, h: Int): (Int, Int, Int, Int) =
    val minX = math.max(0, math.floor(math.min(x0, x1) - r - 1).toInt)
    val minY = math.max(0, math.floor(math.min(y0, y1) - r - 1).toInt)
    val maxX = math.min(w, math.ceil(math.max(x0, x1) + r + 1).toInt)
    val maxY = math.min(h, math.ceil(math.max(y0, y1) + r + 1).toInt)
    (minX, minY, math.max(0, maxX - minX), math.max(0, maxY - minY))

/** Inpaint mask at image resolution (white = inpaint) with undo/redo. Pure Java2D: works headless. */
final class MaskModel(val width: Int, val height: Int, maxHistory: Int = 40):
  require(width > 0 && height > 0, "mask size must be positive")
  private var img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
  private var undoStack = List.empty[Array[Byte]]
  private var redoStack = List.empty[Array[Byte]]

  def image: BufferedImage = img
  private def data: Array[Byte] = img.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData

  private def snapshotBytes(): Array[Byte] = data.clone()
  private def restore(bytes: Array[Byte]): Unit = System.arraycopy(bytes, 0, data, 0, bytes.length)

  /** Records the current state for undo; call once before a stroke or an operation. */
  def checkpoint(): Unit =
    undoStack = (snapshotBytes() :: undoStack).take(maxHistory)
    redoStack = Nil

  def canUndo: Boolean = undoStack.nonEmpty
  def canRedo: Boolean = redoStack.nonEmpty

  def undo(): Boolean = undoStack match
    case h :: t =>
      redoStack = snapshotBytes() :: redoStack
      restore(h)
      undoStack = t
      true
    case Nil => false

  def redo(): Boolean = redoStack match
    case h :: t =>
      undoStack = snapshotBytes() :: undoStack
      restore(h)
      redoStack = t
      true
    case Nil => false

  /** Paints (or erases) a round-capped segment in image coordinates. */
  def stroke(x0: Double, y0: Double, x1: Double, y1: Double, radius: Double, erase: Boolean): Unit =
    val g = img.createGraphics()
    try
      g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
      g.setColor(if erase then Color.BLACK else Color.WHITE)
      g.setStroke(new BasicStroke((radius * 2).toFloat.max(1f), BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND))
      if x0 == x1 && y0 == y1 then
        val r = radius.max(0.5)
        g.fill(new java.awt.geom.Ellipse2D.Double(x0 - r, y0 - r, r * 2, r * 2))
      else g.draw(new java.awt.geom.Line2D.Double(x0, y0, x1, y1))
    finally g.dispose()

  def invert(): Unit =
    val d = data
    var i = 0
    while i < d.length do
      d(i) = (~d(i)).toByte
      i += 1

  def clear(): Unit = java.util.Arrays.fill(data, 0.toByte)

  def isEmpty: Boolean = !data.exists(_ != 0)

  /** Mask value at a pixel (0 or 255). */
  def valueAt(x: Int, y: Int): Int = data(y * width + x) & 0xff

  /** Writes the mask as an 8-bit grayscale PNG. */
  def writePng(file: Path): Path = MaskModel.writePng(img, file)

  /** Independent copy of the current mask (to encode it on another thread while painting continues). */
  def copyImage(): BufferedImage =
    val c = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    c.getRaster.setDataElements(0, 0, img.getRaster)
    c

  /** Loads an existing mask image (any size: scaled to this mask), > 127 counts as masked. */
  def loadFrom(src: BufferedImage): Unit =
    val tmp = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    val g = tmp.createGraphics()
    try g.drawImage(src, 0, 0, width, height, null)
    finally g.dispose()
    val sd = tmp.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val d = data
    var i = 0
    while i < d.length do
      d(i) = if (sd(i) & 0xff) > 127 then 0xff.toByte else 0
      i += 1

object MaskModel:
  def writePng(img: BufferedImage, file: Path): Path =
    Option(file.toAbsolutePath.getParent).foreach(Files.createDirectories(_))
    if !ImageIO.write(img, "png", file.toFile) then throw new IllegalStateException("no PNG writer")
    file
