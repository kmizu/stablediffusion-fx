package sdfx.core.image

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.metadata.{IIOMetadata, IIOMetadataNode}
import javax.imageio.stream.FileImageOutputStream
import javax.imageio.{IIOImage, ImageIO, ImageTypeSpecifier}
import scala.util.Using

/** Animated GIF writer (javax.imageio) for frame sequences (seed travel, prompt travel, loopback videos...). */
object Gif:
  private val Format = "javax_imageio_gif_image_1.0"

  def write(frames: Seq[BufferedImage], dst: Path, frameMillis: Int, loop: Boolean = true): Path =
    require(frames.nonEmpty, "a GIF needs at least one frame")
    val writer = ImageIO.getImageWritersByFormatName("gif").next()
    Option(dst.toAbsolutePath.getParent).foreach(Files.createDirectories(_))
    try
      Using.resource(new FileImageOutputStream(dst.toFile)) { out =>
        writer.setOutput(out)
        writer.prepareWriteSequence(null)
        frames.zipWithIndex.foreach { (f0, i) =>
          val f = toRgb(f0)
          val meta = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(f), writer.getDefaultWriteParam)
          configure(meta, math.max(1, frameMillis / 10), loop && i == 0)
          writer.writeToSequence(new IIOImage(f, null, meta), writer.getDefaultWriteParam)
        }
        writer.endWriteSequence()
      }
      dst
    finally writer.dispose()

  private def toRgb(img: BufferedImage): BufferedImage =
    if img.getType == BufferedImage.TYPE_INT_RGB then img
    else
      val out = new BufferedImage(img.getWidth, img.getHeight, BufferedImage.TYPE_INT_RGB)
      val g = out.createGraphics()
      g.drawImage(img, 0, 0, null)
      g.dispose()
      out

  private def configure(meta: IIOMetadata, delayCs: Int, loop: Boolean): Unit =
    val root = meta.getAsTree(Format).asInstanceOf[IIOMetadataNode]
    val gce = child(root, "GraphicControlExtension")
    gce.setAttribute("disposalMethod", "none")
    gce.setAttribute("userInputFlag", "FALSE")
    gce.setAttribute("transparentColorFlag", "FALSE")
    gce.setAttribute("delayTime", delayCs.toString)
    gce.setAttribute("transparentColorIndex", "0")
    if loop then
      val apps = child(root, "ApplicationExtensions")
      val app = new IIOMetadataNode("ApplicationExtension")
      app.setAttribute("applicationID", "NETSCAPE")
      app.setAttribute("authenticationCode", "2.0")
      app.setUserObject(Array[Byte](1, 0, 0)) // loop forever
      apps.appendChild(app)
    meta.setFromTree(Format, root)

  private def child(root: IIOMetadataNode, name: String): IIOMetadataNode =
    val list = root.getElementsByTagName(name)
    if list.getLength > 0 then list.item(0).asInstanceOf[IIOMetadataNode]
    else
      val n = new IIOMetadataNode(name)
      root.appendChild(n)
      n

  /** Number of frames in a GIF (for tests). */
  def frameCount(path: Path): Int =
    Using.resource(ImageIO.createImageInputStream(path.toFile)) { in =>
      val r = ImageIO.getImageReadersByFormatName("gif").next()
      try
        r.setInput(in)
        r.getNumImages(true)
      finally r.dispose()
    }
