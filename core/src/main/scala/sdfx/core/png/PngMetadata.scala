package sdfx.core.png

import org.w3c.dom.{Element, Node}

import java.awt.image.BufferedImage
import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, StandardCopyOption}
import javax.imageio.metadata.{IIOMetadata, IIOMetadataNode}
import javax.imageio.stream.FileImageOutputStream
import javax.imageio.{IIOImage, ImageIO, ImageTypeSpecifier}
import scala.util.Using

/** Reads and writes PNG text chunks. Compatible with PIL / A1111: Latin-1 text goes to tEXt, anything else to iTXt. */
object PngMetadata:
  val Format = "javax_imageio_png_1.0"
  val ParametersKey = "parameters"

  /** All text entries (tEXt, iTXt, zTXt) keyed by keyword. */
  def readTexts(path: Path): Map[String, String] =
    val in = ImageIO.createImageInputStream(path.toFile)
    if in == null then Map.empty
    else
      Using.resource(in) { in =>
        val readers = ImageIO.getImageReaders(in)
        if !readers.hasNext then Map.empty
        else
          val r = readers.next()
          try
            r.setInput(in, true, false)
            val meta = r.getImageMetadata(0)
            if meta == null || !meta.getMetadataFormatNames.contains(Format) then Map.empty
            else collectTexts(meta.getAsTree(Format))
          finally r.dispose()
    }

  def readParameters(path: Path): Option[String] =
    try readTexts(path).get(ParametersKey).filter(_.nonEmpty)
    catch case _: Exception => None

  private def children(n: Node): Iterator[Node] =
    val list = n.getChildNodes
    Iterator.range(0, list.getLength).map(list.item)

  private def collectTexts(root: Node): Map[String, String] =
    val b = Map.newBuilder[String, String]
    children(root).foreach { chunk =>
      chunk.getNodeName match
        case "tEXt" =>
          children(chunk).foreach { e =>
            val el = e.asInstanceOf[Element]
            b += el.getAttribute("keyword") -> el.getAttribute("value")
          }
        case "iTXt" =>
          children(chunk).foreach { e =>
            val el = e.asInstanceOf[Element]
            b += el.getAttribute("keyword") -> el.getAttribute("text")
          }
        case "zTXt" =>
          children(chunk).foreach { e =>
            val el = e.asInstanceOf[Element]
            b += el.getAttribute("keyword") -> el.getAttribute("text")
          }
        case _ => ()
    }
    b.result()

  private def isLatin1(s: String): Boolean = StandardCharsets.ISO_8859_1.newEncoder().canEncode(s)

  /** Writes `image` as PNG to `dst` with the given text chunks. Written atomically via a temp file. */
  def write(image: BufferedImage, dst: Path, texts: Seq[(String, String)]): Unit =
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    try
      val params = writer.getDefaultWriteParam
      val meta: IIOMetadata = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), params)
      if texts.nonEmpty then
        val root = new IIOMetadataNode(Format)
        val (latin, unicode) = texts.partition((_, v) => isLatin1(v))
        if latin.nonEmpty then
          val t = new IIOMetadataNode("tEXt")
          latin.foreach { (k, v) =>
            val e = new IIOMetadataNode("tEXtEntry")
            e.setAttribute("keyword", k)
            e.setAttribute("value", v)
            t.appendChild(e)
          }
          root.appendChild(t)
        if unicode.nonEmpty then
          val t = new IIOMetadataNode("iTXt")
          unicode.foreach { (k, v) =>
            val e = new IIOMetadataNode("iTXtEntry")
            e.setAttribute("keyword", k)
            e.setAttribute("compressionFlag", "FALSE")
            e.setAttribute("compressionMethod", "0")
            e.setAttribute("languageTag", "")
            e.setAttribute("translatedKeyword", "")
            e.setAttribute("text", v)
            t.appendChild(e)
          }
          root.appendChild(t)
        meta.mergeTree(Format, root)
      Option(dst.toAbsolutePath.getParent).foreach(Files.createDirectories(_))
      val tmp = Files.createTempFile(dst.toAbsolutePath.getParent, ".sdfx-", ".png.part")
      try
        Using.resource(new FileImageOutputStream(tmp.toFile)) { out =>
          writer.setOutput(out)
          writer.write(null, new IIOImage(image, null, meta), params)
        }
        Files.move(tmp, dst, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
      finally Files.deleteIfExists(tmp)
    finally writer.dispose()

  /** Re-encodes `src` (any ImageIO readable image) to `dst` with the A1111 `parameters` chunk. */
  def copyWithParameters(src: Path, dst: Path, parameters: String, extra: Seq[(String, String)] = Nil): Unit =
    val img = ImageIO.read(src.toFile)
    if img == null then throw new IllegalArgumentException(s"not an image: $src")
    write(img, dst, (ParametersKey -> parameters) +: extra)

  def readImage(path: Path): BufferedImage =
    val img = ImageIO.read(new File(path.toString))
    if img == null then throw new IllegalArgumentException(s"not an image: $path")
    img
