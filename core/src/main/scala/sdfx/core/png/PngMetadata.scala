package sdfx.core.png

import org.w3c.dom.{Element, Node}

import java.awt.color.{ColorSpace, ICC_Profile}
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, StandardCopyOption}
import javax.imageio.metadata.{IIOMetadata, IIOMetadataNode}
import javax.imageio.stream.FileImageOutputStream
import javax.imageio.{IIOImage, ImageIO, ImageTypeSpecifier}
import java.util.zip.InflaterInputStream
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

  /** PNG color, EXIF and pHYs, for paths whose raw RGB samples are retained by the caller. */
  private def preservedChunks(path: Path): Seq[Node] =
    val in = ImageIO.createImageInputStream(path.toFile)
    if in == null then Seq.empty
    else Using.resource(in) { stream =>
      val readers = ImageIO.getImageReaders(stream)
      if !readers.hasNext then Seq.empty
      else
        val reader = readers.next()
        try
          reader.setInput(stream, true, false)
          if !reader.getFormatName.equalsIgnoreCase("PNG") then Seq.empty
          else
            val root = reader.getImageMetadata(0).getAsTree(Format)
            val all = children(root).toVector
            val iccs = all.filter(_.getNodeName == "iCCP")
            val density = all.filter(_.getNodeName == "pHYs")
            val srgb = all.filter(_.getNodeName == "sRGB")
            val gamma = all.filter(_.getNodeName == "gAMA")
            val chroma = all.filter(_.getNodeName == "cHRM")
            require(Seq(iccs, density, srgb, gamma, chroma).forall(_.size <= 1), "duplicate PNG color or density metadata")
            require(iccs.isEmpty || (srgb.isEmpty && gamma.isEmpty && chroma.isEmpty), "conflicting or unverifiable PNG color profiles")
            if srgb.nonEmpty then
              gamma.foreach(n => require(n.asInstanceOf[Element].getAttribute("value") == "45455", "conflicting PNG sRGB gamma"))
              val expected = Seq("whitePointX" -> "31270", "whitePointY" -> "32900", "redX" -> "64000", "redY" -> "33000",
                "greenX" -> "30000", "greenY" -> "60000", "blueX" -> "15000", "blueY" -> "6000")
              chroma.foreach { n =>
                val element = n.asInstanceOf[Element]
                require(expected.forall((key, value) => element.getAttribute(key) == value), "conflicting PNG sRGB chromaticities")
              }
            iccs.foreach { n =>
              val bytes = n.asInstanceOf[IIOMetadataNode].getUserObject.asInstanceOf[Array[Byte]]
              require(bytes != null && bytes.length <= 4 * 1024 * 1024, "PNG ICC profile is too large")
              val decoded = new ByteArrayOutputStream()
              Using.resource(new InflaterInputStream(new ByteArrayInputStream(bytes))) { inflated =>
                val buffer = new Array[Byte](8192)
                var count = inflated.read(buffer)
                while count != -1 do
                  require(decoded.size().toLong + count <= 8L * 1024 * 1024, "PNG ICC profile is too large")
                  decoded.write(buffer, 0, count)
                  count = inflated.read(buffer)
              }
              require(ICC_Profile.getInstance(decoded.toByteArray).getColorSpaceType == ColorSpace.TYPE_RGB,
                "PNG ICC profile is not RGB")
            }
            val unknown = all.filter(_.getNodeName == "UnknownChunks").flatMap(children)
            val exif = unknown.filter(n => n.asInstanceOf[Element].getAttribute("type") == "eXIf")
            require(exif.size <= 1, "duplicate PNG EXIF metadata")
            exif.foreach { n =>
              val bytes = n.asInstanceOf[IIOMetadataNode].getUserObject.asInstanceOf[Array[Byte]]
              require(bytes != null && bytes.length >= 8 && bytes.length <= 4 * 1024 * 1024, "invalid or oversized PNG EXIF")
              val little = bytes(0) == 0x49 && bytes(1) == 0x49
              val big = bytes(0) == 0x4d && bytes(1) == 0x4d
              require(little || big, "invalid PNG EXIF byte order")
              val buf = java.nio.ByteBuffer.wrap(bytes).order(if little then java.nio.ByteOrder.LITTLE_ENDIAN else java.nio.ByteOrder.BIG_ENDIAN)
              require((buf.getShort(2) & 0xffff) == 42, "invalid PNG EXIF TIFF header")
              val offset = buf.getInt(4).toLong & 0xffffffffL
              require(offset >= 8 && offset + 2 <= bytes.length, "invalid PNG EXIF IFD offset")
              val entries = buf.getShort(offset.toInt) & 0xffff
              require(offset + 2L + entries * 12L + 4L <= bytes.length, "invalid PNG EXIF IFD length")
            }
            val exifGroup = if exif.isEmpty then Seq.empty else
              val group = new IIOMetadataNode("UnknownChunks")
              exif.foreach(group.appendChild)
              Seq(group)
            iccs ++ srgb ++ gamma ++ chroma ++ density ++ exifGroup
        finally reader.dispose()
    }

  /** Writes `image` as PNG to `dst` with the given text chunks. Written atomically via a temp file. */
  def write(image: BufferedImage, dst: Path, texts: Seq[(String, String)], preserveFrom: Option[Path] = None): Unit =
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    try
      val params = writer.getDefaultWriteParam
      val meta: IIOMetadata = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), params)
      preserveFrom.foreach { source =>
        val chunks = preservedChunks(source)
        if chunks.nonEmpty then
          val root = new IIOMetadataNode(Format)
          chunks.foreach(root.appendChild)
          meta.mergeTree(Format, root)
      }
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
