package sdfx.core

import sdfx.core.png.{PngMetadata, ScrambleBatch, ScrambledPng}
import java.awt.color.{ColorSpace, ICC_Profile}
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.{Files, Path}
import java.util.zip.DeflaterOutputStream
import javax.imageio.{IIOImage, ImageIO, ImageTypeSpecifier}
import javax.imageio.metadata.IIOMetadataNode
import scala.util.Using

class ScrambleMetadataSuite extends munit.FunSuite with TempDir:
  private def tagged(path: Path, gamma: String, exif: Array[Byte]): Unit =
    val image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB)
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    try
      val param = writer.getDefaultWriteParam
      val meta = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), param)
      val root = new IIOMetadataNode(PngMetadata.Format)
      val g = new IIOMetadataNode("gAMA"); g.setAttribute("value", gamma); root.appendChild(g)
      val s = new IIOMetadataNode("sRGB"); s.setAttribute("renderingIntent", "Perceptual"); root.appendChild(s)
      val c = new IIOMetadataNode("cHRM")
      Seq("whitePointX" -> "31270", "whitePointY" -> "32900", "redX" -> "64000", "redY" -> "33000",
        "greenX" -> "30000", "greenY" -> "60000", "blueX" -> "15000", "blueY" -> "6000")
        .foreach((key, value) => c.setAttribute(key, value))
      root.appendChild(c)
      val group = new IIOMetadataNode("UnknownChunks")
      val e = new IIOMetadataNode("UnknownChunk"); e.setAttribute("type", "eXIf"); e.setUserObject(exif)
      group.appendChild(e); root.appendChild(group)
      meta.mergeTree(PngMetadata.Format, root)
      Using.resource(ImageIO.createImageOutputStream(path.toFile)) { out =>
        writer.setOutput(out); writer.write(null, new IIOImage(image, null, meta), param)
      }
    finally writer.dispose()

  private def exif(path: Path): Seq[Byte] =
    Using.resource(ImageIO.createImageInputStream(path.toFile)) { in =>
      val reader = ImageIO.getImageReaders(in).next()
      try
        reader.setInput(in)
        val nodes = reader.getImageMetadata(0).getAsTree(PngMetadata.Format).getChildNodes
        val group = (0 until nodes.getLength).map(nodes.item).find(_.getNodeName == "UnknownChunks").get
        val chunks = group.getChildNodes
        val chunk = (0 until chunks.getLength).map(chunks.item).find(_.asInstanceOf[IIOMetadataNode].getAttribute("type") == "eXIf").get
        chunk.asInstanceOf[IIOMetadataNode].getUserObject.asInstanceOf[Array[Byte]].toSeq
      finally reader.dispose()
    }

  private def colorTags(path: Path): Seq[(String, Seq[(String, String)])] =
    Using.resource(ImageIO.createImageInputStream(path.toFile)) { in =>
      val reader = ImageIO.getImageReaders(in).next()
      try
        reader.setInput(in)
        val nodes = reader.getImageMetadata(0).getAsTree(PngMetadata.Format).getChildNodes
        (0 until nodes.getLength).map(nodes.item).filter(n => Set("sRGB", "gAMA", "cHRM")(n.getNodeName)).map { n =>
          val attrs = n.getAttributes
          n.getNodeName -> (0 until attrs.getLength).map { i =>
            val attr = attrs.item(i); attr.getNodeName -> attr.getNodeValue
          }.sortBy(_._1)
        }.sortBy(_._1)
      finally reader.dispose()
    }

  private def orientationExif: Array[Byte] =
    val b = java.nio.ByteBuffer.allocate(26).order(java.nio.ByteOrder.LITTLE_ENDIAN)
    b.put(0x49.toByte).put(0x49.toByte).putShort(42.toShort).putInt(8)
    b.putShort(1.toShort).putShort(274.toShort).putShort(3.toShort).putInt(1).putShort(6.toShort).putShort(0.toShort).putInt(0)
    b.array()
  private def density(path: Path): (String, String, String) =
    Using.resource(ImageIO.createImageInputStream(path.toFile)) { in =>
      val reader = ImageIO.getImageReaders(in).next()
      try
        reader.setInput(in)
        val nodes = reader.getImageMetadata(0).getAsTree(PngMetadata.Format).getChildNodes
        val node = (0 until nodes.getLength).map(nodes.item).find(_.getNodeName == "pHYs").get.asInstanceOf[IIOMetadataNode]
        (node.getAttribute("pixelsPerUnitXAxis"), node.getAttribute("pixelsPerUnitYAxis"), node.getAttribute("unitSpecifier"))
      finally reader.dispose()
    }
  private def profileChunk(path: Path): Array[Byte] =
    Using.resource(ImageIO.createImageInputStream(path.toFile)) { in =>
      val reader = ImageIO.getImageReaders(in).next()
      try
        reader.setInput(in)
        val root = reader.getImageMetadata(0).getAsTree(PngMetadata.Format)
        val nodes = root.getChildNodes
        (0 until nodes.getLength).map(nodes.item).find(_.getNodeName == "iCCP").get
          .asInstanceOf[IIOMetadataNode].getUserObject.asInstanceOf[Array[Byte]]
      finally reader.dispose()
    }

  private def writeProfiled(path: Path, profile: Array[Byte]): BufferedImage =
    val image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 2; x <- 0 until 3 do image.setRGB(x, y, (x * 50 << 16) | (y * 100 << 8) | 31)
    val compressed = new ByteArrayOutputStream()
    Using.resource(new DeflaterOutputStream(compressed))(_.write(profile))
    val writer = ImageIO.getImageWritersByFormatName("png").next()
    try
      val param = writer.getDefaultWriteParam
      val meta = writer.getDefaultImageMetadata(ImageTypeSpecifier.createFromRenderedImage(image), param)
      val root = new IIOMetadataNode(PngMetadata.Format)
      val chunk = new IIOMetadataNode("iCCP")
      chunk.setAttribute("profileName", "linear RGB")
      chunk.setAttribute("compressionMethod", "deflate")
      chunk.setUserObject(compressed.toByteArray)
      root.appendChild(chunk)
      val density = new IIOMetadataNode("pHYs")
      density.setAttribute("pixelsPerUnitXAxis", "3780")
      density.setAttribute("pixelsPerUnitYAxis", "7560")
      density.setAttribute("unitSpecifier", "meter")
      root.appendChild(density)
      meta.mergeTree(PngMetadata.Format, root)
      Using.resource(ImageIO.createImageOutputStream(path.toFile)) { out =>
        writer.setOutput(out); writer.write(null, new IIOImage(image, null, meta), param)
      }
    finally writer.dispose()
    image

  tmp.test("batch keeps non-sRGB ICC profile and raw samples through scramble and restore") { root =>
    val source = Files.createDirectory(root.resolve("source"))
    val encoded = Files.createDirectory(root.resolve("encoded"))
    val restored = Files.createDirectory(root.resolve("restored"))
    val file = source.resolve("profile.png")
    val image = writeProfiled(file, ICC_Profile.getInstance(ColorSpace.CS_LINEAR_RGB).getData)
    val profile = profileChunk(file).toSeq
    val before = Files.readAllBytes(file).toSeq
    assertEquals(ScrambleBatch.run(source, encoded, "secret", ScrambleBatch.Mode.Encode).head.status, ScrambleBatch.Status.Written)
    assertEquals(profileChunk(encoded.resolve("profile.png")).toSeq, profile)
    assertEquals(density(encoded.resolve("profile.png")), density(file))
    assertEquals(ScrambleBatch.run(encoded, restored, "secret", ScrambleBatch.Mode.Decode).head.status, ScrambleBatch.Status.Written)
    assertEquals(profileChunk(restored.resolve("profile.png")).toSeq, profile)
    assertEquals(density(restored.resolve("profile.png")), density(file))
    assertEquals(Files.readAllBytes(file).toSeq, before)
    val result = PngMetadata.readImage(restored.resolve("profile.png"))
    for y <- 0 until 2; x <- 0 until 3; band <- 0 until 3 do
      assertEquals(result.getRaster.getSample(x, y, band), image.getRaster.getSample(x, y, band))
  }

  tmp.test("batch refuses an oversized inflated ICC profile without publishing output") { root =>
    val source = Files.createDirectory(root.resolve("source")); val output = Files.createDirectory(root.resolve("output"))
    val file = source.resolve("large.png")
    writeProfiled(file, new Array[Byte](8 * 1024 * 1024 + 1))
    val before = Files.readAllBytes(file).toSeq
    val result = ScrambleBatch.run(source, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(result.head.status, ScrambleBatch.Status.Failed)
    assert(!Files.exists(output.resolve("large.png")))
    assertEquals(Files.readAllBytes(file).toSeq, before)
  }

  tmp.test("batch preserves PNG EXIF orientation and compatible sRGB gamma") { root =>
    val source = Files.createDirectory(root.resolve("source")); val encoded = Files.createDirectory(root.resolve("encoded"))
    val restored = Files.createDirectory(root.resolve("restored")); val file = source.resolve("tagged.png")
    tagged(file, "45455", orientationExif)
    assertEquals(ScrambleBatch.run(source, encoded, "secret", ScrambleBatch.Mode.Encode).head.status, ScrambleBatch.Status.Written)
    assertEquals(exif(encoded.resolve("tagged.png")), orientationExif.toSeq)
    assertEquals(colorTags(encoded.resolve("tagged.png")), colorTags(file))
    assertEquals(ScrambleBatch.run(encoded, restored, "secret", ScrambleBatch.Mode.Decode).head.status, ScrambleBatch.Status.Written)
    assertEquals(exif(restored.resolve("tagged.png")), orientationExif.toSeq)
    assertEquals(colorTags(restored.resolve("tagged.png")), colorTags(file))
  }

  tmp.test("batch refuses conflicting sRGB gamma and malformed EXIF without output") { root =>
    val source = Files.createDirectory(root.resolve("source")); val output = Files.createDirectory(root.resolve("output"))
    tagged(source.resolve("conflict.png"), "100000", orientationExif)
    tagged(source.resolve("bad-exif.png"), "45455", Array[Byte](1, 2, 3))
    val result = ScrambleBatch.run(source, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(result.map(_.status), Seq(ScrambleBatch.Status.Failed, ScrambleBatch.Status.Failed))
    assert(!Files.exists(output.resolve("conflict.png")) && !Files.exists(output.resolve("bad-exif.png")))
  }
