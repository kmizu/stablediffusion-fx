package sdfx.core

import sdfx.core.png.{PngMetadata, ScrambleBatch, ScrambledPng}
import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO

class ScrambleBatchSuite extends munit.FunSuite with TempDir:
  tmp.test("batch scrambles lossless WebP with alpha and keeps the source unchanged") { root =>
    val input = Files.createDirectory(root.resolve("input")); val output = Files.createDirectory(root.resolve("output"))
    // A 2x2 lossless WebP made independently with Pillow, including partial alpha.
    val encoded = java.util.Base64.getDecoder.decode("UklGRi4AAABXRUJQVlA4TCEAAAAvAUAAEB8w/wKCIv9HExAU+T+agKDouuUCeGfCOkT0PwIA")
    val source = input.resolve("alpha.webp")
    Files.write(source, encoded)
    val result = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(result.map(_.status), Seq(ScrambleBatch.Status.Written))
    assert(result.head.message.contains("metadata may be lost"))
    assert(Files.readAllBytes(source).sameElements(encoded))
    val restored = ScrambledPng.read(output.resolve("alpha.png"), Some("secret")).image
    assertEquals(restored.getRGB(0, 0), 0xffff0000)
    assertEquals(restored.getRGB(1, 0), 0x8000ff00)
    assertEquals(restored.getRGB(0, 1), 0xff0000ff)
    assertEquals(restored.getRGB(1, 1) >>> 24, 0)
  }

  tmp.test("batch protects PNG and JPEG, never changes originals or overwrites destinations") { root =>
    val input = Files.createDirectory(root.resolve("input"))
    val output = Files.createDirectory(root.resolve("output"))
    val image = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 3; x <- 0 until 4 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    val a = input.resolve("a.png"); val b = input.resolve("b.jpg"); val c = input.resolve("c.png")
    PngMetadata.write(image, a, Seq("parameters" -> "cat"))
    ImageIO.write(image, "jpg", b.toFile)
    PngMetadata.write(image, c, Nil)
    val before = Seq(a, b, c).map(p => Files.readAllBytes(p).toSeq)
    Files.writeString(output.resolve("c.png"), "keep")
    val result = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(result.count(_.status == ScrambleBatch.Status.Written), 2)
    assert(result.find(_.source == b).exists(_.message.contains("metadata may be lost")))
    assertEquals(result.count(_.status == ScrambleBatch.Status.Exists), 1)
    assertEquals(Files.readString(output.resolve("c.png")), "keep")
    assertEquals(Seq(a, b, c).map(p => Files.readAllBytes(p).toSeq), before)
    assertEquals(PngMetadata.readTexts(output.resolve("a.png"))("parameters"), "cat")
    val recovered = ScrambledPng.read(output.resolve("a.png"), Some("secret")).image
    assert(image.getRGB(0, 0, 4, 3, null, 0, 4).sameElements(recovered.getRGB(0, 0, 4, 3, null, 0, 4)))
  }

  tmp.test("decode handles wrong passwords, unmarked files and unverified legacy explicitly") { root =>
    val input = Files.createDirectory(root.resolve("input")); val output = Files.createDirectory(root.resolve("output"))
    val img = new BufferedImage(4, 3, BufferedImage.TYPE_INT_RGB)
    ScrambledPng.write(img, input.resolve("verified.png"), Nil, Some("secret"))
    PngMetadata.write(sdfx.core.png.PixelScramble.encode(img, "secret"), input.resolve("legacy.png"), Seq("Encrypt" -> "pixel_shuffle_3"))
    PngMetadata.write(img, input.resolve("plain.png"), Nil)
    val wrong = ScrambleBatch.run(input, output, "wrong", ScrambleBatch.Mode.Decode)
    assertEquals(wrong.find(_.source.getFileName.toString == "verified.png").get.status, ScrambleBatch.Status.Failed)
    assertEquals(wrong.find(_.source.getFileName.toString == "legacy.png").get.status, ScrambleBatch.Status.Unverified)
    assertEquals(wrong.find(_.source.getFileName.toString == "plain.png").get.status, ScrambleBatch.Status.Skipped)
    assert(!Files.exists(output.resolve("verified.png")) && !Files.exists(output.resolve("legacy.png")))
    val correct = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Decode, allowUnverified = true)
    assertEquals(correct.count(_.status == ScrambleBatch.Status.Written), 2)
    assertEquals(PngMetadata.readTexts(output.resolve("verified.png")).get("Encrypt"), None)
  }

  tmp.test("batch rejects identical source and destination") { root =>
    intercept[IllegalArgumentException](ScrambleBatch.run(root, root, "secret", ScrambleBatch.Mode.Encode))
  }


  tmp.test("same-stem collisions and unreadable images report results without modifying sources") { root =>
    val input = Files.createDirectory(root.resolve("input")); val output = Files.createDirectory(root.resolve("output"))
    val image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB)
    val a = input.resolve("same.png"); val b = input.resolve("same.jpg"); val broken = input.resolve("broken.webp")
    PngMetadata.write(image, a, Nil); ImageIO.write(image, "jpg", b.toFile); Files.writeString(broken, "not an image")
    val before = Seq(a, b, broken).map(p => Files.readAllBytes(p).toSeq)
    val result = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(result.count(_.status == ScrambleBatch.Status.Written), 2)
    assertEquals(result.count(_.status == ScrambleBatch.Status.Exists), 0)
    assertEquals(result.count(_.status == ScrambleBatch.Status.Failed), 1)
    assertEquals(Seq(a, b, broken).map(p => Files.readAllBytes(p).toSeq), before)
    assert(!Files.exists(output.resolve("broken.png")))
    val names = scala.util.Using.resource(Files.list(output))(_.toArray.toSeq.map(_.asInstanceOf[java.nio.file.Path].getFileName.toString))
    assertEquals(names.toSet, Set("same.png", "same.png.png"))
    val restored = result.filter(_.status == ScrambleBatch.Status.Written).map(r =>
      r.source -> ScrambledPng.read(r.destination, Some("secret")).image).toMap
    for source <- Seq(a, b) do
      val expected = ImageIO.read(source.toFile)
      assertEquals(restored(source).getRGB(0, 0), expected.getRGB(0, 0))
    val rerun = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(rerun.count(_.status == ScrambleBatch.Status.Exists), 2)
  }


  tmp.test("cancellation stops between files and reports completed results") { root =>
    val input = Files.createDirectory(root.resolve("input")); val output = Files.createDirectory(root.resolve("output"))
    val image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB)
    for name <- Seq("a.png", "b.png", "c.png") do PngMetadata.write(image, input.resolve(name), Nil)
    var stop = false
    var seen = Vector.empty[ScrambleBatch.Result]
    val result = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Encode,
      shouldStop = () => stop, onResult = r => { seen :+= r; stop = true })
    assertEquals(result.size, 1)
    assertEquals(seen, result.toVector)
    assertEquals(result.head.status, ScrambleBatch.Status.Written)
    assert(Files.exists(output.resolve("a.png")) && !Files.exists(output.resolve("b.png")))
  }

  tmp.test("same-stem collision keeps existing destination and writes the other image") { root =>
    val input = Files.createDirectory(root.resolve("input")); val output = Files.createDirectory(root.resolve("output"))
    val image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB)
    ImageIO.write(image, "jpg", input.resolve("same.jpg").toFile)
    PngMetadata.write(image, input.resolve("same.png"), Nil)
    val existing = output.resolve("same.png")
    Files.writeString(existing, "keep original destination")
    val results = ScrambleBatch.run(input, output, "secret", ScrambleBatch.Mode.Encode)
    assertEquals(results.map(_.status), Seq(ScrambleBatch.Status.Exists, ScrambleBatch.Status.Written))
    assertEquals(results(1).destination.getFileName.toString, "same.png.png")
    assertEquals(Files.readString(existing), "keep original destination")
    assert(ScrambledPng.read(results(1).destination, Some("secret")).scrambled)
  }
