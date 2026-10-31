package sdfx.core

import sdfx.core.modelpresets.*
import java.nio.file.Files
import java.util.concurrent.CancellationException
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import java.io.ByteArrayOutputStream

class ModelPresetServiceSuite extends munit.FunSuite with TempDir:
  tmp.test("identity is full content hash and cancellation does not create a library") { root =>
    val model = root.resolve("model.safetensors")
    Files.write(model, Array[Byte](1, 2, 3))
    val service = new ModelPresetService(root.resolve("library"))
    val identity = service.identify(model, () => ())
    assertEquals(identity.sha256, "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81")
    assertEquals(service.load(identity).document, ModelPresetDocument.empty)
    intercept[CancellationException](service.identify(model, () => throw new CancellationException()))
    assert(!Files.exists(root.resolve("library")))
  }
  tmp.test("thumbnail is bounded, preserves downloaded existing image, and manual replacement is explicit") { root =>
    val service = new ModelPresetService(root.resolve("library"))
    val model = root.resolve("model.safetensors")
    Files.write(model, Array[Byte](1))
    val identity = service.identify(model, () => ())
    def png(color: Int): Array[Byte] =
      val image = new BufferedImage(600, 400, BufferedImage.TYPE_INT_RGB)
      val g = image.createGraphics(); g.setColor(new java.awt.Color(color)); g.fillRect(0, 0, 600, 400); g.dispose()
      val out = new ByteArrayOutputStream(); ImageIO.write(image, "png", out); out.toByteArray
    val path = service.saveThumbnail(identity, png(0xff0000), replace = false, () => ())
    val red = Files.readAllBytes(path)
    service.saveThumbnail(identity, png(0x00ff00), replace = false, () => ())
    assert(java.util.Arrays.equals(red, Files.readAllBytes(path)))
    val image = ImageIO.read(path.toFile)
    assertEquals((image.getWidth, image.getHeight), (300, 200))
    service.saveThumbnail(identity, png(0x00ff00), replace = true, () => ())
    assert(!java.util.Arrays.equals(red, Files.readAllBytes(path)))
    intercept[CancellationException](service.saveThumbnail(identity, png(0), replace = true, () => throw new CancellationException()))
  }
