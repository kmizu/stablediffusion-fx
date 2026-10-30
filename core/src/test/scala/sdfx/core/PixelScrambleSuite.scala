package sdfx.core

import sdfx.core.png.PixelScramble
import java.awt.image.BufferedImage
import java.nio.file.{Files, Paths}

class PixelScrambleSuite extends munit.FunSuite with TempDir:
  private def decode(hex: String): Array[Byte] = hex.grouped(2).map(Integer.parseInt(_, 16).toByte).toArray
  private def image(hex: String, w: Int, h: Int, mode: String): BufferedImage =
    val data = decode(hex)
    val channels = if mode == "RGBA" then 4 else 3
    val result = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
    for y <- 0 until h; x <- 0 until w do
      val i = (y * w + x) * channels
      def c(n: Int) = data(i + n) & 255
      val a = if channels == 4 then c(3) else 255
      result.setRGB(x, y, (a << 24) | (c(0) << 16) | (c(1) << 8) | c(2))
    result
  private def pixels(img: BufferedImage): Seq[Int] =
    (for y <- 0 until img.getHeight; x <- 0 until img.getWidth yield img.getRGB(x, y)).toSeq

  test("all three upstream formats decode and v3 encodes exactly") {
    val cases = ujson.read(Files.readString(Paths.get("python/tests/fixtures/encrypt_image_reference.json")))("cases").arr
    cases.foreach { c =>
      val w = c("width").num.toInt
      val h = c("height").num.toInt
      val mode = c("mode").str
      val pass = c("password").str
      val version = c("version").num.toInt
      val plain = image(c("plain").str, w, h, mode)
      val encoded = image(c("scrambled").str, w, h, mode)
      assertEquals(pixels(PixelScramble.decode(encoded, pass, version)), pixels(plain), s"v$version $w x $h")
      if version == 3 then assertEquals(pixels(PixelScramble.encode(plain, pass)), pixels(encoded))
    }
  }

  test("password marker and malformed arguments") {
    assertEquals(PixelScramble.passwordMarker("秘密🔑"), "b672ea9c4f5a23ada34b74af0e246186156dd61775fbddffa2497ba69605e44c")
    assertEquals(PixelScramble.verifyPassword("x", None), None)
    assertEquals(PixelScramble.verifyPassword("x", Some("💥")), Some(false))
    intercept[IllegalArgumentException](PixelScramble.encode(new BufferedImage(1, 1, 2), ""))
    intercept[IllegalArgumentException](PixelScramble.decode(new BufferedImage(1, 1, 2), "x", 4))
  }

  test("scrambling preserves 16-bit RGBA samples and the source color model") {
    val space = java.awt.color.ColorSpace.getInstance(java.awt.color.ColorSpace.CS_sRGB)
    val model = new java.awt.image.ComponentColorModel(space, Array(16, 16, 16, 16), true, false,
      java.awt.Transparency.TRANSLUCENT, java.awt.image.DataBuffer.TYPE_USHORT)
    val raster = java.awt.image.Raster.createInterleavedRaster(java.awt.image.DataBuffer.TYPE_USHORT, 3, 2, 4, null)
    val original = new BufferedImage(model, raster, false, null)
    for y <- 0 until 2; x <- 0 until 3; channel <- 0 until 4 do
      raster.setSample(x, y, channel, 1000 + y * 9000 + x * 2000 + channel * 71)
    val scrambled = PixelScramble.encode(original, "secret")
    val restored = PixelScramble.decode(scrambled, "secret", 3)
    assertEquals(scrambled.getColorModel, original.getColorModel)
    assertEquals(restored.getColorModel, original.getColorModel)
    for y <- 0 until 2; x <- 0 until 3; channel <- 0 until 4 do
      assertEquals(restored.getRaster.getSample(x, y, channel), raster.getSample(x, y, channel))
  }

  test("scrambling rejects palette and grayscale instead of silently converting samples") {
    intercept[IllegalArgumentException](PixelScramble.encode(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_INDEXED), "secret"))
    intercept[IllegalArgumentException](PixelScramble.encode(new BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY), "secret"))
  }

  tmp.test("protected PNG retains 16-bit RGBA samples across disk round trip") { dir =>
    val space = java.awt.color.ColorSpace.getInstance(java.awt.color.ColorSpace.CS_sRGB)
    val model = new java.awt.image.ComponentColorModel(space, Array(16, 16, 16, 16), true, false,
      java.awt.Transparency.TRANSLUCENT, java.awt.image.DataBuffer.TYPE_USHORT)
    val raster = java.awt.image.Raster.createInterleavedRaster(java.awt.image.DataBuffer.TYPE_USHORT, 3, 2, 4, null)
    val source = new BufferedImage(model, raster, false, null)
    for y <- 0 until 2; x <- 0 until 3; band <- 0 until 4 do
      raster.setSample(x, y, band, 1000 + y * 9000 + x * 2000 + band * 71)
    val file = dir.resolve("16-bit.png")
    sdfx.core.png.ScrambledPng.write(source, file, Nil, Some("secret"))
    val restored = sdfx.core.png.ScrambledPng.read(file, Some("secret")).image
    assertEquals(restored.getColorModel.getComponentSize.toSeq, Seq(16, 16, 16, 16))
    for y <- 0 until 2; x <- 0 until 3; band <- 0 until 4 do
      assertEquals(restored.getRaster.getSample(x, y, band), raster.getSample(x, y, band))
  }


  tmp.test("protected PNG remains scrambled on disk and reloads with correct password") { dir =>
    val path = dir.resolve("protected.png")
    val original = image("ff000000ff000000ffffffff", 2, 2, "RGB")
    sdfx.core.png.ScrambledPng.write(original, path, Seq("parameters" -> "猫"), Some("secret"))
    val texts = sdfx.core.png.PngMetadata.readTexts(path)
    assertEquals(texts("Encrypt"), "pixel_shuffle_3")
    assertEquals(texts("parameters"), "猫")
    assertNotEquals(pixels(sdfx.core.png.PngMetadata.readImage(path)), pixels(original))
    val loaded = sdfx.core.png.ScrambledPng.read(path, Some("secret"))
    assertEquals(pixels(loaded.image), pixels(original))
    assertEquals(loaded.passwordVerified, Some(true))
    intercept[IllegalArgumentException](sdfx.core.png.ScrambledPng.read(path, Some("wrong")))
    intercept[IllegalArgumentException](sdfx.core.png.ScrambledPng.read(path, None))
  }

  tmp.test("legacy without verifier reports unknown and unmarked files remain readable") { dir =>
    val path = dir.resolve("plain.png")
    val original = image("ff000000ff000000ffffffff", 2, 2, "RGB")
    sdfx.core.png.ScrambledPng.write(original, path, Nil, None)
    assertEquals(pixels(sdfx.core.png.ScrambledPng.read(path, None).image), pixels(original))
    val encoded = PixelScramble.encode(original, "secret")
    sdfx.core.png.PngMetadata.write(encoded, path, Seq("Encrypt" -> "pixel_shuffle_3"))
    val decoded = sdfx.core.png.ScrambledPng.read(path, Some("secret"))
    assertEquals(decoded.passwordVerified, None)
    assertEquals(pixels(decoded.image), pixels(original))
  }


  test("session snapshots stay stable across changes without exposing passwords in logs") {
    val session = new sdfx.core.png.ImageScrambleSession
    assertEquals(session.snapshot, None)
    session.unlock("first secret")
    val queued = session.snapshot.get
    session.unlock("second secret")
    assertEquals(queued.use(identity), "first secret")
    assertEquals(session.snapshot.get.use(identity), "second secret")
    assert(!queued.toString.contains("first secret"))
    session.lock()
    assertEquals(session.snapshot, None)
    assertEquals(queued.use(identity), "first secret")
    intercept[IllegalArgumentException](session.unlock(""))
  }


  test("session revision and secret are captured atomically and change on unlock or lock") {
    val session = new sdfx.core.png.ImageScrambleSession
    val off = session.state
    session.unlock("first")
    val first = session.state
    assert(first.revision > off.revision)
    assert(first.secret.exists(_.use(identity) == "first"))
    session.unlock("second")
    val second = session.state
    assert(second.revision > first.revision)
    assert(first.secret.exists(_.use(identity) == "first"))
    session.lock()
    val locked = session.state
    assert(locked.revision > second.revision && locked.secret.isEmpty)
  }

  test("upstream v1 decodes grayscale and indexed samples without changing their color models") {
    // Generated by the pinned upstream encrypt_image using sha256("secret").
    val encoded = decode("090a0003050604070102080b")
    val palette = Array.tabulate[Byte](256)(_.toByte)
    val alpha = Array.tabulate[Byte](256)(i => (255 - i).toByte)
    val indexed = new java.awt.image.IndexColorModel(8, 256, palette, palette.reverse, palette, alpha)
    val images = Seq(new BufferedImage(4, 3, BufferedImage.TYPE_BYTE_GRAY),
      new BufferedImage(4, 3, BufferedImage.TYPE_BYTE_INDEXED, indexed))
    images.foreach { img =>
      for y <- 0 until 3; x <- 0 until 4 do img.getRaster.setSample(x, y, 0, encoded(y * 4 + x) & 255)
      val restored = PixelScramble.decode(img, "secret", 1)
      assertEquals(restored.getColorModel, img.getColorModel)
      for y <- 0 until 3; x <- 0 until 4 do
        assertEquals(restored.getRaster.getSample(x, y, 0), y * 4 + x)
      for version <- Seq(2, 3) do
        intercept[IllegalArgumentException](PixelScramble.decode(img, "secret", version))
    }
  }

  tmp.test("Python and JVM exchange 16-bit RGB and RGBA PNGs without sample loss") { dir =>
    import scala.sys.process.*
    val python = Paths.get("python", ".venv-test", "Scripts", "python.exe").toAbsolutePath.toString
    val helper = Paths.get("python", "tests", "cross_runtime_png_helper.py").toAbsolutePath.toString
    def bridge(action: String, source: java.nio.file.Path, target: java.nio.file.Path): Unit =
      assertEquals(Process(Seq(python, helper, action, source.toString, target.toString)).!, 0)
    for alpha <- Seq(false, true) do
      val channels = if alpha then 4 else 3
      val model = new java.awt.image.ComponentColorModel(
        java.awt.color.ColorSpace.getInstance(java.awt.color.ColorSpace.CS_sRGB),
        Array.fill(channels)(16), alpha, false,
        if alpha then java.awt.Transparency.TRANSLUCENT else java.awt.Transparency.OPAQUE,
        java.awt.image.DataBuffer.TYPE_USHORT)
      val raster = java.awt.image.Raster.createInterleavedRaster(java.awt.image.DataBuffer.TYPE_USHORT, 4, 3, channels, null)
      val image = new BufferedImage(model, raster, false, null)
      for y <- 0 until 3; x <- 0 until 4; c <- 0 until channels do
        raster.setSample(x, y, c, 101 + y * 9001 + x * 2003 + c * 137)
      val prefix = if alpha then "rgba" else "rgb"
      val plain = dir.resolve(s"$prefix-plain.png")
      val pythonEncoded = dir.resolve(s"$prefix-python-encoded.png")
      val jvmEncoded = dir.resolve(s"$prefix-jvm-encoded.png")
      val pythonDecoded = dir.resolve(s"$prefix-python-decoded.png")
      sdfx.core.png.PngMetadata.write(image, plain, Seq("parameters" -> "cross-runtime 猫"))
      bridge("encode", plain, pythonEncoded)
      val decoded = sdfx.core.png.ScrambledPng.read(pythonEncoded, Some("cross-runtime-secret"))
      assertEquals(decoded.passwordVerified, Some(true))
      assertEquals(sdfx.core.png.PngMetadata.readTexts(pythonEncoded).get("parameters"), Some("cross-runtime 猫"))
      sdfx.core.png.ScrambledPng.write(image, jvmEncoded, Seq("parameters" -> "cross-runtime 猫"), Some("cross-runtime-secret"))
      bridge("decode", jvmEncoded, pythonDecoded)
      assertEquals(sdfx.core.png.PngMetadata.readTexts(pythonDecoded).get("parameters"), Some("cross-runtime 猫"))
      assert(!sdfx.core.png.PngMetadata.readTexts(pythonDecoded).contains("Encrypt"))
      val fromPython = sdfx.core.png.PngMetadata.readImage(pythonDecoded)
      for result <- Seq(decoded.image, fromPython); y <- 0 until 3; x <- 0 until 4; c <- 0 until channels do
        assertEquals(result.getRaster.getSample(x, y, c), raster.getSample(x, y, c), s"$prefix ($x,$y,$c)")
  }

  tmp.test("upstream v1 16-bit gray PNG restores raw samples and text") { dir =>
    val order = Seq(9, 10, 0, 3, 5, 6, 4, 7, 1, 2, 8, 11)
    val image = new BufferedImage(4, 3, BufferedImage.TYPE_USHORT_GRAY)
    for y <- 0 until 3; x <- 0 until 4 do image.getRaster.setSample(x, y, 0, order(y * 4 + x) * 5000)
    val path = dir.resolve("legacy16.png")
    sdfx.core.png.PngMetadata.write(image, path, Seq("Encrypt" -> "pixel_shuffle", "parameters" -> "gray16"))
    val decoded = sdfx.core.png.ScrambledPng.read(path, Some("secret"))
    assertEquals(decoded.passwordVerified, None)
    for y <- 0 until 3; x <- 0 until 4 do
      assertEquals(decoded.image.getRaster.getSample(x, y, 0), (y * 4 + x) * 5000)
    val restored = dir.resolve("restored.png")
    sdfx.core.png.ScrambledPng.write(decoded.image, restored, Seq("parameters" -> "gray16"), None, Some(path))
    assertEquals(sdfx.core.png.PngMetadata.readTexts(restored).get("parameters"), Some("gray16"))
    val loaded = sdfx.core.png.PngMetadata.readImage(restored)
    for y <- 0 until 3; x <- 0 until 4 do
      assertEquals(loaded.getRaster.getSample(x, y, 0), (y * 4 + x) * 5000)
  }

  test("upstream LA fixtures preserve grayscale and alpha for all three versions") {
    val fixtures = Seq(
      "ab66be5500ff39cc5faa72994cbb858813ee26dd9877d144",
      "13ee26dd00ff39cc5faa72994cbb8588ab66be559877d144",
      "be559877d144ab6626dd00ff39cc13ee72994cbb85885faa")
    val model = new java.awt.image.ComponentColorModel(
      java.awt.color.ColorSpace.getInstance(java.awt.color.ColorSpace.CS_GRAY),
      Array(8, 8), true, false, java.awt.Transparency.TRANSLUCENT, java.awt.image.DataBuffer.TYPE_BYTE)
    fixtures.zipWithIndex.foreach { (hex, index) =>
      val raster = java.awt.image.Raster.createInterleavedRaster(java.awt.image.DataBuffer.TYPE_BYTE, 4, 3, 2, null)
      val image = new BufferedImage(model, raster, false, null)
      val values = decode(hex)
      for y <- 0 until 3; x <- 0 until 4; band <- 0 until 2 do
        raster.setSample(x, y, band, values((y * 4 + x) * 2 + band) & 255)
      val plain = PixelScramble.decode(image, "secret", index + 1)
      for i <- 0 until 12 do
        assertEquals(plain.getRaster.getSample(i % 4, i / 4, 0), i * 19)
        assertEquals(plain.getRaster.getSample(i % 4, i / 4, 1), 255 - i * 17)
      if index == 2 then
        val encoded = PixelScramble.encode(plain, "secret")
        for y <- 0 until 3; x <- 0 until 4; band <- 0 until 2 do
          assertEquals(encoded.getRaster.getSample(x, y, band), raster.getSample(x, y, band))
    }
  }
