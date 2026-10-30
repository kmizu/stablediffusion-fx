package sdfx.core

import sdfx.core.comparison.ImageComparison.*
import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO

class ImageComparisonSuite extends munit.FunSuite:
  private def solid(w: Int, h: Int, argb: Int): BufferedImage =
    val im = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
    for x <- 0 until w; y <- 0 until h do im.setRGB(x, y, argb)
    im

  test("independent contain-fit preserves asymmetric source ratios and transformed reveal") {
    val view = View(zoom = 2, panX = 10, panY = -5, split = .25)
    assertEquals(plane(100, 80, view), Rect(-40, -45, 200, 160))
    assertEquals(fit(200, 100, plane(100, 80, view)), Rect(-40, -15, 200, 100))
    assertEquals(revealAt(10, 0, 100, 80, view), .25)
    assertEquals(revealAt(0, -5, 100, 80, view.copy(horizontal = false)), .25)
  }
  test("wipe direction, opacity and source-over alpha are pixel-correct without mutating inputs") {
    val a = solid(8, 8, 0xffff0000)
    val b = solid(8, 8, 0xff0000ff)
    val before = b.getRGB(0, 0)
    val lr = render(a, b, 8, 8, View())
    assertEquals(lr.getRGB(1, 4), 0xffff0000)
    assertEquals(lr.getRGB(6, 4), 0xff0000ff)
    val tb = render(a, b, 8, 8, View(horizontal = false))
    assertEquals(tb.getRGB(6, 1), 0xffff0000)
    assertEquals(tb.getRGB(1, 6), 0xff0000ff)
    val mix = render(a, b, 8, 8, View(split = 0, opacity = .5)).getRGB(4, 4)
    assert(math.abs(((mix >> 16) & 255) - 127) <= 1)
    assert(math.abs((mix & 255) - 128) <= 1)
    val translucent = solid(8, 8, 0x800000ff)
    val alpha = render(a, translucent, 8, 8, View(split = 0, opacity = .5)).getRGB(4, 4)
    assert(math.abs((alpha & 255) - 64) <= 1)
    assert(math.abs(((alpha >> 16) & 255) - 191) <= 1)
    assertEquals(b.getRGB(0, 0), before)
    assertEquals(render(a, b, 8, 8, View(opacity = 0)).getRGB(6, 4), 0xffff0000)
  }
  test("view commands clamp and reset transform independently of reveal and opacity") {
    val v = View(split = .2, opacity = .3).zoomBy(99).panBy(100, -100)
    assertEquals(v.zoom, 10d)
    assertEquals(v.zoomBy(-99).zoom, .5)
    assertEquals(v.reset, View(split = .2, opacity = .3))
  }
  test("metadata diffs preserve prompt order and mark missing or changed parameter values") {
    val a = "cat, red, cat <lora:evil:1>\nNegative prompt: blur, dark\nSteps: 10, Seed: 1, Size: 8x8"
    val b = "cat, blue\nNegative prompt: blur\nSteps: 10, Seed: 2, Size: 8x8"
    val diff = differences(Some(a), Some(b))
    assertEquals(diff.positiveA.map(_.text), Seq("cat", "red", "cat <lora:evil:1>"))
    assertEquals(diff.positiveA.map(_.changed), Seq(false, true, true))
    assert(diff.parameters.find(_.key == "Seed").exists(_.changed))
    assert(!diff.parameters.find(_.key == "Steps").get.changed)
    assert(differences(None, Some(b)).positiveB.forall(_.changed))
  }
  test("export preserves original bytes and refuses normalized source or hardlink aliases") {
    val dir = Files.createTempDirectory("image-compare-")
    val source = dir.resolve("a.png")
    val im = solid(8, 8, 0xff00ff00)
    ImageIO.write(im, "png", source.toFile)
    val bytes = Files.readAllBytes(source)
    val out = dir.resolve("comparison.png")
    try
      exportPng(im, out, Seq(source))
      assertEquals(ImageIO.read(out.toFile).getRGB(3, 3), 0xff00ff00)
      intercept[IllegalArgumentException](exportPng(im, dir.resolve(".").resolve("a.png"), Seq(source)))
      val alias = dir.resolve("alias.png")
      Files.createLink(alias, source)
      intercept[IllegalArgumentException](exportPng(im, alias, Seq(source)))
      assert(java.util.Arrays.equals(bytes, Files.readAllBytes(source)))
      Files.delete(alias)
    finally
      Files.deleteIfExists(out)
      Files.deleteIfExists(source)
      Files.deleteIfExists(dir)
  }

  test("protected comparison export uses the captured session secret and preserves rendered pixels") {
    import sdfx.core.png.{ImageScrambleSession, ScrambledPng, PngMetadata}
    val dir = Files.createTempDirectory("comparison-protected-")
    val out = dir.resolve("comparison.png")
    val session = new ImageScrambleSession
    session.unlock("first password")
    val captured = session.snapshot
    session.unlock("second password")
    val image = new BufferedImage(24, 24, BufferedImage.TYPE_INT_RGB)
    for x <- 0 until 24; y <- 0 until 24 do image.setRGB(x, y, 0xff000000 | (x << 16) | (y << 8) | (x * y))
    try
      exportPng(image, out, Nil, captured)
      assertEquals(PngMetadata.readTexts(out).get("Encrypt"), Some("pixel_shuffle_3"))
      intercept[IllegalArgumentException](ScrambledPng.read(out, Some("second password")))
      val decoded = ScrambledPng.read(out, Some("first password")).image
      assert((for x <- 0 until 24; y <- 0 until 24 yield decoded.getRGB(x, y) == image.getRGB(x, y)).forall(identity))
      val encoded = ImageIO.read(out.toFile)
      assert((for x <- 0 until 24; y <- 0 until 24 yield encoded.getRGB(x, y) != image.getRGB(x, y)).exists(identity))
      val entries = Files.list(dir)
      try assertEquals(entries.count(), 1L) finally entries.close()
    finally
      Files.deleteIfExists(out)
      Files.deleteIfExists(dir)
  }

  test("new image pair resets reveal and opacity while preserving the shared view transform") {
    val v = View(horizontal = false, split = .17, opacity = .3, zoom = 3, panX = 42, panY = -71)
    assertEquals(v.forNewPair, View(horizontal = false, split = .5, opacity = 1, zoom = 3, panX = 42, panY = -71))
    assertEquals(v.forNewPair.reset, View(horizontal = false))
  }
