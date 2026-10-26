package sdfx.core

import sdfx.core.image.GridNumbers
import java.awt.Color
import java.awt.image.BufferedImage

class GridNumbersSuite extends munit.FunSuite:
  private val enabled = Map("Grid image numbers" -> "True", "Grid XYZ info" -> "True")
  private def solid(w: Int = 512, h: Int = 512, alpha: Boolean = false): BufferedImage =
    val i = new BufferedImage(w, h, if alpha then BufferedImage.TYPE_INT_ARGB else BufferedImage.TYPE_INT_RGB)
    val g = i.createGraphics()
    try
      g.setColor(new Color(160, 120, 80))
      g.fillRect(0, 0, w, h)
    finally g.dispose()
    i
  private def pixels(i: BufferedImage) = i.getRGB(0, 0, i.getWidth, i.getHeight, null, 0, i.getWidth).toSeq

  test("saved filename prefix preserves leading zeros and rejects non-image/grid names") {
    assertEquals(GridNumbers.number("00042-123.png"), Some("00042"))
    assertEquals(GridNumbers.number("100000--21.png"), Some("100000"))
    Seq("grid-00042.png", "grid.png", "sample.png", "00042.png", "n42-seed.png").foreach(n =>
      assertEquals(GridNumbers.number(n), None))
  }

  test("disabled and both-options-off preserve every pixel and do not label") {
    val image = solid()
    Seq(Map.empty[String, String], Map("Grid image numbers" -> "False", "Grid XYZ info" -> "False")).foreach { options =>
      assertEquals(pixels(GridNumbers.render(image, "00042-1.png", options)), pixels(image))
      assertEquals(GridNumbers.labels(512, 512, "00042-1.png", options), Nil)
    }
  }

  test("number and axes occupy the four source corners with half-transparent background and white text") {
    val image = solid()
    val original = pixels(image)
    val axes = Map("X" -> "Steps: 12", "Y" -> "Clip skip: 2", "Z" -> "Checkpoint name: A")
    val labels = GridNumbers.labels(512, 512, "00042-1.png", enabled, axes)
    assertEquals(labels.map(_.corner).toSet, Set("bottom_left", "top_left", "top_right", "bottom_right"))
    assertEquals(labels.find(_.corner == "bottom_left").get.text, "00042")
    labels.foreach { l =>
      assertEquals(l.fontSize, 24)
      assert(l.box.x >= 10 && l.box.y >= 10 && l.box.x + l.box.width <= 502 && l.box.y + l.box.height <= 502)
    }
    val out = GridNumbers.render(image, "00042-1.png", enabled, axes)
    assertEquals(pixels(image), original)
    assertEquals((out.getWidth, out.getHeight), (512, 512))
    assertEquals(out.getRGB(256, 256), image.getRGB(256, 256))
    labels.foreach { l =>
      val rgb = new Color(out.getRGB(l.box.x + 1, l.box.y + 1))
      assert(math.abs(rgb.getRed - 80) <= 1 && math.abs(rgb.getGreen - 60) <= 1 && math.abs(rgb.getBlue - 40) <= 1)
      val area = for x <- l.box.x until l.box.x + l.box.width; y <- l.box.y until l.box.y + l.box.height yield out.getRGB(x, y) & 0xffffff
      assert(area.exists(_ == 0xffffff), s"missing white text: ${l.text}")
    }
  }

  test("long labels shrink then drop their key; unfit numbers and values are omitted without changing sources") {
    val options = Map("Grid XYZ info" -> "True")
    val shrunk = GridNumbers.labels(180, 128, "00001-1.png", options, Map("X" -> "Sampler: DPM++ 2M"))
    assert(shrunk.nonEmpty && shrunk.head.fontSize < 24)
    val fallback = GridNumbers.labels(100, 128, "00001-1.png", options, Map("X" -> ("very long label" * 30 + ": 7")))
    assertEquals(fallback.map(_.text), Seq("7"))
    assertEquals(GridNumbers.labels(16, 16, "00001-1.png", enabled, Map("X" -> "Steps: 3")), Nil)
    assertEquals(GridNumbers.labels(64, 64, "9" * 100 + "-1.png", Map("Grid image numbers" -> "True")), Nil)
  }

  test("XYZ-only supports absent axes, Unicode and non-numbered source filenames") {
    val src = solid(alpha = true)
    val options = Map("Grid XYZ info" -> "True")
    val labels = GridNumbers.labels(512, 512, "named-source.png", options, Map("X" -> "Prompt: 桜の木", "Y" -> "Nothing"))
    assertEquals(labels.map(_.corner), Seq("top_left"))
    assert(GridNumbers.render(src, "named-source.png", options, Map("X" -> "Prompt: 桜の木")) ne src)
    assertEquals(GridNumbers.labels(512, 512, "grid-1.png", enabled, Map("X" -> "Steps: 3")), Nil)
  }
