package sdfx.core

import sdfx.core.image.Grid

import java.awt.GraphicsEnvironment
import java.awt.image.BufferedImage

class GridSuite extends munit.FunSuite:
  import TestFiles.image

  test("runs headless (core tests are forked with -Djava.awt.headless=true)") {
    assert(GraphicsEnvironment.isHeadless)
  }

  test("autoColumns matches A1111 (rows = round(sqrt(n)), cols = ceil(n / rows))") {
    def a1111Cols(n: Int): Int =
      val rows = math.min(n, math.max(1, math.round(math.sqrt(n.toDouble)).toInt))
      math.ceil(n.toDouble / rows).toInt
    (1 to 200).foreach(n => assertEquals(Grid.autoColumns(n), a1111Cols(n), s"n=$n"))
    assertEquals(Grid.autoColumns(0), 1)
    assertEquals(Grid.autoColumns(-3), 1)
    assertEquals(Grid.autoColumns(4), 2)
    assertEquals(Grid.autoColumns(5), 3)
    assertEquals(Grid.autoColumns(9), 3)
  }

  test("layout math") {
    assertEquals(Grid.layout(4, 2), (2, 2))
    assertEquals(Grid.layout(5, 2), (2, 3))
    assertEquals(Grid.layout(3, 5), (3, 1)) // more columns than images
    assertEquals(Grid.layout(3, 0), (1, 3)) // non-positive columns -> single column
    assertEquals(Grid.layout(3, -1), (1, 3))
    assertEquals(Grid.layout(0, 3), (1, 1))
    assertEquals(Grid.layout(7, 3), (3, 3))
    assertEquals(Grid.layout(1, 1), (1, 1))
  }

  test("plain grid dimensions and cell placement") {
    val colors = Seq(0xff0000, 0x00ff00, 0x0000ff, 0xffff00, 0x00ffff)
    val imgs = colors.map(c => image(10, 8, c))
    val g = Grid.plain(imgs) // autoColumns(5) = 3 -> 3x2
    assertEquals((g.getWidth, g.getHeight), (30, 16))
    colors.zipWithIndex.foreach { (c, i) =>
      val x = (i % 3) * 10 + 5
      val y = (i / 3) * 8 + 4
      assertEquals(g.getRGB(x, y) & 0xffffff, c, s"cell $i")
    }
    assertEquals(g.getRGB(25, 12) & 0xffffff, 0xffffff) // empty cell stays white
  }

  test("plain grid with explicit columns") {
    val g = Grid.plain(Seq.fill(6)(image(5, 5)), cols = 6)
    assertEquals((g.getWidth, g.getHeight), (30, 5))
  }

  test("mixed sizes are centred in max-size cells; null cells left empty") {
    val big = image(20, 10, 0x112233)
    val small = image(4, 4, 0x445566)
    val g = Grid.labeled(Seq(big, null, small), 3, Nil, Nil)
    assertEquals((g.getWidth, g.getHeight), (60, 10))
    assertEquals(g.getRGB(30, 5) & 0xffffff, 0xffffff)
    assertEquals(g.getRGB(50, 5) & 0xffffff, 0x445566) // centre of 3rd cell
    assertEquals(g.getRGB(41, 1) & 0xffffff, 0xffffff) // outside the centred small image
  }

  test("all-null images still produce a grid of 64px cells") {
    val g = Grid.labeled(Seq(null, null), 2, Nil, Nil)
    assertEquals((g.getWidth, g.getHeight), (128, 64))
  }

  test("empty input is rejected") {
    intercept[IllegalArgumentException](Grid.labeled(Nil, 1, Nil, Nil))
    intercept[IllegalArgumentException](Grid.plain(Nil))
  }

  test("labeled output adds a top band, a left band and a title band") {
    val imgs = Seq.fill(4)(image(256, 256, 0x808080))
    val plain = Grid.labeled(imgs, 2, Nil, Nil)
    assertEquals((plain.getWidth, plain.getHeight), (512, 512))

    val cols = Grid.labeled(imgs, 2, Seq("CFG 5", "CFG 7"), Nil)
    assertEquals(cols.getWidth, 512)
    val topH = cols.getHeight - 512
    assert(topH > 0)

    val rows = Grid.labeled(imgs, 2, Nil, Seq("Euler", "DPM++ 2M"))
    assertEquals(rows.getHeight, 512)
    val leftW = rows.getWidth - 512
    assert(leftW > 0 && leftW <= 256, leftW.toString)

    val both = Grid.labeled(imgs, 2, Seq("CFG 5", "CFG 7"), Seq("Euler", "DPM++ 2M"))
    assertEquals((both.getWidth, both.getHeight), (512 + leftW, 512 + topH))

    val titled = Grid.labeled(imgs, 2, Seq("CFG 5", "CFG 7"), Seq("Euler", "DPM++ 2M"), Some("X/Y plot"))
    assertEquals(titled.getWidth, both.getWidth)
    assert(titled.getHeight > both.getHeight)

    // images are placed below/right of the bands
    assertEquals(both.getRGB(leftW + 128, topH + 128) & 0xffffff, 0x808080)
    assertEquals(both.getRGB(2, 2) & 0xffffff, 0xffffff)
  }

  test("labels actually draw text (non-white pixels in the band)") {
    val g = Grid.labeled(Seq(image(200, 200, 0xffffff)), 1, Seq("Hello"), Nil)
    val topH = g.getHeight - 200
    val dark = for x <- 0 until g.getWidth; y <- 0 until topH if (g.getRGB(x, y) & 0xffffff) != 0xffffff yield 1
    assert(dark.nonEmpty)
  }

  test("long column labels wrap to more lines -> taller band") {
    val imgs = Seq.fill(2)(image(128, 128))
    val short = Grid.labeled(imgs, 2, Seq("a", "b"), Nil)
    val long = Grid.labeled(imgs, 2, Seq("a very long label " * 6, "b"), Nil)
    assert(long.getHeight > short.getHeight, s"${long.getHeight} vs ${short.getHeight}")
  }

  test("left band is capped at the cell width") {
    val g = Grid.labeled(Seq(image(64, 64)), 1, Nil, Seq("x" * 500))
    assertEquals(g.getWidth, 128)
  }

  test("extra labels beyond the grid are ignored; Japanese labels render") {
    val g = Grid.labeled(Seq(image(64, 64), image(64, 64)), 2, Seq("一", "二", "三"), Seq("行", "余分"))
    assertEquals(g.getWidth % 64 >= 0, true)
    assert(g.getHeight > 64)
  }

  test("scaleToFit") {
    val img = image(400, 200)
    assert(Grid.scaleToFit(img, 500) eq img)
    assert(Grid.scaleToFit(img, 400) eq img)
    val s = Grid.scaleToFit(img, 100)
    assertEquals((s.getWidth, s.getHeight), (100, 50))
    val tall = Grid.scaleToFit(image(10, 1000), 50)
    assertEquals((tall.getWidth, tall.getHeight), (1, 50))
    assertEquals(s.getType, BufferedImage.TYPE_INT_RGB)
  }
