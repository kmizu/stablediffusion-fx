package sdfx.plugins

import sdfx.plugins.imageops.{ImageOps, Tile}

import java.awt.image.BufferedImage

class ImageOpsSuite extends munit.FunSuite:
  private def covered(w: Int, h: Int, tiles: Seq[Tile]): Boolean =
    (0 until h).forall(y => (0 until w).forall(x => tiles.exists(t => x >= t.x && x < t.x + t.w && y >= t.y && y < t.y + t.h)))

  test("split tiles cover the image, stay inside it and overlap by at least the requested amount") {
    val tiles = ImageOps.splitTiles(1024, 768, 512, 512, 64)
    assert(covered(1024, 768, tiles))
    assert(tiles.forall(t => t.x >= 0 && t.y >= 0 && t.x + t.w <= 1024 && t.y + t.h <= 768))
    val xs = tiles.map(_.x).distinct.sorted
    val ys = tiles.map(_.y).distinct.sorted
    assertEquals(xs.size, 3) // ceil((1024-64)/448) = 3
    assertEquals(ys.size, 2)
    xs.sliding(2).foreach { case Seq(a, b) => assert(a + 512 - b >= 64, s"$a $b") }
    assertEquals(xs.last + 512, 1024)
    assertEquals(ys.last + 512, 768)
  }

  test("image smaller than the tile gives a single clamped tile") {
    assertEquals(ImageOps.splitTiles(300, 200, 512, 512, 64), Seq(Tile(0, 0, 300, 200)))
  }

  test("merge of unmodified tiles reproduces the image exactly") {
    val img = new BufferedImage(700, 500, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 500; x <- 0 until 700 do img.setRGB(x, y, (x % 256) << 16 | (y % 256) << 8 | ((x + y) % 256))
    val tiles = ImageOps.splitTiles(700, 500, 256, 256, 48)
    val merged = ImageOps.mergeTiles(700, 500, tiles.map(t => t -> ImageOps.crop(img, t)), 48)
    assertEquals((merged.getWidth, merged.getHeight), (700, 500))
    for y <- 0 until 500 by 7; x <- 0 until 700 by 5 do assertEquals(merged.getRGB(x, y), img.getRGB(x, y), s"$x,$y")
  }

  test("feather blends seams between differently coloured tiles") {
    val tiles = ImageOps.splitTiles(200, 100, 120, 100, 40)
    assertEquals(tiles.size, 2)
    def solid(rgb: Int, t: Tile) =
      val i = new BufferedImage(t.w, t.h, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until t.h; x <- 0 until t.w do i.setRGB(x, y, rgb)
      i
    val merged = ImageOps.mergeTiles(200, 100, Seq(tiles(0) -> solid(0x000000, tiles(0)), tiles(1) -> solid(0xff0000, tiles(1))), 40)
    assertEquals(merged.getRGB(0, 50) & 0xffffff, 0x000000)
    assertEquals(merged.getRGB(199, 50) & 0xffffff, 0xff0000)
    val mid = (merged.getRGB(100, 50) >> 16) & 0xff
    assert(mid > 40 && mid < 215, s"mid red $mid")
  }

  test("outpaint expansion on all sides rounds to multiples of 8") {
    val e = ImageOps.expansion(500, 300, 64, true, true, true, true)
    assertEquals((e.left, e.top), (64, 64))
    assertEquals(e.canvasW % 8, 0)
    assertEquals(e.canvasH % 8, 0)
    assertEquals(e.canvasW, 500 + 64 + e.right)
    assert(e.right >= 64 && e.right < 72)
  }

  test("outpaint expansion only to the left keeps rounding on the left") {
    val e = ImageOps.expansion(500, 512, 64, true, false, false, false)
    assertEquals((e.right, e.top, e.bottom), (0, 0, 0))
    assertEquals(e.canvasW, 568)
    assertEquals(e.left, 68)
    assertEquals(e.canvasH, 512)
  }

  test("outpaint canvas places the original and replicates edges; mask marks only new area (inset by blur)") {
    val img = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 64; x <- 0 until 64 do img.setRGB(x, y, 0x00ff00)
    img.setRGB(0, 0, 0xff0000)
    val e = ImageOps.expansion(64, 64, 32, true, false, true, false)
    val canvas = ImageOps.expandCanvas(img, e)
    assertEquals((canvas.getWidth, canvas.getHeight), (96, 96))
    assertEquals(canvas.getRGB(32, 32) & 0xffffff, 0xff0000) // original top-left pixel at (left, top)
    assertEquals(canvas.getRGB(0, 0) & 0xffffff, 0xff0000) // replicated corner
    assertEquals(canvas.getRGB(95, 95) & 0xffffff, 0x00ff00)
    val mask = ImageOps.outpaintMask(e, 4)
    def m(x: Int, y: Int) = mask.getRaster.getSample(x, y, 0)
    assertEquals(m(0, 50), 255) // new left strip
    assertEquals(m(50, 0), 255) // new top strip
    assertEquals(m(34, 50), 255) // inside blur inset
    assertEquals(m(36, 50), 0) // kept
    assertEquals(m(95, 95), 0) // right/bottom not expanded -> kept to the edge
    assertEquals(e.keepRect(4), (36, 36, 96, 96))
  }

  test("margin pads every side") {
    val img = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB)
    val m = ImageOps.withMargin(img, 8)
    assertEquals((m.getWidth, m.getHeight), (18, 18))
  }
