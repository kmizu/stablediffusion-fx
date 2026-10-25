package sdfx.plugins

import UltimateSdUpscalePlugin.*

class UltimateSdUpscaleSuite extends munit.FunSuite:
  test("linear order covers the image row by row, edge tiles are clipped"):
    val t = redrawOrder(1100, 600, 512, 512, chess = false)
    assertEquals(t.map(r => (r.x, r.y)), Seq((0, 0), (512, 0), (1024, 0), (0, 512), (512, 512), (1024, 512)))
    assertEquals(t(2).w, 76)
    assertEquals(t(3).h, 88)
    assertEquals(t.map(r => r.w.toLong * r.h).sum, 1100L * 600)

  test("chess order draws one colour of the board first"):
    val t = redrawOrder(1024, 1024, 512, 512, chess = true)
    assertEquals(t.map(r => (r.x, r.y)), Seq((0, 0), (512, 512), (512, 0), (0, 512)))

  test("seam passes: band pass per inner seam, half-tile offset with intersections"):
    assertEquals(seamPasses(1024, 1024, 512, 512, "Band pass", 64).size, 2)
    assertEquals(seamPasses(1536, 1024, 512, 512, "Half tile offset pass", 64).size, 2 * 2 + 1 * 3)
    assertEquals(seamPasses(1536, 1024, 512, 512, "Half tile offset pass + intersections", 64).size, 4 + 3 + 2)
    assertEquals(seamPasses(1024, 1024, 512, 512, "None", 64), Nil)
    val band = seamPasses(1024, 512, 512, 512, "Band pass", 64).head(1024, 512)
    assertEquals(band.getRaster.getSample(512, 100, 0), 255)
    assertEquals(band.getRaster.getSample(400, 100, 0), 0)

  test("gradient mask is white at the seam and dark at the tile edges"):
    val m = gradientMask(1024, 512, Rect(256, 0, 512, 512), vertical = false)
    assert(m.getRaster.getSample(512, 10, 0) > 240)
    assert(m.getRaster.getSample(258, 10, 0) < 10)
    assert(m.getRaster.getSample(766, 10, 0) < 10)
    assertEquals(m.getRaster.getSample(100, 10, 0), 0)
