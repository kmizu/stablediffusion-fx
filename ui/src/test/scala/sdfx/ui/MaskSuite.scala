package sdfx.ui

import sdfx.ui.vm.{MaskGeometry, MaskModel}

import java.nio.file.Files
import javax.imageio.ImageIO

class MaskSuite extends munit.FunSuite:

  test("fit keeps the aspect ratio and centers") {
    val f = MaskGeometry.fit(1000, 500, 400, 400)
    assertEquals(f.scale, 0.4)
    assertEquals((f.width, f.height), (400.0, 200.0))
    assertEquals((f.offsetX, f.offsetY), (0.0, 100.0))
    assertEquals(f.toImage(200, 200), (500.0, 250.0))
    assertEquals(f.toView(500, 250), (200.0, 200.0))
    val tall = MaskGeometry.fit(512, 1024, 600, 300)
    assertEquals(tall.scale, 300.0 / 1024)
    assertEquals(tall.offsetY, 0.0)
  }

  test("brush radius and segment bounds") {
    assertEquals(MaskGeometry.brushRadius(40, 0.5), 40.0)
    assertEquals(MaskGeometry.segmentBounds(10, 10, 20, 15, 5, 100, 100), (4, 4, 22, 17))
    assertEquals(MaskGeometry.segmentBounds(-50, -50, -40, -40, 5, 100, 100)._3, 0)
  }

  test("paint, erase, invert, undo / redo") {
    val m = new MaskModel(64, 32)
    assert(m.isEmpty)
    m.checkpoint()
    m.stroke(10, 16, 30, 16, 4, erase = false)
    assert(!m.isEmpty)
    assertEquals(m.valueAt(20, 16), 255)
    assertEquals(m.valueAt(60, 2), 0)
    m.checkpoint()
    m.stroke(20, 16, 20, 16, 3, erase = true)
    assertEquals(m.valueAt(20, 16), 0)
    assertEquals(m.valueAt(12, 16), 255)
    assert(m.undo())
    assertEquals(m.valueAt(20, 16), 255)
    assert(m.redo())
    assertEquals(m.valueAt(20, 16), 0)
    m.checkpoint()
    m.invert()
    assertEquals(m.valueAt(60, 2), 255)
    m.checkpoint()
    m.clear()
    assert(m.isEmpty)
    assert(m.undo())
    assertEquals(m.valueAt(60, 2), 255)
    assert(m.undo())
    assert(m.undo())
    assert(m.undo())
    assert(m.isEmpty)
    assert(!m.undo())
  }

  test("exported mask PNG is white where painted, at image resolution") {
    val m = new MaskModel(80, 40)
    m.stroke(40, 20, 40, 20, 6, erase = false)
    val f = Files.createTempFile("mask-", ".png")
    m.writePng(f)
    val img = ImageIO.read(f.toFile)
    assertEquals((img.getWidth, img.getHeight), (80, 40))
    assertEquals(img.getRGB(40, 20) & 0xffffff, 0xffffff)
    assertEquals(img.getRGB(2, 2) & 0xffffff, 0)
    val again = new MaskModel(40, 20)
    again.loadFrom(img)
    assertEquals(again.valueAt(20, 10), 255)
    assertEquals(again.valueAt(1, 1), 0)
  }
