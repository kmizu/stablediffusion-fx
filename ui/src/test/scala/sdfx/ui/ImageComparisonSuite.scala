package sdfx.ui

import java.nio.file.Paths
import sdfx.ui.vm.ComparisonHistory

class ImageComparisonSuite extends munit.FunSuite:
  test("previous result is from the prior job, while selections within the same job update current") {
    val h = new ComparisonHistory
    val a = Paths.get("first.png")
    val b = Paths.get("second.png")
    val c = Paths.get("second-batch2.png")
    assertEquals(h.pair, None)
    h.record("one", a)
    assertEquals(h.pair, None)
    h.record("two", b)
    h.record("two", c)
    assertEquals(h.pair, Some(a -> c))
    h.record("three", b)
    assertEquals(h.pair, Some(c -> b))
  }
  test("comparison settings default to source off and survive JSON settings roundtrip") {
    import sdfx.core.config.Settings
    import sdfx.core.json.SnakePickle
    val s = Settings()
    assertEquals(s.comparisonSendButtons, "Off")
    assertEquals(s.comparisonTxtButton, "Off")
    assert(!s.comparisonMouseSend && !s.comparisonMouseAuto)
    val chosen = s.copy(comparisonSendButtons = "Icon", comparisonTxtButton = "Text", comparisonMouseSend = true)
    assertEquals(SnakePickle.read[Settings](SnakePickle.write(chosen)), chosen)
  }

  test("img2img and Inpaint results remain scoped and do not advance previous txt2img history") {
    import sdfx.ui.vm.Mode
    val h = new ComparisonHistory
    val a = Paths.get("a.png")
    val b = Paths.get("b.png")
    h.recordResult("one", a, Mode.Txt2Img)
    h.recordResult("img", b, Mode.Img2Img)
    h.recordResult("paint", a, Mode.Inpaint)
    assertEquals(h.latest(Mode.Img2Img), Some(b))
    assertEquals(h.latest(Mode.Inpaint), Some(a))
    assertEquals(h.pair, None)
    h.recordResult("two", b, Mode.Txt2Img)
    assertEquals(h.pair, Some(a -> b))
  }
