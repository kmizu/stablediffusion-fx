package sdfx.ui

import javafx.scene.input.DataFormat

class NativeAcceptanceSafetySuite extends munit.FunSuite:
  test("null and unreadable advertised clipboard formats reject the entire snapshot before any mutation") {
    val good = DataFormat.PLAIN_TEXT
    val bad = DataFormat.HTML
    assert(NativeAcceptanceSafety.capture(Seq(good, bad), f => if f == good then "keep" else null).isLeft)
    assert(NativeAcceptanceSafety.capture(Seq(good), _ => throw new IllegalStateException("unreadable")).isLeft)
    assert(NativeAcceptanceSafety.capture(Seq(good), _ => new Object()).isLeft)
    assertEquals(NativeAcceptanceSafety.capture(Seq(good), _ => "keep").toOption.get.get(good), "keep")
  }
  test("cleanup attempts every resource and preserves primary and secondary failures") {
    val primary = new IllegalArgumentException("test failed")
    val restore = new IllegalStateException("clipboard restore failed")
    var closed = false
    NativeAcceptanceSafety.cleanup(primary)(() => throw restore, () => { closed = true })
    assert(closed)
    assertEquals(primary.getSuppressed.toSeq, Seq(restore))
    closed = false
    val thrown = intercept[IllegalStateException] {
      NativeAcceptanceSafety.cleanup(null)(() => throw restore, () => { closed = true })
    }
    assert(closed)
    assert(thrown eq restore)
  }
