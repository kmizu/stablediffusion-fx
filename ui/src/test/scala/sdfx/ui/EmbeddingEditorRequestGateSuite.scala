package sdfx.ui.vm

class EmbeddingEditorRequestGateSuite extends munit.FunSuite:
  test("only the owning reply unlocks controls and can replace a draft") {
    var busy = false
    var draft = "unsaved"
    val gate = new EmbeddingEditorRequestGate(value => busy = value)
    val first = gate.begin().get
    assert(busy)
    assertEquals(gate.begin(), None)
    assert(!gate.settle(first + 1) { draft = "stale" })
    assertEquals(draft, "unsaved"); assert(busy)
    assert(gate.settle(first) { () }) // failed save preserves existing draft
    assertEquals(draft, "unsaved"); assert(!busy)
    val second = gate.begin().get
    assert(!gate.settle(first) { draft = "late" })
    assertEquals(draft, "unsaved"); assert(busy)
    assert(gate.settle(second) { draft = "saved" })
    assertEquals(draft, "saved"); assert(!busy)
  }

  test("reply exception always unlocks controls") {
    var busy = false
    val gate = new EmbeddingEditorRequestGate(value => busy = value)
    val token = gate.begin().get
    intercept[IllegalStateException](gate.settle(token) { throw new IllegalStateException("failure") })
    assert(!busy)
    assert(gate.begin().nonEmpty)
  }
