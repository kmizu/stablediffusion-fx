package sdfx.ui.vm

class EmbeddingEditorDraftSuite extends munit.FunSuite:
  test("switching vector and encoder keeps independent unsaved edits") {
    val draft = EmbeddingEditorDraft(Map("clip_l" -> Vector(Vector(1.0, 2.0), Vector(3.0, 4.0)),
      "clip_g" -> Vector(Vector(5.0, 6.0, 7.0), Vector(8.0, 9.0, 10.0))))
    draft.setCoordinate("clip_l", 0, 1, -2.5)
    draft.setCoordinate("clip_g", 1, 2, 42.0)
    assertEquals(draft.vector("clip_l", 0), Vector(1.0, -2.5))
    assertEquals(draft.vector("clip_g", 1), Vector(8.0, 9.0, 42.0))
    assert(draft.dirty)
    draft.resetVector("clip_l", 0)
    assertEquals(draft.vector("clip_l", 0), Vector(1.0, 2.0))
    assert(draft.dirty)
    draft.resetAll()
    assert(!draft.dirty)
  }

  test("invalid edits do not modify the draft") {
    val draft = EmbeddingEditorDraft(Map("clip_l" -> Vector(Vector(1.0, 2.0))))
    intercept[IllegalArgumentException](draft.setCoordinate("clip_l", 0, 1, Double.NaN))
    intercept[IllegalArgumentException](draft.setCoordinate("clip_l", 0, 2, 3.0))
    assertEquals(draft.vector("clip_l", 0), Vector(1.0, 2.0))
    assert(!draft.dirty)
  }
