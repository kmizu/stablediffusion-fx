package sdfx.engine

import sdfx.core.model.GenerationParams

class EditorPreviewMetadataSuite extends munit.FunSuite:
  test("preview marker reflects this job rather than restored image metadata") {
    val restored = GenerationParams(extra = Map("Embedding editor preview" -> "Unsaved embedding edits", "other" -> "kept"))
    val ordinary = EditorPreviewMetadata.forJob(restored, active = false)
    assert(!ordinary.extra.contains("Embedding editor preview"))
    assertEquals(ordinary.extra("other"), "kept")
    val preview = EditorPreviewMetadata.forJob(ordinary, active = true)
    assertEquals(preview.extra("Embedding editor preview"), "Unsaved embedding edits")
  }
