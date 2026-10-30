package sdfx.engine

import java.nio.file.Files

class InpaintDifferencePreviewSuite extends munit.FunSuite:
  test("accepts and cleans only the engine-owned preview files") {
    val root = Files.createTempDirectory("difference-preview-test")
    val dir = Files.createDirectory(root.resolve("inpaint-difference-test"))
    val mask = Files.write(dir.resolve("mask.png"), Array[Byte](1))
    val preview = Files.write(dir.resolve("preview.png"), Array[Byte](2))
    val artifact = InpaintDifferencePreview.accept(root, ujson.Obj("artifact_dir" -> dir.toString,
      "mask_path" -> mask.toString, "preview_path" -> preview.toString, "width" -> 8, "height" -> 8))
    artifact.cleanup()
    assert(!Files.exists(dir) && Files.exists(root))
    Files.delete(root)
  }

  test("rejects traversal, foreign files and invalid dimensions without deleting anything") {
    val root = Files.createTempDirectory("difference-preview-test")
    val foreign = Files.createTempFile("difference-foreign", ".png")
    val data = ujson.Obj("artifact_dir" -> root.toString, "mask_path" -> foreign.toString,
      "preview_path" -> foreign.toString, "width" -> 0, "height" -> 8)
    intercept[IllegalArgumentException](InpaintDifferencePreview.accept(root, data))
    assert(Files.exists(foreign))
    Files.delete(foreign); Files.delete(root)
  }
