package sdfx.plugins

import sdfx.core.plugins.api.Values
import java.nio.file.{Files, Path}
import scala.collection.mutable.ArrayBuffer

class ImageToolsSuite extends munit.FunSuite:
  test("color correction cleans fallback reference after engine failure"):
    var reference: Option[Path] = None
    val ctx = new FakeContext():
      override def engineCall(kind: String, params: ujson.Obj): ujson.Value =
        reference = Some(java.nio.file.Paths.get(params("reference_path").str))
        assert(Files.exists(reference.get))
        throw new IllegalArgumentException("invalid reference")
    val plugin = new ColorCorrectionPlugin
    intercept[IllegalArgumentException] {
      plugin.transform(new java.awt.image.BufferedImage(12, 13, 1),
        sdfx.core.model.GenerationParams(), Values.defaults(plugin.controls), ctx)
    }
    assert(reference.nonEmpty)
    assert(!Files.exists(reference.get))

  test("color correction fallback and comparison files are cleaned"):
    val paths = ArrayBuffer[Path]()
    val ctx = new FakeContext():
      override def engineCall(kind: String, params: ujson.Obj): ujson.Value =
        assertEquals(params("tool").str, "color_correction")
        val reference = java.nio.file.Paths.get(params("reference_path").str)
        assert(Files.exists(reference))
        val main = java.nio.file.Paths.get(initImage(12, 13))
        val comparison = java.nio.file.Paths.get(initImage(12, 13))
        paths ++= Seq(reference, main, comparison)
        ujson.Obj("path" -> main.toString, "outputs" -> ujson.Obj("Wavelet" -> comparison.toString))
    val plugin = new ColorCorrectionPlugin
    val result = plugin.transform(new java.awt.image.BufferedImage(12, 13, 1),
      sdfx.core.model.GenerationParams(), Values.defaults(plugin.controls), ctx)
    assertEquals(result.size, 2)
    assert(paths.forall(p => !Files.exists(p)))

  test("background mask construction deletes all engine side outputs"):
    val outputs = ArrayBuffer[Path]()
    val ctx = new FakeContext():
      override def engineCall(kind: String, params: ujson.Obj): ujson.Value =
        assertEquals(kind, "image_tool")
        assertEquals(params("tool").str, "person_mask")
        assert(params("invert").bool)
        val main = java.nio.file.Paths.get(initImage(16, 16))
        val mask = java.nio.file.Paths.get(initImage(16, 16))
        outputs ++= Seq(main, mask)
        ujson.Obj("path" -> main.toString, "outputs" -> ujson.Obj("mask" -> mask.toString))
    val plugin = new InpaintBackgroundPlugin
    val result = plugin.makeMask(Support.readImage(ctx.initImage(16, 16)), Values.defaults(plugin.controls), ctx)
    assertEquals(result.getWidth, 16)
    assert(outputs.forall(p => !Files.exists(p)), s"leaked output files: $outputs")
