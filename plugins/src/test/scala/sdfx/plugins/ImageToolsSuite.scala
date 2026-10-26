package sdfx.plugins

import sdfx.core.plugins.api.Values
import java.nio.file.{Files, Path}
import scala.collection.mutable.ArrayBuffer

class ImageToolsSuite extends munit.FunSuite:
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
