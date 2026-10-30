package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import java.nio.file.{Files, Path, Paths}

class Waifu2xSuite extends munit.FunSuite:
  test("dedicated native control routes source style noise scale and deletes temporary files"):
    var source: Option[Path] = None
    var output: Option[Path] = None
    val ctx = new FakeContext():
      override def engineCall(kind: String, params: ujson.Obj): ujson.Value =
        assertEquals(kind, "image_tool")
        assertEquals(params("tool").str, "waifu2x")
        assertEquals(params("style").str, "Photo")
        assertEquals(params("noise").num.toInt, 3)
        assertEquals(params("scale").num.toInt, 4)
        source = Some(Paths.get(params("image_path").str))
        assert(Files.exists(source.get))
        output = Some(Paths.get(initImage(12, 20)))
        ujson.Obj("path" -> output.get.toString, "outputs" -> ujson.Obj())
    val plugin = new Waifu2xPlugin
    val values = Values.defaults(plugin.controls).updated("style", "Photo").updated("noise", "High").updated("scale", "4x")
    val result = plugin.transform(new java.awt.image.BufferedImage(3, 5, 1), GenerationParams(), values, ctx)
    assertEquals(result.head._1.getWidth, 12)
    assertEquals(result.head._1.getHeight, 20)
    assert(result.head._2.contains("Waifu2x style" -> "Photo"))
    assert(source.nonEmpty && output.nonEmpty && !Files.exists(source.get) && !Files.exists(output.get))

  test("native plugin is registered with the target id"):
    assertEquals(StandardPlugins.withOwnAxes.count(_.id == "waifu2x"), 1)
