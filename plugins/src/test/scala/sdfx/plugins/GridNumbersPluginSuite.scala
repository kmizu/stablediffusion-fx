package sdfx.plugins

import sdfx.core.infotext.Infotext
import sdfx.core.image.GridNumbers
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import java.nio.file.Files
import javax.imageio.ImageIO

class GridNumbersPluginSuite extends munit.FunSuite:
  private val plugin = new GridNumbersPlugin
  private def values(kv: (String, String)*) = Values(Values.defaults(plugin.controls).map ++ kv)

  test("native plugin is opt-in, separates both settings, persists to infotext and replays without an engine hook") {
    val base = GenerationParams(prompt = "cat", seed = 9)
    assert(!plugin.isActive(values()))
    assertEquals(plugin.process(base, values()), base)
    val enabled = plugin.process(base, values("enabled" -> "true", "image_numbers" -> "false", "xyz_info" -> "true"))
    assertEquals(enabled.hooks, base.hooks)
    assertEquals(enabled.extra("Grid image numbers"), "False")
    assertEquals(enabled.extra("Grid XYZ info"), "True")
    val back = Infotext.toParams(Infotext.parse(Support.infotext(enabled, 9)))
    assert(GridNumbers.enabled(back.extra))
    assertEquals(back.extra("Grid image numbers"), "False")
    assertEquals(StandardPlugins.alwaysOn.count(_.id == "grid_numbers"), 1)
    assertEquals(plugin.requiresEngineFeatures, Nil)
    assertEquals(plugin.enabledBy("xyz_info"), "enabled")
  }

  test("XYZ labels use actual cell axis values despite expensive-axis reordering, and preserve source files") {
    val opts = values("enabled" -> "true", "image_numbers" -> "false", "xyz_info" -> "true")
    val ctx = new FakeContext:
      override def generate(p: GenerationParams) = super.generate(plugin.process(p, opts))
    val xyz = new XyzGridPlugin
    val v = Values(Values.defaults(xyz.controls).map ++ Map("x_type" -> "Checkpoint name", "x_values" -> "A,B",
      "y_type" -> "Steps", "y_values" -> "1,2", "z_type" -> "Clip skip", "z_values" -> "1,2",
      "draw_legend" -> "false", "include_sub_images" -> "true", "include_sub_grids" -> "true"))
    val result = xyz.run(GenerationParams(seed = 4, width = 256, height = 256), v, ctx)
    assertEquals(result.images.size, 8)
    assertEquals(result.grids.size, 3)
    result.images.foreach { o =>
      assertEquals(o.params.extra("Grid X info"), s"Checkpoint name: ${o.params.checkpoint.get}")
      assertEquals(o.params.extra("Grid Y info"), s"Steps: ${o.params.steps}")
      assertEquals(o.params.extra("Grid Z info"), s"Clip skip: ${o.params.clipSkip}")
      val source = ImageIO.read(o.path.toFile)
      assertEquals(source.getRGB(10, 10) & 0xffffff, ctx.colorOf(o.params, o.seed))
      val before = Files.readAllBytes(o.path).toSeq
      val rendered = Support.cellImage(Seq(o)).get
      assertEquals(Files.readAllBytes(o.path).toSeq, before)
      assert(rendered.getRGB(11, 11) != source.getRGB(11, 11))
    }
    ctx.saved.zipWithIndex.foreach { case ((_, _, p, text), index) =>
      assertEquals(p.extra("Grid XYZ info"), "True")
      val names = ujson.read(Infotext.parse(text).get("Grid source images").get).arr.map(_("filename").str).toSet
      val visible = if index == 0 then result.images else result.images.filter(_.params.clipSkip == index)
      assertEquals(names, visible.map(_.path.getFileName.toString).toSet)
    }
  }

  test("shared script cell helper decorates filename numbers without mutating source images") {
    val ctx = new FakeContext
    val p = plugin.process(GenerationParams(width = 256, height = 256, seed = 1), values("enabled" -> "true"))
    val source = ctx.generate(p).head
    val renamed = source.path.resolveSibling("00421-123.png")
    Files.copy(source.path, renamed)
    val output = source.copy(path = renamed)
    val before = Files.readAllBytes(renamed).toSeq
    val decorated = Support.cellImage(Seq(output)).get
    val original = ImageIO.read(renamed.toFile)
    assertEquals(Files.readAllBytes(renamed).toSeq, before)
    assert(decorated.getRGB(11, 230) != original.getRGB(11, 230))
    assertEquals(decorated.getRGB(128, 128), original.getRGB(128, 128))
  }
