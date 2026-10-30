package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class TokenDownsamplingSuite extends munit.FunSuite:
  test("opt-in controls preserve source fields and require the engine hook"):
    val plugin = new TokenDownsamplingPlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    val values = Values(defaults.map ++ Map("enabled" -> "true", "factor" -> "2.5", "factor2" -> "3", "disable_after" -> "0.7"))
    val params = plugin.process(GenerationParams(), values)
    assertEquals(params.hooks(plugin.id), ujson.Obj("factor" -> 2.5, "factor2" -> 3, "disable_after" -> 0.7))
    assertEquals(params.extra("ToDo factor"), "2.5")
    assertEquals(params.extra("ToDo factor2"), "3")
    assertEquals(params.extra("ToDo disable after"), "0.7")
    assertEquals(plugin.requiresEngineFeatures, Seq(plugin.id))
    assertEquals(plugin.xyzAxes.size, 3)
    assertEquals(StandardPlugins.alwaysOn.count(_.id == plugin.id), 1)

  test("all source XYZ axes override controls and source metadata replays safely"):
    val plugin = new TokenDownsamplingPlugin
    val fields = Seq("factor", "factor2", "disable_after")
    plugin.xyzAxes.zip(fields).foreach { (axis, field) =>
      val value = if field == "disable_after" then "0.5" else "2.5"
      val result = plugin.process(axis(GenerationParams(), value), Values.defaults(plugin.controls))
      assertEquals(result.hooks(plugin.id)(field).num, value.toDouble)
      assertEquals(TokenDownsamplingPlugin.replay(result.extra).get.map(field).toDouble, value.toDouble)
    }
    assertEquals(TokenDownsamplingPlugin.replay(Map.empty).get.map("enabled"), "false")
    assertEquals(TokenDownsamplingPlugin.replay(Map("ToDo factor" -> "NaN")), None)
    assertEquals(TokenDownsamplingPlugin.replay(Map("ToDo factor2" -> "9")), None)
