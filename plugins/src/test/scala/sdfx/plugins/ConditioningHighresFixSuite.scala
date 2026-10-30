package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class ConditioningHighresFixSuite extends munit.FunSuite:
  private val plugin = new ConditioningHighresFixPlugin
  private def values(kv: (String, String)*): Values = Values(Values.defaults(plugin.controls).map ++ kv.toMap)

  test("opt in modes reach engine and preserve mode in image metadata") {
    assert(!plugin.isActive(values()))
    for mode <- ConditioningHighresFixPlugin.modes do
      val configured = values("enabled" -> "true", "mode" -> mode)
      val result = plugin.process(GenerationParams(), configured)
      assertEquals(result.hooks("conditioning_highres_fix")("mode").str, mode)
      assertEquals(result.extra("Conditioning Highres Fix"), mode)
    assert(StandardPlugins.alwaysOn.exists(_.id == "conditioning_highres_fix"))
  }

  test("metadata restores valid modes and clears old settings when absent or invalid") {
    assertEquals(ConditioningHighresFixPlugin.replay(Map("Conditioning Highres Fix" -> "high")).get.str("mode"), "high")
    assert(!ConditioningHighresFixPlugin.replay(Map.empty).get.bool("enabled"))
    assert(!ConditioningHighresFixPlugin.replay(Map("Conditioning Highres Fix" -> "invalid")).get.bool("enabled"))
  }
