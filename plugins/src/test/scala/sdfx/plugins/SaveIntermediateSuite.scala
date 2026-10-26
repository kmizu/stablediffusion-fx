package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class SaveIntermediateSuite extends munit.FunSuite:
  test("intermediate recorder is opt-in and exposes complete persisted options"):
    val p = new SaveIntermediatePlugin
    val defaults = Values.defaults(p.controls)
    assert(!p.isActive(defaults))
    val values = defaults.updated("enabled", "true").updated("every_n", "3")
      .updated("image_type", "Noisy").updated("mode", "Video").updated("format", "gif")
      .updated("final_save", "true").updated("smooth", "true").updated("seconds", "5")
    val output = p.process(GenerationParams(), values)
    assertEquals(output.hooks("save_intermediate")("every_n").num, 3.0)
    assertEquals(output.hooks("save_intermediate")("image_type").str, "Noisy")
    assertEquals(output.hooks("save_intermediate")("final_save").bool, true)
    assertEquals(output.hooks("save_intermediate")("smooth").bool, true)
    assertEquals(output.extra("Save intermediate settings"), output.hooks("save_intermediate").render())
    assertEquals(p.requiresEngineFeatures, Seq("save_intermediate"))
    assertEquals(StandardPlugins.alwaysOn.count(_.id == "save_intermediate"), 1)
    assertEquals(p.enabledBy("fps"), "enabled")
