package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class PromptFusionSuite extends munit.FunSuite:
  test("Prompt Fusion preserves syntax, is opt-in, and records all geometry settings"):
    val plugin = new PromptFusionPlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    val input = GenerationParams(prompt = "[cat:dog: , 12:bezier]", negativePrompt = "[red:blue::mean]")
    val output = plugin.process(input, defaults.updated("enabled", "true"))
    assertEquals(output.prompt, input.prompt)
    assertEquals(output.negativePrompt, input.negativePrompt)
    assertEquals(output.hooks("prompt_fusion"), ujson.Obj("slerp_scale" -> 0, "negative_origin" -> true, "slerp_epsilon" -> 0.0001))
    assertEquals(output.extra("Prompt Fusion"), "True")
    assertEquals(plugin.requiresEngineFeatures, Seq("prompt_fusion"))
    assertEquals(StandardPlugins.alwaysOn.count(_.id == "prompt_fusion"), 1)

  test("invalid geometry settings fail clearly"):
    val plugin = new PromptFusionPlugin
    val defaults = Values.defaults(plugin.controls).updated("enabled", "true")
    intercept[IllegalArgumentException](plugin.process(GenerationParams(), defaults.updated("slerp_epsilon", "-1")))
    intercept[IllegalArgumentException](plugin.process(GenerationParams(), defaults.updated("slerp_scale", "NaN")))
