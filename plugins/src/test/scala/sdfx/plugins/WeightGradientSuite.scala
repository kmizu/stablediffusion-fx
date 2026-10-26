package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class WeightGradientSuite extends munit.FunSuite:
  test("weight gradient is opt-in, advertises its hook, and preserves prompt syntax"):
    val p = new WeightGradientPlugin
    val defaults = Values.defaults(p.controls)
    assert(!p.isActive(defaults))
    assert(p.isActive(defaults.updated("enabled", "true")))
    val input = GenerationParams(prompt = "{cat::0-1:e}", negativePrompt = "{dog::1-0}")
    val out = p.process(input, defaults.updated("enabled", "true"))
    assertEquals(out.prompt, input.prompt)
    assertEquals(out.negativePrompt, input.negativePrompt)
    assertEquals(out.hooks("weight_gradient"), ujson.Obj())
    assertEquals(out.extra("Weight Gradient"), "True")
    assertEquals(p.requiresEngineFeatures, Seq("weight_gradient"))
    assertEquals(StandardPlugins.alwaysOn.count(_.id == "weight_gradient"), 1)
