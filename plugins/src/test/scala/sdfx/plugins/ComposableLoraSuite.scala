package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class ComposableLoraSuite extends munit.FunSuite:
  test("all source controls are opt-in, preserve tags and round-trip through PNG infotext"):
    val plugin = new ComposableLoraPlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    val settings = defaults.map.keys.foldLeft(defaults)((v, k) => v.updated(k, "true"))
    val input = GenerationParams(prompt = "a <lora:A:.5> AND b <lyco:A:.8>", negativePrompt = "lowres", seed = 7)
    val out = plugin.process(input, settings)
    assertEquals(out.prompt, input.prompt)
    assertEquals(out.hooks("composable_lora").obj.size, 6)
    assert(out.hooks("composable_lora").obj.values.forall(_.bool))
    assertEquals(Infotext.toParams(Infotext.parse(Infotext.write(out, ImageFacts(seed = 7)))).hooks("composable_lora"),
      out.hooks("composable_lora"))
    assertEquals(StandardPlugins.alwaysOn.count(_.id == "composable_lora"), 1)
    assertEquals(plugin.requiresEngineFeatures, Seq("composable_lora"))
