package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class Img2ImgHiresFixSuite extends munit.FunSuite:
  test("opt-in form keeps base generation unchanged and exports full source settings") {
    val plugin = new Img2ImgHiresFixPlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    val input = GenerationParams(prompt = "cat", steps = 20)
    val output = plugin.process(input, Values(defaults.map ++ Map("enabled" -> "true", "upscaler" -> "Lanczos", "cfg" -> "3")))
    assertEquals(output.prompt, input.prompt)
    assertEquals(output.steps, input.steps)
    assertEquals(output.hooks(plugin.id)("cfg").num, 3.0)
    val restored = Img2ImgHiresFixPlugin.replay(output.extra).get
    assertEquals(restored.str("cfg"), "3")
    assertEquals(Img2ImgHiresFixPlugin.replay(Map.empty).get.bool("enabled"), false)
  }
