package sdfx.plugins

import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.{Tab, Values}

class InpaintDifferenceSuite extends munit.FunSuite:
  test("difference mode carries a base path and typed operation settings") {
    val plugin = new InpaintDifferencePlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    assertEquals(plugin.tabs, Set(Tab.Img2Img))
    val values = Values(defaults.map ++ Map("enabled" -> "true", "base_path" -> "base.png",
      "threshold" -> "0.5", "contours" -> "true", "dilation" -> "2", "erosion" -> "1"))
    val result = plugin.process(GenerationParams(img2img = Some(Img2ImgParams(initImage = "altered.png"))), values)
    val hook = result.hooks("inpaint_difference")
    assertEquals(hook("base_path").str, "base.png")
    assertEquals(hook("threshold").num, .5)
    assertEquals(hook("dilation").num, 2.0)
    assert(hook("contours").bool && result.isInpaint)
  }

  test("empty base and out-of-range controls are rejected") {
    val plugin = new InpaintDifferencePlugin
    val defaults = Values.defaults(plugin.controls)
    intercept[IllegalArgumentException](plugin.hookFields(defaults))
    intercept[IllegalArgumentException](plugin.hookFields(Values(defaults.map ++ Map("base_path" -> "a", "dilation" -> "101"))))
  }
