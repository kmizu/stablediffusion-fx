package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class TemperatureGuardSuite extends munit.FunSuite:
  test("temperature protection is opt-in and serializes numeric settings"):
    val plugin = new TemperatureGuardPlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    val enabled = Values(defaults.map ++ Map("enabled" -> "true", "device" -> "2"))
    val params = plugin.process(GenerationParams(), enabled)
    val hook = params.hooks(plugin.hook)
    assertEquals(hook("device").num, 2.0)
    assertEquals(hook("sleep_temperature").num, 80.0)
    assertEquals(hook("wake_temperature").num, 70.0)
