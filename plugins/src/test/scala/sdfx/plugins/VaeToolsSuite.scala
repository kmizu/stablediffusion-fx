package sdfx.plugins

import sdfx.core.plugins.api.*

class VaeToolsSuite extends munit.FunSuite:
  test("VAE Tools is a native file tool capability, not an accidental generation hook") {
    val plugin = StandardPlugins.withOwnAxes.find(_.id == "vae_tools").get
    assertEquals(plugin.requiresEngineFeatures, Seq("vae_tools"))
    assertEquals(plugin.tabs, Set.empty[Tab])
    assertEquals(plugin.controls, Nil)
    assert(!plugin.isActive(Values.defaults(plugin.controls)))
    assert(!StandardPlugins.withOwnAxes.exists(p => p.id == "vae_tools" && p.tabs.contains(Tab.Img2Img)))
  }
