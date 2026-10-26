package sdfx.ui

import sdfx.core.plugins.api.Values
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.PluginRegistry
import sdfx.plugins.{LatentUpscalePlugin, StandardPlugins}
import sdfx.ui.vm.{Mode, PluginsVM}
import java.nio.file.Files

class LatentUpscaleReplaySuite extends munit.FunSuite:
  test("latent upscale infotext restores script choice and exact native values; ordinary replay clears this script"):
    val p = new LatentUpscalePlugin
    val vm = new PluginsVM(Mode.Img2Img, Seq(p), Nil, _ => Values.defaults(p.controls))
    vm.restoreLatentUpscale(Map("Latent upscale method" -> "bicubic-antialiased", "Latent upscale scheduler" -> "normal"))
    assertEquals(vm.selectedScript.get, "latent_upscale")
    assertEquals(vm.form("latent_upscale").get.values.str("method"), "bicubic-antialiased")
    assertEquals(vm.form("latent_upscale").get.values.str("scheduler"), "normal")
    vm.restoreLatentUpscale(Map.empty)
    assertEquals(vm.selectedScript.get, "")
    vm.selectedScript.set("another_script")
    vm.restoreLatentUpscale(Map.empty)
    assertEquals(vm.selectedScript.get, "another_script")

  test("native Inpaint registry exposes the latent upscale script and restores its own form"):
    val root = Files.createTempDirectory("latent-inpaint-registry")
    var settings = Settings()
    val registry = new PluginRegistry(StandardPlugins.withOwnAxes, AppHome(root), () => settings)
    try
      // Exercise the actual mode -> registry -> form path used by MainWindow, rather than a synthetic plugin list.
      val scripts = registry.scripts(Mode.Inpaint.pluginTab)
      val vm = new PluginsVM(Mode.Inpaint, scripts, registry.alwaysOn(Mode.Inpaint.pluginTab), registry.values)
      assert(vm.form("latent_upscale").nonEmpty)
      assert(!registry.scripts(Mode.Txt2Img.pluginTab).exists(_.id == "latent_upscale"))
      vm.restoreLatentUpscale(Map("Latent upscale method" -> "nearest-exact", "Latent upscale scheduler" -> "simple"))
      assertEquals(vm.mode, Mode.Inpaint)
      val (plugin, values) = vm.scriptJob.get
      assertEquals(plugin.id, "latent_upscale")
      assertEquals(values.str("method"), "nearest-exact")
      assertEquals(values.str("scheduler"), "simple")
      settings = settings.copy(disabledPlugins = Seq("latent_upscale"))
      assert(!registry.scripts(Mode.Inpaint.pluginTab).exists(_.id == "latent_upscale"))
    finally
      registry.close()
      Files.deleteIfExists(root)
