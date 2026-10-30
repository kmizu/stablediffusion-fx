package sdfx.ui

import sdfx.core.plugins.api.Values
import sdfx.plugins.Img2ImgHiresFixPlugin
import sdfx.ui.vm.{GenerationVM, JobBuilder, Mode, PluginsVM, UiState}

class Img2ImgHiresRestoreSuite extends munit.FunSuite:
  for mode <- Seq(Mode.Txt2Img, Mode.Img2Img, Mode.Inpaint) do
    test(s"$mode restores PNG controls into the next job and persists all settings") {
      val plugin = new Img2ImgHiresFixPlugin
      val forms = new PluginsVM(mode, Nil, Seq(plugin), _ => Values.empty)
      val vm = new GenerationVM(mode)
      val text = "cat\nSteps: 10, Seed: 5, Size: 64x64, img2img Hires Fix: \"{'scale':'128x192','upscaler':'Lanczos','denoise':0.4,'steps':7,'cfg':3,'prompt':'dog','negative_prompt':'blur','sampler':'Euler','scheduler':'Karras'}\""
      vm.restoreInfotext(text)
      forms.restoreInfotext(text)
      val job = JobBuilder.build(vm, forms)
      val next = job.alwaysOn.foldLeft(job.params)((p, entry) => entry._1.process(p, entry._2))
      val hook = next.hooks(plugin.id)
      assertEquals(hook("width").num, 128.0)
      assertEquals(hook("height").num, 192.0)
      assertEquals(hook("prompt").str, "dog")
      assertEquals(hook("scheduler").str, "Karras")
      val saved = UiState(pluginValues = Map("mode" -> forms.snapshot))
      val restored = UiState.fromJson(UiState.toJson(saved))
      assertEquals(restored.pluginValues("mode"), forms.snapshot)
      forms.restoreInfotext("cat\nSteps: 10, Seed: 5, Size: 64x64")
      assert(!forms.form(plugin.id).get.isActive)
      forms.restoreInfotext(text)
      forms.restoreInfotext("cat\nSteps: 10, Seed: 5, Size: 64x64, img2img Hires Fix: invalid")
      assert(!forms.form(plugin.id).get.isActive)
    }
