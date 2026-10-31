package sdfx.ui

import sdfx.core.infotext.Infotext
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import sdfx.plugins.ModelKeywordPlugin
import sdfx.ui.vm.{GenerationVM, Mode, PluginsVM}

class ModelKeywordRestoreSuite extends munit.FunSuite:
  test("first use starts enabled while a saved opt-out remains disabled") {
    val fresh = new PluginsVM(Mode.Txt2Img, Nil, Seq(new ModelKeywordPlugin), _ => Values.empty)
    assert(fresh.form("model_keyword").get.isActive)
    val saved = new PluginsVM(Mode.Txt2Img, Nil, Seq(new ModelKeywordPlugin),
      _ => Values(Map("enabled" -> "false")))
    assert(!saved.form("model_keyword").get.isActive)
  }
  test("restoring an effective prompt disables keyword rewriting") {
    val forms = new PluginsVM(Mode.Txt2Img, Nil, Seq(new ModelKeywordPlugin), _ => Values.empty)
    val form = forms.form("model_keyword").get
    form.load(Values(Map("enabled" -> "true")))
    val text = "red cat\nSteps: 5, Seed: 3, Size: 64x64, Model keyword settings: {}"
    forms.restoreInfotext(text)
    assert(!form.isActive)
    val previous = GenerationParams(hooks = Map("model_keyword" -> ujson.Obj("enabled" -> true)))
    val restored = Infotext.toParams(Infotext.parse(text), previous)
    assertEquals(restored.prompt, "red cat")
    assert(!restored.hooks.contains("model_keyword"))
  }

  test("keyword replay spacing marker survives restore and clears after editing") {
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.restoreInfotext("<lora:A:1>red  , cat\nSteps: 5, Seed: 3, Size: 64x64, Model keyword settings: {}")
    assertEquals(vm.toParams.extra.get("Model keyword replay"), Some("True"))
    vm.prompt.set("edited prompt")
    assertEquals(vm.toParams.extra.get("Model keyword replay"), None)
  }
