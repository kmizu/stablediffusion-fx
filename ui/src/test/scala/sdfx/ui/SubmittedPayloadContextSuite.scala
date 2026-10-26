package sdfx.ui

import sdfx.core.plugins.api.*
import sdfx.core.payload.PayloadJsonBudget
import sdfx.engine.CaptureOptions
import sdfx.ui.vm.*

class SubmittedPayloadContextSuite extends munit.FunSuite:
  private class Form(val id: String, val kind: PluginKind) extends Plugin:
    def name = s"Title $id"
    override def controls = Seq(Control.Check("enabled", "Enabled"), Control.Text("literal", "Literal", "<b>猫</b>"))

  test("JobBuilder freezes disabled UI variation gate separately from effective params") {
    val vm = new GenerationVM(Mode.Txt2Img)
    val forms = new PluginsVM(Mode.Txt2Img, Nil, Nil, _ => Values.empty)
    vm.subseed.set(99L)
    vm.subseedStrength.set(0.4)
    vm.variationEnabled.set(false)
    val job = JobBuilder.build(vm, forms)
    assertEquals(job.params.subseed, 99L)
    assertEquals(job.params.subseedStrength, 0.0)
    assertEquals(job.submittedContext.flatMap(_.uiVariationEnabled), Some(false))
    vm.variationEnabled.set(true)
    vm.subseedStrength.set(0.7)
    assertEquals(job.submittedContext.flatMap(_.uiVariationEnabled), Some(false))
    assertEquals(job.params.subseedStrength, 0.0)
  }

  test("actual JobBuilder freezes selected script and all AlwaysOn forms including inactive values") {
    val script = new Form("script", PluginKind.Script)
    val active = new Form("active", PluginKind.AlwaysOn)
    val inactive = new Form("inactive", PluginKind.AlwaysOn)
    val forms = new PluginsVM(Mode.Inpaint, Seq(script), Seq(active, inactive), _ => Values.empty)
    forms.selectedScript.set("script")
    forms.form("active").get.prop("enabled").get.set("true")
    val vm = new GenerationVM(Mode.Inpaint)
    vm.prompt.set("queued prompt")
    vm.initImage.set("original.png")
    vm.maskImage.set("exported-mask.png")
    val job = JobBuilder.build(vm, forms, CaptureOptions(true))
    forms.form("inactive").get.prop("literal").get.set("later edit")
    forms.selectedScript.set("")
    vm.prompt.set("later prompt")
    vm.maskImage.set("later-mask.png")
    val submitted = job.submittedContext.get
    assertEquals(submitted.origin, "ui")
    assertEquals(submitted.sourceMode, Some("inpaint"))
    assertEquals(job.captureOptions, Some(CaptureOptions(true)))
    assertEquals(job.params.prompt, "queued prompt")
    assertEquals(job.params.img2img.flatMap(_.mask), Some("exported-mask.png"))
    assertEquals(job.alwaysOn.map(_._1.id), Seq("active"))
    val config = ujson.read(submitted.configurationJson.get)
    assertEquals(config("selected_script")("id").str, "script")
    assertEquals(config("selected_script")("title").str, "Title script")
    assertEquals(config("selected_script")("values")("literal").str, "<b>猫</b>")
    val all = config("always_on").arr.toVector
    assertEquals(all.map(_("id").str), Vector("active", "inactive"))
    assertEquals(all.map(_("active").bool), Vector(true, false))
    assertEquals(all.last("values")("literal").str, "<b>猫</b>")
  }

  test("submitted form serialization stops at budget and reports a whole unavailable configuration") {
    val plugin = new Form("large", PluginKind.AlwaysOn)
    val forms = new PluginsVM(Mode.Txt2Img, Nil, Seq(plugin), _ => Values.empty)
    forms.form("large").get.prop("literal").get.set("\u0000猫😀" * 1000)
    val json = forms.payloadConfigurationJson(PayloadJsonBudget.Limits(maxBytes = 256))
    val diagnostic = ujson.read(json)
    assertEquals(diagnostic("status").str, "unavailable_limits")
    assert(!diagnostic.obj.contains("always_on"))
    assert(json.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 256)
    assertEquals(forms.form("large").get.values.str("literal"), "\u0000猫😀" * 1000)
  }
