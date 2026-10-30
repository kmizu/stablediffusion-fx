package sdfx.ui

import javafx.beans.property.SimpleStringProperty
import sdfx.ui.vm.InpaintDifferenceInputs

class InpaintDifferenceInputsSuite extends munit.FunSuite:
  test("queued job keeps both paths and options after later form changes") {
    val plugin = new sdfx.plugins.InpaintDifferencePlugin
    val plugins = new sdfx.ui.vm.PluginsVM(sdfx.ui.vm.Mode.Inpaint, Nil, Seq(plugin), _ => sdfx.core.plugins.api.Values.empty)
    val vm = new sdfx.ui.vm.GenerationVM(sdfx.ui.vm.Mode.Inpaint)
    val form = plugins.form(plugin.id).get
    form.props("enabled").set("true")
    form.props("base_path").set("first-base.png")
    form.props("threshold").set("0.5")
    vm.initImage.set("first-altered.png")
    val job = sdfx.ui.vm.JobBuilder.build(vm, plugins)
    form.props("base_path").set("later-base.png")
    form.props("threshold").set("0.8")
    vm.initImage.set("later-altered.png")
    val captured = job.alwaysOn.find(_._1.id == plugin.id).get._2
    assertEquals(job.params.img2img.get.initImage, "first-altered.png")
    assertEquals(captured.str("base_path"), "first-base.png")
    assertEquals(captured.str("threshold"), "0.5")
  }
  test("deleted saved inputs do not override startup's missing-file sanitization") {
    val actual = new SimpleStringProperty("")
    val enabled = new SimpleStringProperty("false")
    val painted = new SimpleStringProperty("missing.png")
    val altered = new SimpleStringProperty("valid.png")
    val state = new InpaintDifferenceInputs(actual, enabled, painted, altered, _ == "valid.png")
    assertEquals(actual.get, "")
    assertEquals(state.painted.get, "")
    enabled.set("true")
    assertEquals(actual.get, "valid.png")
  }
  test("both mode inputs survive saved plugin values across restart") {
    val plugin = new sdfx.plugins.InpaintDifferencePlugin
    val form = new sdfx.ui.vm.PluginValuesVM(plugin)
    val actual = new SimpleStringProperty("paint.png")
    val state = new InpaintDifferenceInputs(actual, form.props("enabled"), form.props("painted_path"), form.props("altered_path"))
    form.props("enabled").set("true")
    state.altered.set("altered.png")
    val saved = sdfx.ui.vm.UiState(pluginValues = Map("inpaint" -> Map(plugin.id -> form.values.map)))
    val restored = sdfx.ui.vm.UiState.fromJson(sdfx.ui.vm.UiState.toJson(saved))
    val newForm = new sdfx.ui.vm.PluginValuesVM(plugin, sdfx.core.plugins.api.Values(restored.pluginValues("inpaint")(plugin.id)))
    val newActual = new SimpleStringProperty("altered.png")
    val reopened = new InpaintDifferenceInputs(newActual, newForm.props("enabled"), newForm.props("painted_path"), newForm.props("altered_path"))
    assertEquals(reopened.altered.get, "altered.png")
    newForm.props("enabled").set("false")
    assertEquals(newActual.get, "paint.png")
    newForm.props("enabled").set("true")
    assertEquals(newActual.get, "altered.png")
  }
  test("switching modes preserves the painted image and restores the altered input") {
    val actual = new SimpleStringProperty("paint.png")
    val enabled = new SimpleStringProperty("false")
    val state = new InpaintDifferenceInputs(actual, enabled)
    enabled.set("true")
    state.altered.set("changed.png")
    assertEquals(actual.get, "changed.png")
    enabled.set("false")
    assertEquals(actual.get, "paint.png")
    actual.set("new-paint.png")
    enabled.set("true")
    assertEquals(actual.get, "changed.png")
    actual.set("sent-to-inpaint.png")
    assertEquals(state.altered.get, "sent-to-inpaint.png")
    enabled.set("false")
    assertEquals(actual.get, "new-paint.png")
  }
