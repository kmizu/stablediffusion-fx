package sdfx.ui

import javafx.beans.property.SimpleBooleanProperty
import java.nio.file.Files
import sdfx.core.presets.*
import sdfx.core.plugins.api.*
import sdfx.core.protocol.ModelEntry
import sdfx.engine.CatalogSnapshot
import sdfx.ui.presets.{NamedGenerationStates, NativePresetFields}
import sdfx.ui.vm.*

class NamedGenerationStatesSuite extends munit.FunSuite:
  private val script = new Plugin:
    val id = "snapshot-script"
    val name = "Snapshot script"
    val kind = PluginKind.Script
    override val controls = Seq(Control.Text("text", "Script input", "default"))
  private val extension = new Plugin:
    val id = "snapshot-extension"
    val name = "Snapshot extension"
    val kind = PluginKind.AlwaysOn
    override val controls = Seq(Control.Check("enabled", "Enabled"), Control.Text("text", "Extension input", "default"))

  private def fixture() =
    val vm = new GenerationVM(Mode.Txt2Img)
    val plugins = new PluginsVM(Mode.Txt2Img, Seq(script), Seq(extension), _ => Values.empty)
    val catalog = new CatalogVM
    val fields = new NativePresetFields(vm, plugins, catalog, new SimpleBooleanProperty(false), namedState = true)
    (vm, plugins, catalog, fields)
  private val empty = PresetLibrary(1, Vector.empty, Vector.empty)

  test("full named state survives typed disk roundtrip and restores inactive plugin controls") {
    val (vm, plugins, _, fields) = fixture()
    vm.prompt.set("saved prompt")
    vm.seed.set(Long.MaxValue)
    vm.eta.set(Some(0.5))
    vm.subseedStrength.set(0.4)
    vm.variationEnabled.set(false)
    vm.shared.clipSkip.set(2)
    plugins.selectedScript.set(script.id)
    plugins.form(script.id).get.prop("text").get.set("script saved")
    plugins.form(extension.id).get.prop("text").get.set("inactive saved")
    val record = NamedGenerationStates.capture("Japanese 日本語", fields).toOption.get
    val directory = Files.createTempDirectory("sdfx-named-state-")
    val store = new PresetStore(directory)
    val saved = store.save(PresetRevision(None), NamedGenerationStates.put(empty, Mode.Txt2Img, record)).toOption.get
    val loaded = store.load().toOption.get
    assertEquals(loaded.library, saved.library)
    vm.prompt.set("changed")
    vm.eta.set(None)
    plugins.selectedScript.set("")
    plugins.form(extension.id).get.prop("text").get.set("changed")
    val fromDisk = NamedGenerationStates.collection(loaded.library, Mode.Txt2Img).presets.head
    val plan = NamedGenerationStates.preflight(fromDisk, fields).fold(e => fail(e.toString), identity)
    assert(fields.write(plan.changes).isRight)
    assertEquals(vm.prompt.get, "saved prompt")
    assertEquals(vm.seed.get, Long.MaxValue)
    assertEquals(vm.eta.get, Some(0.5))
    assertEquals(vm.variationEnabled.get, false)
    assertEquals(plugins.selectedScript.get, script.id)
    assertEquals(plugins.form(extension.id).get.prop("text").get.get, "inactive saved")
    assertEquals(vm.shared.clipSkip.get, 2)
  }

  test("missing checkpoint blocks the entire restore without changing prompt") {
    val (vm, _, catalog, fields) = fixture()
    catalog.snapshot.set(CatalogSnapshot(checkpoints = Seq(ModelEntry("model", "model.safetensors", "sd15"))))
    vm.shared.checkpoint.set("model")
    val record = NamedGenerationStates.capture("model state", fields).toOption.get
    vm.prompt.set("keep this")
    catalog.snapshot.set(CatalogSnapshot())
    val result = NamedGenerationStates.preflight(record, fields)
    assert(result.isLeft)
    assert(result.swap.toOption.get.issues.exists(_.path == "model.checkpoint"))
    assertEquals(vm.prompt.get, "keep this")
    assertEquals(record.fields(FieldKey(Vector("model", "checkpoint"))), PresetAtom.Text("model"))
  }

  test("missing image references block restore; init and mask are explicitly excluded") {
    val (vm, _, _, fields) = fixture()
    vm.controlNet.head.image.set("Z:/missing/sdfx-state-input.png")
    val record = NamedGenerationStates.capture("missing input", fields).toOption.get
    assert(NamedGenerationStates.preflight(record, fields).isLeft)
    assert(!record.fields.keys.exists(_.render.contains("initImage")))
    assert(!record.fields.keys.exists(_.render.contains("maskImage")))
    assert(NamedGenerationStates.scope.contains("NOT saved or restored"))
  }

  test("names, overwrite and delete are mode-isolated and preserve other collections") {
    val (_, _, _, fields) = fixture()
    assert(NamedGenerationStates.capture("  ", fields).isLeft)
    val record = NamedGenerationStates.capture("one", fields).toOption.get
    val first = NamedGenerationStates.put(empty, Mode.Txt2Img, record)
    val second = NamedGenerationStates.put(first, Mode.Inpaint, record.copy(id = java.util.UUID.randomUUID()))
    val replacement = NamedGenerationStates.put(second, Mode.Txt2Img, record.copy(id = java.util.UUID.randomUUID()))
    assertEquals(NamedGenerationStates.collection(replacement, Mode.Txt2Img).presets.head.id, record.id)
    val deleted = NamedGenerationStates.remove(replacement, Mode.Txt2Img, "one")
    assert(NamedGenerationStates.collection(deleted, Mode.Txt2Img).presets.isEmpty)
    assertEquals(NamedGenerationStates.collection(deleted, Mode.Inpaint).presets.size, 1)
  }
