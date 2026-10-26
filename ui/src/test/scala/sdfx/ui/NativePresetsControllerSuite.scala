package sdfx.ui

import java.util.UUID
import javafx.beans.property.{SimpleBooleanProperty, SimpleDoubleProperty, SimpleIntegerProperty}
import sdfx.core.presets.*
import sdfx.core.plugins.api.Values
import sdfx.core.payload.RequestPayload
import sdfx.core.protocol.Json
import sdfx.ui.presets.{NativePresetFields, NativePresetsController}
import sdfx.ui.vm.*

/** Controller tests use the actual editable properties, with no engine or service. */
class NativePresetsControllerSuite extends munit.FunSuite:
  private def key(name: String): FieldKey = FieldKey.parse(name).toOption.get
  private val cfg = key("generation.cfgScale")
  private val steps = key("generation.steps")
  private val width = key("generation.width")
  private val prompt = key("generation.prompt")
  private val sampler = key("generation.sampler")
  private val seed = key("generation.seed")
  private val denoise = key("generation.denoisingStrength")
  private val extra = key("generation.variationEnabled")
  private val subseed = key("generation.subseed")
  private val strength = key("generation.subseedStrength")

  private def record(fields: (FieldKey, PresetAtom)*): PresetRecord =
    PresetRecord(UUID.fromString("00000000-0000-0000-0000-000000000001"), "partial",
      fields.toMap, None, Set.empty, Vector.empty, Vector.empty)

  private def form(mode: Mode, shared: ModelVM = new ModelVM,
      busy: () => Boolean = () => false): (GenerationVM, NativePresetsController) =
    val vm = new GenerationVM(mode, shared)
    val plugins = new PluginsVM(mode, Nil, Nil, _ => Values.empty)
    val fields = new NativePresetFields(vm, plugins, new CatalogVM, new SimpleBooleanProperty(false))
    (vm, new NativePresetsController(fields, busy, () => true))

  private def applied(controller: NativePresetsController, preset: PresetRecord,
      tracked: Set[FieldKey], base: ApplyBase = ApplyBase.Current): PresetApplyPlan =
    controller.apply(preset, tracked, base).fold(error => fail(s"unexpected apply error: $error"), identity)

  test("current_overlay_changes_only_saved_tracked_fields") {
    val shared = new ModelVM
    shared.checkpoint.set("loaded checkpoint")
    shared.clipSkip.set(3)
    val (txt, controller) = form(Mode.Txt2Img, shared)
    val (img, _) = form(Mode.Img2Img, shared)
    txt.cfgScale.set(9)
    txt.steps.set(20)
    txt.prompt.set("keep prompt")
    img.steps.set(41)
    applied(controller, record(steps -> PresetAtom.Integer(28)), Set(cfg, steps))
    assertEquals(txt.steps.get, 28)
    assertEquals(txt.cfgScale.get, 9.0)
    assertEquals(txt.prompt.get, "keep prompt")
    assertEquals(img.steps.get, 41)
    assertEquals(shared.checkpoint.get, "loaded checkpoint")
    assertEquals(shared.clipSkip.get, 3)
  }

  test("controller_applies_known_value_and_returns_retained_opaque_warning") {
    val (vm, controller) = form(Mode.Txt2Img)
    val raw = PresetJson.readSource("""{"n":9007199254740993}"""
      .getBytes(java.nio.charset.StandardCharsets.UTF_8), PresetLimits()).toOption.get
    val saved = record(steps -> PresetAtom.Integer(28)).copy(
      opaque = Vector("foreign_source_value" -> raw))
    vm.steps.set(20)
    val result = applied(controller, saved, Set(steps))
    assertEquals(vm.steps.get, 28)
    assertEquals(result.changes, Map(steps -> PresetAtom.Integer(28)))
    assertEquals(result.issues.map(_.path), Vector("foreign_source_value"))
    assert(result.issues.head.message.contains("not applied"))
    assertEquals(saved.opaque.head._2.text, """{"n":9007199254740993}""")
  }

  test("defaults_reset_only_tracked_available_values") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.cfgScale.set(9)
    vm.prompt.set("untracked prompt")
    applied(controller, record(), Set(cfg), ApplyBase.Defaults)
    assertEquals(vm.cfgScale.get, 7.0)
    assertEquals(vm.prompt.get, "untracked prompt")
  }

  test("invalid_known_number_blocks_every_visible_write") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.prompt.set("original")
    vm.steps.set(20)
    val outcome = controller.apply(record(prompt -> PresetAtom.Text("would be partial"),
      steps -> PresetAtom.Integer(1000000)), Set(prompt, steps), ApplyBase.Current)
    assert(outcome.isLeft)
    assertEquals(vm.prompt.get, "original")
    assertEquals(vm.steps.get, 20)
  }

  test("setter_failure_rolls_back_other_touched_properties") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.cfgScale.set(9)
    vm.width.bind(new SimpleIntegerProperty(512))
    val result = controller.apply(record(cfg -> PresetAtom.Decimal(5),
      width -> PresetAtom.Integer(768)), Set(cfg, width), ApplyBase.Current)
    assert(result.isLeft)
    assertEquals(vm.cfgScale.get, 9.0)
    assertEquals(vm.width.get, 512)
  }

  test("rollback_failure_is_reported_separately_from_the_setter_failure") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.cfgScale.set(9)
    vm.width.set(512)
    var armed = false
    def bindAfterFirstChange(): Unit =
      if !armed then
        armed = true
        vm.cfgScale.bind(new SimpleDoubleProperty(vm.cfgScale.get))
        vm.width.bind(new SimpleIntegerProperty(vm.width.get))
    vm.cfgScale.addListener((_, _, _) => bindAfterFirstChange())
    vm.width.addListener((_, _, _) => bindAfterFirstChange())
    val result = controller.apply(record(cfg -> PresetAtom.Decimal(5),
      width -> PresetAtom.Integer(768)), Set(cfg, width), ApplyBase.Current)
    assert(armed, "a first setter must run before the second becomes bound")
    assert(result.isLeft)
    val issues = result.swap.toOption.get.issues
    assert(issues.exists(i => (i.code + " " + i.message).toLowerCase.contains("rollback")),
      s"expected a separately visible rollback failure, got $issues")
  }

  test("mask_export_blocks_apply_without_changing_the_form") {
    val (vm, controller) = form(Mode.Inpaint, busy = () => true)
    vm.prompt.set("painted draft")
    val result = controller.apply(record(prompt -> PresetAtom.Text("new prompt")),
      Set(prompt), ApplyBase.Current)
    assert(result.isLeft)
    assertEquals(vm.prompt.get, "painted draft")
  }

  test("apply_admission_checks_the_same_busy_gate_before_an_editor_commit") {
    var exporting = true
    val (vm, controller) = form(Mode.Inpaint, busy = () => exporting)
    vm.steps.set(34)
    assertEquals(controller.admitApply().swap.toOption.get.issues.map(_.code),
      Vector("mask-export-busy"))
    assertEquals(vm.steps.get, 34)
    exporting = false
    assertEquals(controller.admitApply(), Right(()))

    val offThreadVm = new GenerationVM(Mode.Inpaint)
    val offThreadFields = new NativePresetFields(offThreadVm,
      new PluginsVM(Mode.Inpaint, Nil, Nil, _ => Values.empty),
      new CatalogVM, new SimpleBooleanProperty(false))
    val offThread = new NativePresetsController(offThreadFields, () => false, () => false)
    assertEquals(offThread.admitApply().swap.toOption.get.issues.map(_.code),
      Vector("wrong-thread"))
  }

  test("apply_admission_rejects_reentry_during_an_actual_form_write") {
    val (vm, controller) = form(Mode.Txt2Img)
    var reentry: Option[PresetError] = None
    vm.steps.addListener((_, _, _) => {
      reentry = controller.admitApply().swap.toOption
      ()
    })
    applied(controller, record(steps -> PresetAtom.Integer(42)), Set(steps))
    assertEquals(reentry.toVector.flatMap(_.issues.map(_.code)), Vector("reentrant"))
    assertEquals(vm.steps.get, 42)
  }

  test("wrong_fx_thread_rejects_preview_apply_and_capture_without_mutation") {
    val vm = new GenerationVM(Mode.Txt2Img)
    val plugins = new PluginsVM(Mode.Txt2Img, Nil, Nil, _ => Values.empty)
    val fields = new NativePresetFields(vm, plugins, new CatalogVM, new SimpleBooleanProperty(false))
    val controller = new NativePresetsController(fields, () => false, () => false)
    val preset = record(prompt -> PresetAtom.Text("would change"))
    val collection = PresetCollection("utilities", Set(prompt), Vector.empty, Vector.empty, Vector.empty)
    vm.prompt.set("original")
    var observedWrites = 0
    vm.prompt.addListener((_, _, _) => observedWrites += 1)
    assert(controller.preview(preset, Set(prompt), ApplyBase.Current).isLeft)
    assert(controller.apply(preset, Set(prompt), ApplyBase.Current).isLeft)
    assert(controller.captureSelected("must not gather", collection, Set(prompt)).isLeft)
    assert(controller.inspect().isLeft)
    assertEquals(vm.prompt.get, "original")
    assertEquals(observedWrites, 0)
  }

  test("inspect_returns_a_guarded_immutable_schema_and_current_pair") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.steps.set(36)
    val first = controller.inspect().fold(error => fail(s"inspect failed: $error"), identity)
    assert(first._1.fields.exists(_.key == steps))
    assertEquals(first._2.values.get(steps), Some(PresetAtom.Integer(36)))
    vm.steps.set(48)
    val second = controller.inspect().fold(error => fail(s"inspect failed: $error"), identity)
    assertEquals(first._2.values.get(steps), Some(PresetAtom.Integer(36)))
    assertEquals(second._2.values.get(steps), Some(PresetAtom.Integer(48)))
  }

  test("inspect_rejects_a_reentrant_form_write_before_property_reads") {
    val (vm, controller) = form(Mode.Txt2Img)
    var nested: Option[Either[PresetError, (PresetSchema, FieldSnapshot)]] = None
    vm.steps.addListener((_, _, _) => nested = Some(controller.inspect()))
    applied(controller, record(steps -> PresetAtom.Integer(39)), Set(steps))
    assert(nested.exists(_.left.toOption.exists(_.issues.exists(_.code == "reentrant"))))
  }

  test("utilities_denoise_roundtrips_and_applies_across_modes") {
    val (txt, txtController) = form(Mode.Txt2Img)
    val (img, imgController) = form(Mode.Img2Img)
    val (paint, paintController) = form(Mode.Inpaint)
    val collection = PresetCollection("utilities", Set(denoise), Vector.empty, Vector.empty, Vector.empty)
    txt.hiresDenoise.set(0.4)
    txt.hiresEnabled.set(false)
    val saved = txtController.captureSelected("shared denoise", collection, Set(denoise))
      .fold(error => fail(s"unexpected capture error: $error"), identity)
    assertEquals(saved.fields.get(denoise), Some(PresetAtom.Decimal(BigDecimal("0.4"))))
    applied(imgController, saved, collection.tracked)
    applied(paintController, saved, collection.tracked)
    assertEquals(img.denoise.get, 0.4)
    assertEquals(paint.denoise.get, 0.4)
    assertEquals(txt.hiresEnabled.get, false)
    img.denoise.set(0.6)
    val reverse = imgController.captureSelected("reverse", collection, Set(denoise))
      .fold(error => fail(s"unexpected capture error: $error"), identity)
    applied(txtController, reverse, collection.tracked)
    assertEquals(txt.hiresDenoise.get, 0.6)
    assertEquals(txt.hiresEnabled.get, false)
  }

  test("capture_is_immutable_and_keeps_unresolved_seed") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.seed.set(-1L)
    vm.prompt.set("first")
    val collection = PresetCollection("utilities", Set(seed, prompt), Vector.empty, Vector.empty, Vector.empty)
    val saved = controller.captureSelected("fixed snapshot", collection, Set(seed, prompt))
      .fold(error => fail(s"unexpected capture error: $error"), identity)
    vm.seed.set(12345L)
    vm.prompt.set("later")
    assertEquals(saved.fields.get(seed), Some(PresetAtom.Integer(-1L)))
    assertEquals(saved.fields.get(prompt), Some(PresetAtom.Text("first")))
    assertEquals(saved.id, PresetRecord.initialId("utilities", "fixed snapshot"))
  }

  test("renamed_record_keeps_its_id_and_freed_name_can_be_saved_again") {
    val (_, controller) = form(Mode.Txt2Img)
    val empty = PresetCollection("utilities", Set(prompt), Vector.empty, Vector.empty, Vector.empty)
    val originalA = controller.captureSelected("A", empty, Set(prompt)).toOption.get
    val renamedB = originalA.copy(name = "B") // An external name edit preserves stored identity.
    val collection = empty.copy(presets = Vector(renamedB))
    val newA = controller.captureSelected("A", collection, Set(prompt))
      .fold(error => fail(s"the free name A must remain savable: $error"), identity)
    assertEquals(renamedB.id, originalA.id)
    assertNotEquals(newA.id, renamedB.id)
    assertEquals(newA.name, "A")
    val library = PresetLibrary(1, Vector(collection.copy(presets = Vector(renamedB, newA))), Vector.empty)
    assert(PresetJson.writeNative(library, PresetLimits()).isRight,
      "both records must pass the native codec's duplicate-ID validation")
  }

  test("occupied_id_fallback_is_deterministic_for_the_same_collection") {
    val (_, controller) = form(Mode.Txt2Img)
    val empty = PresetCollection("utilities", Set(prompt), Vector.empty, Vector.empty, Vector.empty)
    val renamedB = controller.captureSelected("A", empty, Set(prompt)).toOption.get.copy(name = "B")
    val collection = empty.copy(presets = Vector(renamedB))
    val first = controller.captureSelected("A", collection, Set(prompt)).toOption.get
    val second = controller.captureSelected("A", collection, Set(prompt)).toOption.get
    assertEquals(first.id, second.id)
    assertNotEquals(first.id, renamedB.id)
  }

  test("occupied_alternate_id_search_avoids_a_second_collision") {
    val (_, controller) = form(Mode.Txt2Img)
    val empty = PresetCollection("utilities", Set(prompt), Vector.empty, Vector.empty, Vector.empty)
    val renamedB = controller.captureSelected("A", empty, Set(prompt)).toOption.get.copy(name = "B")
    val firstOccupied = empty.copy(presets = Vector(renamedB))
    val renamedC = controller.captureSelected("A", firstOccupied, Set(prompt)).toOption.get.copy(name = "C")
    val twiceOccupied = empty.copy(presets = Vector(renamedB, renamedC))
    val newA = controller.captureSelected("A", twiceOccupied, Set(prompt))
      .fold(error => fail(s"a second free ID should be found: $error"), identity)
    assert(!Set(renamedB.id, renamedC.id)(newA.id))
    val library = PresetLibrary(1, Vector(twiceOccupied.copy(presets = Vector(renamedB, renamedC, newA))),
      Vector.empty)
    assert(PresetJson.writeNative(library, PresetLimits()).isRight)
  }

  test("same_name_overwrite_keeps_id_and_unused_name_uses_base_id") {
    val (vm, controller) = form(Mode.Txt2Img)
    val empty = PresetCollection("utilities", Set(prompt), Vector.empty, Vector.empty, Vector.empty)
    val originalA = controller.captureSelected("A", empty, Set(prompt)).toOption.get
    val renamedB = originalA.copy(name = "B")
    val collection = empty.copy(presets = Vector(renamedB))
    vm.prompt.set("edited")
    val overwrittenB = controller.captureSelected("B", collection, Set(prompt)).toOption.get
    assertEquals(overwrittenB.id, renamedB.id)
    assertEquals(overwrittenB.fields(prompt), PresetAtom.Text("edited"))
    val unused = controller.captureSelected("new unused name", collection, Set(prompt)).toOption.get
    assertEquals(unused.id, PresetRecord.initialId("utilities", "new unused name"))
  }

  test("extra_only_apply_keeps_positive_seed_and_strength_drafts") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.subseed.set(99L)
    vm.subseedStrength.set(0.4)
    vm.variationEnabled.set(true)
    applied(controller, record(extra -> PresetAtom.Flag(false)), Set(extra, subseed, strength))
    assert(!vm.variationEnabled.get)
    assertEquals(vm.subseed.get, 99L)
    assertEquals(vm.subseedStrength.get, 0.4)
    assertEquals(vm.toParams.subseedStrength, 0.0)
    applied(controller, record(extra -> PresetAtom.Flag(true)), Set(extra, subseed, strength))
    assertEquals(vm.toParams.subseedStrength, 0.4)
  }

  test("variation_strength_only_apply_preserves_disabled_gate") {
    val (vm, controller) = form(Mode.Img2Img)
    vm.variationEnabled.set(false)
    applied(controller, record(strength -> PresetAtom.Decimal(BigDecimal("0.4"))),
      Set(extra, strength))
    assert(!vm.variationEnabled.get)
    assertEquals(vm.subseedStrength.get, 0.4)
    assertEquals(vm.toParams.subseedStrength, 0.0)
  }

  test("changed_form_is_used_instead_of_an_old_preview") {
    val (vm, controller) = form(Mode.Txt2Img)
    vm.cfgScale.set(9)
    val preset = record(steps -> PresetAtom.Integer(30))
    val old = controller.preview(preset, Set(cfg, steps), ApplyBase.Defaults)
      .fold(error => fail(s"unexpected preview error: $error"), identity)
    vm.cfgScale.set(11)
    val fresh = applied(controller, preset, Set(cfg, steps), ApplyBase.Defaults)
    assertEquals(vm.cfgScale.get, 7.0)
    assertEquals(vm.steps.get, 30)
    assert(old.currentRevision != fresh.currentRevision,
      "apply must plan from the edited form, rather than commit the stale preview")
  }

  test("changed_catalog_replans_unavailable_choice_with_warning") {
    val vm = new GenerationVM(Mode.Txt2Img)
    val catalog = new CatalogVM
    val fields = new NativePresetFields(vm,
      new PluginsVM(Mode.Txt2Img, Nil, Nil, _ => Values.empty),
      catalog, new SimpleBooleanProperty(false))
    val controller = new NativePresetsController(fields, () => false, () => true)
    vm.sampler.set("Euler")
    val preset = record(sampler -> PresetAtom.Text("DDIM"))
    assert(controller.preview(preset, Set(sampler), ApplyBase.Current).isRight)
    catalog.samplers.setAll("Euler")
    val result = applied(controller, preset, Set(sampler))
    assertEquals(vm.sampler.get, "Euler")
    assert(result.issues.exists(_.path.contains(sampler.render)))
  }

  test("queued_job_remains_frozen_after_preset_apply") {
    val vm = new GenerationVM(Mode.Txt2Img)
    val plugins = new PluginsVM(Mode.Txt2Img, Nil, Nil, _ => Values.empty)
    val fields = new NativePresetFields(vm, plugins, new CatalogVM, new SimpleBooleanProperty(false))
    val controller = new NativePresetsController(fields, () => false, () => true)
    vm.prompt.set("queued prompt")
    vm.seed.set(12345L)
    val queued = JobBuilder.build(vm, plugins)
    val oldBody = RequestPayload.freeze(Json.toJs(queued.params))
    applied(controller, record(prompt -> PresetAtom.Text("later prompt"),
      seed -> PresetAtom.Integer(-1L)), Set(prompt, seed))
    assertEquals(vm.prompt.get, "later prompt")
    assertEquals(vm.seed.get, -1L)
    assertEquals(queued.params.prompt, "queued prompt")
    assertEquals(queued.params.seed, 12345L)
    assertEquals(RequestPayload.freeze(Json.toJs(queued.params)), oldBody)
  }

  test("apply_reentrancy_is_rejected_while_property_observers_run") {
    val (vm, controller) = form(Mode.Txt2Img)
    val preset = record(prompt -> PresetAtom.Text("outer"))
    var nested: Option[Either[PresetError, PresetApplyPlan]] = None
    vm.prompt.addListener((_, _, _) =>
      nested = Some(controller.apply(record(prompt -> PresetAtom.Text("nested")),
        Set(prompt), ApplyBase.Current)))
    applied(controller, preset, Set(prompt))
    assert(nested.exists(_.isLeft), "nested apply must report its reentrancy error")
    assertEquals(vm.prompt.get, "outer")
  }
