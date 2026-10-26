package sdfx.core.presets

import java.util.UUID

/** Literal form fixtures keep planning expectations independent of native bindings. */
class PresetApplySuite extends munit.FunSuite:
  private def key(name: String): FieldKey = FieldKey.parse(name).toOption.get
  private val cfg = key("generation.cfgScale")
  private val steps = key("generation.steps")
  private val prompt = key("generation.prompt")
  private val sampler = key("generation.sampler")
  private val denoise = key("generation.denoisingStrength")
  private val extra = key("generation.variationEnabled")
  private val strength = key("generation.subseedStrength")
  private val pluginEnabled = key("plugin.face_restore.enabled")
  private val unknown = key("plugin.absent.enabled")

  private def spec(
    id: FieldKey,
    kind: PresetFieldKind,
    default: PresetAtom,
    choices: Vector[PresetAtom] = Vector.empty,
    min: Option[BigDecimal] = None,
    max: Option[BigDecimal] = None
  ): PresetFieldSpec = PresetFieldSpec(id, id.render, kind, default, choices, min, max)

  private def schema(fields: PresetFieldSpec*): PresetSchema =
    PresetSchema(PresetMode.Txt2Img, "schema-1", fields.toVector, Vector.empty)

  private def current(values: (FieldKey, PresetAtom)*): FieldSnapshot =
    FieldSnapshot(values.toMap, "form-1")

  private def record(fields: (FieldKey, PresetAtom)*): PresetRecord =
    PresetRecord(UUID.fromString("00000000-0000-0000-0000-000000000001"), "partial",
      fields.toMap, None, Set.empty, Vector.empty, Vector.empty)

  private def planned(
    preset: PresetRecord,
    tracked: Set[FieldKey],
    destination: PresetSchema,
    snapshot: FieldSnapshot,
    base: ApplyBase
  ): PresetApplyPlan =
    PresetApply.plan(preset, tracked, destination, snapshot, base)
      .fold(error => fail(s"unexpected plan error: $error"), identity)

  test("overlay_retains_missing_values") {
    val result = planned(record(steps -> PresetAtom.Integer(28)), Set(cfg, steps),
      schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7)),
        spec(steps, PresetFieldKind.Integer, PresetAtom.Integer(20))),
      current(cfg -> PresetAtom.Decimal(9), steps -> PresetAtom.Integer(20)), ApplyBase.Current)
    assertEquals(result.changes.get(steps), Some(PresetAtom.Integer(28)))
    assert(!result.changes.contains(cfg), "missing CFG must keep the current value 9")
    assertEquals(result.schemaRevision, "schema-1")
    assertEquals(result.currentRevision, "form-1")
  }

  test("defaults_reset_only_tracked_available") {
    val result = planned(record(steps -> PresetAtom.Integer(28)), Set(cfg, steps, unknown),
      schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7)),
        spec(steps, PresetFieldKind.Integer, PresetAtom.Integer(20)),
        spec(pluginEnabled, PresetFieldKind.Flag, PresetAtom.Flag(false))),
      current(cfg -> PresetAtom.Decimal(9), steps -> PresetAtom.Integer(20),
        pluginEnabled -> PresetAtom.Flag(true)), ApplyBase.Defaults)
    assertEquals(result.changes.get(cfg), Some(PresetAtom.Decimal(7)))
    assertEquals(result.changes.get(steps), Some(PresetAtom.Integer(28)))
    assert(!result.changes.contains(pluginEnabled), "an untracked plugin toggle must survive")
    assert(!result.changes.contains(unknown), "an unavailable tracked field has no setter")
  }

  test("empty_config_none_is_noop") {
    val result = planned(record(), Set(cfg),
      schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7))),
      current(cfg -> PresetAtom.Decimal(9)), ApplyBase.Current)
    assertEquals(result.changes, Map.empty[FieldKey, PresetAtom])
    assertEquals(result.issues, Vector.empty[PresetIssue])
  }

  test("utilities_empty_default_preset_resets_tracked") {
    val result = planned(record(), Set(cfg),
      schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7)),
        spec(prompt, PresetFieldKind.Text, PresetAtom.Text(""))),
      current(cfg -> PresetAtom.Decimal(9), prompt -> PresetAtom.Text("keep this prompt")),
      ApplyBase.Defaults)
    assertEquals(result.changes, Map(cfg -> PresetAtom.Decimal(7)))
  }

  test("utilities_denoise_roundtrips_and_applies_across_modes") {
    val saved = record(denoise -> PresetAtom.Decimal(BigDecimal("0.4")))
    Seq(PresetMode.Txt2Img, PresetMode.Img2Img, PresetMode.Inpaint).foreach { mode =>
      val destination = PresetSchema(mode, mode.toString,
        Vector(spec(denoise, PresetFieldKind.Decimal, PresetAtom.Decimal(BigDecimal("0.75")),
          min = Some(BigDecimal(0)), max = Some(BigDecimal(1)))), Vector.empty)
      val result = planned(saved, Set(denoise), destination,
        current(denoise -> PresetAtom.Decimal(BigDecimal("0.9"))), ApplyBase.Current)
      assertEquals(result.changes.get(denoise), Some(PresetAtom.Decimal(BigDecimal("0.4"))), mode.toString)
    }
    assertEquals(saved.fields(denoise), PresetAtom.Decimal(BigDecimal("0.4")))
  }

  test("invalid_choice_falls_back_with_diagnostic") {
    val result = planned(record(sampler -> PresetAtom.Text("removed sampler")), Set(sampler),
      schema(spec(sampler, PresetFieldKind.Choice, PresetAtom.Text("Euler"),
        Vector(PresetAtom.Text("Euler"), PresetAtom.Text("DDIM")))),
      current(sampler -> PresetAtom.Text("DDIM")), ApplyBase.Current)
    assert(!result.changes.contains(sampler))
    assert(result.issues.exists(_.path.contains(sampler.render)))
  }

  test("invalid_choice_falls_back_to_declared_default_in_default_mode") {
    val result = planned(record(sampler -> PresetAtom.Text("removed sampler")), Set(sampler),
      schema(spec(sampler, PresetFieldKind.Choice, PresetAtom.Text("Euler"),
        Vector(PresetAtom.Text("Euler"), PresetAtom.Text("DDIM")))),
      current(sampler -> PresetAtom.Text("DDIM")), ApplyBase.Defaults)
    assertEquals(result.changes.get(sampler), Some(PresetAtom.Text("Euler")))
    assert(result.issues.exists(_.path.contains(sampler.render)))
  }

  test("invalid_default_choice_leaves_current") {
    val result = planned(record(sampler -> PresetAtom.Text("removed sampler")), Set(sampler),
      schema(spec(sampler, PresetFieldKind.Choice, PresetAtom.Text("old default"),
        Vector(PresetAtom.Text("DDIM")))),
      current(sampler -> PresetAtom.Text("DDIM")), ApplyBase.Defaults)
    assert(!result.changes.contains(sampler))
    assert(result.issues.exists(_.path.contains(sampler.render)))
  }

  test("out_of_range_numeric_choice_default_leaves_current") {
    val result = planned(record(steps -> PresetAtom.Integer(9)), Set(steps),
      schema(spec(steps, PresetFieldKind.Choice, PresetAtom.Integer(100),
        Vector(PresetAtom.Integer(1), PresetAtom.Integer(100)),
        min = Some(BigDecimal(0)), max = Some(BigDecimal(10)))),
      current(steps -> PresetAtom.Integer(1)), ApplyBase.Defaults)
    assert(!result.changes.contains(steps), "out-of-range fallback must never reach an FX setter")
    assert(result.issues.exists(i => i.code == "unavailable-default" && i.path == steps.render))
  }

  test("duplicate_destination_descriptors_are_ambiguous_and_unwritable") {
    val result = planned(record(cfg -> PresetAtom.Decimal(5)), Set(cfg),
      schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7)),
        spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(8))),
      current(cfg -> PresetAtom.Decimal(9)), ApplyBase.Current)
    assert(!result.changes.contains(cfg))
    assert(result.issues.exists(i => i.code == "ambiguous" && i.path == cfg.render))
  }

  test("known_bad_numeric_rejects_all_mutations") {
    val destination = schema(spec(prompt, PresetFieldKind.Text, PresetAtom.Text("")),
      spec(steps, PresetFieldKind.Integer, PresetAtom.Integer(20),
        min = Some(BigDecimal(1)), max = Some(BigDecimal(150))))
    val proposed = record(prompt -> PresetAtom.Text("must not write"),
      steps -> PresetAtom.Integer(1000))
    assert(PresetApply.plan(proposed, Set(prompt, steps), destination,
      current(prompt -> PresetAtom.Text("original"), steps -> PresetAtom.Integer(20)),
      ApplyBase.Current).isLeft)
  }

  test("known_wrong_atom_kind_rejects_all_mutations") {
    val destination = schema(spec(prompt, PresetFieldKind.Text, PresetAtom.Text("")),
      spec(extra, PresetFieldKind.Flag, PresetAtom.Flag(false)))
    assert(PresetApply.plan(record(prompt -> PresetAtom.Text("must not write"),
      extra -> PresetAtom.Text("true")), Set(prompt, extra), destination,
      current(prompt -> PresetAtom.Text("original"), extra -> PresetAtom.Flag(false)),
      ApplyBase.Current).isLeft)
  }

  test("tracked_field_unavailable_in_destination_is_skipped_with_diagnostic") {
    val result = planned(record(unknown -> PresetAtom.Flag(true), cfg -> PresetAtom.Decimal(5)),
      Set(unknown, cfg), schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7))),
      current(cfg -> PresetAtom.Decimal(9)), ApplyBase.Current)
    assertEquals(result.changes, Map(cfg -> PresetAtom.Decimal(5)))
    assert(result.issues.exists(_.path.contains(unknown.render)))
  }

  test("untracked_source_fields_do_not_apply") {
    val result = planned(record(cfg -> PresetAtom.Decimal(5),
      pluginEnabled -> PresetAtom.Flag(false), unknown -> PresetAtom.Flag(true)), Set(cfg),
      schema(spec(cfg, PresetFieldKind.Decimal, PresetAtom.Decimal(7)),
        spec(pluginEnabled, PresetFieldKind.Flag, PresetAtom.Flag(false))),
      current(cfg -> PresetAtom.Decimal(9), pluginEnabled -> PresetAtom.Flag(true)), ApplyBase.Current)
    assertEquals(result.changes, Map(cfg -> PresetAtom.Decimal(5)))
    assert(result.issues.exists(_.path.contains(pluginEnabled.render)))
    assert(result.issues.exists(_.path.contains(unknown.render)))
  }

  test("opaque_source_values_warn_per_key_without_changing_typed_overlay_or_raw_bytes") {
    val refiner = RawJson.validated("true")
    val distilled = RawJson.validated("3.5000e+0")
    val opaque = Vector("txt2img_enable-checkbox" -> refiner,
      "txt2img_distilled_cfg_scale" -> distilled)
    val saved = record(steps -> PresetAtom.Integer(28)).copy(opaque = opaque)
    val result = planned(saved, Set(steps),
      schema(spec(steps, PresetFieldKind.Integer, PresetAtom.Integer(20))),
      current(steps -> PresetAtom.Integer(20)), ApplyBase.Current)
    assertEquals(result.changes, Map(steps -> PresetAtom.Integer(28)))
    assertEquals(result.issues.map(_.path).toSet, opaque.map(_._1).toSet)
    assertEquals(result.issues.size, opaque.size)
    assert(result.issues.forall(i => i.code == "opaque-unapplied" &&
      i.message.contains("retained") && i.message.contains("not applied")))
    assertEquals(saved.opaque, opaque)
    assertEquals(saved.opaque.last._2.text, "3.5000e+0")
  }

  test("opaque_warning_count_keeps_the_last_of_210_keys_reachable") {
    val opaque = Vector.tabulate(210)(i => f"unknown_source_$i%03d" -> RawJson.validated("1"))
    val result = planned(record().copy(opaque = opaque), Set.empty,
      schema(), current(), ApplyBase.Current)
    assertEquals(result.changes, Map.empty[FieldKey, PresetAtom])
    assertEquals(result.issues.size, opaque.size)
    assert(result.issues.exists(_.path == "unknown_source_209"))
  }

  test("opaque_values_do_not_make_invalid_known_fields_partially_applicable") {
    val saved = record(prompt -> PresetAtom.Text("must not write"),
      steps -> PresetAtom.Integer(1000)).copy(
        opaque = Vector("foreign" -> RawJson.validated("9007199254740993")))
    val result = PresetApply.plan(saved, Set(prompt, steps),
      schema(spec(prompt, PresetFieldKind.Text, PresetAtom.Text("")),
        spec(steps, PresetFieldKind.Integer, PresetAtom.Integer(20),
          min = Some(BigDecimal(1)), max = Some(BigDecimal(150)))),
      current(prompt -> PresetAtom.Text("original"), steps -> PresetAtom.Integer(20)),
      ApplyBase.Current)
    assert(result.isLeft)
    assertEquals(result.swap.toOption.get.issues.map(_.code), Vector("invalid-range"))
    assertEquals(saved.opaque.head._2.text, "9007199254740993")
  }

  test("extra_only_preset_preserves_positive_drafts") {
    val result = planned(record(extra -> PresetAtom.Flag(false)), Set(extra, strength),
      schema(spec(extra, PresetFieldKind.Flag, PresetAtom.Flag(false)),
        spec(strength, PresetFieldKind.Decimal, PresetAtom.Decimal(0))),
      current(extra -> PresetAtom.Flag(true),
        strength -> PresetAtom.Decimal(BigDecimal("0.4"))), ApplyBase.Current)
    assertEquals(result.changes.get(extra), Some(PresetAtom.Flag(false)))
    assert(!result.changes.contains(strength))
  }

  test("variation_strength_only_preset_preserves_gate") {
    val result = planned(record(strength -> PresetAtom.Decimal(BigDecimal("0.4"))),
      Set(extra, strength),
      schema(spec(extra, PresetFieldKind.Flag, PresetAtom.Flag(false)),
        spec(strength, PresetFieldKind.Decimal, PresetAtom.Decimal(0),
          min = Some(BigDecimal(0)), max = Some(BigDecimal(1)))),
      current(extra -> PresetAtom.Flag(false), strength -> PresetAtom.Decimal(0)), ApplyBase.Current)
    assertEquals(result.changes.get(strength), Some(PresetAtom.Decimal(BigDecimal("0.4"))))
    assert(!result.changes.contains(extra))
  }
