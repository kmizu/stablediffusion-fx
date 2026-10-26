package sdfx.core.presets

class PresetCompatibilityMetadataSuite extends munit.FunSuite:
  private val limits = PresetLimits()
  private def field(id: String): FieldKey = FieldKey.parse(id).toOption.get
  private def sourced(raw: RawJson): PresetRecord =
    PresetRecord(PresetRecord.initialId("config-txt2img", "One"), "One", Map.empty,
      Some(PresetSource("config-presets", "One", Some(PresetMode.Txt2Img), raw)),
      Set.empty, Vector.empty, Vector.empty)
  private def schema(mode: PresetMode, ids: String*): PresetSchema =
    PresetSchema(mode, "metadata-test", ids.toVector.map { id =>
      PresetFieldSpec(field(id), id, PresetFieldKind.Text, PresetAtom.Text(""),
        Vector.empty, None, None)
    }, Vector.empty)
  private val txt = Some(PresetMode.Txt2Img)
  private val img = Some(PresetMode.Img2Img)
  private val config = "config-presets"
  private val utilities = "preset-utilities"
  private val noBindings = Vector.empty[SourceBinding]
  private val allTiling = Set(
    field("plugin.asymmetric_tiling.enabled"), field("plugin.asymmetric_tiling.tile_x"),
    field("plugin.asymmetric_tiling.tile_y"), field("plugin.asymmetric_tiling.start_step"),
    field("plugin.asymmetric_tiling.stop_step"))

  test("source_field_ids_decode_escaped_names_in_source_order_without_changing_raw_lexemes") {
    val original = "{\"" + "\\u0053eed\":9007199254740993,\"fraction\":1.2500e+2," +
      "\"prompt\":\"" + ("p" * 2048) + "\",\"txt2img_steps\":41}"
    val raw = PresetJson.readSource(original.getBytes(java.nio.charset.StandardCharsets.UTF_8),
      limits).toOption.get
    val record = sourced(raw)
    assertEquals(PresetCompatibility.sourceFieldIds(record),
      Right(Vector("Seed", "fraction", "prompt", "txt2img_steps")))
    assertEquals(record.source.get.originalFields.text, original)
    assert(record.source.get.originalFields.text.contains("9007199254740993"))
    assert(record.source.get.originalFields.text.contains("1.2500e+2"))
  }

  test("source_field_ids_without_provenance_returns_empty_and_null_record_fails") {
    val native = sourced(RawJson.validated("{}")).copy(source = None)
    assertEquals(PresetCompatibility.sourceFieldIds(native), Right(Vector.empty[String]))
    assert(PresetCompatibility.sourceFieldIds(null).isLeft)
    assert(PresetCompatibility.sourceFieldIds(native.copy(source = null)).isLeft)
  }

  test("source_field_ids_reject_invalid_manually_constructed_raw_source") {
    for invalid <- Vector("[]", "17", "{", "{\"x\":1,\"x\":2}") do
      assert(PresetCompatibility.sourceFieldIds(sourced(RawJson.validated(invalid))).isLeft,
        s"invalid source was accepted: $invalid")
    assert(PresetCompatibility.sourceFieldIds(sourced(null)).isLeft)
  }

  test("source_field_ids_accept_2048_fields_and_reject_2049_without_truncation") {
    def rawWith(count: Int): RawJson = RawJson.validated(
      (0 until count).map(n => s"\"k$n\":$n").mkString("{", ",", "}"))
    val accepted = PresetCompatibility.sourceFieldIds(sourced(rawWith(2048))).toOption.get
    assertEquals(accepted.size, 2048)
    assertEquals(accepted.head, "k0")
    assertEquals(accepted.last, "k2047")
    assert(PresetCompatibility.sourceFieldIds(sourced(rawWith(2049))).isLeft)
  }

  test("source_field_ids_enforce_512_utf8_byte_key_boundary") {
    val exact = "界" * 170 + "ab"
    val tooLong = exact + "c"
    assertEquals(PresetCompatibility.sourceFieldIds(sourced(
      RawJson.validated(s"{\"$exact\":1}"))), Right(Vector(exact)))
    assert(PresetCompatibility.sourceFieldIds(sourced(
      RawJson.validated(s"{\"$tooLong\":1}"))).isLeft)
  }

  test("source_field_ids_use_default_source_byte_depth_and_node_limits") {
    val tooManyBytes = RawJson.validated("{\"a\":\"" + ("x" * 4194304) + "\"}")
    assert(PresetCompatibility.sourceFieldIds(sourced(tooManyBytes)).isLeft)
    val tooDeep = RawJson.validated("{\"a\":" + ("[" * 33) + "0" + ("]" * 33) + "}")
    assert(PresetCompatibility.sourceFieldIds(sourced(tooDeep)).isLeft)
    val tooManyNodes = RawJson.validated("{\"a\":[" + ("0," * 100000) + "0]}")
    assert(PresetCompatibility.sourceFieldIds(sourced(tooManyNodes)).isLeft)
  }

  test("resolve_known_config_id_keeps_canonical_target_when_current_schema_lacks_it") {
    val schemas = Map(PresetMode.Txt2Img -> schema(PresetMode.Txt2Img, "generation.steps"))
    assertEquals(PresetCompatibility.resolveSourceId("txt2img_hires_steps", config, txt,
      noBindings, schemas), Right(Vector(field("generation.hiresSteps"))))
    assertEquals(PresetCompatibility.resolveSourceId("txt2img_steps_alt", config, txt,
      noBindings, schemas), Right(Vector(field("generation.hiresSteps"))))
    assertEquals(PresetCompatibility.resolveSourceId("txt2img_hires_steps", config, img,
      noBindings, schemas).isLeft, true)
  }

  test("resolve_bare_resize_mode_and_xyz_ids_only_in_their_declared_modes") {
    val schemas = Map.empty[PresetMode, PresetSchema]
    assertEquals(PresetCompatibility.resolveSourceId("resize_mode", config, img,
      noBindings, schemas), Right(Vector(field("generation.resizeMode"))))
    assert(PresetCompatibility.resolveSourceId("resize_mode", config, txt,
      noBindings, schemas).isLeft)
    for axis <- Vector("x", "y", "z"); part <- Vector("type", "values") do
      val id = s"script_img2img_xyz_plot_${axis}_${part}"
      assertEquals(PresetCompatibility.resolveSourceId(id, config, img,
        noBindings, schemas), Right(Vector(field(s"plugin.xyz_grid.${axis}_${part}"))))
      assert(PresetCompatibility.resolveSourceId(id, config, txt,
        noBindings, schemas).isLeft)
    assertEquals(PresetCompatibility.resolveSourceId("X Values", utilities, None,
      noBindings, schemas), Right(Vector(field("plugin.xyz_grid.x_values"))))
  }

  test("standard_tiling_resolves_all_five_fields_and_explicit_plain_alias_resolves_one") {
    val schemas = Map.empty[PresetMode, PresetSchema]
    val enabled = field("plugin.asymmetric_tiling.enabled")
    val binding = Vector(SourceBinding(utilities, "my-tiling-enabled", enabled))
    assertEquals(PresetCompatibility.resolveSourceId("Tiling", utilities, None,
      binding, schemas).toOption.get.toSet, allTiling)
    assertEquals(PresetCompatibility.resolveSourceId("my-tiling-enabled", utilities, None,
      binding, schemas), Right(Vector(enabled)))
  }

  test("resolve_requires_exact_format_mode_and_unambiguous_verified_binding") {
    val schemas = Map.empty[PresetMode, PresetSchema]
    val height = field("generation.height")
    val width = field("generation.width")
    val aliases = Vector(SourceBinding(config, "my-size", height))
    assertEquals(PresetCompatibility.resolveSourceId("my-size", config, txt,
      aliases, schemas), Right(Vector(height)))
    assert(PresetCompatibility.resolveSourceId("my-size", utilities, None,
      aliases, schemas).isLeft)
    assert(PresetCompatibility.resolveSourceId("txt2img_height", config, None,
      noBindings, schemas).isLeft)
    assert(PresetCompatibility.resolveSourceId("Height", utilities, txt,
      noBindings, schemas).isLeft)
    assert(PresetCompatibility.resolveSourceId("txt2img_height", "unknown", txt,
      noBindings, schemas).isLeft)
    assert(PresetCompatibility.resolveSourceId("not-a-source-id", config, txt,
      noBindings, schemas).isLeft)
    val conflict = aliases :+ SourceBinding(config, "my-size", width)
    assertEquals(PresetCompatibility.resolveSourceId("my-size", config, txt,
      conflict, schemas).swap.toOption.get.code, "alias-conflict")
    val standardConflict = Vector(SourceBinding(config, "txt2img_height", width))
    assertEquals(PresetCompatibility.resolveSourceId("txt2img_height", config, txt,
      standardConflict, schemas).swap.toOption.get.code, "alias-conflict")
  }

  test("declared_synonyms_inherit_only_verified_bindings_and_detect_graph_conflicts") {
    val source = "tiledvae-t2i-enc-size"
    val synonym = "MD-t2i-enc-size"
    val target = field("plugin.tiled_vae.encoderTileSize")
    val other = field("plugin.tiled_vae.decoderTileSize")
    val schemas = Map.empty[PresetMode, PresetSchema]
    assert(PresetCompatibility.resolveSourceId(source, config, txt,
      noBindings, schemas).isLeft)
    assertEquals(PresetCompatibility.resolveSourceId(source, config, txt,
      Vector(SourceBinding(config, synonym, target)), schemas), Right(Vector(target)))
    val conflict = Vector(SourceBinding(config, source, target),
      SourceBinding(config, synonym, other))
    assertEquals(PresetCompatibility.resolveSourceId(source, config, txt,
      conflict, schemas).swap.toOption.get.code, "alias-conflict")
  }

  test("reverse_metadata_lists_declared_synonym_only_after_its_binding_is_verified") {
    val target = field("plugin.tiled_vae.encoderTileSize")
    val source = "tiledvae-t2i-enc-size"
    val bound = "MD-t2i-enc-size"
    assertEquals(PresetCompatibility.sourceIdsForField(target, config, txt,
      noBindings), Right(Vector.empty[String]))
    assertEquals(PresetCompatibility.sourceIdsForField(target, config, txt,
      Vector(SourceBinding(config, bound, target))), Right(Vector(bound, source)))
    val conflicting = Vector(SourceBinding(config, bound, target),
      SourceBinding(config, source, field("plugin.tiled_vae.decoderTileSize")))
    assertEquals(PresetCompatibility.sourceIdsForField(target, config, txt,
      conflicting).swap.toOption.get.code, "alias-conflict")
  }

  test("source_ids_for_field_prefers_verified_explicit_binding_and_never_invents_a_native_key") {
    val height = field("generation.height")
    val custom = field("plugin.my_native_only.level")
    val bindings = Vector(SourceBinding(config, "my-height", height),
      SourceBinding(config, "my-level", custom))
    assertEquals(PresetCompatibility.sourceIdsForField(height, config, txt,
      bindings).toOption.get.head, "my-height")
    assertEquals(PresetCompatibility.sourceIdsForField(custom, config, txt,
      bindings), Right(Vector("my-level")))
    assertEquals(PresetCompatibility.sourceIdsForField(custom, config, txt,
      noBindings), Right(Vector.empty[String]))
    assert(PresetCompatibility.sourceIdsForField(height, config, None,
      bindings).isLeft)
    val conflict = bindings :+ SourceBinding(config, "my-height", field("generation.width"))
    assert(PresetCompatibility.sourceIdsForField(height, config, txt,
      conflict).isLeft)
  }

  test("export_tracked_ids_requires_whole_set_and_omits_only_with_explicit_supported_only") {
    val tracked = Set(field("generation.steps"), field("plugin.native_only.level"))
    val rejected = PresetCompatibility.exportTrackedIds(tracked, config, txt,
      noBindings, supportedOnly = false).swap.toOption.get
    assert(rejected.issues.exists(i => i.code == "unsupported" &&
      i.path == "plugin.native_only.level"))
    assertEquals(PresetCompatibility.exportTrackedIds(tracked, config, txt,
      noBindings, supportedOnly = true), Right(Vector("txt2img_steps")))
  }

  test("export_tracked_tiling_requires_complete_group_unless_plain_alias_covers_a_constituent") {
    assertEquals(PresetCompatibility.exportTrackedIds(allTiling, utilities, None,
      noBindings, supportedOnly = false), Right(Vector("Tiling")))
    val enabled = field("plugin.asymmetric_tiling.enabled")
    val partial = Set(enabled, field("plugin.asymmetric_tiling.tile_x"))
    assert(PresetCompatibility.exportTrackedIds(partial, utilities, None,
      noBindings, supportedOnly = false).isLeft)
    assertEquals(PresetCompatibility.exportTrackedIds(partial, utilities, None,
      noBindings, supportedOnly = true), Right(Vector.empty[String]))
    val plain = Vector(SourceBinding(utilities, "my-tile-x", field("plugin.asymmetric_tiling.tile_x")))
    assertEquals(PresetCompatibility.exportTrackedIds(Set(field("plugin.asymmetric_tiling.tile_x")),
      utilities, None, plain, supportedOnly = false), Right(Vector("my-tile-x")))
  }

  test("export_tracked_ids_reports_all_unsupported_fields_and_alias_collisions") {
    val a = field("plugin.native.a")
    val b = field("plugin.native.b")
    val error = PresetCompatibility.exportTrackedIds(Set(a, b), config, txt,
      noBindings, supportedOnly = false).swap.toOption.get
    assertEquals(error.issues.map(_.path).toSet, Set("plugin.native.a", "plugin.native.b"))
    val collision = Vector(SourceBinding(config, "shared", a), SourceBinding(config, "shared", b))
    assert(PresetCompatibility.exportTrackedIds(Set(a, b), config, txt,
      collision, supportedOnly = false).isLeft)
  }

  test("supported_only_never_suppresses_a_conflicting_standard_scalar_binding") {
    val conflict = Vector(SourceBinding(config, "txt2img_height", field("generation.width")))
    for supportedOnly <- Vector(false, true) do
      val result = PresetCompatibility.exportTrackedIds(Set(field("generation.height")),
        config, txt, conflict, supportedOnly).swap.toOption.get
      assert(result.issues.exists(_.code == "alias-conflict"))
  }

  test("supported_only_never_suppresses_a_conflicting_complete_tiling_binding") {
    val conflict = Vector(SourceBinding(config, "txt2img_tiling", field("generation.width")))
    for supportedOnly <- Vector(false, true) do
      val result = PresetCompatibility.exportTrackedIds(allTiling,
        config, txt, conflict, supportedOnly).swap.toOption.get
      assert(result.issues.exists(_.code == "alias-conflict"))
  }

  test("metadata_bounds_accept_512_utf8_bytes_and_reject_513_before_materialization") {
    val exact = "界" * 170 + "ab"
    val tooLong = exact + "c"
    val target = field("generation.height")
    val schemas = Map.empty[PresetMode, PresetSchema]
    assertEquals(PresetCompatibility.resolveSourceId(exact, config, txt,
      Vector(SourceBinding(config, exact, target)), schemas), Right(Vector(target)))
    assert(PresetCompatibility.resolveSourceId(tooLong, config, txt,
      Vector(SourceBinding(config, tooLong, target)), schemas).isLeft)
    assert(PresetCompatibility.sourceIdsForField(target, config, txt,
      Vector(SourceBinding(config, tooLong, target))).isLeft)
    val oversizedField = FieldKey(Vector("plugin", "x" * 513))
    assert(PresetCompatibility.exportTrackedIds(Set(oversizedField), config, txt,
      noBindings, supportedOnly = true).isLeft)
  }

  test("metadata_rejects_oversized_bindings_schemas_and_tracked_sets_even_in_supported_only") {
    val height = field("generation.height")
    val binding = SourceBinding(config, "my-height", height)
    val manyBindings = Vector.fill(2049)(binding)
    val schemas = Map(PresetMode.Txt2Img -> schema(PresetMode.Txt2Img,
      (1 to 4097).map(n => s"generation.f$n")*))
    assert(PresetCompatibility.resolveSourceId("my-height", config, txt,
      manyBindings, Map.empty).isLeft)
    assert(PresetCompatibility.resolveSourceId("txt2img_height", config, txt,
      noBindings, schemas).isLeft)
    assert(PresetCompatibility.sourceIdsForField(height, config, txt,
      manyBindings).isLeft)
    val manyTracked = (1 to 4097).map(n => field(s"plugin.native.f$n")).toSet
    assert(PresetCompatibility.exportTrackedIds(manyTracked, config, txt,
      noBindings, supportedOnly = true).isLeft)
  }

  test("metadata_accepts_exact_binding_schema_and_tracked_count_limits") {
    val height = field("generation.height")
    val bindings = Vector.fill(2048)(SourceBinding(config, "my-height", height))
    val schemas = Map(PresetMode.Txt2Img -> schema(PresetMode.Txt2Img,
      (1 to 4096).map(n => s"generation.f$n")*))
    assertEquals(PresetCompatibility.resolveSourceId("my-height", config, txt,
      bindings, schemas), Right(Vector(height)))
    val tracked = (1 to 4095).map(n => field(s"plugin.native.f$n")).toSet +
      field("generation.height")
    assertEquals(PresetCompatibility.exportTrackedIds(tracked, config, txt,
      noBindings, supportedOnly = true), Right(Vector("txt2img_height")))
  }
