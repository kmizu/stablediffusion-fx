package sdfx.core.presets

import java.nio.charset.StandardCharsets.UTF_8
import java.util.UUID

class PresetCompatibilitySuite extends munit.FunSuite:
  private val limits = PresetLimits()
  private def key(id: String): FieldKey = FieldKey.parse(id).toOption.get
  private def spec(id: String, default: PresetAtom, label: String = ""): PresetFieldSpec =
    val kind = default match
      case _: PresetAtom.Integer => PresetFieldKind.Integer
      case _: PresetAtom.Decimal => PresetFieldKind.Decimal
      case _: PresetAtom.Flag => PresetFieldKind.Flag
      case _ => PresetFieldKind.Text
    PresetFieldSpec(key(id), if label.isEmpty then id else label, kind, default, Vector.empty, None, None)
  private def schema(mode: PresetMode, fields: PresetFieldSpec*): PresetSchema =
    PresetSchema(mode, "test-catalog-v1", fields.toVector, Vector.empty)
  private val common = Vector(
    spec("generation.prompt", PresetAtom.Text(""), "Prompt"),
    spec("generation.negativePrompt", PresetAtom.Text(""), "Negative prompt"),
    spec("generation.seed", PresetAtom.Integer(-1L), "Seed"),
    spec("generation.steps", PresetAtom.Integer(20L), "Sampling Steps"),
    spec("generation.sampler", PresetAtom.Text("Euler a"), "Sampling method"),
    spec("generation.scheduler", PresetAtom.Text("Normal"), "Scheduler"),
    spec("generation.width", PresetAtom.Integer(512L), "Width"),
    spec("generation.height", PresetAtom.Integer(512L), "Height"),
    spec("generation.batchCount", PresetAtom.Integer(1L), "Batch count"),
    spec("generation.batchSize", PresetAtom.Integer(1L), "Batch size"),
    spec("generation.cfgScale", PresetAtom.Decimal(BigDecimal(7)), "CFG Scale"),
    spec("generation.variationEnabled", PresetAtom.Flag(false), "Extra"),
    spec("generation.subseed", PresetAtom.Integer(-1L), "Variation seed"),
    spec("generation.subseedStrength", PresetAtom.Decimal(BigDecimal(0)), "Variation strength")
  )
  private val txt = schema(PresetMode.Txt2Img,
    (common ++ Vector(
      spec("generation.hiresEnabled", PresetAtom.Flag(false), "Hires. fix"),
      spec("generation.hiresUpscaler", PresetAtom.Text(""), "Upscaler"),
      spec("generation.hiresSteps", PresetAtom.Integer(0), "Hires. steps"),
      spec("generation.hiresResizeX", PresetAtom.Integer(0), "Resize width to"),
      spec("generation.hiresResizeY", PresetAtom.Integer(0), "Resize height to"),
      spec("generation.hiresPrompt", PresetAtom.Text(""), "Hires prompt"),
      spec("generation.hiresNegative", PresetAtom.Text(""), "Hires negative prompt"),
      PresetFieldSpec(key("generation.hiresScale"), "Upscale by", PresetFieldKind.Decimal,
        PresetAtom.Decimal(BigDecimal(2)), Vector.empty, Some(BigDecimal(1)), Some(BigDecimal(5))),
      spec("generation.denoisingStrength", PresetAtom.Decimal(BigDecimal("0.7")), "Denoising strength"),
      spec("generation.hiresSampler", PresetAtom.Text(""), "Hires sampling method"),
      spec("plugin.face_restore.enabled", PresetAtom.Flag(false), "Restore faces"),
      spec("plugin.asymmetric_tiling.enabled", PresetAtom.Flag(false), "Asymmetric tiling"),
      spec("plugin.asymmetric_tiling.tile_x", PresetAtom.Flag(true), "Tile X"),
      spec("plugin.asymmetric_tiling.tile_y", PresetAtom.Flag(true), "Tile Y"),
      spec("plugin.asymmetric_tiling.start_step", PresetAtom.Integer(0), "Start step"),
      spec("plugin.asymmetric_tiling.stop_step", PresetAtom.Integer(-1), "Stop step")
    ))*)
  private val img = schema(PresetMode.Img2Img,
    (common ++ Vector(
      spec("generation.denoisingStrength", PresetAtom.Decimal(BigDecimal("0.75")), "Denoising strength"),
      spec("generation.maskBlur", PresetAtom.Integer(4), "Mask blur"),
      PresetFieldSpec(key("generation.resizeMode"), "Resize mode", PresetFieldKind.Choice,
        PresetAtom.Text("Just resize"),
        Vector(PresetAtom.Text("Resize and fill"), PresetAtom.Text("Just resize"),
          PresetAtom.Text("Crop and resize")), None, None)
    ))*)
  private val inp = schema(PresetMode.Inpaint, (img.fields ++ Vector(
    spec("generation.inpaintingFill", PresetAtom.Text("original"), "Masked content")
  ))*)
  private val schemas = Map(PresetMode.Txt2Img -> txt, PresetMode.Img2Img -> img, PresetMode.Inpaint -> inp)
  private def document(json: String, format: String, mode: Option[PresetMode]): SourceDocument =
    SourceDocument(format, mode, PresetJson.readSource(json.getBytes(UTF_8), limits).toOption.get, "test-sha256")
  private def imported(json: String, format: String, mode: Option[PresetMode],
      bindings: Vector[SourceBinding] = Vector.empty): CompatibilityResult =
    PresetCompatibility.importRecords(document(json, format, mode), schemas, bindings).toOption.get.head
  private def issueFor(result: CompatibilityResult, fragment: String): Boolean =
    result.issues.exists(i => (i.path + " " + i.message).contains(fragment))
  private def fresh(fields: Map[FieldKey, PresetAtom], name: String = "One"): PresetRecord =
    PresetRecord(UUID.fromString("68d0e2a3-e2a6-46b1-95da-35d9069e0e21"), name, fields,
      None, Set.empty, Vector.empty, Vector.empty)
  private def sourceText(bytes: Array[Byte]): String = new String(bytes, UTF_8)
  private def numericSchema(mode: PresetMode): PresetSchema =
    schema(mode,
      spec("generation.resizeMode", PresetAtom.Integer(0), "Resize mode"),
      spec("generation.inpaintingFill", PresetAtom.Integer(1), "Masked content"),
      spec("generation.maskInvert", PresetAtom.Flag(false), "Mask mode"),
      spec("generation.inpaintFullRes", PresetAtom.Flag(false), "Inpaint area"))
  private def numericSchemas: Map[PresetMode, PresetSchema] =
    Map(PresetMode.Txt2Img -> numericSchema(PresetMode.Txt2Img),
      PresetMode.Img2Img -> numericSchema(PresetMode.Img2Img),
      PresetMode.Inpaint -> numericSchema(PresetMode.Inpaint))
  private val tilingFields = Map[FieldKey, PresetAtom](
    key("plugin.asymmetric_tiling.enabled") -> PresetAtom.Flag(true),
    key("plugin.asymmetric_tiling.tile_x") -> PresetAtom.Flag(true),
    key("plugin.asymmetric_tiling.tile_y") -> PresetAtom.Flag(true),
    key("plugin.asymmetric_tiling.start_step") -> PresetAtom.Integer(0),
    key("plugin.asymmetric_tiling.stop_step") -> PresetAtom.Integer(-1))
  private val xyzNames = Vector(
    "x_type" -> "X Type", "y_type" -> "Y Type", "z_type" -> "Z Type",
    "x_values" -> "X Values", "y_values" -> "Y Values", "z_values" -> "Z Values")
  private val xyzSchema = schema(PresetMode.Txt2Img,
    (txt.fields ++ xyzNames.map { (control, label) =>
      spec(s"plugin.xyz_grid.$control", PresetAtom.Text(""), label)
    } ++ Vector(spec("plugin.xyz_grid.draw_legend", PresetAtom.Flag(true), "Draw legend")))*)
  private val xyzSchemas = schemas.updated(PresetMode.Txt2Img, xyzSchema)

  test("raw_json_accessors_keep_escaped_key_and_exact_numeric_lexemes") {
    val source = "{\"" + "\\u0053eed\":9007199254740993,\"fraction\":1.2500e+2,\"nil\":null,\"nested\":{\"x\":1}}"
    val raw = PresetJson.readSource(source.getBytes(UTF_8), limits).toOption.get
    val members = PresetJson.objectMembers(raw, limits).toOption.get
    assertEquals(members.map(_.name), Vector("Seed", "fraction", "nil", "nested"))
    assertEquals(members.head.keyToken.text, "\"\\u0053eed\"")
    assertEquals(members.head.value.text, "9007199254740993")
    assertEquals(PresetJson.scalarAtom(members.head.value, limits).toOption.get,
      Some(PresetAtom.Integer(9007199254740993L)))
    assertEquals(PresetJson.scalarAtom(members(1).value, limits).toOption.get,
      Some(PresetAtom.Decimal(BigDecimal("125.00"))))
    assertEquals(PresetJson.scalarAtom(members(2).value, limits).toOption.get, None)
    assertEquals(PresetJson.scalarAtom(members(3).value, limits).toOption.get, None)
    assert(PresetJson.objectMembers(members.head.value, limits).isLeft)
  }

  test("source_seed_long_is_exact_and_fractional_or_overflow_seed_is_not_applied") {
    val exact = imported("""{"Large":{"txt2img_seed":9223372036854775807}}""",
      "config-presets", Some(PresetMode.Txt2Img))
    assertEquals(exact.record.fields(key("generation.seed")), PresetAtom.Integer(Long.MaxValue))
    for token <- Vector("9223372036854775808", "1.5") do
      val result = PresetCompatibility.importRecords(
        document(s"""{"Bad":{"txt2img_seed":$token}}""", "config-presets", Some(PresetMode.Txt2Img)),
        schemas, Vector.empty)
      result match
        case Left(error) => assert(error.issues.nonEmpty, s"invalid seed $token has no error")
        case Right(records) =>
          val row = records.head
          assert(!row.record.fields.contains(key("generation.seed")),
            s"invalid seed $token became a native field")
          assert(row.issues.nonEmpty, s"invalid seed $token was skipped without a diagnostic")
          assert(row.record.source.get.originalFields.text.contains(token),
            s"invalid seed $token lost its source token")
  }

  test("source_format_and_config_mode_are_explicit_not_inferred_from_filename") {
    val bytes = """{"One":{"txt2img_steps":20}}""".getBytes(UTF_8)
    val raw = PresetJson.readSource(bytes, limits).toOption.get
    val wrong = PresetCompatibility.importRecords(
      SourceDocument("sdfx-native-presets", Some(PresetMode.Txt2Img), raw, "hash"), schemas, Vector.empty)
    val missingMode = PresetCompatibility.importRecords(
      SourceDocument("config-presets", None, raw, "hash"), schemas, Vector.empty)
    assert(wrong.isLeft)
    assert(missingMode.isLeft)
    assert(wrong.swap.toOption.get.issues.nonEmpty)
    assert(missingMode.swap.toOption.get.issues.nonEmpty)
  }

  test("config_standard_aliases_use_field_identity_and_source_mode") {
    val result = imported("""{"One":{"txt2img_steps":31,"txt2img_cfg_scale":6.5,"txt2img_restore_faces":true,"txt2img_enable_hr":true,"txt2img_subseed_show":false}}""",
      "config-presets", Some(PresetMode.Txt2Img))
    assertEquals(result.record.fields(key("generation.steps")), PresetAtom.Integer(31))
    assertEquals(result.record.fields(key("generation.cfgScale")), PresetAtom.Decimal(BigDecimal("6.5")))
    assertEquals(result.record.fields(key("plugin.face_restore.enabled")), PresetAtom.Flag(true))
    assertEquals(result.record.fields(key("generation.hiresEnabled")), PresetAtom.Flag(true))
    assertEquals(result.record.fields(key("generation.variationEnabled")), PresetAtom.Flag(false))
    assertEquals(result.record.source.get.mode, Some(PresetMode.Txt2Img))
    assertEquals(result.record.dirty, Set.empty[FieldKey])
  }

  test("standard_txt2img_ids_map_to_declared_fields_without_label_guessing") {
    val source = """{"One":{"txt2img_prompt":"p","txt2img_neg_prompt":"n","txt2img_sampling":"Euler","txt2img_scheduler":"Simple","txt2img_steps":30,"txt2img_width":768,"txt2img_height":432,"txt2img_batch_count":2,"txt2img_batch_size":3,"txt2img_cfg_scale":7.5,"txt2img_hr_scale":5,"txt2img_hires_steps":15,"txt2img_denoising_strength":0.3,"txt2img_hr_resize_x":1920,"txt2img_hr_resize_y":1080,"txt2img_subseed":99,"txt2img_subseed_strength":0.4,"hires_prompt":"high","hires_neg_prompt":"low"}}"""
    val result = imported(source, "config-presets", Some(PresetMode.Txt2Img))
    val expected = Map(
      "generation.prompt" -> PresetAtom.Text("p"),
      "generation.negativePrompt" -> PresetAtom.Text("n"),
      "generation.sampler" -> PresetAtom.Text("Euler"),
      "generation.scheduler" -> PresetAtom.Text("Simple"),
      "generation.steps" -> PresetAtom.Integer(30),
      "generation.width" -> PresetAtom.Integer(768),
      "generation.height" -> PresetAtom.Integer(432),
      "generation.batchCount" -> PresetAtom.Integer(2),
      "generation.batchSize" -> PresetAtom.Integer(3),
      "generation.cfgScale" -> PresetAtom.Decimal(BigDecimal("7.5")),
      "generation.hiresScale" -> PresetAtom.Decimal(BigDecimal(5)),
      "generation.hiresSteps" -> PresetAtom.Integer(15),
      "generation.denoisingStrength" -> PresetAtom.Decimal(BigDecimal("0.3")),
      "generation.hiresResizeX" -> PresetAtom.Integer(1920),
      "generation.hiresResizeY" -> PresetAtom.Integer(1080),
      "generation.subseed" -> PresetAtom.Integer(99),
      "generation.subseedStrength" -> PresetAtom.Decimal(BigDecimal("0.4")),
      "generation.hiresPrompt" -> PresetAtom.Text("high"),
      "generation.hiresNegative" -> PresetAtom.Text("low"))
    expected.foreach { (id, value) => assertEquals(result.record.fields.get(key(id)), Some(value), id) }
  }

  test("unavailable_optional_and_custom_fields_keep_raw_values_and_diagnostics") {
    val result = imported("""{"One":{"txt2img_steps":20,"txt2img_enable-checkbox":true,"txt2img_distilled_cfg_scale":3.5,"script_txt2img_adetailer_ad_model":"face","foreign":9007199254740993}}""",
      "config-presets", Some(PresetMode.Txt2Img))
    assertEquals(result.record.fields(key("generation.steps")), PresetAtom.Integer(20))
    for id <- Vector("txt2img_enable-checkbox", "txt2img_distilled_cfg_scale",
      "script_txt2img_adetailer_ad_model", "foreign") do
      assert(result.record.opaque.exists(_._1 == id), s"$id lost")
      assert(issueFor(result, id), s"$id needs a diagnostic")
    assertEquals(result.record.opaque.find(_._1 == "foreign").get._2.text, "9007199254740993")
  }

  test("synonym_conflict_never_picks_an_arbitrary_winner") {
    val result = imported("""{"One":{"txt2img_enable_hr":true,"txt2img_hr-checkbox":false}}""",
      "config-presets", Some(PresetMode.Txt2Img))
    assert(!result.record.fields.contains(key("generation.hiresEnabled")))
    assert(issueFor(result, "txt2img_enable_hr") || issueFor(result, "txt2img_hr-checkbox"))
    assert(result.record.source.get.originalFields.text.contains("txt2img_enable_hr"))
    assert(result.record.source.get.originalFields.text.contains("txt2img_hr-checkbox"))
  }

  test("utilities_label_collision_and_wrong_z_id_are_reported") {
    val collision = Vector(
      SourceBinding("preset-utilities", "Duplicated", key("generation.width")),
      SourceBinding("preset-utilities", "Duplicated", key("generation.height")))
    val result = PresetCompatibility.importRecords(
      document("{\"One\":{\"Duplicated\":640,\"Z type\":\"Seed\"}}", "preset-utilities", None),
      schemas, collision)
    assert(result.isLeft || result.toOption.get.head.issues.exists(i =>
      (i.path + i.message).contains("Duplicated") || (i.path + i.message).contains("Z type")))
    assert(result.isLeft || !result.toOption.get.head.record.fields.contains(key("generation.width")))
  }

  test("utilities_wrong_z_id_is_detected_with_real_xyz_destinations") {
    val wrong = SourceBinding("preset-utilities", "Z type", key("plugin.xyz_grid.y_type"))
    val result = PresetCompatibility.importRecords(
      document("""{"One":{"Z type":"Seed"}}""", "preset-utilities", None),
      xyzSchemas, Vector(wrong)).toOption.get.head
    assert(issueFor(result, "Z type"))
    assert(!result.record.fields.contains(key("plugin.xyz_grid.y_type")))
    assert(!result.record.fields.contains(key("plugin.xyz_grid.z_type")))
    assert(result.record.source.get.originalFields.text.contains("\"Z type\""))
  }

  test("utilities_xyz_supported_labels_reverse_for_dirty_and_fresh_records") {
    xyzNames.foreach { (control, label) =>
      val field = key(s"plugin.xyz_grid.$control")
      val initial = fresh(Map(field -> PresetAtom.Text("Seed")))
      val exported = PresetCompatibility.exportRecords(Vector(initial), "preset-utilities",
        None, Vector.empty, supportedOnly = false).toOption.get
      assert(sourceText(exported).contains(s"\"${label}\":\"Seed\""), label)
      val recognized = PresetCompatibility.importRecords(
        document(s"""{"One":{"${label}":"Seed"}}""", "preset-utilities", None),
        xyzSchemas, Vector.empty).toOption.get.head.record
      assertEquals(recognized.fields.get(field), Some(PresetAtom.Text("Seed")))
      val changed = recognized.copy(fields = recognized.fields.updated(field, PresetAtom.Text("Steps")),
        dirty = Set(field))
      val updated = PresetCompatibility.exportRecords(Vector(changed), "preset-utilities",
        None, Vector.empty, supportedOnly = false).toOption.get
      assert(sourceText(updated).contains(s"\"${label}\":\"Steps\""), label)
    }
  }

  test("config_xyz_reverse_uses_only_six_verified_axis_keys") {
    xyzNames.foreach { (control, _) =>
      val field = key(s"plugin.xyz_grid.$control")
      val output = PresetCompatibility.exportRecords(Vector(fresh(Map(field -> PresetAtom.Text("Seed")))),
        "config-presets", Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).toOption.get
      val id = s"script_txt2img_xyz_plot_${control}"
      assert(sourceText(output).contains(s"\"${id}\":\"Seed\""))
      val again = PresetCompatibility.importRecords(
        document(sourceText(output), "config-presets", Some(PresetMode.Txt2Img)),
        xyzSchemas, Vector.empty).toOption.get.head.record
      assertEquals(again.fields.get(field), Some(PresetAtom.Text("Seed")))
    }
    val unsupported = fresh(Map(key("plugin.xyz_grid.draw_legend") -> PresetAtom.Flag(false)))
    assert(PresetCompatibility.exportRecords(Vector(unsupported), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
  }

  test("all_189_and_232_commented_template_ids_are_classified_without_loss") {
    val stream = Option(getClass.getResourceAsStream("/sdfx/core/presets/config-template-ids.txt")).get
    val source = scala.io.Source.fromInputStream(stream, "UTF-8")
    val rows = try source.getLines().filterNot(_.startsWith("#")).map { line =>
      val parts = line.split("\\|", 2)
      parts(0) -> parts(1)
    }.toVector finally source.close()
    assertEquals(rows.count(_._1 == "txt2img"), 189)
    assertEquals(rows.count(_._1 == "img2img"), 232)
    assertEquals(rows.distinct.size, 421)
    rows.foreach { (modeName, id) =>
      val mode = if modeName == "txt2img" then PresetMode.Txt2Img else PresetMode.Img2Img
      val input = ujson.write(ujson.Obj("One" -> ujson.Obj(id -> ujson.Num(1))))
      val result = PresetCompatibility.importRecords(
        document(input, "config-presets", Some(mode)), schemas, Vector.empty)
      result match
        case Left(error) => assert(error.issues.nonEmpty, s"$modeName $id has no classification")
        case Right(records) =>
          val row = records.head
          assert(row.record.fields.nonEmpty || row.record.opaque.exists(_._1 == id) ||
            issueFor(row, id), s"$modeName $id silently lost")
          assert(row.record.source.get.originalFields.text.contains(ujson.write(id)),
            s"$modeName $id missing from provenance")
    }
  }

  test("source_index_uses_fixed_option_table_not_current_choice_order") {
    val indexed = imported("""{"One":{"resize_mode":2}}""",
      "config-presets", Some(PresetMode.Img2Img))
    assertEquals(indexed.record.fields.get(key("generation.resizeMode")),
      Some(PresetAtom.Text("Resize and fill")))
    val outOfRange = imported("""{"One":{"resize_mode":99}}""",
      "config-presets", Some(PresetMode.Img2Img))
    assert(!outOfRange.record.fields.contains(key("generation.resizeMode")))
    assert(issueFor(outOfRange, "resize_mode"))
  }

  test("fixed source choice honors declared choice atom type and membership") {
    val resize = key("generation.resizeMode")
    val source = document("""{"One":{"resize_mode":2}}""",
      "config-presets", Some(PresetMode.Img2Img))
    def withChoice(default: PresetAtom, options: Vector[PresetAtom]): CompatibilityResult =
      val field = PresetFieldSpec(resize, "Resize mode", PresetFieldKind.Choice,
        default, options, None, None)
      PresetCompatibility.importRecords(source,
        Map(PresetMode.Img2Img -> schema(PresetMode.Img2Img, field)), Vector.empty)
        .toOption.get.head

    val numeric = withChoice(PresetAtom.Integer(0),
      Vector(PresetAtom.Integer(2), PresetAtom.Integer(0), PresetAtom.Integer(1)))
    assertEquals(numeric.record.fields.get(resize), Some(PresetAtom.Integer(2)))
    assert(numeric.issues.isEmpty)
    val absentNumber = withChoice(PresetAtom.Integer(0),
      Vector(PresetAtom.Integer(0), PresetAtom.Integer(1)))
    assert(!absentNumber.record.fields.contains(resize))
    assert(absentNumber.record.opaque.exists(_._1 == "resize_mode"))
    assert(issueFor(absentNumber, "resize_mode"))

    val textual = withChoice(PresetAtom.Text("Just resize"),
      Vector(PresetAtom.Text("Resize and fill"), PresetAtom.Text("Just resize"),
        PresetAtom.Text("Crop and resize")))
    assertEquals(textual.record.fields.get(resize), Some(PresetAtom.Text("Resize and fill")))
    val absentLabel = withChoice(PresetAtom.Text("Just resize"),
      Vector(PresetAtom.Text("Just resize"), PresetAtom.Text("Crop and resize")))
    assert(!absentLabel.record.fields.contains(resize))
    assert(absentLabel.record.opaque.exists(_._1 == "resize_mode"))
    assert(issueFor(absentLabel, "resize_mode"))
  }

  test("dirty_config_resize_mode_reexports_bare_id_for_choice_schema") {
    val original = imported("""{"One":{"resize_mode":2}}""",
      "config-presets", Some(PresetMode.Img2Img)).record
    val changed = original.copy(
      fields = original.fields.updated(key("generation.resizeMode"), PresetAtom.Text("Crop and resize")),
      dirty = Set(key("generation.resizeMode")))
    val output = PresetCompatibility.exportRecords(Vector(changed), "config-presets",
      Some(PresetMode.Img2Img), Vector.empty, supportedOnly = false).toOption.get
    assert(sourceText(output).contains("\"resize_mode\":1"))
    assert(!sourceText(output).contains("img2img_resize_mode"))
    val importedAgain = PresetCompatibility.importRecords(
      document(sourceText(output), "config-presets", Some(PresetMode.Img2Img)),
      schemas, Vector.empty).toOption.get.head.record
    assertEquals(importedAgain.fields.get(key("generation.resizeMode")),
      Some(PresetAtom.Text("Crop and resize")))
  }

  test("fresh_config_resize_mode_reexports_bare_id_for_integer_schema") {
    val record = fresh(Map(key("generation.resizeMode") -> PresetAtom.Integer(2)))
    val output = PresetCompatibility.exportRecords(Vector(record), "config-presets",
      Some(PresetMode.Img2Img), Vector.empty, supportedOnly = false).toOption.get
    assert(sourceText(output).contains("\"resize_mode\":2"))
    assert(!sourceText(output).contains("img2img_resize_mode"))
    val importedAgain = PresetCompatibility.importRecords(
      document(sourceText(output), "config-presets", Some(PresetMode.Img2Img)),
      numericSchemas, Vector.empty).toOption.get.head.record
    assertEquals(importedAgain.fields.get(key("generation.resizeMode")),
      Some(PresetAtom.Integer(2)))
  }

  test("utilities_choice_export_uses_labels_even_with_integer_and_flag_native_fields") {
    val native = Map[FieldKey, PresetAtom](
      key("generation.resizeMode") -> PresetAtom.Integer(2),
      key("generation.inpaintingFill") -> PresetAtom.Integer(1),
      key("generation.maskInvert") -> PresetAtom.Flag(true),
      key("generation.inpaintFullRes") -> PresetAtom.Flag(true))
    val output = PresetCompatibility.exportRecords(Vector(fresh(native)),
      "preset-utilities", None, Vector.empty, supportedOnly = false).toOption.get
    val json = sourceText(output)
    assert(json.contains("\"Resize mode\":\"Resize and fill\""))
    assert(json.contains("\"Masked content\":\"original\""))
    assert(json.contains("\"Mask mode\":\"Inpaint not masked\""))
    assert(json.contains("\"Inpaint area\":\"Only masked\""))
    val again = PresetCompatibility.importRecords(
      document(json, "preset-utilities", None), numericSchemas, Vector.empty).toOption.get.head.record
    assertEquals(again.fields, native)
  }

  test("hires_same_sampler_is_native_empty_string_and_roundtrips") {
    val record = imported("{\"One\":{\"hr_sampler\":\"Use same sampler\"}}",
      "config-presets", Some(PresetMode.Txt2Img)).record
    assertEquals(record.fields.get(key("generation.hiresSampler")), Some(PresetAtom.Text("")))
    val changed = record.copy(dirty = Set(key("generation.hiresSampler")))
    val output = PresetCompatibility.exportRecords(Vector(changed), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).toOption.get
    assert(new String(output, UTF_8).contains("\"hr_sampler\":\"Use same sampler\""))
  }

  test("tiling_true_requires_full_equivalent_compound_not_only_enabled") {
    val result = imported("{\"One\":{\"Tiling\":true}}", "preset-utilities", None)
    val fields = result.record.fields
    assertEquals(fields.get(key("plugin.asymmetric_tiling.enabled")), Some(PresetAtom.Flag(true)))
    assertEquals(fields.get(key("plugin.asymmetric_tiling.tile_x")), Some(PresetAtom.Flag(true)))
    assertEquals(fields.get(key("plugin.asymmetric_tiling.tile_y")), Some(PresetAtom.Flag(true)))
    assertEquals(fields.get(key("plugin.asymmetric_tiling.start_step")), Some(PresetAtom.Integer(0)))
    assertEquals(fields.get(key("plugin.asymmetric_tiling.stop_step")), Some(PresetAtom.Integer(-1)))
    val disabled = imported("{\"One\":{\"Tiling\":false}}", "preset-utilities", None)
    assertEquals(disabled.record.fields,
      Map(key("plugin.asymmetric_tiling.enabled") -> PresetAtom.Flag(false)))
  }

  test("utilities_default_tracking_carries_all_five_tiling_constituents") {
    val tracked = PresetDefaults.library(schemas).collections.find(_.id == "utilities").get.tracked
    assert(tilingFields.keySet.subsetOf(tracked))
    val importedTrue = imported("""{"One":{"Tiling":true}}""", "preset-utilities", None)
    assert(tilingFields.keySet.subsetOf(importedTrue.record.fields.keySet))
  }

  test("fresh_full_tiling_true_exports_one_semantically_equivalent_source_value") {
    val output = PresetCompatibility.exportRecords(Vector(fresh(tilingFields)),
      "preset-utilities", None, Vector.empty, supportedOnly = false).toOption.get
    assert(sourceText(output).contains("\"Tiling\":true"))
    val again = PresetCompatibility.importRecords(
      document(sourceText(output), "preset-utilities", None), schemas, Vector.empty).toOption.get.head.record
    assertEquals(again.fields.filter((k, _) => tilingFields.contains(k)), tilingFields)
  }

  test("tiling_true_missing_constituents_or_partial_drafts_never_exports_true") {
    val enabledOnly = fresh(Map(key("plugin.asymmetric_tiling.enabled") -> PresetAtom.Flag(true)))
    val partial = fresh(tilingFields.updated(key("plugin.asymmetric_tiling.tile_y"), PresetAtom.Flag(false)))
    val timed = fresh(tilingFields.updated(key("plugin.asymmetric_tiling.start_step"), PresetAtom.Integer(2)))
    for record <- Vector(enabledOnly, partial, timed) do
      assert(PresetCompatibility.exportRecords(Vector(record), "preset-utilities",
        None, Vector.empty, supportedOnly = false).isLeft)
      val supported = PresetCompatibility.exportRecords(Vector(record), "preset-utilities",
        None, Vector.empty, supportedOnly = true)
      assert(supported.isLeft || !sourceText(supported.toOption.get).contains("\"Tiling\":true"))
  }

  test("tiling_false_preserves_disabled_drafts_in_native_and_requires_explicit_source_omission") {
    val disabled = fresh(tilingFields.updated(key("plugin.asymmetric_tiling.enabled"), PresetAtom.Flag(false))
      .updated(key("plugin.asymmetric_tiling.tile_y"), PresetAtom.Flag(false)))
    assertEquals(disabled.fields(key("plugin.asymmetric_tiling.tile_y")), PresetAtom.Flag(false))
    assert(PresetCompatibility.exportRecords(Vector(disabled), "preset-utilities",
      None, Vector.empty, supportedOnly = false).isLeft)
    val supported = PresetCompatibility.exportRecords(Vector(disabled), "preset-utilities",
      None, Vector.empty, supportedOnly = true)
    assert(supported.isLeft || !sourceText(supported.toOption.get).contains("\"Tiling\":true"))
  }

  test("dirty removal of imported false tiling emits an empty source object") {
    val enabled = key("plugin.asymmetric_tiling.enabled")
    val original = imported("""{"One":{"Tiling":false}}""", "preset-utilities", None).record
    assertEquals(original.fields, Map(enabled -> PresetAtom.Flag(false)))
    val removed = original.copy(fields = original.fields - enabled, dirty = Set(enabled))
    val output = PresetCompatibility.exportRecords(Vector(removed),
      "preset-utilities", None, Vector.empty, supportedOnly = false).toOption.get
    assertEquals(sourceText(output), """{"One":{}}""")
    val again = imported(sourceText(output), "preset-utilities", None).record
    assertEquals(again.fields, Map.empty[FieldKey, PresetAtom])
  }

  test("dirty removal of complete imported true tiling emits an empty source object") {
    val original = imported("""{"One":{"Tiling":true}}""", "preset-utilities", None).record
    assertEquals(original.fields, tilingFields)
    val partialRemoval = original.copy(fields = original.fields - key("plugin.asymmetric_tiling.enabled"),
      dirty = Set(key("plugin.asymmetric_tiling.enabled")))
    val rejected = PresetCompatibility.exportRecords(Vector(partialRemoval),
      "preset-utilities", None, Vector.empty, supportedOnly = false).swap.toOption.get
    assert(rejected.issues.exists(issue =>
      tilingFields.keySet.exists(_.render == issue.path)))
    val removed = original.copy(fields = original.fields -- tilingFields.keySet,
      dirty = tilingFields.keySet)
    val output = PresetCompatibility.exportRecords(Vector(removed),
      "preset-utilities", None, Vector.empty, supportedOnly = false).toOption.get
    assertEquals(sourceText(output), """{"One":{}}""")
    val again = imported(sourceText(output), "preset-utilities", None).record
    assertEquals(again.fields, Map.empty[FieldKey, PresetAtom])
  }

  test("plain alias to tiling enabled roundtrips fresh true and false independently") {
    val enabled = key("plugin.asymmetric_tiling.enabled")
    val binding = SourceBinding("preset-utilities", "my-enable", enabled)
    for flag <- Vector(true, false) do
      val output = PresetCompatibility.exportRecords(
        Vector(fresh(Map(enabled -> PresetAtom.Flag(flag)))),
        "preset-utilities", None, Vector(binding), supportedOnly = false).toOption.get
      assertEquals(sourceText(output), s"""{"One":{"my-enable":$flag}}""")
      val again = imported(sourceText(output), "preset-utilities", None, Vector(binding)).record
      assertEquals(again.fields, Map(enabled -> PresetAtom.Flag(flag)))
  }

  test("dirty plain alias to tiling enabled replaces its original without inventing Tiling") {
    val enabled = key("plugin.asymmetric_tiling.enabled")
    val binding = SourceBinding("preset-utilities", "my-enable", enabled)
    for (before, after) <- Vector(true -> false, false -> true) do
      val original = imported(s"""{"One":{"my-enable":$before}}""",
        "preset-utilities", None, Vector(binding)).record
      assertEquals(original.fields, Map(enabled -> PresetAtom.Flag(before)))
      val changed = original.copy(fields = original.fields.updated(enabled, PresetAtom.Flag(after)),
        dirty = Set(enabled))
      val output = PresetCompatibility.exportRecords(Vector(changed),
        "preset-utilities", None, Vector(binding), supportedOnly = false).toOption.get
      assertEquals(sourceText(output), s"""{"One":{"my-enable":$after}}""")
      val again = imported(sourceText(output), "preset-utilities", None, Vector(binding)).record
      assertEquals(again.fields, Map(enabled -> PresetAtom.Flag(after)))
  }

  test("plain alias to a tiling constituent exports without compound enabled") {
    val tileX = key("plugin.asymmetric_tiling.tile_x")
    val binding = SourceBinding("preset-utilities", "my-tile-x", tileX)
    val freshOutput = PresetCompatibility.exportRecords(
      Vector(fresh(Map(tileX -> PresetAtom.Flag(true)))),
      "preset-utilities", None, Vector(binding), supportedOnly = false).toOption.get
    assertEquals(sourceText(freshOutput), """{"One":{"my-tile-x":true}}""")
    val original = imported(sourceText(freshOutput), "preset-utilities", None, Vector(binding)).record
    assertEquals(original.fields, Map(tileX -> PresetAtom.Flag(true)))
    val changed = original.copy(fields = original.fields.updated(tileX, PresetAtom.Flag(false)),
      dirty = Set(tileX))
    val output = PresetCompatibility.exportRecords(Vector(changed),
      "preset-utilities", None, Vector(binding), supportedOnly = false).toOption.get
    assertEquals(sourceText(output), """{"One":{"my-tile-x":false}}""")
    val again = imported(sourceText(output), "preset-utilities", None, Vector(binding)).record
    assertEquals(again.fields, Map(tileX -> PresetAtom.Flag(false)))
  }

  test("imported true tiling changed to false cannot silently discard four saved drafts") {
    val enabled = key("plugin.asymmetric_tiling.enabled")
    val original = imported("""{"One":{"Tiling":true}}""", "preset-utilities", None).record
    assertEquals(original.fields, tilingFields)
    val disabled = original.copy(fields = original.fields.updated(enabled, PresetAtom.Flag(false)),
      dirty = Set(enabled))
    assertEquals(disabled.fields.keySet, tilingFields.keySet)
    assert(PresetCompatibility.exportRecords(Vector(disabled), "preset-utilities",
      None, Vector.empty, supportedOnly = false).isLeft)
    val supported = PresetCompatibility.exportRecords(Vector(disabled), "preset-utilities",
      None, Vector.empty, supportedOnly = true).toOption.get
    assertEquals(sourceText(supported), """{"One":{"Tiling":false}}""")
    val again = imported(sourceText(supported), "preset-utilities", None).record
    assertEquals(again.fields, Map(enabled -> PresetAtom.Flag(false)))
    assertEquals(disabled.fields.filter((field, _) => field != enabled),
      tilingFields - enabled)
  }

  test("unedited_known_alias_retains_original_raw_value_on_export") {
    val value = """{"One":{"txt2img_hr-checkbox":true,"txt2img_steps":2.0e1,"unknown":9007199254740993}}"""
    val record = imported(value, "config-presets", Some(PresetMode.Txt2Img)).record
    val output = PresetCompatibility.exportRecords(Vector(record), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).toOption.get
    val json = new String(output, UTF_8)
    assert(json.contains("\"txt2img_hr-checkbox\":true"))
    assert(json.contains("2.0e1"))
    assert(json.contains("9007199254740993"))
    assert(!json.contains("\"txt2img_enable_hr\""))
  }

  test("edited_custom_alias_uses_explicit_reverse_binding_and_reimports") {
    val binding = SourceBinding("config-presets", "my-height", key("generation.height"))
    val original = imported("""{"One":{"my-height":512}}""", "config-presets",
      Some(PresetMode.Txt2Img), Vector(binding)).record
    assertEquals(original.fields(key("generation.height")), PresetAtom.Integer(512))
    val edited = original.copy(fields = original.fields.updated(key("generation.height"), PresetAtom.Integer(768)),
      dirty = Set(key("generation.height")))
    val output = PresetCompatibility.exportRecords(Vector(edited), "config-presets",
      Some(PresetMode.Txt2Img), Vector(binding), supportedOnly = false).toOption.get
    assert(new String(output, UTF_8).contains("\"my-height\":768"))
    val again = PresetCompatibility.importRecords(
      document(new String(output, UTF_8), "config-presets", Some(PresetMode.Txt2Img)),
      schemas, Vector(binding)).toOption.get.head.record
    assertEquals(again.fields(key("generation.height")), PresetAtom.Integer(768))
  }

  test("custom_plain_script_alias_roundtrips_native_id_without_title_conversion") {
    val binding = SourceBinding("config-presets", "my-script", key("script.selected"))
    val custom = schemas.updated(PresetMode.Txt2Img,
      schema(PresetMode.Txt2Img, (txt.fields :+
        spec("script.selected", PresetAtom.Text(""), "Selected script"))*))
    val original = PresetCompatibility.importRecords(
      document("""{"One":{"my-script":"xyz_grid"}}""", "config-presets", Some(PresetMode.Txt2Img)),
      custom, Vector(binding)).toOption.get.head.record
    assertEquals(original.fields.get(key("script.selected")), Some(PresetAtom.Text("xyz_grid")))
    val changed = original.copy(dirty = Set(key("script.selected")))
    val output = PresetCompatibility.exportRecords(Vector(changed), "config-presets",
      Some(PresetMode.Txt2Img), Vector(binding), supportedOnly = false).toOption.get
    assert(sourceText(output).contains("\"my-script\":\"xyz_grid\""))
    val again = PresetCompatibility.importRecords(
      document(sourceText(output), "config-presets", Some(PresetMode.Txt2Img)),
      custom, Vector(binding)).toOption.get.head.record
    assertEquals(again.fields.get(key("script.selected")), Some(PresetAtom.Text("xyz_grid")))
  }

  test("custom_plain_hires_sampler_alias_preserves_native_empty_string") {
    val binding = SourceBinding("config-presets", "my-sampler", key("generation.hiresSampler"))
    val original = imported("""{"One":{"my-sampler":""}}""", "config-presets",
      Some(PresetMode.Txt2Img), Vector(binding)).record
    val output = PresetCompatibility.exportRecords(
      Vector(original.copy(dirty = Set(key("generation.hiresSampler")))),
      "config-presets", Some(PresetMode.Txt2Img), Vector(binding), supportedOnly = false).toOption.get
    assert(sourceText(output).contains("\"my-sampler\":\"\""))
    assert(!sourceText(output).contains("Use same sampler"))
  }

  test("source_export_refuses_native_only_field_without_explicit_omission") {
    val record = imported("""{"One":{"txt2img_steps":20}}""", "config-presets",
      Some(PresetMode.Txt2Img)).record
    val native = key("plugin.my%2Eplugin.private")
    val changed = record.copy(fields = record.fields.updated(native, PresetAtom.Text("draft")),
      dirty = Set(native))
    assert(PresetCompatibility.exportRecords(Vector(changed), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
    val reduced = PresetCompatibility.exportRecords(Vector(changed), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = true).toOption.get
    assert(!new String(reduced, UTF_8).contains("draft"))
  }

  test("reverse_binding_rejects_one_source_key_with_two_targets") {
    val bindings = Vector(
      SourceBinding("config-presets", "shared", key("generation.width")),
      SourceBinding("config-presets", "shared", key("generation.height")))
    val record = fresh(Map(key("generation.width") -> PresetAtom.Integer(640)))
    assert(PresetCompatibility.exportRecords(Vector(record), "config-presets",
      Some(PresetMode.Txt2Img), bindings, supportedOnly = false).isLeft)
    val supported = PresetCompatibility.exportRecords(Vector(record), "config-presets",
      Some(PresetMode.Txt2Img), bindings, supportedOnly = true)
    assert(supported.isLeft || !sourceText(supported.toOption.get).contains("\"shared\""))
  }

  test("reverse_binding_rejects_custom_target_on_standard_source_key") {
    val bindings = Vector(SourceBinding("config-presets", "txt2img_width", key("generation.height")))
    val record = fresh(Map(key("generation.height") -> PresetAtom.Integer(768)))
    assert(PresetCompatibility.exportRecords(Vector(record), "config-presets",
      Some(PresetMode.Txt2Img), bindings, supportedOnly = false).isLeft)
    val supported = PresetCompatibility.exportRecords(Vector(record), "config-presets",
      Some(PresetMode.Txt2Img), bindings, supportedOnly = true)
    assert(supported.isLeft || !sourceText(supported.toOption.get).contains("\"txt2img_width\""))
  }

  test("source_export_rejects_invalid_names_and_2049_opaque_fields") {
    val blank = fresh(Map(key("generation.width") -> PresetAtom.Integer(512)), "")
    assert(PresetCompatibility.exportRecords(Vector(blank), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
    val raw = PresetJson.readSource("1".getBytes(UTF_8), limits).toOption.get
    val excess = fresh(Map.empty).copy(
      opaque = Vector.tabulate(2049)(i => s"custom-$i" -> raw))
    assert(PresetCompatibility.exportRecords(Vector(excess), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
  }

  test("source_export_utf8_scalar_boundary_is_byte_based") {
    val exact = fresh(Map(key("generation.prompt") -> PresetAtom.Text("é" * 131072)))
    val output = PresetCompatibility.exportRecords(Vector(exact), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false)
    assert(output.isRight)
    val tooLong = exact.copy(fields =
      Map(key("generation.prompt") -> PresetAtom.Text("é" * 131073)))
    assert(PresetCompatibility.exportRecords(Vector(tooLong), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
  }

  test("source_export_aggregate_and_decimal_limits_reject_without_truncation") {
    // The same modest 5 KiB atom is shared by all records; this is a bounded
    // acceptance check, while before-allocation behavior also needs code review.
    val shared = PresetAtom.Text("x" * 5000)
    val records = Vector.tabulate(1000) { i =>
      fresh(Map(key("generation.prompt") -> shared), s"record-$i")
    }
    assert(PresetCompatibility.exportRecords(records, "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
    val coefficient = BigDecimal(new java.math.BigDecimal("9" * 262145))
    val decimal = fresh(Map(key("generation.cfgScale") -> PresetAtom.Decimal(coefficient)))
    assert(PresetCompatibility.exportRecords(Vector(decimal), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).isLeft)
  }

  test("oversized_registered_schema_is_rejected_before_source_mapping") {
    val huge = schema(PresetMode.Txt2Img,
      Vector.fill(4097)(spec("generation.width", PresetAtom.Integer(512)))*)
    val input = document("""{"One":{"txt2img_width":640}}""",
      "config-presets", Some(PresetMode.Txt2Img))
    assert(PresetCompatibility.importRecords(input,
      schemas.updated(PresetMode.Txt2Img, huge), Vector.empty).isLeft)
  }

  test("verified_custom_endpoint_resolves_exact_declared_rename_only") {
    val target = key("plugin.example.enabled")
    val custom = schemas.updated(PresetMode.Txt2Img,
      schema(PresetMode.Txt2Img, (txt.fields :+
        spec("plugin.example.enabled", PresetAtom.Flag(false), "Example enabled"))*))
    val binding = SourceBinding("config-presets", "MD-t2i-enabled-checkbox", target)
    val oldName = PresetCompatibility.importRecords(
      document("""{"One":{"MD-t2i-enabled":true}}""",
        "config-presets", Some(PresetMode.Txt2Img)),
      custom, Vector(binding)).toOption.get.head
    assertEquals(oldName.record.fields.get(target), Some(PresetAtom.Flag(true)))
    assert(oldName.record.source.get.originalFields.text.contains("\"MD-t2i-enabled\""))
    val unbound = PresetCompatibility.importRecords(
      document("""{"One":{"MD-t2i-enabled":true}}""",
        "config-presets", Some(PresetMode.Txt2Img)),
      custom, Vector.empty).toOption.get.head
    assert(!unbound.record.fields.contains(target))
    assert(unbound.record.opaque.exists(_._1 == "MD-t2i-enabled"))
    assert(issueFor(unbound, "MD-t2i-enabled"))
  }

  test("verified_custom_rename_conflict_does_not_select_either_value") {
    val target = key("plugin.example.enabled")
    val custom = schemas.updated(PresetMode.Txt2Img,
      schema(PresetMode.Txt2Img, (txt.fields :+
        spec("plugin.example.enabled", PresetAtom.Flag(false), "Example enabled"))*))
    val binding = SourceBinding("config-presets", "MD-t2i-enabled-checkbox", target)
    val result = PresetCompatibility.importRecords(
      document("""{"One":{"MD-t2i-enabled":true,"MD-t2i-enabled-checkbox":false}}""",
        "config-presets", Some(PresetMode.Txt2Img)),
      custom, Vector(binding)).toOption.get.head
    assert(!result.record.fields.contains(target))
    assert(result.issues.exists(_.code == "alias-conflict"))
    assert(result.record.source.get.originalFields.text.contains("MD-t2i-enabled"))
    assert(result.record.source.get.originalFields.text.contains("MD-t2i-enabled-checkbox"))
  }

  test("utilities_import_retains_known_values_for_other_destination_modes") {
    val result = imported("""{"Mixed":{"Upscale by":2.5,"Mask blur":8,"Denoising strength":0.4}}""",
      "preset-utilities", None)
    assertEquals(result.record.fields(key("generation.hiresScale")), PresetAtom.Decimal(BigDecimal("2.5")))
    assertEquals(result.record.fields(key("generation.maskBlur")), PresetAtom.Integer(8))
    assertEquals(result.record.fields(key("generation.denoisingStrength")), PresetAtom.Decimal(BigDecimal("0.4")))
    assertEquals(result.record.source.get.mode, None)
  }

  test("utilities_source_seed_and_extra_do_not_lose_disabled_drafts") {
    val result = imported("""{"One":{"Seed":9007199254740993,"Variation seed":99,"Variation strength":0.4,"Extra":false}}""",
      "preset-utilities", None)
    assertEquals(result.record.fields(key("generation.seed")), PresetAtom.Integer(9007199254740993L))
    assertEquals(result.record.fields(key("generation.subseed")), PresetAtom.Integer(99))
    assertEquals(result.record.fields(key("generation.subseedStrength")), PresetAtom.Decimal(BigDecimal("0.4")))
    assertEquals(result.record.fields(key("generation.variationEnabled")), PresetAtom.Flag(false))
  }

  test("defaults_include_full_config_names_order_and_unsupported_source_values") {
    val library = PresetDefaults.library(schemas)
    val t = library.collections.find(_.id == "config-txt2img").get
    val i = library.collections.find(_.id == "config-img2img").get
    assertEquals(t.presets.size, 14)
    assertEquals(i.presets.size, 4)
    assertEquals(t.presets.map(_.name), Vector(
      "None", "SD1.5 - 512x512", "SD2.1 - 768x768", "SDXL --- 1024x1024",
      "SDXL --- 1024x1024 with Refiner",
      "Flux.1 Dev - 256x256 to 1920x1080 (0.1 to 2.0 megapixels), 20 steps, CFG 1, Distilled CFG 3.5, Euler Simple",
      "Flux.1 Schnell - 256x256 to 1920x1080 (0.1 to 2.0 megapixels), 4 steps, CFG 1, Distilled CFG 3.5, Euler Simple",
      "High res ----------- [Hires fix - Upscale by: 2, Denoising: 0.3, Hires steps: 10]",
      "SD1.5 Low quality ------ steps: 10, batch size: 4, DPM++ 2M",
      "SD1.5 Medium quality - steps: 15, batch size: 4, DPM++ 2M",
      "SD1.5 High quality ------ steps: 30, batch size: 4, DPM++ 2S a",
      "SD1.5 - 1080p --- 432x768 -> 1920x1080, [Hires fix - Upscale by: 2.5, Denoising: 0.4, Hires steps: 10]",
      "SD1.5 - 1440p --- 432x768 -> 2560x1440, [Hires fix - Upscale by: 3.3334, Denoising: 0.35, Hires steps: 10]",
      "SD1.5 - 4k -------- 432x768 -> 3840x2160, [Upscale by: 5, Denoising: 0.3, Hires steps: 15]"))
    assertEquals(i.presets.map(_.name), Vector("None",
      "Low denoising ------- denoising: 0.25, steps: 20, DPM++ 2M",
      "Medium denoising -- denoising: 0.40, steps: 20, DPM++ 2M",
      "High denoising ------- denoising: 0.75, steps: 30, DPM++ 2M"))
    assertEquals(t.presets.head.fields, Map.empty[FieldKey, PresetAtom])
    assertEquals(i.presets.head.fields, Map.empty[FieldKey, PresetAtom])
    val refiner = t.presets.find(_.name.endsWith("with Refiner")).get
    assert(refiner.opaque.exists(_._1 == "txt2img_enable-checkbox"))
    assertEquals(t.presets.last.fields.get(key("generation.hiresScale")),
      Some(PresetAtom.Decimal(BigDecimal(5))))
    assertEquals(t.presets.last.fields.get(key("generation.width")), Some(PresetAtom.Integer(768)))
    assertEquals(t.presets.last.fields.get(key("generation.height")), Some(PresetAtom.Integer(432)))
    val scaleSpec = txt.fields.find(_.key == key("generation.hiresScale")).get
    assertEquals(scaleSpec.max, Some(BigDecimal(5)))
    val exported4k = PresetCompatibility.exportRecords(Vector(t.presets.last), "config-presets",
      Some(PresetMode.Txt2Img), Vector.empty, supportedOnly = false).toOption.get
    val imported4k = PresetCompatibility.importRecords(
      document(new String(exported4k, UTF_8), "config-presets", Some(PresetMode.Txt2Img)),
      schemas, Vector.empty).toOption.get.head
    assertEquals(imported4k.record.fields.get(key("generation.hiresScale")),
      Some(PresetAtom.Decimal(BigDecimal(5))))
    assert(t.presets.last.source.get.originalFields.text.contains("\"txt2img_hr_scale\": 5") ||
      t.presets.last.source.get.originalFields.text.contains("\"txt2img_hr_scale\":5"))
    for name <- t.presets.map(_.name).filter(_.startsWith("Flux.1")) do
      assert(t.presets.find(_.name == name).get.opaque.exists(_._1 == "txt2img_distilled_cfg_scale"))
  }

  test("config_default_tracking_excludes_prompt_and_seed_until_selected") {
    val library = PresetDefaults.library(schemas)
    val config = library.collections.find(_.id == "config-txt2img").get
    val utilities = library.collections.find(_.id == "utilities").get
    assert(!config.tracked.contains(key("generation.prompt")))
    assert(!config.tracked.contains(key("generation.seed")))
    assert(utilities.tracked.contains(key("generation.prompt")))
    assert(utilities.tracked.contains(key("generation.seed")))
    assert(utilities.tracked.contains(key("generation.variationEnabled")))
    assertEquals(utilities.presets.map(_.name), Vector("Reset common controls"))
    assert(!utilities.presets.head.fields.contains(key("generation.denoisingStrength")))
    assertEquals(utilities.presets.head.fields.get(key("generation.variationEnabled")),
      Some(PresetAtom.Flag(false)))
  }

  test("mode_relative_default_difference_omits_denoise_from_shared_reset") {
    val library = PresetDefaults.library(schemas)
    val reset = library.collections.find(_.id == "utilities").get.presets.head
    assert(!reset.fields.contains(key("generation.denoisingStrength")))
    assertEquals(reset.fields(key("generation.cfgScale")), PresetAtom.Decimal(BigDecimal(7)))
  }
