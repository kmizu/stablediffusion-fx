package sdfx.core.presets

import java.nio.charset.StandardCharsets.UTF_8

class PresetExportDiagnosticsSuite extends munit.FunSuite:
  private val config = "config-presets"
  private val utilities = "preset-utilities"
  private val txt = Some(PresetMode.Txt2Img)
  private val limits = PresetLimits()
  private def key(id: String): FieldKey = FieldKey.parse(id).toOption.get
  private def fresh(name: String, fields: Map[FieldKey, PresetAtom]): PresetRecord =
    PresetRecord(PresetRecord.initialId("config-txt2img", name), name, fields,
      None, Set.empty, Vector.empty, Vector.empty)
  private def nativeFields(count: Int): Map[FieldKey, PresetAtom] =
    (0 until count).map(n => key(f"plugin.native_only.field$n%04d") -> PresetAtom.Integer(n.toLong)).toMap
  private def sourced(name: String, raw: String,
      fields: Map[FieldKey, PresetAtom] = Map.empty): PresetRecord =
    fresh(name, fields).copy(source = Some(PresetSource(config, name, txt, RawJson.validated(raw))))
  private def diagnostics(records: Vector[PresetRecord], format: String = config,
      mode: Option[PresetMode] = txt, bindings: Vector[SourceBinding] = Vector.empty)
      : Either[PresetError, Vector[PresetIssue]] =
    PresetCompatibility.exportRecordIssues(records, format, mode, bindings)
  private def omissions(records: Vector[PresetRecord], format: String = config,
      mode: Option[PresetMode] = txt, bindings: Vector[SourceBinding] = Vector.empty): Vector[PresetIssue] =
    diagnostics(records, format, mode, bindings) match
      case Right(issues) => issues
      case Left(error) => fail(s"expected omission review, got $error")
  private def hardFailure(result: Either[PresetError, Vector[PresetIssue]], code: String): Unit =
    result match
      case Left(error) => assert(error.issues.exists(_.code == code), error.toString)
      case Right(issues) => fail(s"expected hard $code error, got ${issues.size} omissions")
  private def identifies(issue: PresetIssue, record: String, field: String): Boolean =
    val context = issue.path + " " + issue.message
    context.contains(record) && context.contains(field)
  private def exported(records: Vector[PresetRecord], supportedOnly: Boolean,
      format: String = config, mode: Option[PresetMode] = txt,
      bindings: Vector[SourceBinding] = Vector.empty): String =
    new String(PresetCompatibility.exportRecords(records, format, mode, bindings, supportedOnly).toOption.get, UTF_8)
  private val tiling = Map[FieldKey, PresetAtom](
    key("plugin.asymmetric_tiling.enabled") -> PresetAtom.Flag(true),
    key("plugin.asymmetric_tiling.tile_x") -> PresetAtom.Flag(true),
    key("plugin.asymmetric_tiling.tile_y") -> PresetAtom.Flag(true),
    key("plugin.asymmetric_tiling.start_step") -> PresetAtom.Integer(0),
    key("plugin.asymmetric_tiling.stop_step") -> PresetAtom.Integer(-1))

  test("all 210 native-only omissions include the final key and their record context") {
    val fields = nativeFields(210)
    val issues = omissions(Vector(fresh("Many fields", fields)))
    assertEquals(issues.size, 210)
    assert(issues.forall(_.code == "unsupported"))
    fields.keys.foreach(k => assertEquals(issues.count(identifies(_, "Many fields", k.render)), 1))
    assert(issues.exists(identifies(_, "Many fields", "plugin.native_only.field0209")))
  }

  test("collect every named record even when earlier records already have omissions") {
    val records = Vector(fresh("First record", nativeFields(210)), fresh("Second record", nativeFields(211)))
    val issues = omissions(records)
    assertEquals(issues.size, 421)
    records.foreach(r => r.fields.keys.foreach(k =>
      assertEquals(issues.count(identifies(_, r.name, k.render)), 1)))
  }

  test("supported native fields and complete tiling have no omissions in either source format") {
    val record = fresh("Supported", tiling ++ Map(
      key("generation.seed") -> PresetAtom.Integer(9007199254740993L),
      key("generation.prompt") -> PresetAtom.Text("literal prompt"),
      key("generation.steps") -> PresetAtom.Integer(30)))
    assertEquals(omissions(Vector(record)), Vector.empty[PresetIssue])
    assertEquals(omissions(Vector(record), utilities, None), Vector.empty[PresetIssue])
    assertEquals(omissions(Vector.empty), Vector.empty[PresetIssue])
  }

  test("untouched provenance and opaque numeric lexemes stay representable and byte-identical") {
    val raw = """{"txt2img_seed":9007199254740993,"txt2img_cfg_scale":1.2500e+2,"foreign":{"n":9223372036854775808123}}"""
    val record = sourced("Original", raw, Map(
      key("generation.seed") -> PresetAtom.Integer(9007199254740993L),
      key("generation.cfgScale") -> PresetAtom.Decimal(BigDecimal("125.00")),
      key("plugin.unavailable.draft") -> PresetAtom.Text("untouched native draft")))
      .copy(opaque = Vector("foreign" -> RawJson.validated("""{"n":9223372036854775808123}"""),
        "opaque-extra" -> RawJson.validated("1.2300e-30")))
    val expected = """{"Original":{"txt2img_seed":9007199254740993,"txt2img_cfg_scale":1.2500e+2,"foreign":{"n":9223372036854775808123},"opaque-extra":1.2300e-30}}"""
    assertEquals(exported(Vector(record), false), expected)
    assertEquals(omissions(Vector(record)), Vector.empty[PresetIssue])
    assertEquals(exported(Vector(record), false), expected)
    assertEquals(exported(Vector(record), true), expected)
    assertEquals(record.source.get.originalFields.text, raw)
    assertEquals(record.dirty, Set.empty[FieldKey])
    assertEquals(record.opaque.last._2.text, "1.2300e-30")
  }

  test("only dirty provenance fields are assessed and a dirty removal uses its original alias") {
    val steps = key("generation.steps")
    val missing = key("plugin.native_only.changed")
    val untouched = key("plugin.native_only.untouched")
    val record = sourced("Edited", """{"custom-steps":3.000e1,"foreign":null}""",
      Map(missing -> PresetAtom.Integer(1), untouched -> PresetAtom.Integer(2)))
      .copy(dirty = Set(steps, missing))
    val bindings = Vector(SourceBinding(config, "custom-steps", steps))
    val issues = omissions(Vector(record), bindings = bindings)
    assertEquals(issues.size, 1)
    assert(identifies(issues.head, "Edited", missing.render))
    assertEquals(exported(Vector(record), true, bindings = bindings), """{"Edited":{"foreign":null}}""")
  }

  test("original verified plain aliases take precedence over standard scalar conversion") {
    val script = key("script.selected")
    val record = sourced("Alias", """{"my-script":"old"}""",
      Map(script -> PresetAtom.Text("native_only_script"))).copy(dirty = Set(script))
    val bindings = Vector(SourceBinding(config, "my-script", script))
    assertEquals(omissions(Vector(record), bindings = bindings), Vector.empty[PresetIssue])
    assertEquals(exported(Vector(record), false, bindings = bindings),
      """{"Alias":{"my-script":"native_only_script"}}""")
    val standard = fresh("Standard", record.fields ++ nativeFields(1))
    val issues = omissions(Vector(standard))
    assertEquals(issues.size, 2)
    assert(issues.exists(identifies(_, "Standard", "script_list")))
    assert(issues.exists(identifies(_, "Standard", "plugin.native_only.field0000")))
  }

  test("partial compound tiling reports its source key and every omitted constituent and other field") {
    val record = fresh("Partial tiling", tiling.updated(key("plugin.asymmetric_tiling.tile_x"),
      PresetAtom.Flag(false)) ++ nativeFields(1))
    val issues = omissions(Vector(record))
    assertEquals(issues.size, 6)
    assert(issues.forall(_.code == "unsupported"))
    assert(issues.exists(identifies(_, record.name, "txt2img_tiling")))
    for id <- Vector("tile_x", "tile_y", "start_step", "stop_step") do
      assert(issues.exists(identifies(_, record.name, s"plugin.asymmetric_tiling.$id")))
    assert(issues.exists(identifies(_, record.name, "plugin.native_only.field0000")))
    assertEquals(exported(Vector(record), true), """{"Partial tiling":{}}""")
  }

  test("disabled original tiling includes its preserved draft constituents when enabled is dirty") {
    val enabled = key("plugin.asymmetric_tiling.enabled")
    val record = sourced("Disabled", """{"txt2img_tiling":true}""",
      tiling.updated(enabled, PresetAtom.Flag(false))).copy(dirty = Set(enabled))
    val issues = omissions(Vector(record))
    assertEquals(issues.size, 4)
    for id <- Vector("tile_x", "tile_y", "start_step", "stop_step") do
      assert(issues.exists(identifies(_, record.name, s"plugin.asymmetric_tiling.$id")))
    assertEquals(exported(Vector(record), true), """{"Disabled":{"txt2img_tiling":false}}""")
  }

  test("hard alias conflicts after unsupported fields or records cannot become an omission review") {
    val width = key("generation.width")
    val height = key("generation.height")
    val bindings = Vector(SourceBinding(config, "same-size", width), SourceBinding(config, "same-size", height))
    val records = Vector(fresh("Omissions first", nativeFields(210)),
      fresh("Conflict last", Map(width -> PresetAtom.Integer(512), height -> PresetAtom.Integer(768))))
    hardFailure(diagnostics(records, bindings = bindings), "alias-conflict")
    hardFailure(diagnostics(Vector(fresh("Standard conflict", Map(width -> PresetAtom.Integer(512)))),
      bindings = Vector(SourceBinding(config, "txt2img_height", width))), "alias-conflict")
    hardFailure(diagnostics(Vector(sourced("Untouched conflict", "{}")),
      bindings = bindings), "alias-conflict")
  }

  test("invalid format mode provenance and null inputs fail rather than returning partial issues") {
    val records = Vector(fresh("Valid", nativeFields(1)))
    hardFailure(diagnostics(records, "unknown"), "format")
    hardFailure(diagnostics(records, mode = None), "mode")
    hardFailure(diagnostics(records, mode = Some(PresetMode.Inpaint)), "mode")
    hardFailure(diagnostics(records, utilities, txt), "mode")
    assert(diagnostics(null).isLeft)
    assert(diagnostics(records, bindings = null).isLeft)
    assert(diagnostics(Vector(null)).isLeft)
    assert(diagnostics(Vector(records.head.copy(source = null))).isLeft)
    assert(diagnostics(Vector(records.head.copy(source = Some(null)))).isLeft)
    hardFailure(diagnostics(Vector(sourced("Other format", "{}")), utilities, None), "format")
  }

  test("invalid record names keys atoms and bindings are hard errors") {
    val valid = fresh("Valid", nativeFields(1))
    assert(diagnostics(Vector(valid.copy(name = " "))).isLeft)
    hardFailure(diagnostics(Vector(valid, valid)), "duplicate-name")
    assert(diagnostics(Vector(valid.copy(fields = Map(FieldKey(Vector("")) -> PresetAtom.Integer(1))))).isLeft)
    assert(diagnostics(Vector(valid.copy(fields = Map(key("generation.prompt") -> PresetAtom.Text(null))))).isLeft)
    assert(diagnostics(Vector(valid.copy(fields = Map(key("generation.cfgScale") -> PresetAtom.Decimal(null))))).isLeft)
    assert(diagnostics(Vector(valid), bindings = Vector(null)).isLeft)
    assert(diagnostics(Vector(valid), bindings = Vector(SourceBinding(config, "x" * 513,
      key("generation.steps")))).isLeft)
    assert(diagnostics(Vector(valid.copy(opaque = Vector("x" * 513 -> RawJson.validated("0"))))).isLeft)
  }

  test("valid 2048 fields and 1000 records are accepted completely without an arbitrary diagnostic cap") {
    val fields = omissions(Vector(fresh("Field boundary", nativeFields(2048))))
    assertEquals(fields.size, 2048)
    assert(fields.exists(identifies(_, "Field boundary", "plugin.native_only.field2047")))
    val records = (0 until 1000).map(n => fresh(s"Record $n", nativeFields(1))).toVector
    val issues = omissions(records)
    assertEquals(issues.size, 1000)
    assert(issues.exists(identifies(_, "Record 999", "plugin.native_only.field0000")))
    hardFailure(diagnostics(Vector(fresh("Too many fields", nativeFields(2049)))), "limit")
    hardFailure(diagnostics(records :+ fresh("Record 1000", Map.empty)), "limit")
    hardFailure(diagnostics(Vector.empty, bindings = Vector.fill(2049)(
      SourceBinding(config, "my-steps", key("generation.steps")))), "limit")
  }

  test("invalid or oversized saved raw JSON rejects the entire review even after valid omissions") {
    val prefix = fresh("Earlier omissions", nativeFields(210))
    val invalid = Vector("[]", "{", """{"x":1,"x":2}""",
      "{\"x\":" + "[" * 33 + "0" + "]" * 33 + "}",
      "{\"x\":[" + "0," * 100000 + "0]}",
      "{\"x\":\"" + "x" * limits.maxBytes + "\"}",
      (0 until 2049).map(n => s"\"k$n\":0").mkString("{", ",", "}"))
    invalid.foreach(raw => assert(diagnostics(Vector(prefix, sourced("Invalid raw", raw))).isLeft))
    val malformedOpaque = fresh("Invalid opaque", Map.empty)
      .copy(opaque = Vector("opaque" -> RawJson.validated("[")))
    assert(diagnostics(Vector(prefix, malformedOpaque)).isLeft)
  }

  test("scalar and aggregate limits return hard errors rather than successful partial reviews") {
    val prompt = key("generation.prompt")
    val exact = fresh("Exact scalar", Map(prompt -> PresetAtom.Text("x" * limits.maxScalarBytes)))
    assertEquals(omissions(Vector(exact)), Vector.empty[PresetIssue])
    hardFailure(diagnostics(Vector(exact.copy(fields = Map(prompt ->
      PresetAtom.Text("x" * (limits.maxScalarBytes + 1)))))), "limit")
    val oversized = PresetAtom.Text("x" * (limits.maxScalarBytes + 1))
    hardFailure(diagnostics(Vector(fresh("Unsupported invalid scalar",
      Map(key("plugin.native_only.large") -> oversized)))), "limit")
    hardFailure(diagnostics(Vector(sourced("Untouched invalid scalar", "{}",
      Map(prompt -> oversized)))), "limit")
    val tooManyBytes = (0 until 17).map(n => exact.copy(name = s"Large $n")).toVector
    hardFailure(diagnostics(tooManyBytes), "limit")
    val array = RawJson.validated("[" + Vector.fill(60000)("0").mkString(",") + "]")
    val nested = Vector("First", "Last").map(n => fresh(n, Map.empty).copy(opaque = Vector("array" -> array)))
    hardFailure(diagnostics(nested), "limit")
  }

  test("writer-accepted near-limit source remains supported despite duplicate native provenance storage") {
    val value = "x" * limits.maxScalarBytes
    val raw = "{\"txt2img_prompt\":\"" + value + "\"}"
    // Each record stores both raw and typed text; only the actual source output owns the 4 MiB budget.
    val records = (0 until 15).map(n => sourced(s"Near limit $n", raw,
      Map(key("generation.prompt") -> PresetAtom.Text(value)))).toVector
    val before = exported(records, false)
    assert(before.length > 3900000 && before.length < limits.maxBytes)
    assertEquals(omissions(records), Vector.empty[PresetIssue])
    assertEquals(exported(records, false), before)
  }

  test("diagnostic accumulation has an explicit node-derived limit and never returns a truncated review") {
    val fields = nativeFields(2048)
    val records = (0 until 50).map(n => fresh(s"Omission-heavy $n", fields)).toVector
    hardFailure(diagnostics(records), "limit")
  }

  test("512-byte source keys and exact decimal extremes follow existing writer acceptance") {
    val field = key("generation.cfgScale")
    val sourceKey = "界" * 170 + "ab"
    val bindings = Vector(SourceBinding(config, sourceKey, field))
    for number <- Vector(BigDecimal("9007199254740993.12500"), BigDecimal("1e-2147483647")) do
      val records = Vector(fresh("Exact decimal", Map(field -> PresetAtom.Decimal(number))))
      val before = exported(records, false, bindings = bindings)
      assertEquals(omissions(records, bindings = bindings), Vector.empty[PresetIssue])
      assertEquals(exported(records, false, bindings = bindings), before)
    val oversized = PresetAtom.Decimal(BigDecimal(new java.math.BigDecimal(
      java.math.BigInteger.TEN.pow(limits.maxScalarBytes))))
    hardFailure(diagnostics(Vector(fresh("Oversized decimal", Map(field -> oversized)))), "limit")
  }

  private val integerScalarBoundaries = Vector(
    (9L, 1, true), (12L, 1, false),
    (-1L, 2, true), (-12L, 2, false), (-1L, 1, false))
  private def oneSourceAtom(atom: PresetAtom)
      : Iterator[Either[PresetError, (String, Vector[(String, SourceValue)])]] =
    Iterator(Right("N" -> Vector("n" -> SourceValue.Atom(atom))))
  private def assertScalarBoundary[A](result: Either[PresetError, A], accepted: Boolean,
      clue: String): Unit =
    assertEquals(result.isRight, accepted, clue)
    if !accepted then assert(result.swap.toOption.get.issues.exists(_.code == "limit"), clue)

  test("ordinary source writer counts every integer digit and negative sign in the scalar budget") {
    integerScalarBoundaries.foreach { (value, cap, accepted) =>
      val result = PresetJson.writeSourceRecords(oneSourceAtom(PresetAtom.Integer(value)),
        limits.copy(maxScalarBytes = cap))
      assertScalarBoundary(result, accepted, s"integer $value with scalar cap $cap")
      if accepted then assertEquals(new String(result.toOption.get, UTF_8), s"{\"N\":{\"n\":$value}}")
    }
  }

  test("count-only record validation matches integer scalar boundaries including the negative sign") {
    integerScalarBoundaries.foreach { (value, cap, accepted) =>
      val result = PresetJson.validateSourceRecords(oneSourceAtom(PresetAtom.Integer(value)),
        limits.copy(maxScalarBytes = cap))
      assertScalarBoundary(result, accepted, s"integer $value with scalar cap $cap")
    }
  }

  test("count-only atom validation matches integer scalar boundaries including the negative sign") {
    integerScalarBoundaries.foreach { (value, cap, accepted) =>
      val result = PresetJson.validateSourceAtom(PresetAtom.Integer(value), limits.copy(maxScalarBytes = cap))
      assertScalarBoundary(result, accepted, s"integer $value with scalar cap $cap")
    }
  }

  test("boolean literals retain reader semantics when their token exceeds the scalar budget") {
    val small = limits.copy(maxScalarBytes = 1)
    for value <- Vector(true, false) do
      val atom = PresetAtom.Flag(value)
      val written = PresetJson.writeSourceRecords(oneSourceAtom(atom), small)
      assert(written.isRight)
      assertEquals(new String(written.toOption.get, UTF_8), s"{\"N\":{\"n\":$value}}")
      assertEquals(PresetJson.validateSourceRecords(oneSourceAtom(atom), small), Right(()))
      assertEquals(PresetJson.validateSourceAtom(atom, small), Right(()))
  }
