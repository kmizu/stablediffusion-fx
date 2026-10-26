package sdfx.core.presets

import java.nio.charset.StandardCharsets.UTF_8
import java.util.UUID

class PresetJsonSuite extends munit.FunSuite:
  private val seedKey = FieldKey(Vector("generation", "seed"))
  private val seedId = UUID.fromString("79ac2471-a722-470d-8931-cbc5ea350bbc")

  private def bytes(json: String): Array[Byte] = json.getBytes(UTF_8)

  private def raw(json: String): RawJson =
    PresetJson.readSource(bytes(json), PresetLimits()).toOption.get

  private def record(name: String, id: UUID = seedId, seed: Long = 9007199254740993L): PresetRecord =
    PresetRecord(
      id = id,
      name = name,
      fields = Map(seedKey -> PresetAtom.Integer(seed)),
      source = None,
      dirty = Set.empty,
      opaque = Vector.empty,
      extra = Vector.empty
    )

  private def collection(id: String, records: Vector[PresetRecord]): PresetCollection =
    PresetCollection(id, Set(seedKey), Vector.empty, records, Vector.empty)

  private def library(records: Vector[PresetRecord] = Vector(record("portrait"))): PresetLibrary =
    PresetLibrary(1, Vector(collection("config-txt2img", records)), Vector.empty)

  private def write(library: PresetLibrary, limits: PresetLimits = PresetLimits()): Array[Byte] =
    PresetJson.writeNative(library, limits).toOption.get

  private def read(json: String): Either[PresetError, PresetLibrary] =
    PresetJson.readNative(bytes(json), PresetLimits())

  test("native_roundtrip_keeps_unknown_numeric_lexeme") {
    val original = raw("{\"mystery\":1.2300e+14,\"nested\":[9.007199254740993e+15]}")
    val source = PresetSource("config", "portrait", Some(PresetMode.Txt2Img), original)
    val entry = record("portrait").copy(
      source = Some(source),
      opaque = Vector("upstream" -> raw("9.007199254740993e+15")),
      extra = Vector("recordFuture" -> raw("1.2300e+14"))
    )
    val doc = PresetLibrary(
      1,
      Vector(collection("config-txt2img", Vector(entry)).copy(extra = Vector("collectionFuture" -> raw("2.500e-8")))),
      Vector("libraryFuture" -> raw("123456789012345678901234567890"))
    )
    val encoded = write(doc)
    val result = PresetJson.readNative(encoded, PresetLimits()).toOption.get
    assertEquals(result, doc)
    assertEquals(result.collections.head.presets.head.source.get.originalFields.text, original.text)
    assertEquals(result.collections.head.presets.head.opaque.head._2.text, "9.007199254740993e+15")
    assertEquals(result.extra.head._2.text, "123456789012345678901234567890")
    assert(new String(encoded, UTF_8).contains("1.2300e+14"))
  }

  test("seed_long_is_exact") {
    for seed <- Vector(9007199254740993L, Long.MaxValue, Long.MinValue, -1L) do
      val encoded = write(library(Vector(record("portrait", seed = seed))))
      val decoded = PresetJson.readNative(encoded, PresetLimits()).toOption.get
      assertEquals(decoded.collections.head.presets.head.fields(seedKey), PresetAtom.Integer(seed))
      assert(new String(encoded, UTF_8).contains(seed.toString))

    val good = new String(write(library()), UTF_8)
    assert(read(good.replace("9007199254740993", "9007199254740993.25")).isLeft)
    assert(read(good.replace("9007199254740993", "9223372036854775808")).isLeft)
    assert(read(good.replace("9007199254740993", "-1.0")).isRight)
    assertEquals(
      read(good.replace("9007199254740993", "-1.0")).toOption.get.collections.head.presets.head.fields(seedKey),
      PresetAtom.Integer(-1L)
    )
  }

  test("duplicate_keys_rejected_at_every_depth") {
    for json <- Vector(
      "{\"a\":1,\"a\":2}",
      "{\"a\":1,\"\\u0061\":2}",
      "{\"outer\":[{\"x\":0,\"\\u0078\":1}]}"
    ) do assert(PresetJson.readSource(bytes(json), PresetLimits()).isLeft, json)

    val native = new String(write(library()), UTF_8)
    assert(read(native.replace("\"version\":1", "\"version\":1,\"\\u0076ersion\":1")).isLeft)
  }

  test("boundaries_are_inclusive") {
    val max = PresetLimits()
    assertEquals(max.maxBytes, 4194304)
    assert(PresetJson.readSource(bytes("null" + " " * (4194304 - 4)), max).isRight)
    assert(PresetJson.readSource(bytes("null" + " " * (4194304 - 3)), max).isLeft)
    assert(PresetJson.readSource(bytes("[" * 32 + "0" + "]" * 32), max).isRight)
    assert(PresetJson.readSource(bytes("[" * 33 + "0" + "]" * 33), max).isLeft)

    val atNodeLimit = "[" + Vector.fill(99999)("0").mkString(",") + "]"
    assert(PresetJson.readSource(bytes(atNodeLimit), max).isRight)
    assert(PresetJson.readSource(bytes(atNodeLimit.dropRight(1) + ",0]"), max).isLeft)
    assert(PresetJson.readSource(bytes("\"" + "x" * 262144 + "\""), max).isRight)
    assert(PresetJson.readSource(bytes("\"" + "x" * 262145 + "\""), max).isLeft)

    val emptyDoc = PresetLibrary(1, Vector.empty, Vector.empty)
    val encoded = write(emptyDoc)
    assert(PresetJson.writeNative(emptyDoc, max.copy(maxBytes = encoded.length)).isRight)
    assert(PresetJson.writeNative(emptyDoc, max.copy(maxBytes = encoded.length - 1)).isLeft)
  }

  test("invalid_utf8_surrogates_nonfinite_rejected") {
    assert(PresetJson.readSource(Array(0xc3.toByte, 0x28.toByte), PresetLimits()).isLeft)
    for json <- Vector("\"\\uD800\"", "\"\\uDC00\"", "{\"\\uD800\":1}", "NaN", "Infinity", "-Infinity", "1e", "01") do
      assert(PresetJson.readSource(bytes(json), PresetLimits()).isLeft, json)
    assertEquals(PresetJson.readSource(bytes("\"\\uD83D\\uDE00\""), PresetLimits()).toOption.get.text, "\"\\uD83D\\uDE00\"")
    assert(PresetJson.readNative(bytes("{\"format\":\"sdfx-native-presets\",\"version\":2,\"collections\":[]}"), PresetLimits()).isLeft)
  }

  test("unicode_escapes_require_ascii_hex_digits_in_keys_and_values") {
    for escape <- Vector("\\u００６１", "\\uＦＦ21", "\\u٠٠٦١") do
      assert(PresetJson.readSource(bytes("\"" + escape + "\""), PresetLimits()).isLeft, escape)
      assert(PresetJson.readSource(bytes("{\"" + escape + "\":0}"), PresetLimits()).isLeft, escape)
    for escape <- Vector("\\u0061", "\\u0041", "\\u00af", "\\uD83D\\uDE00", "\\uFF21") do
      assert(PresetJson.readSource(bytes("\"" + escape + "\""), PresetLimits()).isRight, escape)
      assert(PresetJson.readSource(bytes("{\"" + escape + "\":0}"), PresetLimits()).isRight, escape)
  }

  test("field_key_encoding_is_unambiguous") {
    val dotted = FieldKey(Vector("plugin", "xyz.grid", "%value", "line\nbreak"))
    assertEquals(dotted.render, "plugin.xyz%2Egrid.%25value.line%0Abreak")
    assertEquals(FieldKey.parse(dotted.render).toOption.get, dotted)
    assert(FieldKey(Vector("a.b")).render != FieldKey(Vector("a", "b")).render)
    for id <- Vector("a.%2e", "a.%41", "a.%", "a.%GG", "a..b", ".a", "a.") do
      assert(FieldKey.parse(id).isLeft, id)
    assert(FieldKey.parse("a." + "x" * 511).isLeft)
    assert(FieldKey.parse("a." + "x" * 510).isRight)
  }

  test("field_key_utf8_limit_is_inclusive_for_multibyte_ids") {
    assert(FieldKey.parse("x" * 512).isRight)
    assert(FieldKey.parse("x" * 513).isLeft)
    assert(FieldKey.parse("é" * 256).isRight)
    assert(FieldKey.parse("é" * 257).isLeft)
    assert(FieldKey.parse("😀" * 128).isRight)
    assert(FieldKey.parse("😀" * 129).isLeft)
    assert(FieldKey.parse("界" * 2048).isLeft)
    assert(FieldKey.parse("\ud800").isLeft)
  }

  test("decimal_writer_bounds_coefficient_and_preserves_extreme_scale") {
    val decimalKey = FieldKey(Vector("generation", "cfg"))
    def decimalDoc(n: BigDecimal): PresetLibrary =
      library(Vector(record("portrait").copy(fields = Map(decimalKey -> PresetAtom.Decimal(n)))))
    val scalar64 = BigDecimal(new java.math.BigDecimal("9" * 64))
    val scalar65 = BigDecimal(new java.math.BigDecimal("9" * 65))
    val cap = PresetLimits(maxScalarBytes = 64)
    assert(PresetJson.writeNative(decimalDoc(scalar64), cap).isRight)
    assert(PresetJson.writeNative(decimalDoc(scalar65), cap).isLeft)

    val extreme = BigDecimal(new java.math.BigDecimal("1E+1000000"))
    val encoded = write(decimalDoc(extreme))
    assert(new String(encoded, UTF_8).contains("1E+1000000"))
    val decoded = PresetJson.readNative(encoded, PresetLimits()).toOption.get
    assertEquals(decoded.collections.head.presets.head.fields(decimalKey), PresetAtom.Decimal(extreme))

    val utf8Doc = library(Vector(record("portrait").copy(fields = Map(decimalKey -> PresetAtom.Text("猫猫")))))
    val utf8Bytes = write(utf8Doc)
    assert(PresetJson.writeNative(utf8Doc, PresetLimits(maxBytes = utf8Bytes.length)).isRight)
    assert(PresetJson.writeNative(utf8Doc, PresetLimits(maxBytes = utf8Bytes.length - 1)).isLeft)
  }

  test("native_aggregate_limits_reject_whole_document") {
    val first = collection("config-txt2img", Vector(record("one")))
    val second = collection("config-img2img", Vector(record("two", UUID.fromString("b5929c7b-67d0-4c6f-8732-c3632083a5c7"))))
    val both = PresetLibrary(1, Vector(first, second), Vector.empty)
    assert(PresetJson.writeNative(both, PresetLimits(maxCollections = 2, maxRecords = 2)).isRight)
    assert(PresetJson.writeNative(both, PresetLimits(maxCollections = 1)).isLeft)
    assert(PresetJson.writeNative(both, PresetLimits(maxRecords = 1)).isLeft)

    val oneOpaque = record("portrait").copy(opaque = Vector("u1" -> raw("1")))
    val twoOpaque = oneOpaque.copy(opaque = oneOpaque.opaque :+ ("u2" -> raw("2")))
    assert(PresetJson.writeNative(library(Vector(oneOpaque)), PresetLimits(maxFieldsPerRecord = 2)).isRight)
    assert(PresetJson.writeNative(library(Vector(twoOpaque)), PresetLimits(maxFieldsPerRecord = 2)).isLeft)
  }

  test("identity_and_names_are_not_paths") {
    val pathLike = record("../../x")
    val sameNameOtherCollection = pathLike.copy(id = UUID.fromString("b5929c7b-67d0-4c6f-8732-c3632083a5c7"))
    val doc = PresetLibrary(
      1,
      Vector(collection("config-txt2img", Vector(pathLike)), collection("config-img2img", Vector(sameNameOtherCollection))),
      Vector.empty
    )
    assertEquals(PresetJson.readNative(write(doc), PresetLimits()).toOption.get, doc)
    assertEquals(PresetJson.readNative(write(doc), PresetLimits()).toOption.get.collections.head.presets.head.name, "../../x")

    val otherId = pathLike.copy(id = sameNameOtherCollection.id)
    assert(PresetJson.writeNative(library(Vector(pathLike, otherId)), PresetLimits()).isLeft)
    assert(PresetJson.writeNative(library(Vector(pathLike, pathLike.copy(name = "other"))), PresetLimits()).isLeft)
    for invalid <- Vector("", "  ", "bad\nname", "bad\u0000name", "x" * 257) do
      assert(PresetJson.writeNative(library(Vector(record(invalid))), PresetLimits()).isLeft, invalid)
    assert(PresetJson.writeNative(library(Vector(record("x" * 256))), PresetLimits()).isRight)
  }

  test("writer_rejects_duplicate_and_colliding_unknown_properties") {
    val duplicateExtra = library().copy(extra = Vector("future" -> raw("1"), "future" -> raw("2")))
    assert(PresetJson.writeNative(duplicateExtra, PresetLimits()).isLeft)
    assert(PresetJson.writeNative(library().copy(extra = Vector("version" -> raw("2"))), PresetLimits()).isLeft)
    val collidingRecord = record("portrait").copy(extra = Vector("fields" -> raw("{}")))
    assert(PresetJson.writeNative(library(Vector(collidingRecord)), PresetLimits()).isLeft)
    val duplicateOpaque = record("portrait").copy(opaque = Vector("upstream" -> raw("1"), "upstream" -> raw("2")))
    assert(PresetJson.writeNative(library(Vector(duplicateOpaque)), PresetLimits()).isLeft)
  }
