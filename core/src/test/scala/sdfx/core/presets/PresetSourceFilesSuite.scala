package sdfx.core.presets

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files
import java.security.MessageDigest
import sdfx.core.TempDir

class PresetSourceFilesSuite extends munit.FunSuite with TempDir:
  private val limits = PresetLimits()

  private def sha256(value: Array[Byte]): String =
    MessageDigest.getInstance("SHA-256").digest(value).map(b => f"${b & 0xff}%02x").mkString

  test("text_comments_and_blank_lines_do_not_become_tracked_ids") {
    val source = "# comment\n\ntxt2img_steps\n# another comment\r\ncustom.script-value\n".getBytes(UTF_8)
    assertEquals(PresetSourceFiles.readTrackedIds(source, limits).toOption.get,
      Vector("txt2img_steps", "custom.script-value"))
  }

  test("tracked_ids_writer_roundtrips_literal_ids_as_one_line_each") {
    val ids = Vector("txt2img_steps", "custom.script-value")
    val written = PresetSourceFiles.writeTrackedIds(ids, limits).toOption.get
    assertEquals(new String(written, UTF_8).linesIterator.toVector, ids)
    assertEquals(PresetSourceFiles.readTrackedIds(written, limits).toOption.get, ids)
  }

  test("tracked_text_limits_reject_excess_before_semantic_import") {
    val tooLarge = Array.fill[Byte](262145)('#'.toByte)
    assert(PresetSourceFiles.readTrackedIds(tooLarge, limits).isLeft)
    val tooMany = Vector.fill(2049)("x")
    assert(PresetSourceFiles.writeTrackedIds(tooMany, limits).isLeft)
  }

  test("tracked_id_writer_has_inclusive_ascii_and_multibyte_512_byte_boundaries") {
    val asciiAtLimit = "a" * 512
    val multibyteAtLimit = "🙂" * 128 // 128 four-byte code points
    assertEquals(asciiAtLimit.getBytes(UTF_8).length, 512)
    assertEquals(multibyteAtLimit.getBytes(UTF_8).length, 512)
    for id <- Vector(asciiAtLimit, multibyteAtLimit) do
      val written = PresetSourceFiles.writeTrackedIds(Vector(id), limits).toOption.get
      assertEquals(PresetSourceFiles.readTrackedIds(written, limits).toOption.get, Vector(id))
      assert(PresetSourceFiles.writeTrackedIds(Vector(id + "x"), limits).isLeft)
  }

  test("additional_json_writer_has_inclusive_multibyte_label_and_id_boundaries") {
    val atLimit = "🙂" * 128
    val asciiAtLimit = "a" * 512
    assertEquals(atLimit.getBytes(UTF_8).length, 512)
    assertEquals(asciiAtLimit.getBytes(UTF_8).length, 512)
    val entries = Vector(atLimit -> Some(atLimit), asciiAtLimit -> Some(asciiAtLimit))
    val written = PresetSourceFiles.writeAdditionalComponents(entries, limits).toOption.get
    assertEquals(PresetSourceFiles.readAdditionalComponents(written, limits).toOption.get, entries)
    assert(PresetSourceFiles.writeAdditionalComponents(Vector((atLimit + "x") -> None), limits).isLeft)
    assert(PresetSourceFiles.writeAdditionalComponents(Vector("label" -> Some(atLimit + "x")), limits).isLeft)
    assert(PresetSourceFiles.writeAdditionalComponents(Vector(("a" * 513) -> None), limits).isLeft)
    assert(PresetSourceFiles.writeAdditionalComponents(Vector("label" -> Some("a" * 513)), limits).isLeft)
  }

  test("source_writers_reject_unpaired_surrogates") {
    val high = new String(Array(0xd800.toChar))
    val low = new String(Array(0xdc00.toChar))
    assertEquals(PresetSourceFiles.writeTrackedIds(Vector(high), limits).swap.toOption.get.issues.head.code, "invalid-unicode")
    assertEquals(PresetSourceFiles.writeTrackedIds(Vector(low), limits).swap.toOption.get.issues.head.code, "invalid-unicode")
    assertEquals(PresetSourceFiles.writeAdditionalComponents(Vector(high -> None), limits).swap.toOption.get.issues.head.code,
      "invalid-unicode")
    assertEquals(PresetSourceFiles.writeAdditionalComponents(Vector("label" -> Some(low)), limits).swap.toOption.get.issues.head.code,
      "invalid-unicode")
  }

  test("legacy_additional_null_stays_null_and_is_not_a_native_id") {
    val source = """{"additionalComponents":{"Optional":null,"X type":"script_xyz_x_type"}}""".getBytes(UTF_8)
    val parsed = PresetSourceFiles.readAdditionalComponents(source, limits).toOption.get
    assertEquals(parsed, Vector("Optional" -> None, "X type" -> Some("script_xyz_x_type")))
    val written = PresetSourceFiles.writeAdditionalComponents(parsed, limits).toOption.get
    assertEquals(PresetSourceFiles.readAdditionalComponents(written, limits).toOption.get, parsed)
    assert("\"Optional\"\\s*:\\s*null".r.findFirstIn(new String(written, UTF_8)).nonEmpty)
    assert("\"Optional\"\\s*:\\s*\"null\"".r.findFirstIn(new String(written, UTF_8)).isEmpty)
  }

  test("additional_components_reject_wrong_value_type_or_duplicate_label") {
    val wrongType = """{"additionalComponents":{"Bad":42}}""".getBytes(UTF_8)
    val duplicate = """{"additionalComponents":{"A":null,"\u0041":"id"}}""".getBytes(UTF_8)
    assert(PresetSourceFiles.readAdditionalComponents(wrongType, limits).isLeft)
    assert(PresetSourceFiles.readAdditionalComponents(duplicate, limits).isLeft)
  }

  tmp.test("source_read_is_explicitly_formatted_read_only_and_has_exact_sha256") { dir =>
    val path = dir.resolve("deliberately-unrelated-name.json")
    val original = "{\"One\":{\"txt2img_steps\":7,\"foreign\":9007199254740993}}".getBytes(UTF_8)
    Files.write(path, original)
    val doc = PresetSourceFiles.read(path, "config-presets", Some(PresetMode.Txt2Img), limits).toOption.get
    assertEquals(doc.format, "config-presets")
    assertEquals(doc.mode, Some(PresetMode.Txt2Img))
    assertEquals(doc.sha256, sha256(original))
    assert(doc.raw.text.contains("9007199254740993"))
    assertEquals(Files.readAllBytes(path).toVector, original.toVector)
  }

  tmp.test("explicit_source_format_has_inclusive_512_byte_limit_and_rejects_surrogates") { dir =>
    val path = dir.resolve("source.json")
    Files.writeString(path, "{}")
    assert(PresetSourceFiles.read(path, "f" * 512, None, limits).isRight)
    assert(PresetSourceFiles.read(path, "f" * 513, None, limits).isLeft)
    assertEquals(PresetSourceFiles.read(path, new String(Array(0xd800.toChar)), None, limits).swap.toOption.get.issues.head.code,
      "invalid-unicode")
  }

  tmp.test("source_read_rejects_file_above_cap_without_modifying_it") { dir =>
    val path = dir.resolve("source.json")
    val oversized = Array.fill[Byte](4194305)(' '.toByte)
    Files.write(path, oversized)
    assert(PresetSourceFiles.read(path, "config-presets", Some(PresetMode.Txt2Img), limits).isLeft)
    assertEquals(Files.size(path), 4194305L)
  }

  tmp.test("explicit_export_rejects_stale_revision_and_keeps_external_bytes") { dir =>
    val path = dir.resolve("chosen-export.json")
    val first = "{\"One\":{}}".getBytes(UTF_8)
    val revision = PresetSourceFiles.exportFile(path, first, PresetRevision(None)).toOption.get
    assertEquals(revision.sha256, Some(sha256(first)))
    val external = "{ \"manually edited\": true }\n".getBytes(UTF_8)
    Files.write(path, external)
    assert(PresetSourceFiles.exportFile(path, "{\"Two\":{}}".getBytes(UTF_8), revision).isLeft)
    assertEquals(Files.readAllBytes(path).toVector, external.toVector)
  }

  tmp.test("explicit_export_refuses_oversize_without_creating_destination") { dir =>
    val path = dir.resolve("chosen-export.json")
    assert(PresetSourceFiles.exportFile(path, Array.fill[Byte](4194305)('x'.toByte), PresetRevision(None)).isLeft)
    assert(!Files.exists(path))
  }
