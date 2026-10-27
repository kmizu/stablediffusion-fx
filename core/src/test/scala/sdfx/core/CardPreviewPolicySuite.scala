package sdfx.core

import java.nio.file.Path
import sdfx.core.modelcards.*

class CardPreviewPolicySuite extends munit.FunSuite:
  test("literal_source_only_ascii_case_insensitive_without_url_decoding") {
    val cases = Vector(
      None -> false,
      Some("") -> false,
      Some("file:///safe/NSFW-preview.png") -> true,
      Some("file:///safe/benignsfword.png") -> true,
      Some("file:///NSfW/safe.png") -> true,
      Some("file:///safe/n%73fw.png") -> false,
      Some("file:///safe/nſfw.png") -> false,
      Some("file:///safe/ＮＳＦＷ.png") -> false,
      Some("file:///safe/ns-fw.png") -> false
    )
    cases.foreach { (source, want) => assertEquals(CardPreviewPolicy.matches(source), want, source.toString) }
  }

  test("startup_exact_values_and_invalid_or_null_fails_closed") {
    import CardPreviewMode.*
    Vector("Blur" -> Blur, "Hide" -> Hide, "Show" -> Show,
      "blur" -> Hide, "SHOW" -> Hide, " Blur" -> Hide, "Blur " -> Hide,
      "" -> Hide, "unknown" -> Hide, (null: String) -> Hide).foreach { (saved, want) =>
      assertEquals(CardPreviewPolicy.startup(saved), want)
    }
  }

  test("all_modes_and_reveal_hide_never_reveals_and_missing_source_is_normal") {
    import CardPreviewMode.*
    import PreviewPresentation.*
    val marked = Some("file:///NSFW.png")
    val cases = Vector((Blur, false, Blurred), (Blur, true, Normal),
      (Hide, false, Hidden), (Hide, true, Hidden), (Show, false, Normal), (Show, true, Normal))
    cases.foreach { (mode, revealed, want) =>
      assertEquals(CardPreviewPolicy.presentation(marked, mode, revealed), want)
      assertEquals(CardPreviewPolicy.presentation(None, mode, revealed), Normal)
      assertEquals(CardPreviewPolicy.presentation(Some("file:///safe.png"), mode, revealed), Normal)
    }
  }

  test("identity_key_roundtrip_has_two_json_strings_without_delimiter_collisions") {
    val path = Path.of("models", "日本語 % model.safetensors").toAbsolutePath.normalize.toString
    val id = CardId("future:\"kind\\name", path)
    val json = ujson.read(id.storageKey)
    assertEquals(json.arr.toVector.map(_.str), Vector("future:\"kind\\name", path))
    assertEquals(CardId.fromStorageKey(id.storageKey), Right(id))
    assertEquals(id.storageKey, ujson.write(ujson.Arr(id.kind, path)))
    assertNotEquals(CardId("a:b", path).storageKey, CardId("a", path).storageKey)
  }

  test("identity_normalizes_dot_components_but_keeps_selected_path_spelling") {
    val root = Path.of("models").toAbsolutePath
    val id = CardId("hypernetwork", root.resolve("MiXeD").resolve("..").resolve("日本語.pt").toString)
    assertEquals(id.modelPath, root.resolve("日本語.pt").normalize.toString)
    assertEquals(CardId("LoRA", root.resolve("MiXeD.pt").toString).kind, "LoRA")
  }

  test("identity_rejects_empty_kind_relative_path_and_malformed_storage_keys") {
    val absolute = Path.of("model.pt").toAbsolutePath.normalize.toString
    intercept[IllegalArgumentException](CardId("", absolute))
    intercept[IllegalArgumentException](CardId("lora", "relative.pt"))
    Vector("null", "{}", "[]", "[\"lora\"]", "[1,2]", "[\"lora\",\"relative.pt\"]",
      ujson.write(ujson.Arr("", absolute)), ujson.write(ujson.Arr("lora", absolute, "extra")), "broken"
    ).foreach(key => assert(CardId.fromStorageKey(key).isLeft, key))
  }
