package sdfx.core

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import sdfx.core.modelcards.*

class ModelPreviewResolverSuite extends munit.FunSuite with TempDir:
  private def resolver = new ModelPreviewResolver(LocalPreviewFiles.scoped(new PreviewTestPlatform))
  private def write(dir: Path, name: String): Path =
    val path = dir.resolve(name)
    Files.write(path, ("fixture:" + name).getBytes(UTF_8))
    path
  private def selected(r: Either[String, Option[PreviewReference]]): PreviewReference =
    r.fold(fail(_), _.getOrElse(fail("Expected selected preview")))
  private def sha(path: Path): Vector[Byte] = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).toVector

  tmp.test("override_is_authoritative_even_when_missing_or_unsupported") { dir =>
    val model = write(dir, "model.safetensors")
    write(dir, "model.preview.png")
    val id = CardId("checkpoint", model.toString)
    assert(resolver.resolve(id, false, Some(dir.resolve("missing.png").toString)).isLeft)
    val unsupported = write(dir, "chosen.svg")
    assert(resolver.resolve(id, false, Some(unsupported.toString)).isLeft)
    assert(resolver.resolve(id, false, Some("")).isLeft)
    assert(resolver.resolve(id, false, Some("https://example.invalid/NSFW.png")).isLeft)
  }

  tmp.test("literal_source_only_model_name_and_kind_never_classify_a_safe_override") { dir =>
    val model = write(dir, "NSFW-model.pt")
    val safe = write(dir, "safe.png")
    val marked = write(dir, "NSFW-preview.png")
    for kind <- Vector("checkpoint", "lora", "embedding", "hypernetwork", "future-kind") do
      val safeRef = selected(resolver.resolve(CardId(kind, model.toString), false, Some(safe.toString)))
      assert(!CardPreviewPolicy.matches(Some(safeRef.source)))
      val markedRef = selected(resolver.resolve(CardId(kind, dir.resolve("safe.pt").toString), false, Some(marked.toString)))
      assert(CardPreviewPolicy.matches(Some(markedRef.source)))
      assertEquals(safeRef.allowedRealParent, None)
  }

  tmp.test("sidecar_priority_covers_every_supported_file_candidate_in_order") { dir =>
    val model = write(dir, "model.v2.safetensors")
    val names = Vector("model.v2.preview.png", "model.v2.preview.jpg", "model.v2.preview.jpeg",
      "model.v2.png", "model.v2.jpg", "model.v2.jpeg", "model.v2.preview.gif", "model.v2.gif",
      "model.v2.preview.bmp", "model.v2.bmp")
    names.reverse.foreach(write(dir, _))
    val id = CardId("checkpoint", model.toString)
    names.foreach { name =>
      val ref = selected(resolver.resolve(id, false, None))
      assertEquals(ref.localPath, dir.resolve(name))
      assertEquals(ref.allowedRealParent, Some(dir.toRealPath()))
      Files.delete(dir.resolve(name))
    }
    assertEquals(resolver.resolve(id, false, None), Right(None))
  }

  tmp.test("unreadable_high_priority_candidate_reports_problem_without_lower_priority_fallback") { dir =>
    val model = write(dir, "model.pt")
    val unavailable = write(dir, "model.preview.png")
    write(dir, "model.jpg")
    val id = CardId("lora", model.toString)
    assertEquals(selected(resolver.resolve(id, false, None)).localPath, unavailable,
      "Native control must select the winning preview before denial is injected")
    var reached = false
    val platform = new PreviewTestPlatform:
      override def stamp(p: Path) =
        if p == unavailable then
          reached = true
          Left("fixture denied")
        else super.stamp(p)
    val r = new ModelPreviewResolver(LocalPreviewFiles.scoped(platform))
    assertEquals(r.resolve(id, false, None), Left("fixture denied"))
    assert(reached, "Winning-candidate denial fixture was not reached")
  }

  tmp.test("directory_models_use_only_direct_preview_candidates_in_order") { dir =>
    val folder = Files.createDirectory(dir.resolve("model"))
    val nested = Files.createDirectory(folder.resolve("nested"))
    write(nested, "preview.png")
    val names = Vector("preview.png", "preview.jpg", "preview.jpeg", "preview.gif", "preview.bmp")
    names.reverse.foreach(write(folder, _))
    names.foreach { name =>
      val ref = selected(resolver.resolve(CardId("generic", folder.toString), true, None))
      assertEquals(ref.localPath, folder.resolve(name))
      assertEquals(ref.allowedRealParent, Some(folder.toRealPath()))
      Files.delete(folder.resolve(name))
    }
    assertEquals(resolver.resolve(CardId("generic", folder.toString), true, None), Right(None))
  }

  tmp.test("sidecar_case_matching_on_windows_retains_actual_selected_spelling") { dir =>
    val model = write(dir, "MODEL.pt")
    val preview = write(dir, "MoDeL.PrEvIeW.PnG")
    val windows = new PreviewTestPlatform:
      override def windows = true
    val r = new ModelPreviewResolver(LocalPreviewFiles.scoped(windows))
    val ref = selected(r.resolve(CardId("lora", model.toString), false, None))
    assertEquals(ref.localPath, preview)
    assertEquals(ref.source, preview.toUri.toASCIIString)
  }

  tmp.test("case_only_ambiguity_is_a_problem_on_case_sensitive_directories") { dir =>
    val lower = write(dir, "model.preview.png")
    val upper = dir.resolve("MODEL.PREVIEW.PNG")
    // Windows usually aliases these names. Only a filesystem that can hold both supplies this fixture.
    if !Files.exists(upper) then
      write(dir, "MODEL.PREVIEW.PNG")
      val windows = new PreviewTestPlatform:
        override def windows = true
      val r = new ModelPreviewResolver(LocalPreviewFiles.scoped(windows))
      assert(r.resolve(CardId("lora", dir.resolve("model.pt").toString), false, None).isLeft)
      assert(Files.exists(lower) && Files.exists(upper))
  }

  tmp.test("local_uri_encoding_keeps_percent_unicode_space_and_selected_spelling") { dir =>
    val path = write(dir, "日本語 n%73fw # preview.PNG")
    val ref = resolver.validateExplicit(path).fold(fail(_), identity)
    assertEquals(ref.source, path.toAbsolutePath.normalize.toUri.toASCIIString)
    assert(ref.source.contains("%2573fw%20%23%20preview.PNG"))
    assert(ref.source.contains("%E6%97%A5%E6%9C%AC%E8%AA%9E"))
    assert(!CardPreviewPolicy.matches(Some(ref.source)))
  }

  tmp.test("resolving_and_policy_never_modify_model_or_preview_bytes") { dir =>
    val model = write(dir, "NSFW-model.safetensors")
    val preview = write(dir, "chosen.NSFW.png")
    val before = Vector(sha(model), sha(preview))
    val ref = selected(resolver.resolve(CardId("checkpoint", model.toString), false, Some(preview.toString)))
    for mode <- CardPreviewMode.values; revealed <- Vector(false, true) do
      CardPreviewPolicy.presentation(Some(ref.source), mode, revealed)
    assertEquals(Vector(sha(model), sha(preview)), before)
  }
