package sdfx.core

import sdfx.core.config.*
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.security.MessageDigest

class CardPreviewSettingsSuite extends munit.FunSuite with TempDir:
  private def sha(bytes: Array[Byte]): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 0xff}%02x").mkString

  tmp.test("missing file and old valid settings default to Blur without recovery") { dir =>
    val file = dir.resolve("settings.json")
    val fresh = Settings.loadResult(file)
    assertEquals(fresh.settings.nsfwCardBlurDefault, "Blur")
    assertEquals(fresh.recovery, None)
    Files.writeString(file, """{"theme":"light","last_model":"kept"}""")
    val old = Settings.loadResult(file)
    assertEquals(old.settings.nsfwCardBlurDefault, "Blur")
    assertEquals(old.settings.theme, "light")
    assertEquals(old.settings.lastModel, Some("kept"))
    assertEquals(old.recovery, None)
  }

  tmp.test("invalid existing mode is Hide with recovery and preserves other settings") { dir =>
    val file = dir.resolve("settings.json")
    Seq("null", "42", "true", "[]", "\"blur\"", "\"unexpected\"").foreach { mode =>
      val bytes = s"""{"theme":"light","last_model":"kept","plugin_values":{"p":{"x":"y"}},"nsfw_card_blur_default":$mode}""".getBytes(StandardCharsets.UTF_8)
      Files.write(file, bytes)
      val result = Settings.loadResult(file)
      assertEquals(result.settings.nsfwCardBlurDefault, "Hide", mode)
      assertEquals(result.settings.theme, "light")
      assertEquals(result.settings.lastModel, Some("kept"))
      assertEquals(result.settings.pluginValues, Map("p" -> Map("x" -> "y")))
      assertEquals(result.recovery.flatMap(_.originalSha256), Some(sha(bytes)))
      assert(result.recovery.exists(_.reason.nonEmpty))
      assertEquals(Files.readAllBytes(file).toSeq, bytes.toSeq)
    }
  }

  tmp.test("malformed and null documents recover to Hide with original hash") { dir =>
    val file = dir.resolve("settings.json")
    Seq("{broken", "null", "[]", "").foreach { text =>
      val bytes = text.getBytes(StandardCharsets.UTF_8)
      Files.write(file, bytes)
      val result = Settings.loadResult(file)
      assertEquals(result.settings.nsfwCardBlurDefault, "Hide")
      assertEquals(result.recovery.flatMap(_.originalSha256), Some(sha(bytes)))
      assertEquals(Settings.load(file), result.settings)
      assertEquals(Files.readAllBytes(file).toSeq, bytes.toSeq)
    }
  }

  tmp.test("input larger than 16 MiB is rejected and retained") { dir =>
    val file = dir.resolve("settings.json")
    val out = Files.newOutputStream(file)
    try
      out.write("{\"padding\":\"".getBytes(StandardCharsets.UTF_8))
      val block = Array.fill[Byte](8192)('a'.toByte)
      (0 until 2048).foreach(_ => out.write(block))
      out.write("\"}".getBytes(StandardCharsets.UTF_8))
    finally out.close()
    val before = Files.readAllBytes(file)
    val result = Settings.loadResult(file)
    assertEquals(result.settings.nsfwCardBlurDefault, "Hide")
    assert(result.recovery.exists(_.reason.nonEmpty))
    assertEquals(result.recovery.flatMap(_.originalSha256), None)
    assertEquals(sha(Files.readAllBytes(file)), sha(before))
    assertEquals(Files.readAllBytes(file).toSeq, before.toSeq)
  }

  tmp.test("invalid preview maps recover without discarding unrelated settings") { dir =>
    val file = dir.resolve("settings.json")
    val cases = Seq(
      "\"model_preview_overrides\":null",
      "\"model_preview_overrides\":{\"a\":42}",
      "\"model_preview_folders\":{\"kind\":[null]}",
      "\"model_preview_folders\":[]",
      "\"model_preview_overrides\":{" + (0 to 10000).map(i => s"\"$i\":\"x\"").mkString(",") + "}",
      "\"model_preview_folders\":{" + (0 to 64).map(i => s"\"k$i\":[]").mkString(",") + "}",
      "\"model_preview_folders\":{\"k\":[" + (0 to 256).map(_ => "\"x\"").mkString(",") + "]}",
      "\"model_preview_overrides\":{\"k\":\"" + ("x" * 8193) + "\"}",
      "\"model_preview_folders\":{\"" + ("k" * 129) + "\":[]}",
      "\"model_preview_overrides\":{" + (0 until 130).map(i => s"\"$i\":\"${"x" * 8000}\"").mkString(",") + "}"
    )
    cases.foreach { field =>
      val bytes = s"""{"theme":"light","last_model":"kept",$field}""".getBytes(StandardCharsets.UTF_8)
      Files.write(file, bytes)
      val result = Settings.loadResult(file)
      assert(result.recovery.exists(_.reason.nonEmpty), field.take(80))
      assertEquals(result.settings.theme, "light")
      assertEquals(result.settings.lastModel, Some("kept"))
      assertEquals(result.recovery.flatMap(_.originalSha256), Some(sha(bytes)))
      assertEquals(Files.readAllBytes(file).toSeq, bytes.toSeq)
    }
  }

  tmp.test("valid maps round trip with snake case and no recovery metadata") { dir =>
    val file = dir.resolve("settings.json")
    val settings = Settings(nsfwCardBlurDefault = "Show", modelPreviewOverrides = Map("[key]" -> "D:/画像.png"),
      modelPreviewFolders = Map("hypernetwork" -> Seq("D:/models")))
    Settings.save(file, settings)
    val json = ujson.read(Files.readString(file))
    assertEquals(json("nsfw_card_blur_default").str, "Show")
    assertEquals(json("model_preview_overrides")("[key]").str, "D:/画像.png")
    assert(!json.obj.contains("recovery"))
    assertEquals(Settings.loadResult(file), SettingsLoadResult(settings, None))
  }

  tmp.test("field-local errors preserve the independently valid preview field in either order") { dir =>
    val file = dir.resolve("settings.json")
    val validOverrides = "\"model_preview_overrides\":{\"id\":\"D:/kept.png\"}"
    val validFolders = "\"model_preview_folders\":{\"hyper\":[\"D:/models\"]}"
    val invalidOverrides = Seq("\"model_preview_overrides\":null",
      "\"model_preview_overrides\":{" + (0 to 10000).map(i => s"\"$i\":\"x\"").mkString(",") + "}")
    val invalidFolders = Seq("\"model_preview_folders\":null",
      "\"model_preview_folders\":{" + (0 to 64).map(i => s"\"kind$i\":[]").mkString(",") + "}")
    val pairs = invalidOverrides.map(_ -> validFolders) ++ invalidFolders.map(_ -> validOverrides)
    pairs.foreach { (bad, good) =>
      Seq(Seq(bad, good), Seq(good, bad)).foreach { fields =>
        val source = s"""{"theme":"light",${fields.mkString(",")}}""".getBytes(StandardCharsets.UTF_8)
        Files.write(file, source)
        val result = Settings.loadResult(file)
        assert(result.recovery.nonEmpty)
        assertEquals(result.settings.theme, "light")
        if good == validOverrides then
          assertEquals(result.settings.modelPreviewOverrides, Map("id" -> "D:/kept.png"))
          assertEquals(result.settings.modelPreviewFolders, Map.empty)
        else
          assertEquals(result.settings.modelPreviewFolders, Map("hyper" -> Seq("D:/models")))
          assertEquals(result.settings.modelPreviewOverrides, Map.empty)
        assertEquals(result.recovery.flatMap(_.originalSha256), Some(sha(source)))
        assertEquals(Files.readAllBytes(file).toSeq, source.toSeq)
      }
    }
  }

  tmp.test("aggregate cap applies across both fields and rejects only the field that crosses it") { dir =>
    val file = dir.resolve("settings.json")
    val overrides = "\"model_preview_overrides\":{" +
      (0 until 80).map(i => s"\"$i\":\"${"x" * 8000}\"").mkString(",") + "}"
    val folders = "\"model_preview_folders\":{\"kind\":[" +
      (0 until 55).map(_ => "\"" + ("y" * 8000) + "\"").mkString(",") + "]}"
    Seq((Seq(overrides, folders), true), (Seq(folders, overrides), false)).foreach { (fields, overridesFirst) =>
      Files.writeString(file, s"""{"theme":"light",${fields.mkString(",")}}""")
      val result = Settings.loadResult(file)
      assert(result.recovery.nonEmpty)
      assertEquals(result.settings.theme, "light")
      if overridesFirst then
        assertEquals(result.settings.modelPreviewOverrides.size, 80)
        assertEquals(result.settings.modelPreviewFolders, Map.empty)
      else
        assertEquals(result.settings.modelPreviewFolders("kind").size, 55)
        assertEquals(result.settings.modelPreviewOverrides, Map.empty)
    }
  }

  tmp.test("UTF-8 errors and nesting boundary recover while valid unknown nested fields are accepted") { dir =>
    val file = dir.resolve("settings.json")
    val malformed = Array[Byte](0x7b, 0x22, 0x78, 0x22, 0x3a, 0x22, 0xc3.toByte, 0x28, 0x22, 0x7d)
    Files.write(file, malformed)
    val bad = Settings.loadResult(file)
    assert(bad.recovery.nonEmpty)
    assertEquals(bad.settings.nsfwCardBlurDefault, "Hide")
    assertEquals(Files.readAllBytes(file).toSeq, malformed.toSeq)
    def nested(depth: Int) = "[" * depth + "0" + "]" * depth
    Files.writeString(file, s"""{"theme":"light","future":${nested(127)}}""")
    val accepted = Settings.loadResult(file)
    assertEquals(accepted.recovery, None)
    assertEquals(accepted.settings.theme, "light")
    Files.writeString(file, s"""{"theme":"light","future":${nested(128)}}""")
    val rejected = Settings.loadResult(file)
    assert(rejected.recovery.nonEmpty)
    assertEquals(rejected.settings.nsfwCardBlurDefault, "Hide")
  }

  test("preflight sanitizes oversized preview maps before general JSON materialization") {
    val entries = (0 until 150).map(i => s"\"$i\":\"${"x" * 8000}\"").mkString(",")
    val source = s"""{"theme":"light","model_preview_overrides":{$entries},"last_model":"kept"}"""
    val result = SettingsPreviewPreflight(source)
    assert(result.problem.nonEmpty)
    assert(result.retainedUnits <= 1000000L)
    assertEquals(result.overrides, Map.empty)
    val json = ujson.read(result.json)
    assertEquals(json("model_preview_overrides").obj.size, 0)
    assertEquals(json("theme").str, "light")
    assertEquals(json("last_model").str, "kept")
  }

  test("preflight counts decoded escapes and JSON encoding and validates syntax in rejected fields") {
    val escaped = (0 until 90).map(i => s"\"$i\":\"${"\\u0000" * 8000}\"").mkString(",")
    val result = SettingsPreviewPreflight(s"""{"model_preview_overrides":{$escaped},"theme":"light"}""")
    assert(result.problem.nonEmpty)
    assertEquals(result.overrides, Map.empty)
    assertEquals(ujson.read(result.json)("theme").str, "light")
    intercept[IllegalArgumentException] {
      SettingsPreviewPreflight("""{"model_preview_overrides":{"bad":"\q"},"theme":"light"}""")
    }
    val valid = SettingsPreviewPreflight("""{"model_preview_folders":{"hy\u0070er":["D:/a\\b"]}}""")
    assertEquals(valid.problem, None)
    assertEquals(valid.folders, Map("hyper" -> Seq("D:/a\\b")))
  }
