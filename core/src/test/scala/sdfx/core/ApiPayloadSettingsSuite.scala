package sdfx.core

import sdfx.core.config.Settings
import java.nio.file.Files

class ApiPayloadSettingsSuite extends munit.FunSuite:
  test("old settings decode image diagnostics false; only explicit preference persists") {
    val dir = Files.createTempDirectory("sdfx-payload-settings-")
    val file = dir.resolve("settings.json")
    try
      Files.writeString(file, "{\"theme\":\"light\"}")
      val old = Settings.load(file)
      assertEquals(old.apiPayloadIncludeImageContents, false)
      Settings.save(file, old.copy(apiPayloadIncludeImageContents = true))
      assertEquals(Settings.load(file).apiPayloadIncludeImageContents, true)
      val saved = ujson.read(Files.readString(file))
      assertEquals(saved("api_payload_include_image_contents"), ujson.Bool(true))
      assert(!saved.obj.keys.exists(k => Set("payloads", "body_json", "assets_json", "context_json")(k)))
    finally
      Files.deleteIfExists(file)
      Files.deleteIfExists(dir)
  }
