package sdfx.engine

import munit.FunSuite
import sdfx.core.config.{AppHome, Settings}
import java.nio.file.Files

class WdSettingsSuite extends FunSuite:
  test("explicit WD files are passed as separate process arguments and dry run stays unchanged") {
    val root = Files.createTempDirectory("wd-settings-test")
    val home = AppHome(root)
    val settings = Settings(wdV3Model = Some("C:\\Models With Spaces\\model.onnx"),
      wdV3Tags = Some("C:\\Models With Spaces\\tags.csv"))
    val args = PythonEnv.wdArguments(home, settings, false)
    assertEquals(args.take(4), Seq("--wd-v3-model", "C:\\Models With Spaces\\model.onnx", "--wd-v3-tags", "C:\\Models With Spaces\\tags.csv"))
    assertEquals(args(4), "--wd-v3-temp")
    assert(Files.isDirectory(java.nio.file.Paths.get(args(5))))
    assertEquals(PythonEnv.wdArguments(home, settings, true), Nil)
    assertEquals(PythonEnv.wdArguments(home, settings.copy(wdV3Tags = None), false), Nil)
    assertEquals(PythonEnv.wdArguments(home, Settings(), false), Nil)
  }

  test("WD choices persist and older settings retain empty defaults") {
    val file = Files.createTempDirectory("wd-settings-roundtrip").resolve("settings.json")
    Settings.save(file, Settings(wdV3Model = Some("/local/model.onnx"), wdV3Tags = Some("/local/tags.csv")))
    val loaded = Settings.load(file)
    assertEquals(loaded.wdV3Model, Some("/local/model.onnx"))
    assertEquals(loaded.wdV3Tags, Some("/local/tags.csv"))
    Files.writeString(file, "{}")
    val older = Settings.load(file)
    assertEquals(older.wdV3Model, None)
    assertEquals(older.wdV3Tags, None)
  }

  test("invalid output temp path disables only optional WD arguments") {
    val root = Files.createTempDirectory("wd-invalid-temp")
    val settings = Settings(outputDir = "\u0000", wdV3Model = Some("model"), wdV3Tags = Some("tags"))
    assertEquals(PythonEnv.wdArguments(AppHome(root), settings, false), Nil)
  }
