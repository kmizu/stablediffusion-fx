package sdfx.core

import sdfx.core.config.*
import sdfx.core.json.SnakePickle

import java.nio.file.{Files, Path, Paths}

class SettingsSuite extends munit.FunSuite with TempDir:

  val custom: Settings = Settings(
    modelDirs = ModelDirs().withA1111("D:/sd"),
    outputDir = "D:/out",
    pythonExe = Some("C:/py/python.exe"),
    dtype = "bf16",
    vramMode = "lowvram",
    vaeTiling = true,
    previewEvery = 5,
    previewMethod = "taesd",
    lastModel = Some("animagineXL.safetensors"),
    lastVae = None,
    mcpHttpEnabled = true,
    mcpHttpPort = 9000,
    mcpToken = Some("secret"),
    disabledPlugins = Seq("loopback"),
    pluginValues = Map("freeU" -> Map("b1" -> "1.3", "enabled" -> "true"), "xyz_grid" -> Map.empty),
    hfOffline = true,
    theme = "light"
  )

  tmp.test("save / load round trip") { dir =>
    val f = dir.resolve("config/settings.json")
    Settings.save(f, custom)
    assertEquals(Settings.load(f), custom)
  }

  tmp.test("saved JSON uses snake_case keys and writes defaults, map keys verbatim") { dir =>
    val f = dir.resolve("settings.json")
    Settings.save(f, custom)
    val js = ujson.read(Files.readString(f))
    Seq("model_dirs", "output_dir", "python_exe", "vae_tiling", "preview_every", "last_vae", "mcp_http_port",
      "mcp_token", "disabled_plugins", "plugin_values", "hf_offline").foreach(k => assert(js.obj.contains(k), k))
    assertEquals(js("last_vae"), ujson.Null)
    assertEquals(js("plugin_values").obj.keys.toSet, Set("freeU", "xyz_grid"))
    assert(js("model_dirs").obj.contains("checkpoints"))
  }

  tmp.test("save overwrites atomically and leaves no temp files") { dir =>
    val f = dir.resolve("settings.json")
    Settings.save(f, Settings())
    Settings.save(f, custom)
    assertEquals(Settings.load(f), custom)
    assertEquals(Files.list(dir).count(), 1L)
  }

  tmp.test("missing file -> defaults") { dir =>
    assertEquals(Settings.load(dir.resolve("none.json")), Settings())
  }

  tmp.test("unknown keys ignored, missing keys default") { dir =>
    val f = dir.resolve("s.json")
    Files.writeString(f, """{"theme":"light","future_option":{"a":1},"model_dirs":{"lora":["x"],"new_kind":[]}}""")
    val s = Settings.load(f)
    assertEquals(s.theme, "light")
    assertEquals(s.modelDirs.lora, Seq("x"))
    assertEquals(s.modelDirs.checkpoints, ModelDirs().checkpoints)
    assertEquals(s.copy(theme = "dark", modelDirs = ModelDirs()), Settings())
  }

  tmp.test("empty object -> defaults") { dir =>
    val f = dir.resolve("s.json")
    Files.writeString(f, "{}")
    assertEquals(Settings.load(f), Settings())
  }

  tmp.test("corrupt / wrongly typed file -> defaults") { dir =>
    val f = dir.resolve("s.json")
    Seq("{not json", "", "[]", "null", """{"mcp_http_port":"abc"}""", "\u0000\u0001garbage").foreach { c =>
      Files.writeString(f, c)
      val original = Files.readAllBytes(f)
      val result = Settings.loadResult(f)
      assertEquals(result.settings, Settings(nsfwCardBlurDefault = "Hide"), c)
      assert(result.recovery.nonEmpty, c)
      assertEquals(Files.readAllBytes(f).toSeq, original.toSeq, c)
    }
  }

  tmp.test("UTF-8 values survive") { dir =>
    val f = dir.resolve("s.json")
    val s = Settings(outputDir = "D:/出力/画像", pluginValues = Map("p" -> Map("prompt" -> "桜, 夜")))
    Settings.save(f, s)
    assertEquals(Settings.load(f), s)
  }

  test("ModelDirs.withA1111 adds A1111 folders") {
    val root = "D:/stable-diffusion-webui"
    val d = ModelDirs().withA1111(root)
    def p(sub: String) = Paths.get(root, sub).toString
    assertEquals(d.checkpoints, Seq("models/checkpoints", p("models/Stable-diffusion")))
    assertEquals(d.vae, Seq("models/vae", p("models/VAE")))
    assertEquals(d.lora, Seq("models/lora", p("models/Lora"), p("models/LyCORIS")))
    assertEquals(d.controlnet, Seq("models/controlnet", p("models/ControlNet")))
    assertEquals(d.embeddings, Seq("models/embeddings", p("embeddings")))
    assertEquals(d.upscalers, Seq("models/upscalers", p("models/ESRGAN"), p("models/RealESRGAN"), p("models/SwinIR"),
      p("models/ScuNET"), p("models/DAT")))
    assertEquals(d.faceRestore, Seq("models/face_restore", p("models/GFPGAN"), p("models/Codeformer")))
    assertEquals(d.svd, Seq("models/svd", p("models/svd")))
    assertEquals(d.zero123, Seq("models/zero123", p("models/z123")))
  }

  test("ModelDirs.withA1111 is idempotent") {
    val once = ModelDirs().withA1111("D:/sd")
    assertEquals(once.withA1111("D:/sd"), once)
    val two = once.withA1111("E:/forge")
    assertEquals(two.checkpoints.size, 3)
  }

  test("ModelDirs JSON round trip") {
    val d = ModelDirs(checkpoints = Nil, lora = Seq("a", "b"))
    assertEquals(SnakePickle.read[ModelDirs](SnakePickle.write(d)), d)
  }

class AppHomeSuite extends munit.FunSuite with TempDir:

  test("resolve: relative against root, normalized") {
    val root = Paths.get("D:/app").toAbsolutePath
    val h = AppHome(root)
    assertEquals(h.resolve("outputs"), root.resolve("outputs"))
    assertEquals(h.resolve("models/../outputs/./x"), root.resolve("outputs/x"))
    assertEquals(h.resolve(""), root)
  }

  test("resolve: absolute stays") {
    val abs = Paths.get("E:/elsewhere/out").toAbsolutePath
    assertEquals(AppHome(Paths.get("D:/app").toAbsolutePath).resolve(abs.toString), abs)
  }

  test("derived paths") {
    val root = Paths.get("D:/app").toAbsolutePath
    val h = AppHome(root)
    assertEquals(h.pythonDir, root.resolve("python"))
    assertEquals(h.configDir, root.resolve("config"))
    assertEquals(h.settingsFile, root.resolve("config").resolve("settings.json"))
    assertEquals(h.stylesFile, root.resolve("config").resolve("styles.csv"))
    assertEquals(h.pluginsDir, root.resolve("plugins-ext"))
    assertEquals(h.outputs(Settings()), root.resolve("outputs"))
    assertEquals(h.outputs(Settings(outputDir = "x/../y")), root.resolve("y"))
  }

  tmp.test("tmpDir is created under outputs") { dir =>
    val h = AppHome(dir)
    val t = h.tmpDir(Settings(outputDir = "out"))
    assertEquals(t, dir.resolve("out").resolve(".tmp"))
    assert(Files.isDirectory(t))
  }

  private def withProp[A](k: String, v: String)(body: => A): A =
    val old = System.getProperty(k)
    System.setProperty(k, v)
    try body
    finally if old == null then System.clearProperty(k) else System.setProperty(k, old)

  tmp.test("detect honours -Dsdfx.home when it contains python/sdfx") { dir =>
    Files.createDirectories(dir.resolve("python/sdfx"))
    withProp("sdfx.home", dir.toString) {
      assertEquals(AppHome.detect().root, dir.toAbsolutePath.normalize())
    }
  }

  tmp.test("detect ignores an invalid -Dsdfx.home and still returns an absolute root") { dir =>
    withProp("sdfx.home", dir.resolve("not-a-home").toString) {
      val r = AppHome.detect().root
      assert(r.isAbsolute)
      assertNotEquals(r, dir.resolve("not-a-home").toAbsolutePath.normalize())
    }
  }

  tmp.test("detect normalizes the property path") { dir =>
    Files.createDirectories(dir.resolve("app/python/sdfx"))
    withProp("sdfx.home", dir.resolve("app/../app/.").toString) {
      assertEquals(AppHome.detect().root, dir.resolve("app").toAbsolutePath.normalize())
    }
  }
