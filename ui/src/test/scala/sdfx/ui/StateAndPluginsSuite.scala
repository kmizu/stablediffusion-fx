package sdfx.ui

import sdfx.core.gallery.GalleryItem
import sdfx.core.model.*
import sdfx.core.plugins.api.*
import sdfx.ui.vm.*

import java.nio.file.{Files, Paths}

class StateAndPluginsSuite extends munit.FunSuite:

  test("legacy_state_without_gate_defaults_true") {
    val old = UiState(params = Mode.values.map(m => m.key -> GenerationParams(subseed = 99L, subseedStrength = 0.4)).toMap)
    val legacyJson = ujson.read(UiState.toJson(old)).obj
    legacyJson.remove("variation_enabled")
    val decoded = UiState.fromJson(ujson.write(ujson.Obj.from(legacyJson)))
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.load(decoded.params(mode.key), variationEnabledState = decoded.variationEnabled.getOrElse(mode.key, true))
      assert(vm.variationEnabled.get)
      assertEquals(vm.toParams.subseedStrength, 0.4)
    }
  }

  test("disabled_restart_keeps_subseed_strength_and_gate") {
    val originals = Mode.values.map { mode =>
      val vm = new GenerationVM(mode)
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      vm.variationEnabled.set(false)
      mode -> vm
    }.toMap
    val saved = UiState(
      params = Mode.values.map(m => m.key -> originals(m).snapshot).toMap,
      variationEnabled = Mode.values.map(m => m.key -> originals(m).variationEnabled.get).toMap
    )
    val restored = UiState.fromJson(UiState.toJson(saved))
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.load(restored.params(mode.key), variationEnabledState = restored.variationEnabled.getOrElse(mode.key, true))
      assert(!vm.variationEnabled.get)
      assertEquals(vm.subseed.get, 99L)
      assertEquals(vm.subseedStrength.get, 0.4)
      assertEquals(vm.toParams.subseedStrength, 0.0)
    }
  }

  test("UiState JSON round trip") {
    val s = UiState(
      window = WindowState(Some(10.5), Some(20), 1600, 1000, maximized = true),
      dividers = Map("main" -> Seq(0.4), "right" -> Seq(0.7)),
      selectedTab = "inpaint",
      params = Map(
        "txt2img" -> GenerationParams(prompt = "p, <lora:x:0.5>", hires = HiresParams(enabled = true)),
        "inpaint" -> GenerationParams(img2img = Some(Img2ImgParams("i.png", None, inpaintFullRes = true)))
      ),
      pluginValues = Map("txt2img" -> Map("freeu" -> Map("enabled" -> "true", "b1" -> "1.1"))),
      selectedScript = Map("txt2img" -> "xyz_grid"),
      expanded = Map("txt2img.hires" -> true),
      logVisible = true,
      galleryQuery = "cat",
      lastSeed = 42,
      extrasUpscaler = "R-ESRGAN 4x+",
      extrasScale = 4
    )
    val json = UiState.toJson(s)
    assert(json.contains("\"selected_tab\""), json)
    assertEquals(UiState.fromJson(json), s)
  }

  test("UiState load/save via files, unreadable files give defaults") {
    val dir = Files.createTempDirectory("sdfx-ui-state")
    val f = dir.resolve("config").resolve("ui-state.json")
    assertEquals(UiState.load(f), UiState())
    val s = UiState(selectedTab = "img2img", lastSeed = 7)
    UiState.save(f, s)
    assertEquals(UiState.load(f), s)
    Files.writeString(f, "{ not json")
    assertEquals(UiState.load(f), UiState())
  }

  private object Demo extends Plugin:
    def id = "demo"
    def name = "Demo"
    def kind = PluginKind.AlwaysOn
    override def controls: Seq[Control] = Seq(
      Control.Check("enabled", "Enabled"),
      Control.Slider("strength", "Strength", 0, 2, 0.05, 0.75),
      Control.IntField("count", "Count", 1, 10, 3),
      Control.Choice("mode", "Mode", Seq("a", "b"), "b"),
      Control.DynamicChoice("sampler", "Sampler", ChoiceSource.Samplers),
      Control.Text("text", "Text", "hello", multiline = true),
      Control.Text("line", "Line"),
      Control.Note("note", "Just a note")
    )

  private object Script extends Plugin:
    def id = "script"
    def name = "Script"
    def kind = PluginKind.Script
    override def controls: Seq[Control] = Seq(Control.Text("values", "Values", "1,2"))

  test("plugin controls map to widgets and default values") {
    import Widget.*
    assertEquals(Demo.controls.map(PluginValuesVM.widgetFor), Seq(CheckBox, SliderSpinner, IntSpinner, Combo, DynamicCombo, TextArea, TextField, Label))
    val vm = new PluginValuesVM(Demo)
    assertEquals(
      vm.values,
      Values(Map("enabled" -> "false", "strength" -> "0.75", "count" -> "3", "mode" -> "b", "sampler" -> "", "text" -> "hello", "line" -> ""))
    )
    assertEquals(vm.values, PluginValuesVM.defaults(Demo.controls))
    assert(!vm.props.contains("note"))
    assert(!vm.isActive)
    vm.prop("enabled").get.set("true")
    assert(vm.isActive)
    vm.reset()
    assert(!vm.isActive)
    val dyn = Demo.controls.collectFirst { case d: Control.DynamicChoice => d }.get
    assertEquals(PluginValuesVM.options(dyn, _ => Seq("Euler", "DDIM")), Seq("None", "Euler", "DDIM"))
  }

  test("PluginsVM builds the job script and active AlwaysOn plugins") {
    val pv = new PluginsVM(Mode.Txt2Img, Seq(Script), Seq(Demo), p => if p.id == "demo" then Values(Map("enabled" -> "true")) else Values.empty)
    assertEquals(pv.scriptJob, None)
    assertEquals(pv.alwaysOnJob.map(_._1.id), Seq("demo"))
    assertEquals(pv.alwaysOnJob.head._2.str("strength"), "0.75")
    pv.selectedScript.set("script")
    assertEquals(pv.scriptJob.map(_._2.str("values")), Some("1,2"))
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.prompt.set("x")
    val job = JobBuilder.build(vm, pv)
    assertEquals(job.origin, "ui")
    assertEquals(job.params.prompt, "x")
    assertEquals(job.script.map(_._1.id), Some("script"))
    assertEquals(job.alwaysOn.map(_._1.id), Seq("demo"))
    assertEquals(pv.snapshot("demo")("enabled"), "true")
  }

  test("gallery filter and navigation") {
    def item(n: String, params: String) = GalleryItem(Paths.get(n), 0, 1, Some(params))
    val g = new GalleryVM
    g.setAll(Seq(item("b.png", "a red car"), item("a.png", "a blue cat")))
    g.addNewest(item("c.png", "red cat"))
    assertEquals(g.items.get(0).fileName, "c.png")
    g.query.set("red")
    assertEquals(g.filtered.size, 2)
    g.query.set("red cat")
    assertEquals(g.filtered.size, 1)
    g.query.set("")
    assertEquals(g.neighbor(Paths.get("c.png"), 1).map(_.fileName), Some("b.png"))
    assertEquals(g.neighbor(Paths.get("c.png"), -1), None)
    g.remove(Paths.get("b.png"))
    assertEquals(g.items.size, 2)
  }

  test("cli options") {
    val o = CliOptions.parse(Seq("--dry-run", "--home", "X", "--screenshot", "s.png", "--exit-after", "3", "--autogen", "--tab", "inpaint", "--other"))
    assertEquals(o, CliOptions(dryRun = true, home = Some("X"), screenshot = Some("s.png"), exitAfter = Some(3.0), autogen = true, tab = Some("inpaint")))
    assert(CliOptions.parse(Seq("--mcp")).mcp)
  }
