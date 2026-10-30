package sdfx.ui

import javafx.beans.property.{SimpleDoubleProperty, SimpleIntegerProperty, SimpleObjectProperty}
import sdfx.core.config.Settings
import sdfx.core.gallery.GalleryItem
import sdfx.core.model.*
import sdfx.core.plugins.api.*
import sdfx.ui.vm.*

import java.nio.file.{Files, Path, Paths}

class FixRoundSuite extends munit.FunSuite:
  test("temp cleanup ignores malformed saved paths") {
    val dir = Files.createTempDirectory("invalid-cleanup-path")
    val tracker = new TempTracker(() => dir)
    assertEquals(tracker.closeInputs(Set("\u0000")), 0)
    Files.delete(dir)
  }
  test("closed window removes an abandoned input after the last queued job releases it") {
    val dir = Files.createTempDirectory("closed-held-input")
    val tracker = new TempTracker(() => dir)
    val image = tracker.createInputTemp("paste-", ".png")
    assert(tracker.completeInputWrite(image))
    tracker.holdJobInputs("first", Seq(image)); tracker.holdJobInputs("second", Seq(image))
    tracker.closeInputs(Set.empty)
    assert(Files.exists(image))
    tracker.releaseJob("first")
    assert(Files.exists(image))
    tracker.releaseJob("second")
    assert(!Files.exists(image))
    Files.delete(dir)
  }
  test("closing input tracking preserves referenced images and cleans abandoned or late writes") {
    val dir = Files.createTempDirectory("closing-inputs")
    val tracker = new TempTracker(() => dir)
    val kept = tracker.createInputTemp("paste-", ".png")
    val abandoned = tracker.createInputTemp("drop-", ".png")
    val late = tracker.createInputTemp("paste-", ".png")
    assert(tracker.completeInputWrite(kept))
    assert(tracker.completeInputWrite(abandoned))
    tracker.closeInputs(Set(kept.toString))
    assert(Files.exists(kept) && !Files.exists(abandoned) && Files.exists(late))
    assert(!tracker.completeInputWrite(late))
    assert(!Files.exists(late))
    intercept[IllegalStateException](tracker.createInputTemp("paste-", ".png"))
    Files.delete(kept); Files.delete(dir)
  }
  test("a newly created pasted image remains pinned while its UI callback is pending") {
    val dir = Files.createTempDirectory("pending-paste")
    val tracker = new TempTracker(() => dir)
    val image = tracker.createInputTemp("paste-", ".png")
    assertEquals(tracker.sweep(Set.empty), 0)
    assert(Files.exists(image))
    tracker.forgetInput(image)
    assertEquals(tracker.sweep(Set.empty), 1)
    Files.delete(dir)
  }
  test("queued input references survive sweeps and are not deleted on completion") {
    val dir = Files.createTempDirectory("queued-inputs")
    val image = Files.write(dir.resolve("paste-base.png"), Array[Byte](1))
    val tracker = new TempTracker(() => dir)
    tracker.holdJobInputs("a", Seq(image))
    tracker.holdJobInputs("b", Seq(image))
    assertEquals(tracker.sweep(Set.empty), 0)
    tracker.releaseJob("a")
    assert(Files.exists(image))
    assertEquals(tracker.sweep(Set.empty), 0)
    tracker.releaseJob("b")
    assert(Files.exists(image))
    assertEquals(tracker.sweep(Set.empty), 1)
    Files.delete(dir)
  }

  // ---- JobBuilder ----------------------------------------------------------------------------------------

  private object Hook extends Plugin:
    def id = "hook"
    def name = "Hook"
    def kind = PluginKind.AlwaysOn
    override def controls: Seq[Control] = Seq(Control.Check("enabled", "Enabled"), Control.Slider("s", "S", 0, 1, 0.1, 0.5))

  private object Grid extends Plugin:
    def id = "grid"
    def name = "Grid"
    def kind = PluginKind.Script
    override def controls: Seq[Control] = Seq(Control.Text("x", "X", "1,2"))

  private def plugins(mode: Mode, hookOn: Boolean = false) =
    new PluginsVM(mode, Seq(Grid), Seq(Hook), p => if p.id == "hook" && hookOn then Values(Map("enabled" -> "true")) else Values.empty)

  test("JobBuilder: txt2img carries hires, no img2img, no script") {
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.prompt.set("cat")
    vm.hiresEnabled.set(true)
    vm.initImage.set("ignored.png")
    val job = JobBuilder.build(vm, plugins(Mode.Txt2Img))
    assert(job.params.hires.enabled)
    assertEquals(job.params.img2img, None)
    assertEquals(job.script, None)
    assertEquals(job.alwaysOn, Nil)
    assertEquals(job.origin, "ui")
    assert(job.id.nonEmpty)
  }

  test("JobBuilder captures the image session snapshot without serializing its secret") {
    val session = new sdfx.core.png.ImageScrambleSession
    session.unlock("queued secret")
    val job = JobBuilder.build(new GenerationVM(Mode.Txt2Img), plugins(Mode.Txt2Img), imageSecret = session.snapshot)
    session.lock()
    assert(job.imageSecret.exists(_.use(identity) == "queued secret"))
    assert(!job.toString.contains("queued secret"))
    assert(!job.submittedContext.toString.contains("queued secret"))
  }

  test("JobBuilder: img2img has the init image and no mask; hires is dropped") {
    val vm = new GenerationVM(Mode.Img2Img)
    vm.initImage.set("init.png")
    vm.maskImage.set("stray-mask.png")
    vm.hiresEnabled.set(true)
    val job = JobBuilder.build(vm, plugins(Mode.Img2Img))
    assertEquals(job.params.img2img.map(_.initImage), Some("init.png"))
    assertEquals(job.params.img2img.flatMap(_.mask), None)
    assert(!job.params.hires.enabled)
  }

  test("JobBuilder: inpaint sends init image + mask") {
    val vm = new GenerationVM(Mode.Inpaint)
    vm.initImage.set("init.png")
    vm.maskImage.set("mask.png")
    vm.inpaintFullRes.set(true)
    val job = JobBuilder.build(vm, plugins(Mode.Inpaint))
    assert(job.params.isInpaint)
    assertEquals(job.params.img2img.flatMap(_.mask), Some("mask.png"))
    assert(job.params.img2img.exists(_.inpaintFullRes))
  }

  test("JobBuilder: selected script and only enabled always-on plugins") {
    val pv = plugins(Mode.Txt2Img, hookOn = true)
    pv.selectedScript.set("grid")
    pv.form("grid").get.prop("x").get.set("3,4")
    val job = JobBuilder.build(new GenerationVM(Mode.Txt2Img), pv)
    assertEquals(job.script.map((p, v) => p.id -> v.str("x")), Some("grid" -> "3,4"))
    assertEquals(job.alwaysOn.map((p, v) => p.id -> v.str("s")), Seq("hook" -> "0.5"))
    val off = JobBuilder.build(new GenerationVM(Mode.Txt2Img), plugins(Mode.Txt2Img))
    assertEquals(off.alwaysOn, Nil)
    assertEquals(off.limits, sdfx.engine.JobLimits())
  }

  // ---- stale mask ----------------------------------------------------------------------------------------

  test("a mask exported for an earlier job does not satisfy validation once the canvas is cleared") {
    val vm = new GenerationVM(Mode.Inpaint)
    vm.initImage.set("a.png")
    vm.hasMask.set(true)
    vm.maskImage.set("old-mask.png") // exported by a previous generate
    assertEquals(vm.validate(_ => true), Nil)
    vm.hasMask.set(false) // user pressed Clear
    assert(vm.validate(_ => true).exists(_.contains("Paint a mask")))
  }

  test("changing the init image drops the exported mask") {
    val vm = new GenerationVM(Mode.Inpaint)
    vm.initImage.set("a.png")
    vm.maskImage.set("mask-a.png")
    vm.initImage.set("b.png")
    assertEquals(vm.maskImage.get, "")
    assertEquals(vm.toParams.img2img.flatMap(_.mask), None)
  }

  // ---- slider / spinner clamping ------------------------------------------------------------------------

  test("int slider shows the clamped value but keeps a larger typed model value") {
    val model = new SimpleIntegerProperty(512)
    val slider = new SimpleDoubleProperty(0)
    NumberSync.bindIntSlider(model, slider, 64, 2048, 8)
    assertEquals(slider.get, 512.0)
    model.set(3000) // typed into the spinner (max 4096)
    assertEquals(model.get, 3000)
    assertEquals(slider.get, 2048.0)
    slider.set(1003) // user drags: rounds to the step
    assertEquals(model.get, 1000)
    model.set(4000)
    assertEquals(slider.get, 2048.0)
    assertEquals(model.get, 4000) // the sync itself never writes the clamp back
    slider.set(1500)
    assertEquals(model.get, 1504) // 64 + k*8
  }

  test("double slider rounds to the step and keeps out-of-range model values") {
    val model = new SimpleDoubleProperty(7)
    val slider = new SimpleDoubleProperty(0)
    NumberSync.bindDoubleSlider(model, slider, 1, 30, 0.5)
    slider.set(7.26)
    assertEquals(model.get, 7.5)
    model.set(35)
    assertEquals(model.get, 35.0)
    assertEquals(slider.get, 30.0)
  }

  test("spinner value factory sync keeps values within its own range") {
    val model = new SimpleIntegerProperty(512)
    val factory = new SimpleObjectProperty[Integer](512)
    NumberSync.bindIntFactory(model, factory)
    model.set(3000)
    assertEquals(factory.get.intValue, 3000)
    factory.set(4096)
    assertEquals(model.get, 4096)
  }

  // ---- MCP / temps / gallery / settings -----------------------------------------------------------------

  test("McpController starts, restarts on port / token change and stops") {
    var started = List.empty[(Int, Option[String])]
    var closed = 0
    val c = new McpController((port, tok) =>
      started = started :+ (port -> tok)
      () => closed += 1
    )
    assert(!c.apply(enabled = false, 7861, None))
    assert(c.apply(enabled = true, 7861, Some(" ")))
    assertEquals(started, List(7861 -> None))
    assert(!c.apply(enabled = true, 7861, None)) // unchanged
    assert(c.apply(enabled = true, 7862, Some("t")))
    assertEquals(closed, 1)
    assertEquals(c.port, Some(7862))
    assert(c.apply(enabled = false, 7862, Some("t")))
    assertEquals(closed, 2)
    assert(!c.isRunning)
  }

  test("McpController propagates bind failures and stays stopped") {
    val c = new McpController((_, _) => throw new java.net.BindException("in use"))
    intercept[java.net.BindException](c.apply(enabled = true, 1, None))
    assert(!c.isRunning)
  }

  test("TempTracker deletes job files and sweeps unreferenced UI temps") {
    val dir = Files.createTempDirectory("sdfx-tmp")
    val t = new TempTracker(() => dir)
    def mk(n: String): Path = Files.writeString(dir.resolve(n), "x")
    val mask = mk("mask-1.png")
    val held = mk("mask-2.png")
    val paste = mk("paste-1.png")
    val kept = mk("paste-2.png")
    val engine = mk("sdfx-engine.png")
    t.registerJob("j1", Seq(mask))
    t.registerJob("j2", Seq(held))
    assertEquals(t.releaseJob("j1"), 1)
    assert(!Files.exists(mask))
    assertEquals(t.sweep(Set(kept.toString)), 1)
    assert(!Files.exists(paste))
    assert(Files.exists(kept) && Files.exists(held) && Files.exists(engine))
    assert(!t.isUiTemp(Paths.get("elsewhere/mask-1.png")))
  }

  test("gallery merge keeps images saved during the scan") {
    def item(n: String, mod: Long) = GalleryItem(Paths.get(n), mod, 1, None)
    val g = new GalleryVM
    g.setAll(Seq(item("old.png", 10)))
    g.addNewest(item("new.png", 200)) // saved while scanning (since = 100)
    g.merge(Seq(item("a.png", 50), item("b.png", 40)), since = 100)
    assertEquals(g.items.toArray.toSeq.map(_.asInstanceOf[GalleryItem].fileName), Seq("new.png", "a.png", "b.png"))
  }

  test("settings merge only applies dialog fields") {
    val cur = Settings(lastModel = Some("m"), lastVae = Some("v"), pluginValues = Map("p" -> Map("k" -> "1")))
    val edited = Settings(mcpHttpEnabled = true, mcpHttpPort = 9000, theme = "light", lastModel = None)
    val r = SettingsMerge(cur, edited)
    assertEquals(r.lastModel, Some("m"))
    assertEquals(r.lastVae, Some("v"))
    assertEquals(r.pluginValues, cur.pluginValues)
    assert(r.mcpHttpEnabled)
    assertEquals(r.mcpHttpPort, 9000)
    assertEquals(r.theme, "light")
  }
