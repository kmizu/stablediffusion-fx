package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.event.Event
import javafx.scene.control.*
import javafx.scene.input.{KeyCode, KeyEvent, MouseButton, MouseEvent, ScrollEvent}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.comparison.ImageComparison
import sdfx.core.png.PngMetadata
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*

/** MANUAL native display acceptance; existing trained SD15/XL images only, no generation or GPU requests.
  * Args: image-A image-B output-directory. Run with each model family's existing PNG artifacts.
  */
object RealImageComparisonCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String)(body: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while !body && System.nanoTime() < end do Thread.sleep(40)
    require(body, "Timed out: " + label)
  private def hash(path: Path): String = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString
  def main(args: Array[String]): Unit =
    require(args.length == 3, "image-A image-B output-directory")
    val a = Paths.get(args(0)).toAbsolutePath.normalize()
    val b = Paths.get(args(1)).toAbsolutePath.normalize()
    require(Files.isRegularFile(a) && Files.isRegularFile(b))
    val before = Seq(hash(a), hash(b))
    val output = Paths.get(args(2)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "comparison-home-"))
    val services = new AppServices(home, Settings(outputDir = output.resolve("images").toString), dryRun = true,
      StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var checks = Vector.empty[String]
    def lookup[A <: javafx.scene.Node](id: String): A =
      val n = window.scene.lookup("#" + id)
      require(n != null, "Missing control: " + id)
      n.asInstanceOf[A]
    def button(id: String): Unit = fx { lookup[Button](id).fire() }
    def loaded(first: Path, second: Path): Unit = eventually("loaded image pair") {
      fx { window.comparison.loadedPaths.contains(first -> second) }
    }
    def snapshot(name: String): Unit = fx {
      window.root.applyCss()
      window.root.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", output.resolve(name).toFile))
    }
    def mouse(kind: javafx.event.EventType[MouseEvent], x: Double, y: Double, shift: Boolean): Unit = fx {
      Event.fireEvent(window.comparison.canvas, new MouseEvent(kind, x, y, x, y, MouseButton.PRIMARY, 1,
        shift, false, false, false, true, false, false, false, false, false, null))
    }
    try
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(selectedTab = "comparison"), CliOptions())
        w.show()
        w.comparison.inputA.set(a.toString)
        w.comparison.inputB.set(b.toString)
        w
      }
      button("comparison-upload")
      loaded(a, b)
      fx {
        require(window.state.selectedTab == "comparison")
        window.comparison.canvas.requestFocus()
        Event.fireEvent(window.comparison.canvas, new KeyEvent(KeyEvent.KEY_PRESSED, "+", "+", KeyCode.PLUS, false, false, false, false))
        require(window.comparison.canvas.view.zoom == 1.5)
        Event.fireEvent(window.comparison.canvas, new KeyEvent(KeyEvent.KEY_PRESSED, "", "", KeyCode.LEFT, false, false, false, false))
        require(window.comparison.canvas.view.panX == 100)
      }
      button("comparison-reset")
      val width = fx { window.comparison.canvas.getWidth }
      mouse(MouseEvent.MOUSE_PRESSED, width * .25, 100, false)
      fx { require(math.abs(window.comparison.canvas.view.split - .25) < .01) }
      mouse(MouseEvent.MOUSE_PRESSED, 100, 100, true)
      mouse(MouseEvent.MOUSE_DRAGGED, 140, 120, true)
      fx {
        require(window.comparison.canvas.view.panX == 40 && window.comparison.canvas.view.panY == 20)
        Event.fireEvent(window.comparison.canvas, new ScrollEvent(ScrollEvent.SCROLL, 100, 100, 100, 100,
          true, false, false, false, false, false, 0, 40, 0, 40,
          ScrollEvent.HorizontalTextScrollUnits.NONE, 0, ScrollEvent.VerticalTextScrollUnits.NONE, 0, 0, null))
        require(window.comparison.canvas.view.zoom == 1.5)
      }
      button("comparison-reset")
      fx {
        lookup[Slider]("comparison-opacity").setValue(.35)
        val direction = lookup[CheckBox]("comparison-horizontal")
        direction.fire()
        require(!window.comparison.canvas.view.horizontal && window.comparison.canvas.view.opacity == 1)
        lookup[Slider]("comparison-reveal").setValue(.4)
        lookup[Slider]("comparison-opacity").setValue(.65)
      }
      button("comparison-swap")
      fx {
        require(window.comparison.loadedPaths.contains(b -> a))
        require(window.comparison.canvas.view.split == .4 && window.comparison.canvas.view.opacity == .65)
      }
      checks :+= "native keyboard, mouse reveal, Shift pan/wheel, reset, direction, opacity and swap"
      val exported = output.resolve("comparison-export.png")
      fx { window.comparison.exportTo(exported) }
      require(Files.isRegularFile(exported) && PngMetadata.readParameters(exported).isEmpty)
      val actual = ImageIO.read(exported.toFile)
      val view = fx { window.comparison.canvas.view }
      val expected = ImageComparison.render(ImageIO.read(b.toFile), ImageIO.read(a.toFile), actual.getWidth, actual.getHeight, view)
      require(actual.getRGB(0, 0, actual.getWidth, actual.getHeight, null, 0, actual.getWidth)
        .sameElements(expected.getRGB(0, 0, expected.getWidth, expected.getHeight, null, 0, expected.getWidth)))
      fx {
        try
          window.comparison.exportTo(a)
          throw new AssertionError("Source overwrite was allowed")
        catch case _: IllegalArgumentException => ()
      }
      checks :+= "export exact pixels, no falsely attributed metadata and source overwrite refusal"
      fx {
        window.comparisonHistory.recordResult("prior", a, Mode.Txt2Img)
        window.comparisonHistory.recordResult("current", b, Mode.Txt2Img)
        window.comparisonHistory.recordResult("img", b, Mode.Img2Img)
        window.comparisonHistory.recordResult("paint", b, Mode.Inpaint)
        window.vms(Mode.Img2Img).initImage.set(a.toString)
        window.vms(Mode.Inpaint).initImage.set(a.toString)
        window.extras.image.set(a.toString)
        window.extras.showResult(b)
      }
      Seq("comparison-img2img", "comparison-inpaint", "comparison-extras", "comparison-previous").foreach { id =>
        fx { if window.comparison.loadedPaths.contains(a -> b) then window.comparison.swap() }
        button(id)
        loaded(a, b)
      }
      checks :+= "native source buttons for existing img2img/Inpaint/Extras and previous-job controller state"
      // Failed asynchronous loads must retain the displayed pair.
      fx { window.comparison.compare(output.resolve("missing.png"), b) }
      eventually("unreadable input error") { fx { lookup[Label]("comparison-status").getText.startsWith("Cannot compare:") } }
      fx { require(window.comparison.loadedPaths.contains(a -> b)) }
      checks :+= "failed input preserves displayed pair"
      fx { window.stage.setWidth(1280); window.stage.setHeight(800) }
      eventually("compact viewport") { fx { window.scene.getWidth < 1300 } }
      snapshot("comparison-compact.png")
      fx { window.stage.setWidth(1600); window.stage.setHeight(1000) }
      eventually("expanded viewport") { fx { window.scene.getWidth > 1500 } }
      snapshot("comparison-expanded.png")
      fx {
        lookup[TitledPane]("comparison-infotext").setExpanded(true)
        lookup[ScrollPane]("comparison-tool").setVvalue(1)
      }
      snapshot("comparison-infotext.png")
      fx {
        services.updateSettings(_.copy(comparisonMouseSend = true, comparisonMouseAuto = true))
        window.select(Mode.Txt2Img)
        window.preview.show(a)
        Event.fireEvent(window.preview.zoom, new MouseEvent(MouseEvent.MOUSE_CLICKED, 10, 10, 10, 10,
          MouseButton.PRIMARY, 1, false, true, true, false, false, false, false, false, false, true, null))
        require(window.comparison.inputA.get == a.toString && window.state.selectedTab == "comparison")
        window.select(Mode.Txt2Img)
        window.preview.show(b)
        Event.fireEvent(window.preview.zoom, new MouseEvent(MouseEvent.MOUSE_CLICKED, 10, 10, 10, 10,
          MouseButton.SECONDARY, 1, false, true, true, false, false, false, false, false, false, true, null))
        require(window.comparison.inputB.get == b.toString && window.state.selectedTab == "comparison")
      }
      loaded(a, b)
      checks :+= "opt-in CtrlAlt left/right native preview sends preserve original paths and switch tabs"
      // Real asynchronous interleaving: input edits happen in the same FX turn as the request,
      // before its callback can publish. Wait for actual completion before checking retained state.
      val draft = output.resolve("new-draft.png")
      Files.copy(a, draft, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      fx {
        window.comparison.canvas.update(window.comparison.canvas.view.copy(zoom = 2, panX = 43, panY = -27))
        window.comparison.compare(b, a)
        window.comparison.inputA.set(draft.toString)
      }
      eventually("superseded pair decode completed") { fx { !window.comparison.isLoading } }
      fx {
        require(window.comparison.inputA.get == draft.toString, "Stale pair callback overwrote a new draft")
        require(window.comparison.loadedPaths.contains(a -> b), "Superseded decode replaced the displayed pair")
        window.comparison.compare(b, a)
        window.comparison.inputA.set("")
      }
      eventually("cleared input decode completed") { fx { !window.comparison.isLoading } }
      fx {
        require(window.comparison.inputA.get.isEmpty, "Stale pair callback undid Clear")
        require(window.comparison.loadedPaths.contains(a -> b))
        window.comparison.compare(b, a)
      }
      loaded(b, a)
      fx {
        val v = window.comparison.canvas.view
        require(v.zoom == 2 && v.panX == 43 && v.panY == -27 && v.split == .5 && v.opacity == 1,
          "New source pair lost zoom/pan or failed to reset reveal/opacity")
      }
      checks :+= "actual decode completion preserves later draft/Clear and shared transforms"
      val savedView = fx {
        window.select(Mode.Txt2Img)
        window.preview.show(a)
        window.preview.zoom.image
      }
      fx { window.preview.showLive(java.util.Base64.getEncoder.encodeToString(Files.readAllBytes(b))) }
      eventually("live replacement published") { fx { window.preview.zoom.image ne savedView } }
      fx {
        require(window.preview.zoom.getUserData == null, "Live pixels retained the previous saved path")
        Event.fireEvent(window.preview.zoom, new MouseEvent(MouseEvent.MOUSE_CLICKED, 10, 10, 10, 10,
          MouseButton.SECONDARY, 1, false, true, true, false, false, false, false, false, false, true, null))
      }
      eventually("displayed live pixels sent to comparison") { fx {
        val path = window.comparison.inputB.get
        path.nonEmpty && path != a.toString && path != b.toString && !window.comparison.isLoading
      } }
      val live = fx { Paths.get(window.comparison.inputB.get) }
      require(PngMetadata.readParameters(live).isEmpty, "Live copy inherited saved-image metadata")
      val copied = ImageIO.read(live.toFile)
      val original = ImageIO.read(b.toFile)
      require(copied.getWidth == original.getWidth && copied.getHeight == original.getHeight)
      require(copied.getRGB(0, 0, copied.getWidth, copied.getHeight, null, 0, copied.getWidth)
        .sameElements(original.getRGB(0, 0, original.getWidth, original.getHeight, null, 0, original.getWidth)),
        "Shortcut sent stale saved pixels instead of the displayed live image")
      checks :+= "saved-to-live preview sends displayed pixels with no stale path or metadata"
      checks :+= "compact, expanded and metadata-difference native screenshots"
      require(Seq(hash(a), hash(b)) == before, "Original image bytes changed")
      require(services.supervisor.hello.isEmpty, "Display acceptance unexpectedly started a diffusion engine")
      Files.writeString(output.resolve("comparison-report.json"), ujson.write(ujson.Obj("success" -> true,
        "source_hashes" -> ujson.Arr.from(before), "checks" -> ujson.Arr.from(checks), "engine_started" -> false,
        "export" -> exported.toString), indent = 2))
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown()
      Platform.exit()
