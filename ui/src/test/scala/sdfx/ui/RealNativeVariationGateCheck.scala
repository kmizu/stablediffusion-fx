package sdfx.ui

import javafx.application.Platform
import javafx.scene.control.CheckBox
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.model.GenerationParams
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

import java.awt.image.BufferedImage
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO

/** Root-only offline native check of the actual variation pane, checkbox, restart and image routes.
  * Arg: output directory. No generation, Python worker, model load, or system clipboard access.
  */
object RealNativeVariationGateCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def check(condition: Boolean, message: String): Unit = require(condition, message)

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "variation-gate-home-"))
    val imagePath = home.root.resolve("source.png")
    check(ImageIO.write(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB), "png", imagePath.toFile), "fixture PNG write failed")
    val services = new AppServices(home, Settings(), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    check(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var first: MainWindow = null
    var second: MainWindow = null
    var failure: Throwable = null
    var checks = Vector.empty[String]
    def checked(label: String): Unit = { checks :+= label; println("VARIATION GATE CHECK: " + label) }
    def checkbox(window: MainWindow, mode: Mode): CheckBox =
      val pane = window.panels(mode).sections(s"${mode.key}.variation")
      val found = pane.getContent.lookup(s"#variation-enabled-${mode.key}")
      check(found != null, s"missing checkbox for ${mode.key}")
      found.asInstanceOf[CheckBox]
    def noEngine(): Unit =
      check(services.supervisor.pid.isEmpty && services.supervisor.loadedModel.isEmpty,
        "variation UI started an engine or loaded a model")
    try
      val initial = UiState(
        params = Mode.values.map(m => m.key -> GenerationParams(subseed = 99L, subseedStrength = 0.4)).toMap,
        variationEnabled = Map("txt2img" -> false, "img2img" -> true, "inpaint" -> false),
        expanded = Map("txt2img.variation" -> false, "img2img.variation" -> true, "inpaint.variation" -> false))
      first = fx {
        val window = new MainWindow(new Stage(), services, initial, CliOptions(), _ => ())
        window.show()
        window
      }
      fx {
        Mode.values.foreach { mode =>
          val vm = first.vms(mode)
          val pane = first.panels(mode).sections(s"${mode.key}.variation")
          val control = checkbox(first, mode)
          val expectedGate = initial.variationEnabled(mode.key)
          check(control.isSelected == expectedGate && vm.variationEnabled.get == expectedGate, s"${mode.key} startup gate")
          check(pane.isExpanded == initial.expanded(s"${mode.key}.variation"), s"${mode.key} disclosure restore")
          check(vm.subseed.get == 99L && vm.subseedStrength.get == 0.4, s"${mode.key} startup drafts")
          pane.setExpanded(!pane.isExpanded)
          pane.setExpanded(!pane.isExpanded)
          check(vm.variationEnabled.get == expectedGate, s"${mode.key} collapse changed gate")
          check(vm.toParams.subseedStrength == (if expectedGate then 0.4 else 0.0), s"${mode.key} collapse changed effective strength")
          control.fire()
          check(vm.variationEnabled.get != expectedGate, s"${mode.key} checkbox is not bound")
          check(vm.subseed.get == 99L && vm.subseedStrength.get == 0.4, s"${mode.key} checkbox destroyed drafts")
          control.fire()
          check(vm.variationEnabled.get == expectedGate, s"${mode.key} checkbox re-toggle failed")
        }
        val txt = first.vms(Mode.Txt2Img)
        txt.subseedStrength.set(0.6)
        check(!txt.variationEnabled.get && txt.toParams.subseedStrength == 0.0 && txt.snapshot.subseedStrength == 0.6,
          "editing strength while disabled changed the gate or draft")
      }
      checked("actual pane disclosure and checkbox are independent in all three modes")
      val saved = fx { first.state }
      check(saved.variationEnabled == initial.variationEnabled, "state capture lost per-mode gates")
      check(saved.params("txt2img").subseedStrength == 0.6, "state capture lost disabled draft")
      UiState.save(UiState.file(home), saved)
      fx { first.dispose(); first.stage.close() }
      first = null
      second = fx {
        val window = new MainWindow(new Stage(), services, UiState.load(UiState.file(home)), CliOptions(), _ => ())
        window.show()
        window
      }
      fx {
        Mode.values.foreach { mode =>
          check(second.vms(mode).variationEnabled.get == initial.variationEnabled(mode.key), s"${mode.key} restart gate")
          check(checkbox(second, mode).isSelected == initial.variationEnabled(mode.key), s"${mode.key} restart checkbox")
          check(second.vms(mode).subseed.get == 99L, s"${mode.key} restart subseed")
        }
        check(second.vms(Mode.Txt2Img).subseedStrength.get == 0.6, "restart strength draft")
        second.select(Mode.Txt2Img)
        second.restore("house\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512, Variation seed: 99, Variation seed strength: 0.4")
        check(second.vms(Mode.Txt2Img).variationEnabled.get && second.vms(Mode.Txt2Img).toParams.subseedStrength == 0.4,
          "MainWindow Restore did not enable variation metadata")
        second.restore("house\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512")
        check(!second.vms(Mode.Txt2Img).variationEnabled.get && second.vms(Mode.Txt2Img).subseed.get == -1L &&
          second.vms(Mode.Txt2Img).subseedStrength.get == 0.0, "MainWindow Restore did not clear absent variation")
        second.sendTo(Mode.Img2Img, imagePath,
          Some("house\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512, Variation seed: 99, Variation seed strength: 0.4"))
        check(second.vms(Mode.Img2Img).variationEnabled.get && second.vms(Mode.Img2Img).toParams.subseedStrength == 0.4,
          "MainWindow Send-to did not enable variation metadata")
        second.sendTo(Mode.Inpaint, imagePath,
          Some("house\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512"))
        check(!second.vms(Mode.Inpaint).variationEnabled.get && second.vms(Mode.Inpaint).subseedStrength.get == 0.0,
          "MainWindow Send-to did not clear absent variation")
      }
      checked("UiState file restart keeps all gates and drafts; MainWindow Restore and Send-to follow metadata")
      noEngine()
      Files.writeString(output.resolve("variation-gate-report.json"), ujson.write(ujson.Obj(
        "status" -> "passed", "home" -> home.root.toString, "checks" -> ujson.Arr.from(checks)), indent = 2))
      println("VARIATION GATE VERIFIED: " + output.resolve("variation-gate-report.json"))
    catch case error: Throwable => failure = error
    finally
      NativeAcceptanceSafety.cleanup(failure)(
        () => if first != null then fx { first.dispose(); first.stage.close() },
        () => if second != null then fx { second.dispose(); second.stage.close() },
        () => services.shutdown(),
        () => Platform.exit())
    if failure != null then throw failure
