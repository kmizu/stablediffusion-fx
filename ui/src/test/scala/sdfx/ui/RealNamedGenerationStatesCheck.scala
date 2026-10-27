package sdfx.ui

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.presets.{PresetJson, PresetLimits}
import sdfx.core.protocol.ModelEntry
import sdfx.engine.{AppServices, CatalogSnapshot}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Root-run offline acceptance of the real MainWindow and named-state buttons/dialogs.
  * No engine startup, inference, model download or clipboard access. Arg: output directory.
  */
object RealNamedGenerationStatesCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String)(condition: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while !condition && System.nanoTime() < end do Thread.sleep(25)
    require(condition, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "named-states-home-"))
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var services: AppServices = null
    var window: MainWindow = null
    var failure: Throwable = null
    var checks = Vector.empty[String]
    var screenshots = Vector.empty[String]
    val mode = Mode.Img2Img
    val library = home.configDir.resolve("generation-states/library-v1.json")
    val image = output.resolve("named-state-current-input.png")
    val bitmap = new java.awt.image.BufferedImage(64, 64, java.awt.image.BufferedImage.TYPE_INT_RGB)
    ImageIO.write(bitmap, "png", image.toFile)

    def checked(label: String): Unit =
      checks :+= label
      println("NAMED STATES PASS: " + label)
    def stage: Stage = window.namedStatesStage.getOrElse(sys.error("Named-state stage unavailable"))
    def node[A <: Node](id: String): A =
      val found = stage.getScene.lookup("#" + id)
      require(found != null, s"Actual control missing: $id")
      found.asInstanceOf[A]
    def noWork(): Unit =
      require(services.supervisor.pid.isEmpty && services.supervisor.loadedModel.isEmpty,
        "Named-state operation started the engine or loaded a model")
      require(services.queue.isIdle && services.payloads.snapshots.isEmpty,
        "Named-state operation submitted generation work")
    def screenshot(name: String): Unit = fx {
      stage.getScene.getRoot.applyCss()
      stage.getScene.getRoot.layout()
      val path = output.resolve(name + ".png")
      require(ImageIO.write(SwingFXUtils.fromFXImage(stage.getScene.snapshot(null), null), "png", path.toFile))
      screenshots :+= path.toString
    }
    def open(): Unit =
      services = new AppServices(home, Settings(), dryRun = true,
        builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
      window = fx {
        val created = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        created.show()
        created.catalogVM.snapshot.set(CatalogSnapshot(checkpoints = Seq(ModelEntry("fixture-model", "offline-model", "sd15"))))
        created.select(mode)
        created.openGenerationStates(mode)
        created
      }
      eventually("library available") { fx { !node[Button]("generation-states-save").isDisabled } }
      noWork()
    def close(): Unit =
      if window != null then
        fx { window.dispose(); window.stage.close() }
        window = null
      if services != null then
        services.shutdown()
        services = null
    def select(name: String): Unit = fx {
      val list = node[ListView[String]]("generation-states-list")
      require(list.getItems.contains(name), s"Saved state is missing from visible list: $name")
      list.getSelectionModel.select(name)
    }
    def save(name: String): Unit =
      fx {
        node[TextField]("generation-states-name").setText(name)
        node[Button]("generation-states-save").fire()
      }
      eventually(s"saved $name") { fx {
        !node[Button]("generation-states-save").isDisabled &&
          node[ListView[String]]("generation-states-list").getItems.contains(name)
      }}
    def confirmButton(id: String): Unit =
      val task = new FutureTask[Unit](new Callable[Unit]:
        def call(): Unit = node[Button](id).fire())
      Platform.runLater(task)
      eventually(s"confirmation for $id") { fx {
        Window.getWindows.asScala.exists(w => w.isShowing && w.getScene != null &&
          w.getScene.getRoot.isInstanceOf[DialogPane]) || task.isDone
      }}
      fx {
        val pane = Window.getWindows.asScala.filter(_.isShowing).flatMap(w => Option(w.getScene))
          .map(_.getRoot).collectFirst { case dialog: DialogPane => dialog }
          .getOrElse(sys.error("Expected real confirmation dialog; status: " + node[TextArea]("generation-states-status").getText))
        pane.lookupButton(ButtonType.OK).asInstanceOf[Button].fire()
      }
      task.get(30, TimeUnit.SECONDS)

    try
      open()
      fx {
        val vm = window.vms(mode)
        vm.prompt.set("named state native roundtrip 日本語")
        vm.width.set(640)
        vm.height.set(512)
        vm.seed.set(123456789L)
        vm.eta.set(Some(0.35))
        window.model.checkpoint.set("fixture-model")
        window.model.clipSkip.set(2)
        val freeu = window.plugins(mode).form("freeu").get
        freeu.prop("enabled").get.set("false")
        freeu.prop("b1").get.set("1.37")
      }
      save("native roundtrip")
      require(Files.isRegularFile(library), "Save did not persist the separate named-state library")
      require(!Files.exists(home.configDir.resolve("native-presets/library-v1.json")), "Named states touched native preset storage")
      val parsed = PresetJson.readNative(Files.readAllBytes(library), PresetLimits()).toOption.get
      require(parsed.collections.exists(_.id == "state-img2img"), "Saved mode was lost")
      checked("actual Save persists a mode-specific full snapshot in separate storage")
      select("native roundtrip")
      screenshot("named-states-saved")
      fx {
        val vm = window.vms(mode)
        vm.prompt.set("changed prompt")
        vm.width.set(768)
        vm.height.set(768)
        vm.eta.set(None)
        vm.initImage.set(image.toString)
        window.plugins(mode).form("freeu").get.prop("b1").get.set("1.8")
      }
      confirmButton("generation-states-restore")
      fx {
        val vm = window.vms(mode)
        require(vm.prompt.get == "named state native roundtrip 日本語")
        require(vm.width.get == 640 && vm.height.get == 512)
        require(vm.eta.get.contains(0.35))
        require(vm.initImage.get == image.toString, "Restoring settings changed current init image")
        require(window.plugins(mode).form("freeu").get.prop("enabled").get.get == "false")
        require(window.plugins(mode).form("freeu").get.prop("b1").get.get.toDouble == 1.37)
        require(window.model.clipSkip.get == 2)
      }
      noWork()
      checked("actual Restore confirmation recovers dimensions, Eta and disabled FreeU values while retaining init image")
      fx {
        window.catalogVM.snapshot.set(CatalogSnapshot())
        window.vms(mode).prompt.set("must remain untouched")
        window.vms(mode).width.set(896)
        node[Button]("generation-states-restore").fire()
        require(window.vms(mode).prompt.get == "must remain untouched" && window.vms(mode).width.get == 896)
        require(node[TextArea]("generation-states-status").getText.contains("model.checkpoint"))
      }
      screenshot("named-states-missing-model")
      checked("missing checkpoint visibly blocks Restore without partial form mutation")
      close()
      open()
      select("native roundtrip")
      fx { window.vms(mode).initImage.set(image.toString) }
      confirmButton("generation-states-restore")
      fx {
        require(window.vms(mode).prompt.get == "named state native roundtrip 日本語")
        require(window.vms(mode).width.get == 640)
        require(window.vms(mode).initImage.get == image.toString)
        require(window.plugins(mode).form("freeu").get.prop("b1").get.get.toDouble == 1.37)
      }
      checked("new MainWindow and AppServices on the same home reload and restore disk snapshot")
      screenshot("named-states-reopened")
      confirmButton("generation-states-delete")
      eventually("delete persisted") { fx {
        !node[Button]("generation-states-save").isDisabled &&
          !node[ListView[String]]("generation-states-list").getItems.contains("native roundtrip")
      }}
      val deleted = PresetJson.readNative(Files.readAllBytes(library), PresetLimits()).toOption.get
      require(deleted.collections.flatMap(_.presets).isEmpty)
      noWork()
      checked("actual Delete confirmation removes the state from visible list and disk")
      Files.writeString(output.resolve("named-states-native-report.json"), ujson.write(ujson.Obj(
        "status" -> "passed", "home" -> home.root.toString, "library" -> library.toString,
        "checks" -> ujson.Arr.from(checks), "screenshots" -> ujson.Arr.from(screenshots)), indent = 2))
    catch
      case error: Throwable =>
        failure = error
        if window != null then
          try screenshot("named-states-failure")
          catch case _: Throwable => ()
    finally
      NativeAcceptanceSafety.cleanup(failure)(() => close(), () => Platform.exit())
    if failure != null then throw failure
