package sdfx.ui

import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, ConcurrentLinkedQueue, CountDownLatch, FutureTask, TimeUnit}
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.event.Event
import javafx.scene.Node
import javafx.scene.control.{Button, ButtonType, DialogPane, ListView, TextField}
import javafx.scene.input.{KeyCode, KeyEvent}
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.png.PngMetadata
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Root-run real-model acceptance: local tag completion -> named-state Save/Restore -> ordinary Generate.
  * Args: output-directory python checkpoint. No downloads; one 256px, two-step output.
  */
object RealV003GenerationCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task)
    task.get(45, TimeUnit.SECONDS)

  private def eventually(label: String, seconds: Int = 45)(condition: => Boolean): Unit =
    val end = System.nanoTime() + seconds.seconds.toNanos
    while !condition && System.nanoTime() < end do Thread.sleep(50)
    require(condition, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 3, "output-directory python checkpoint")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    val python = Paths.get(args(1)).toAbsolutePath.normalize()
    val checkpoint = Paths.get(args(2)).toAbsolutePath.normalize()
    Seq(python, checkpoint).foreach(p => require(Files.isRegularFile(p), s"Missing file: $p"))
    Files.createDirectories(output)
    val run = Files.createTempDirectory(output, "v003-generation-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    val csv = run.resolve("local-tags.csv")
    Files.writeString(csv, "blue_hair,0,20,azure_hair\nblue_eyes,0,10,\n")
    val engineDir = Paths.get("python").toAbsolutePath.normalize()
    val settings = Settings(pythonExe = Some(python.toString), outputDir = home.root.resolve("outputs").toString,
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString)),
      hfOffline = true, previewEvery = 1, previewMethod = "approx", tagAutocompleteCsv = Some(csv.toString))
    val launch = () => EngineLaunch(Seq(python.toString, "-m", "sdfx"), engineDir,
      Map("PYTHONPATH" -> engineDir.toString, "HF_HUB_OFFLINE" -> "1", "TRANSFORMERS_OFFLINE" -> "1",
          "OPENBLAS_NUM_THREADS" -> "1", "OMP_NUM_THREADS" -> "1", "USE_TF" -> "0"))
    val services = new AppServices(home, settings, dryRun = false, builtins = StandardPlugins.withOwnAxes,
      launchOverride = Some(launch), persistSettings = false, timeouts = EngineTimeouts(load = 10.minutes),
      installShutdownHook = false)
    val events = new ConcurrentLinkedQueue[JobEvent]()
    services.queue.addListener(e => events.add(e))
    val started = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); started.countDown() })
    require(started.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var window: MainWindow = null
    var failure: Throwable = null
    val name = "completed tag and exact generation settings"
    def control[A <: Node](id: String): A =
      val node = window.namedStatesStage.get.getScene.lookup("#" + id)
      require(node != null, s"Missing production control: $id")
      node.asInstanceOf[A]
    def screenshot(name: String): Unit =
      window.root.applyCss(); window.root.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", run.resolve(name).toFile))
    try
      require(!services.start().dryRun)
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      require(Set("sd15", "sdxl")(loaded.kind))
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        w.show(); w.select(Mode.Txt2Img); w.stage.setWidth(1280); w.stage.setHeight(850)
        w.model.checkpoint.set(loaded.name)
        w.panels(Mode.Txt2Img).prompts.promptArea.requestFocus()
        w
      }
      eventually("local CSV and prompt focus") { fx {
        window.ctx.tagDictionary.tags.size == 2 && window.panels(Mode.Txt2Img).prompts.promptArea.isFocused
      } }
      val before = fx {
        val editor = window.panels(Mode.Txt2Img).prompts
        val area = editor.promptArea
        area.setText("blue, detailed portrait"); area.positionCaret(4)
        require(editor.completionVisible, "Actual completion popup did not appear")
        Event.fireEvent(area, new KeyEvent(KeyEvent.KEY_PRESSED, "", "", KeyCode.TAB, false, false, false, false))
        require(area.getText == "blue hair, detailed portrait", area.getText)
        require(!editor.completionVisible && services.queue.isIdle)
        val vm = window.vms(Mode.Txt2Img)
        vm.negativePrompt.set("blurry, low quality")
        vm.width.set(256); vm.height.set(256); vm.steps.set(2); vm.seed.set(7303)
        vm.batchSize.set(1); vm.batchCount.set(1); vm.cfgScale.set(6.5)
        vm.sampler.set("Euler"); vm.scheduler.set("Automatic")
        screenshot("01-completed-tag.png")
        val expected = vm.toParams
        window.openGenerationStates(Mode.Txt2Img)
        expected
      }
      eventually("named-state library") { fx { !control[Button]("generation-states-save").isDisabled } }
      fx {
        control[TextField]("generation-states-name").setText(name)
        control[Button]("generation-states-save").fire()
      }
      eventually("named-state persisted") { fx {
        !control[Button]("generation-states-save").isDisabled && control[ListView[String]]("generation-states-list").getItems.contains(name)
      } }
      require(Files.isRegularFile(home.configDir.resolve("generation-states/library-v1.json")))
      fx {
        control[ListView[String]]("generation-states-list").getSelectionModel.select(name)
        val vm = window.vms(Mode.Txt2Img)
        vm.prompt.set("mutated prompt"); vm.negativePrompt.set("mutated negative")
        vm.width.set(384); vm.height.set(320); vm.steps.set(4); vm.seed.set(99)
        vm.batchSize.set(2); vm.batchCount.set(2); vm.cfgScale.set(3)
      }
      val restore = new FutureTask[Unit](new Callable[Unit] {
        def call(): Unit = control[Button]("generation-states-restore").fire()
      })
      Platform.runLater(restore)
      eventually("real restore confirmation") { fx {
        Window.getWindows.asScala.exists(w => w.isShowing && w.getScene != null && w.getScene.getRoot.isInstanceOf[DialogPane]) || restore.isDone
      } }
      fx {
        val pane = Window.getWindows.asScala.filter(_.isShowing).flatMap(w => Option(w.getScene))
          .map(_.getRoot).collectFirst { case p: DialogPane => p }.getOrElse(sys.error("Restore did not reach confirmation"))
        pane.lookupButton(ButtonType.OK).asInstanceOf[Button].fire()
      }
      restore.get(30, TimeUnit.SECONDS)
      fx {
        require(window.vms(Mode.Txt2Img).toParams == before, "Full GenerationParams snapshot was not restored")
        require(services.queue.isIdle && events.asScala.forall(!_.isInstanceOf[JobEvent.JobQueued]), "Save/Restore submitted a job")
        window.namedStatesStage.foreach(_.hide())
        screenshot("02-restored-settings.png")
        window.generate(false)
      }
      eventually("generation queued") { events.asScala.exists(_.isInstanceOf[JobEvent.JobQueued]) }
      val queued = events.asScala.collectFirst { case e: JobEvent.JobQueued => e.job }.get
      require(queued.params == before, s"Submitted snapshot differs from restored settings: ${queued.params}")
      eventually("generation terminal", 900) { events.asScala.exists(e => e.jobId == queued.id &&
        (e.isInstanceOf[JobEvent.JobFinished] || e.isInstanceOf[JobEvent.JobFailed] || e.isInstanceOf[JobEvent.JobCancelled])) }
      val result = events.asScala.collectFirst { case e: JobEvent.JobFinished if e.jobId == queued.id => e.output }
        .getOrElse(sys.error(s"Generation failed: ${events.asScala.filter(_.jobId == queued.id).mkString("\n")}"))
      require(result.images.size == 1)
      val image = result.images.head
      require(Files.isRegularFile(image.path) && image.seed == before.seed)
      val pixels = ImageIO.read(image.path.toFile)
      require(pixels != null && pixels.getWidth == 256 && pixels.getHeight == 256)
      val text = PngMetadata.readParameters(image.path).getOrElse(sys.error("Generated PNG lacks parameters"))
      val parsed = Infotext.parse(text)
      require(parsed.prompt == before.prompt && parsed.negativePrompt == before.negativePrompt)
      require(parsed.get("Seed").contains(before.seed.toString) && parsed.get("Steps").contains("2"))
      eventually("native gallery result") { fx { window.galleryVM.items.asScala.exists(_.path == image.path) } }
      eventually("final preview pixels loaded") { fx {
        val shown = window.preview.zoom.image
        window.preview.currentPath.contains(image.path) && shown != null && !shown.isError && shown.getProgress >= 1
      } }
      fx { screenshot("03-generated-result.png") }
      val hash = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(image.path)).map(b => f"${b & 255}%02x").mkString
      Files.writeString(run.resolve("report.json"), ujson.write(ujson.Obj("status" -> "passed", "family" -> loaded.kind,
        "prompt" -> before.prompt, "seed" -> before.seed, "output" -> image.path.toString, "sha256" -> hash,
        "full_generation_params_restored" -> true, "queued_params_match" -> true,
        "tag_csv" -> csv.toString, "named_state" -> name, "output_count" -> 1), indent = 2))
      println(s"V003_GENERATION_OK artifacts=$run family=${loaded.kind}")
    catch case error: Throwable => failure = error
    finally
      NativeAcceptanceSafety.cleanup(failure)(
        () => if window != null then fx { window.dispose(); window.stage.close() },
        () => services.shutdown(), () => Platform.exit())
    if failure != null then throw failure
