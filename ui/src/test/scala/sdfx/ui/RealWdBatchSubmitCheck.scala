package sdfx.ui

import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import java.security.MessageDigest
import java.util.concurrent.{Callable, ConcurrentLinkedQueue, CountDownLatch, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.{Button, ButtonType, CheckBox, DialogPane, Label, Slider, Spinner, TabPane, TextField}
import javafx.scene.layout.{HBox, VBox}
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.concurrent.duration.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Manual real worker/GPU acceptance: actual UI submit, queue and gallery.
  * Args: output-dir python checkpoint-file wd-model wd-tags image-file.
  * One run per checkpoint; chooser result is injected at its text field.
  */
object RealWdBatchSubmitCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task)
    task.get(45, TimeUnit.SECONDS)
  private def waitFor(label: String, seconds: Int = 120)(test: => Boolean): Unit =
    val until = System.nanoTime() + seconds.seconds.toNanos
    while !test && System.nanoTime() < until do Thread.sleep(100)
    require(test, s"Timeout: $label")
  private def sha(path: Path): String =
    val md = MessageDigest.getInstance("SHA-256")
    val in = Files.newInputStream(path)
    try
      val b = new Array[Byte](1024 * 1024)
      var n = in.read(b)
      while n >= 0 do
        if n > 0 then md.update(b, 0, n)
        n = in.read(b)
    finally in.close()
    md.digest().map(b => f"${b & 255}%02x").mkString

  def main(args: Array[String]): Unit =
    require(args.length == 6 || args.length == 7, "output-dir python checkpoint-file wd-model wd-tags image-file [protected]")
    val protectedMode = args.length == 7 && args(6) == "protected"
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    val python = Paths.get(args(1)).toAbsolutePath.normalize()
    val checkpoint = Paths.get(args(2)).toAbsolutePath.normalize()
    val wdModel = Paths.get(args(3)).toAbsolutePath.normalize()
    val wdTags = Paths.get(args(4)).toAbsolutePath.normalize()
    val input = Paths.get(args(5)).toAbsolutePath.normalize()
    Seq(python, checkpoint, wdModel, wdTags, input).foreach(p => require(Files.isRegularFile(p), s"Missing file $p"))
    Files.createDirectories(output)
    val run = Files.createTempDirectory(output, "wd-submit-native-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    val inputs = Files.createDirectories(run.resolve("image inputs"))
    val copied = inputs.resolve("input" + input.getFileName.toString.dropWhile(_ != '.'))
    if protectedMode then
      sdfx.core.png.ScrambledPng.write(ImageIO.read(input.toFile), copied, Nil, Some("wd-native-test"))
    else Files.copy(input, copied, StandardCopyOption.COPY_ATTRIBUTES)
    val copiedSecond = inputs.resolve("second" + input.getFileName.toString.dropWhile(_ != '.'))
    if protectedMode then Files.copy(copied, copiedSecond)
    else Files.copy(input, copiedSecond, StandardCopyOption.COPY_ATTRIBUTES)
    val watched = Seq(checkpoint, wdModel, wdTags, input, copied, copiedSecond)
    val hashes = watched.map(p => p -> sha(p)).toMap
    val temp = Files.createDirectories(home.root.resolve("wd-temp"))
    val engineDir = Paths.get("python").toAbsolutePath.normalize()
    val settings = Settings(pythonExe = Some(python.toString), outputDir = home.root.resolve("outputs").toString,
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString)), wdV3Model = Some(wdModel.toString),
      wdV3Tags = Some(wdTags.toString), hfOffline = true)
    val launch = () => EngineLaunch(Seq(python.toString, "-m", "sdfx", "--wd-v3-model", wdModel.toString,
      "--wd-v3-tags", wdTags.toString, "--wd-v3-temp", temp.toString), engineDir,
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
    try
      val hello = services.start()
      require(!hello.dryRun && hello.features.contains("batch_interrogate_wd_v3_partial"), "real WD worker unavailable")
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      if protectedMode then services.imageScrambleSession.unlock("wd-native-test")
      require(Set("sd15", "sdxl")(loaded.kind), s"Unexpected model family ${loaded.kind}")
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        w.show(); w.select(Mode.Img2Img); w.stage.setWidth(1280); w.stage.setHeight(800)
        w.model.checkpoint.set(loaded.name)
        w
      }
      val settingsError = new AtomicReference[Throwable]()
      fx {
        Platform.runLater(() => {
          try
            val stage = Window.getWindows.asScala.collectFirst { case s: Stage if s.isShowing && s.getTitle == "Settings" => s }.get
            val pane = stage.getScene.getRoot.asInstanceOf[DialogPane]
            val tabs = pane.lookupAll(".tab-pane").asScala.collectFirst { case t: TabPane => t }.get
            require(tabs.getTabs.asScala.exists(_.getText == "Model folders"), "Model folders not visible")
            require(tabs.getSelectionModel.getSelectedItem.getText == "Model folders")
            require(ImageIO.write(SwingFXUtils.fromFXImage(stage.getScene.snapshot(null), null), "png", run.resolve("settings-model-folders.png").toFile))
            pane.lookupButton(ButtonType.CANCEL).asInstanceOf[Button].fire()
          catch
            case e: Throwable =>
              settingsError.set(e)
              Window.getWindows.asScala.collect { case s: Stage if s.getTitle == "Settings" => s }.foreach(_.close())
        })
        window.openSettings()
      }
      if settingsError.get() != null then throw settingsError.get()
      fx {
        val vm = window.vms(Mode.Img2Img)
        vm.width.set(256); vm.height.set(320); vm.steps.set(2); vm.seed.set(42)
        vm.prompt.set("a detailed portrait"); vm.negativePrompt.set("low quality")
        val panel = window.panels(Mode.Img2Img)
        panel.content.applyCss(); panel.content.layout()
        val sizeCard = panel.content.lookupAll(".card").asScala.collectFirst {
          case box: VBox if box.getChildren.asScala.exists {
            case label: Label => label.getText == "Size"
            case _ => false
          } => box
        }.getOrElse(throw new IllegalStateException("Size card missing"))
        val rows = sizeCard.lookupAll(".slider-row").asScala.collect { case h: HBox => h }.toSeq
        require(rows.size == 2, s"Expected width and height slider rows, found ${rows.size}")
        def positions(row: HBox): (Double, Double, Double, Double) =
          val slider = row.getChildren.asScala.collectFirst { case s: Slider => s }.get
          val spinner = row.getChildren.asScala.collectFirst { case s: Spinner[?] => s }.get
          val a = slider.localToScene(slider.getBoundsInLocal); val b = spinner.localToScene(spinner.getBoundsInLocal)
          (a.getMinX, a.getWidth, b.getMinX, b.getWidth)
        require(positions(rows(0)).productIterator.zip(positions(rows(1)).productIterator)
          .forall((a, b) => math.abs(a.asInstanceOf[Double] - b.asInstanceOf[Double]) < 2.0), "Width/Height alignment mismatch")
        val enable = panel.content.lookup("#wd-batch-enabled").asInstanceOf[CheckBox]
        val imageFolder = panel.content.lookup("#wd-batch-input").asInstanceOf[TextField]
        val outputName = panel.content.lookup("#wd-batch-output-name").asInstanceOf[TextField]
        enable.fire(); imageFolder.setText(inputs.toString); outputName.setText("wd-result")
        val targetY = panel.content.sceneToLocal(enable.localToScene(enable.getBoundsInLocal)).getMinY - 35
        val scrollable = panel.content.getLayoutBounds.getHeight - panel.node.getViewportBounds.getHeight
        panel.node.setVvalue(if scrollable > 0 then math.max(0, math.min(1, targetY / scrollable)) else 0)
        window.root.applyCss(); window.root.layout()
        require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", run.resolve("wd-submit.png").toFile))
        window.generate(false)
        vm.prompt.set("changed after submission")
        imageFolder.setText("later edit")
      }
      waitFor("WD queued", 90) { events.asScala.exists(_.isInstanceOf[JobEvent.JobQueued]) }
      val queued = events.asScala.collectFirst { case e: JobEvent.JobQueued => e.job }.get
      require(queued.wdBatch.exists(_.imagesRoot == inputs.toString) && queued.params.prompt == "a detailed portrait")
      waitFor("WD terminal", 900) { events.asScala.exists(e => e.jobId == queued.id &&
        (e.isInstanceOf[JobEvent.JobFinished] || e.isInstanceOf[JobEvent.JobFailed] || e.isInstanceOf[JobEvent.JobCancelled])) }
      val terminal = events.asScala.find(e => e.jobId == queued.id && (e.isInstanceOf[JobEvent.JobFinished] ||
        e.isInstanceOf[JobEvent.JobFailed] || e.isInstanceOf[JobEvent.JobCancelled])).get
      val result = terminal match
        case e: JobEvent.JobFinished => e.output
        case other => throw new IllegalStateException(s"WD job did not finish: $other")
      require(result.images.size == 2 && result.images.forall(i => Files.isRegularFile(i.path)), "expected two batch output images")
      if protectedMode then
        result.images.foreach { image =>
          require(sdfx.core.png.PngMetadata.readTexts(image.path).get("Encrypt").contains("pixel_shuffle_3"))
          val decoded = sdfx.core.png.ScrambledPng.read(image.path, Some("wd-native-test"))
          require(decoded.passwordVerified.contains(true))
          ImageIO.write(decoded.image, "png", run.resolve("decoded-" + image.path.getFileName.toString).toFile)
        }
      require(result.images.map(_.path).distinct.size == 2, "batch outputs overlap")
      require(result.images.forall(_.params.prompt != queued.params.prompt), "WD caption did not alter each prompt")
      waitFor("gallery output", 60) { fx { window.galleryVM.items.asScala.exists(_.path == result.images.head.path) } }
      require(watched.forall(p => sha(p) == hashes(p)), "source asset changed")
      val tempEntries = Files.list(temp)
      try require(!tempEntries.iterator().hasNext, "WD temporary staging retained")
      finally tempEntries.close()
      Files.writeString(run.resolve("report.json"), ujson.write(ujson.Obj("status" -> "passed", "family" -> loaded.kind,
        "checkpoint" -> checkpoint.toString, "source_sha256" -> hashes(input), "model_sha256" -> hashes(wdModel),
        "tags_sha256" -> hashes(wdTags), "output" -> result.images.head.path.toString,
        "prompt" -> result.images.head.params.prompt, "output_count" -> result.images.size,
        "chooser" -> "field-injected"), indent = 2))
      println(s"WD_NATIVE_SUBMIT_OK artifacts=$run family=${loaded.kind}")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown()
      Platform.exit()
