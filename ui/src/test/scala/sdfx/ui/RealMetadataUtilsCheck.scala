package sdfx.ui

import javafx.application.Platform
import javafx.scene.control.{Button, ComboBox, Label, TabPane, TextArea, TextField}
import javafx.stage.Stage
import javafx.embed.swing.SwingFXUtils
import javax.imageio.ImageIO
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.protocol.MetadataUtilsRequest
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode}
import scala.concurrent.Await
import scala.concurrent.duration.*

/** Actual MainWindow file workflow. Inputs must be isolated staged copies, never original model folders. */
object RealMetadataUtilsCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length == 4 || args.length == 5, "home python staged-checkpoint staged-lora [sd15|sdxl|metadata-only]")
    val largeFiles = args.length == 5
    val real = largeFiles && !args.lift(4).contains("metadata-only")
    val sdxl = args.lift(4).contains("sdxl")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val checkpoint = Paths.get(args(2)).toAbsolutePath
    val lora = Paths.get(args(3)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(args(1)),
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString), lora = Seq(lora.getParent.toString))),
      dryRun = !real, builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var lastImage: Option[java.nio.file.Path] = None
    def selectMetadata(): Unit =
      val tabs = window.scene.getRoot.lookupAll(".tab-pane").toArray.collect { case t: TabPane => t }
      tabs.find(_.getTabs.stream().anyMatch(_.getId == "metadata-utils-tab")).foreach(t =>
        t.getTabs.stream().filter(_.getId == "metadata-utils-tab").findFirst().ifPresent(t.getSelectionModel.select))
    def hash(path: java.nio.file.Path): String =
      val digest = java.security.MessageDigest.getInstance("SHA-256")
      val stream = Files.newInputStream(path)
      try
        val bytes = new Array[Byte](1024 * 1024)
        var count = stream.read(bytes)
        while count >= 0 do
          digest.update(bytes, 0, count)
          count = stream.read(bytes)
      finally stream.close()
      java.util.HexFormat.of().formatHex(digest.digest())
    def waitFor(condition: => Boolean): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(120)
      var nextDiagnostic = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
      while !condition && System.nanoTime() < deadline do
        Thread.sleep(25)
        if window != null && System.nanoTime() >= nextDiagnostic then
          val status = fx(Option(window.scene.lookup("#metadata-status")).collect {
            case label: Label => label.getText
          }.getOrElse("metadata controls unavailable"))
          println(s"${java.time.Instant.now()} native wait: $status")
          nextDiagnostic = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
      if !condition && window != null then fx {
        selectMetadata()
        ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png",
          home.root.resolve("metadata-timeout.png").toFile)
      }
      require(condition, "native metadata operation timed out")
    try
      services.start()
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show()
        selectMetadata()
      }
      def generate(modelPath: java.nio.file.Path, loraPath: java.nio.file.Path): Array[Int] =
        waitFor(services.queue.isIdle && fx(!window.progress.get.running))
        services.loadCheckpoint(modelPath.toString, remember = false)
        fx {
          window.model.checkpoint.set(modelPath.toString)
          val side = if sdxl then 768 else 512
          window.vms(Mode.Txt2Img).load(GenerationParams(prompt = "a woman in a garden, detailed illustration, portrait",
            negativePrompt = "blurry, low quality", seed = 173, width = side, height = side,
            steps = 16, cfgScale = if sdxl then 4 else 6,
            loras = Seq(LoraRef(loraPath.getFileName.toString.stripSuffix(".safetensors"), .3))))
          window.statusBar.generateBtn.fire()
        }
        val output = finished.poll(300, TimeUnit.SECONDS)
        require(output != null, "native model generation timed out")
        val image = output.fold(e => throw new IllegalStateException(e), _.head)
        lastImage = Some(image.path)
        val rgb = ImageIO.read(image.path.toFile)
        rgb.getRGB(0, 0, rgb.getWidth, rgb.getHeight, null, 0, rgb.getWidth)
      val baseline = if real then Some(generate(checkpoint, lora)) else None
      fx(selectMetadata())
      for (kind, source) <- Seq("Checkpoint" -> checkpoint, "LoRA" -> lora) do
        val original = hash(source)
        val target = source.resolveSibling(source.getFileName.toString.stripSuffix(".safetensors") + "_md.safetensors")
        require(!Files.exists(target), "staged destination must be fresh")
        fx {
          window.scene.lookup("#metadata-kind").asInstanceOf[ComboBox[String]].setValue(kind)
          window.scene.lookup("#metadata-refresh").asInstanceOf[Button].fire()
        }
        waitFor(fx(!window.scene.lookup("#metadata-refresh").isDisabled))
        fx {
          window.scene.lookup("#metadata-file").asInstanceOf[ComboBox[String]].setValue(source.toString)
          window.scene.lookup("#metadata-load").asInstanceOf[Button].fire()
        }
        if largeFiles && kind == "Checkpoint" then
          waitFor(fx(window.scene.lookup("#metadata-status").asInstanceOf[Label].getText.startsWith("Reading source fingerprint:")))
          fx(window.scene.lookup("#metadata-cancel").asInstanceOf[Button].fire())
          waitFor(fx(!window.scene.lookup("#metadata-load").isDisabled))
          require(fx(window.scene.lookup("#metadata-save").isDisabled), "cancelled read produced a save token")
          fx(window.scene.lookup("#metadata-load").asInstanceOf[Button].fire())
        waitFor(fx(!window.scene.lookup("#metadata-save").isDisabled))
        if !largeFiles then
          require(Files.size(source) < 1024 * 1024, "stale-source fixture must be tiny and isolated")
          val originalBytes = Files.readAllBytes(source)
          for replace <- Seq(true, false) do
            val timestamp = Files.getLastModifiedTime(source)
            val changed = originalBytes.clone()
            changed(changed.length - 1) = (changed.last ^ 1).toByte
            if replace then
              val replacement = source.resolveSibling("replacement.tmp")
              Files.write(replacement, changed)
              Files.move(replacement, source, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
            else Files.write(source, changed)
            Files.setLastModifiedTime(source, timestamp)
            fx(window.scene.lookup("#metadata-save").asInstanceOf[Button].fire())
            waitFor(fx(!window.scene.lookup("#metadata-load").isDisabled))
            require(!Files.exists(target), "stale editor was applied to replacement source")
            require(fx(window.scene.lookup("#metadata-save").isDisabled), "stale token still permits save")
            require(fx(window.scene.lookup("#metadata-status").asInstanceOf[Label].getText.contains("reload metadata")))
            require(java.util.Arrays.equals(Files.readAllBytes(source), changed))
            Files.write(source, originalBytes)
            Files.setLastModifiedTime(source, timestamp)
            fx(window.scene.lookup("#metadata-load").asInstanceOf[Button].fire())
            waitFor(fx(!window.scene.lookup("#metadata-save").isDisabled))
        if largeFiles && kind == "Checkpoint" then
          val cancelled = source.resolveSibling("cancelled-metadata-copy.safetensors")
          require(!Files.exists(cancelled))
          fx {
            window.scene.lookup("#metadata-name").asInstanceOf[TextField].setText("cancelled-metadata-copy")
            window.scene.lookup("#metadata-save").asInstanceOf[Button].fire()
          }
          waitFor(fx(window.scene.lookup("#metadata-status").asInstanceOf[Label].getText.startsWith("Copying tensor bytes:")))
          fx(window.scene.lookup("#metadata-cancel").asInstanceOf[Button].fire())
          waitFor(fx(!window.scene.lookup("#metadata-save").isDisabled))
          require(!Files.exists(cancelled), "cancelled copy was published")
          val leftovers = Files.list(source.getParent)
          try require(!leftovers.anyMatch(_.getFileName.toString.startsWith(".metadata-")), "cancelled copy left a temporary file")
          finally leftovers.close()
          require(hash(source) == original)
          fx(window.scene.lookup("#metadata-name").asInstanceOf[TextField].clear())
        fx {
          val editor = window.scene.lookup("#metadata-json").asInstanceOf[TextArea]
          val metadata = ujson.read(editor.getText).obj
          metadata.update("NUI test", ujson.Str("日本語"))
          metadata.update("NUI nested", ujson.Obj("x" -> 2))
          editor.setText(ujson.Obj.from(metadata).render(indent = 2))
          window.scene.lookup("#metadata-save").asInstanceOf[Button].fire()
        }
        waitFor(fx(!window.scene.lookup("#metadata-save").isDisabled))
        require(Files.isRegularFile(target), fx(window.scene.lookup("#metadata-status").asInstanceOf[Label].getText))
        val savedMetadata = Await.result(services.client.metadataUtilsOwned(MetadataUtilsRequest(path = target.toString)).result, 30.seconds)
        require(savedMetadata.metadata("NUI test") == "日本語", "native editor contents were not saved")
        require(ujson.read(savedMetadata.metadata("NUI nested"))("x").num == 2)
        require(hash(source) == original)
        val saved = Files.size(target)
        fx(window.scene.lookup("#metadata-save").asInstanceOf[Button].fire())
        waitFor(fx(!window.scene.lookup("#metadata-save").isDisabled))
        require(Files.size(target) == saved, "existing destination changed")
        require(fx(window.scene.lookup("#metadata-status").asInstanceOf[Label].getText.contains("Metadata Utils")))
      if real then
        val editedCheckpoint = checkpoint.resolveSibling(checkpoint.getFileName.toString.stripSuffix(".safetensors") + "_md.safetensors")
        val editedLora = lora.resolveSibling(lora.getFileName.toString.stripSuffix(".safetensors") + "_md.safetensors")
        require(java.util.Arrays.equals(baseline.get, generate(editedCheckpoint, editedLora)), "edited model/LoRA changed native RGB")
        waitFor(fx(lastImage == window.preview.currentPath && window.preview.zoom.image != null))
      Files.writeString(home.root.resolve("metadata-native-report.json"), ujson.Obj(
        "real_model" -> real, "family" -> args.lift(4).getOrElse("dry-run"),
        "checkpoint" -> checkpoint.toString, "lora" -> lora.toString,
        "last_image" -> lastImage.fold("")(_.toString), "exact_rgb" -> real,
        "source_hashes_preserved" -> true, "checkpoint_copy_cancelled" -> largeFiles,
        "checkpoint_read_cancelled" -> largeFiles, "stale_source_rejected" -> !largeFiles).render(indent = 2))
      fx {
        selectMetadata()
        window.scene.getRoot.applyCss(); window.scene.getRoot.layout()
        ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", home.root.resolve("metadata-controls.png").toFile)
      }
      println("Metadata Utils actual MainWindow checkpoint/LoRA JSON edit, copy and collision passed")
    finally
      if window != null then fx(window.dispose())
      services.shutdown()
      Platform.exit()
