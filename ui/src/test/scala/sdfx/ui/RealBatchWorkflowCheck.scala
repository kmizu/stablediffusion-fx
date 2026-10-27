package sdfx.ui

import java.io.ByteArrayInputStream
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.Base64
import java.util.concurrent.{Callable, ConcurrentLinkedQueue, CountDownLatch, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import java.util.zip.ZipFile
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.Label
import javafx.stage.Stage
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Manual native acceptance. Args: output-dir [python checkpoint-file].
  * Uses the actual MainWindow submit, queue, preview, generation gallery and ZIP writer.
  * No model downloads; the optional real-engine mode must be launched by the GPU owner.
  */
object RealBatchWorkflowCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task)
    task.get(45, TimeUnit.SECONDS)

  private def waitFor(label: String, seconds: Int = 120)(test: => Boolean): Unit =
    val until = System.nanoTime() + seconds.seconds.toNanos
    while !test && System.nanoTime() < until do Thread.sleep(100)
    require(test, s"Timeout: $label")

  private def sha(bytes: Array[Byte]): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString

  def main(args: Array[String]): Unit =
    require(args.length == 1 || args.length == 3, "output-dir [python checkpoint-file]")
    val real = args.length == 3
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val run = Files.createTempDirectory(output, "batch-native-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    val python = if real then Some(Paths.get(args(1)).toAbsolutePath.normalize()) else None
    val checkpoint = if real then Some(Paths.get(args(2)).toAbsolutePath.normalize()) else None
    (python.toSeq ++ checkpoint.toSeq).foreach(p => require(Files.isRegularFile(p), s"Missing file $p"))
    val engineDir = Paths.get("python").toAbsolutePath.normalize()
    val settings = Settings(pythonExe = python.map(_.toString), outputDir = home.root.resolve("outputs").toString,
      modelDirs = ModelDirs(checkpoints = checkpoint.toSeq.map(_.getParent.toString)),
      hfOffline = true, previewEvery = 1, previewMethod = "approx")
    val launch = python.map(p => () => EngineLaunch(Seq(p.toString, "-m", "sdfx"), engineDir,
      Map("PYTHONPATH" -> engineDir.toString, "HF_HUB_OFFLINE" -> "1", "TRANSFORMERS_OFFLINE" -> "1",
          "OPENBLAS_NUM_THREADS" -> "1", "OMP_NUM_THREADS" -> "1", "USE_TF" -> "0")))
    val services = new AppServices(home, settings, dryRun = !real, builtins = StandardPlugins.withOwnAxes,
      launchOverride = launch, persistSettings = false, timeouts = EngineTimeouts(load = 10.minutes),
      installShutdownHook = false)
    val events = new ConcurrentLinkedQueue[JobEvent]()
    services.queue.addListener(e => events.add(e))
    val started = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); started.countDown() })
    require(started.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    val previewSnapshot = new AtomicReference[Path]()
    val snapshotError = new AtomicReference[Throwable]()
    var window: MainWindow = null
    try
      val hello = services.start()
      require(hello.dryRun == !real)
      val loaded = checkpoint.map(p => services.loadCheckpoint(p.toString, remember = false))
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        w.show(); w.select(Mode.Txt2Img); w.stage.setWidth(1440); w.stage.setHeight(900)
        loaded.foreach(m => w.model.checkpoint.set(m.name))
        w.root.applyCss(); w.root.layout()
        val badge = w.preview.imageNode.lookupAll(".live-badge").asScala.collectFirst { case l: Label => l }.get
        // Snapshot synchronously at the actual preview visibility transition, not after the job finishes.
        badge.visibleProperty.addListener((_, _, visible) => {
          if visible.booleanValue && previewSnapshot.get() == null then
            try
              w.root.applyCss(); w.root.layout()
              val path = run.resolve("live-batch-preview.png")
              require(ImageIO.write(SwingFXUtils.fromFXImage(w.scene.snapshot(null), null), "png", path.toFile))
              previewSnapshot.set(path)
            catch case e: Throwable => snapshotError.set(e)
        })
        val vm = w.vms(Mode.Txt2Img)
        vm.width.set(256); vm.height.set(256); vm.steps.set(3); vm.seed.set(1200)
        vm.batchSize.set(4); vm.batchCount.set(5)
        vm.prompt.set("a small wooden cabin in a green meadow, detailed illustration")
        vm.negativePrompt.set("low quality, blurry")
        w.generate(false)
        w
      }
      waitFor("batch queued") { events.asScala.exists(_.isInstanceOf[JobEvent.JobQueued]) }
      val queued = events.asScala.collectFirst { case e: JobEvent.JobQueued => e.job }.get
      require(queued.params.batchSize == 4 && queued.params.nIter == 5)
      waitFor("batch terminal", 900) { events.asScala.exists(e => e.jobId == queued.id &&
        (e.isInstanceOf[JobEvent.JobFinished] || e.isInstanceOf[JobEvent.JobFailed] || e.isInstanceOf[JobEvent.JobCancelled])) }
      val result = events.asScala.collectFirst { case e: JobEvent.JobFinished if e.jobId == queued.id => e.output }
        .getOrElse(throw new IllegalStateException(s"Batch failed: ${events.asScala.filter(_.jobId == queued.id).mkString("\n")}"))
      require(result.images.size == 20 && result.images.map(_.path).distinct.size == 20)
      require(result.images.map(_.seed) == (1200L until 1220L))
      require(result.images.forall(i => Files.isRegularFile(i.path)))
      waitFor("all twenty gallery results") { fx {
        result.images.forall(i => window.galleryVM.items.asScala.exists(_.path == i.path))
      } }
      if snapshotError.get() != null then throw snapshotError.get()
      require(previewSnapshot.get() != null, "No live preview was rendered before completion")
      val previews = events.asScala.collect { case e: JobEvent.Preview if e.jobId == queued.id => e }.toSeq
      require(previews.size >= 2, s"Expected multiple live preview frames, got ${previews.size}")
      val previewDimensions = previews.map { e =>
        val image = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder.decode(e.jpegB64)))
        require(image != null, "Invalid preview JPEG")
        if real then
          require(image.getWidth == 512 && image.getHeight == 512, "Expected a readable 2x2 batch preview grid")
          val tileWidth = image.getWidth / 2
          val tileHeight = image.getHeight / 2
          val tiles = for y <- 0 until 2; x <- 0 until 2 yield
            image.getRGB(x * tileWidth, y * tileHeight, tileWidth, tileHeight, null, 0, tileWidth).toSeq
          require(tiles.distinct.size >= 2, "Live batch grid repeats the same image in every tile")
        (image.getWidth, image.getHeight)
      }
      val paths = fx {
        window.galleryVM.selectedGeneration.set(queued.id)
        val selected = window.galleryVM.selectedGenerationPaths
        require(selected.toSet == result.images.map(_.path).toSet, "ZIP selection differs from the twenty outputs")
        window.root.applyCss(); window.root.layout()
        require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png",
          run.resolve("final-batch-gallery.png").toFile))
        selected
      }
      val zipPath = run.resolve("batch-twenty-images.zip")
      require(GalleryZipExporter.write(paths, zipPath) == 20)
      val zip = new ZipFile(zipPath.toFile)
      try
        val entries = zip.entries().asScala.toSeq
        require(entries.size == 20 && entries.map(_.getName).distinct.size == 20)
        val expectedHashes = paths.map(p => sha(Files.readAllBytes(p))).sorted
        val actualHashes = entries.map { e =>
          val in = zip.getInputStream(e)
          try sha(in.readAllBytes()) finally in.close()
        }.sorted
        require(actualHashes == expectedHashes, "ZIP entries differ from the generated PNG bytes")
      finally zip.close()
      Files.writeString(run.resolve("report.json"), ujson.write(ujson.Obj("status" -> "passed",
        "engine" -> (if real then "real" else "dry-run"), "family" -> loaded.map(_.kind).getOrElse("dry-run"),
        "batch_size" -> 4, "batch_count" -> 5, "output_count" -> 20, "preview_frames" -> previews.size,
        "preview_dimensions" -> ujson.Arr.from(previewDimensions.map((w, h) => ujson.Arr(w, h))),
        "zip" -> zipPath.toString, "zip_count" -> 20, "zip_sha256" -> sha(Files.readAllBytes(zipPath)),
        "outputs" -> ujson.Arr.from(paths.map(p => ujson.Obj("path" -> p.toString,
          "sha256" -> sha(Files.readAllBytes(p))))), "zip_target" -> "injected destination, production writer"), indent = 2))
      println(s"BATCH_NATIVE_OK artifacts=$run family=${loaded.map(_.kind).getOrElse("dry-run")}")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown()
      Platform.exit()
