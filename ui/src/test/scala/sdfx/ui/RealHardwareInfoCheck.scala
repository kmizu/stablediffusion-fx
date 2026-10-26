package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.{HardwareInfo, Infotext}
import sdfx.core.model.{GenerationParams, HiresParams, Img2ImgParams}
import sdfx.core.png.PngMetadata
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, Job}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.collection.immutable.ListMap
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL root-only gate. python output-directory [checkpoint-directory checkpoint-name].
  * No checkpoint args means actual dry-run stdio + native controls; trained models are parent-only.
  * Every setting/output is owned by a new isolated AppHome below output-directory. No clipboard or network.
  */
object RealHardwareInfoCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(40, TimeUnit.SECONDS)
  private def eventually(label: String)(body: => Boolean): Unit =
    val end = System.nanoTime() + 60.seconds.toNanos
    while !body && System.nanoTime() < end do Thread.sleep(30)
    require(body, "Timed out: " + label)
  private def hash(path: Path): String = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path))
    .map(b => f"${b & 255}%02x").mkString
  private def pixels(path: Path): Seq[Int] =
    val image = ImageIO.read(path.toFile)
    image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth).toSeq

  def main(args: Array[String]): Unit =
    require(args.length == 2 || args.length == 4, "python output-directory [checkpoint-directory checkpoint-name]")
    val real = args.length == 4
    val output = Paths.get(args(1)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "hardware-app-home-"))
    val engineDir = AppHome.detect().pythonDir
    val settings = Settings(outputDir = output.resolve("images").toString, previewEvery = 0,
      modelDirs = if real then ModelDirs(checkpoints = Seq(Paths.get(args(2)).toAbsolutePath.toString)) else ModelDirs())
    val services = new AppServices(home, settings, dryRun = !real, builtins = StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(Paths.get(args(0)).toAbsolutePath.toString, "-m", "sdfx") ++
        (if real then Nil else Seq("--dry-run")), engineDir, Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = true, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var primaryFailure: Throwable = null
    var checks = Vector.empty[String]
    var galleryExpected = 0
    def checked(message: String): Unit = { checks :+= message; println("HARDWARE CHECK: " + message) }
    def find[A <: Node](id: String): A =
      val node = window.scene.lookup("#" + id)
      require(node != null, "Missing native control #" + id)
      node.asInstanceOf[A]
    def systemTab(): Unit = fx {
      val tabs = find[TabPane]("main-tabs")
      tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "system-info-tab").get)
    }
    def toggle(value: Boolean): Unit =
      fx { val control = find[CheckBox]("hardware-info-enabled"); if control.isSelected != value then control.fire() }
      eventually("native metadata setting") { services.settings.hardwareInfoEnabled == value }
    def generate(params: GenerationParams) =
      val result = Await.result(services.queue.submit(Job(params)), 10.minutes).images
      galleryExpected += result.size
      result
    def verify(path: Path, remark: String): String =
      val parsed = Infotext.parse(PngMetadata.readParameters(path).getOrElse(sys.error("Missing PNG parameters: " + path)))
      val profile = parsed.get("Hardware Info").getOrElse(sys.error("Missing Hardware Info: " + path))
      require(profile.nonEmpty && profile != "unknown", "Actual hardware detection is unavailable")
      require(parsed.get("Remark").contains(remark))
      require(parsed.get("Time taken").exists(_.matches("(?:[0-9]+ min\\. )?[0-9]+\\.[0-9] sec\\.")))
      profile
    def snapshot(name: String): Unit = fx {
      window.root.applyCss(); window.root.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", output.resolve(name).toFile))
    }
    def clearToasts(): Unit = fx { window.ctx.toaster.layer.getChildren.clear() }
    def notices: Seq[String] = fx {
      window.ctx.toaster.layer.lookupAll(".label").asScala.collect {
        case label: Label if label.getText.startsWith("Recorded device differs:") => label.getText
      }.toSeq
    }
    def imported(hardware: Option[String]): String = Infotext.render("restored cat", "bad",
      ListMap("Steps" -> "4", "Seed" -> "901", "Size" -> "64x64") ++
        hardware.map("Hardware Info" -> _) ++ ListMap("Time taken" -> "999.0 sec.", "Remark" -> "imported remark"))
    try
      window = fx { val value = new MainWindow(new Stage(), services, UiState(), CliOptions()); value.show(); value }
      fx { window.restore(imported(Some("Remote offline device, CPU, RAM"))) }
      require(services.supervisor.pid.isEmpty && services.supervisor.hello.isEmpty,
        "Offline metadata restoration started the engine")
      systemTab()
      fx {
        require(!find[CheckBox]("hardware-info-enabled").isSelected)
        find[TitledPane]("hardware-info-settings").setExpanded(true)
      }
      checked("native hardware metadata starts opt-in Off; offline Restore does not start a model or engine")
      val hello = services.start()
      require(hello.dryRun == !real)
      val loaded = if real then Some(services.loadCheckpoint(args(3), remember = false)) else None
      loaded.foreach(model => fx { window.model.checkpoint.set(model.name) })
      require(!real || loaded.exists(model => Set("sd15", "sdxl")(model.kind)))
      val remark = "机, 日本語\nquiet test"
      fx {
        find[TextArea]("hardware-info-remark").setText(remark)
        find[TitledPane]("system-info-metadata").setExpanded(true)
        window.root.applyCss(); window.root.layout()
        val selection = find[Node]("system-info-metadata-fields").lookupAll(".check-box").asScala.collectFirst {
          case check: CheckBox if check.getText == "platform.python" => check
        }.get
        if !selection.isSelected then selection.fire()
      }
      eventually("remark saved through native controls") { services.settings.hardwareInfoRemark == remark }
      require(!services.settings.hardwareInfoEnabled)
      val size = if loaded.exists(_.kind == "sdxl") then 384 else 256
      val params = GenerationParams(prompt = "a small lighthouse at sunset", negativePrompt = "blurry", seed = 197,
        subseed = 18, steps = 4, width = size, height = size, batchSize = 2, nIter = 2, checkpoint = loaded.map(_.name))
      val baseline = generate(params)
      require(baseline.size == 4)
      require(baseline.forall(image => Infotext.parse(image.infotext).get("Hardware Info").isEmpty))
      toggle(true)
      val enabled = generate(params)
      require(enabled.map(_.seed) == Seq(197L, 198L, 199L, 200L))
      val profiles = enabled.map(image => verify(image.path, remark))
      require(profiles.distinct.size == 1)
      baseline.zip(enabled).foreach((a, b) => require(pixels(a.path) == pixels(b.path), "Metadata changed image pixels"))
      require(enabled.forall(image => Infotext.parse(image.infotext).get("System Info").exists(_.contains("platform.python"))))
      val persisted = Settings.load(home.settingsFile)
      require(persisted.hardwareInfoEnabled && persisted.hardwareInfoRemark == remark)
      require(persisted.systemInfoMetadataFields.contains("platform.python"))
      checked("actual native controls persist literal remark and independent System Info selection; four per-image records keep seed association and exact enabled/off pixels")
      val source = enabled.head.path
      val sourceHash = hash(source)
      val img2img = generate(params.copy(seed = 201, batchSize = 1, nIter = 1,
        img2img = Some(Img2ImgParams(source.toString, denoisingStrength = .5)))).head
      verify(img2img.path, remark)
      val hires = generate(params.copy(seed = 202, batchSize = 1, nIter = 1,
        hires = HiresParams(enabled = true, scale = 1.5, secondPassSteps = 4))).head
      verify(hires.path, remark)
      require(hash(source) == sourceHash)
      checked("img2img and hires save hardware/timing fields after final processing and preserve source bytes")
      toggle(false)
      val extrasBaseline = services.generator.upscaleImage(source, "Lanczos", 1.5)
      toggle(true)
      fx {
        val tabs = find[TabPane]("main-tabs")
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getText == "Extras").get)
        window.extras.image.set(source.toString); window.extras.upscaler.set("Lanczos"); window.extras.scale.set(1.5)
        val run = window.extras.node.lookupAll(".button").asScala.collectFirst {
          case button: Button if button.getText.contains("Upscale") => button
        }.get
        run.fire()
      }
      eventually("actual native Extras result") { fx { window.extras.comparisonPair.exists((_, path) => Files.isRegularFile(path)) } }
      val extras = fx(window.extras.comparisonPair.get._2)
      verify(extras, remark)
      val extrasInfo = Infotext.parse(PngMetadata.readParameters(extras).get)
      require(extrasInfo.prompt.isEmpty && extrasInfo.get("Postprocess upscaler").contains("Lanczos"))
      require(pixels(extrasBaseline) == pixels(extras) && hash(source) == sourceHash)
      checked("actual native Extras writes whole-operation provenance, exact output pixels and processing identity without replacing its input")
      systemTab()
      eventually("actual compact hardware preview") { fx { find[Label]("hardware-info-preview").getText.contains(profiles.head) } }
      val originalPid = services.supervisor.pid
      val originalModel = fx(window.model.checkpoint.get -> window.model.vae.get)
      clearToasts()
      fx { window.restore(imported(Some(profiles.head))) }
      Thread.sleep(350)
      require(notices.isEmpty)
      clearToasts()
      fx { window.restore(imported(Some("unknown"))) }
      Thread.sleep(350)
      require(notices.isEmpty)
      clearToasts()
      fx { window.restore(imported(Some("Remote device A, CPU, RAM"))) }
      eventually("first actual Restore provenance notice") { notices.exists(_.contains("Remote device A")) }
      clearToasts()
      fx { window.sendTo(Mode.Img2Img, source, Some(imported(Some("Remote device B, CPU, RAM")))) }
      eventually("second actual Send-to provenance notice") { notices.exists(_.contains("Remote device B")) }
      clearToasts()
      fx { window.sendTo(Mode.Inpaint, source, Some(imported(Some("Remote device C, CPU, RAM")))) }
      Thread.sleep(350)
      require(notices.isEmpty, "Mismatch notices exceeded the two-per-window source budget")
      require(services.supervisor.pid == originalPid && fx(window.model.checkpoint.get -> window.model.vae.get) == originalModel)
      require(services.settings.hardwareInfoEnabled && services.settings.hardwareInfoRemark == remark)
      require(fx(window.vms(Mode.Inpaint).prompt.get == "restored cat" && window.vms(Mode.Inpaint).seed.get == 901))
      checked("actual Restore/Send-to ignore equal/unknown hardware, share exactly two mismatch notices and never change the local device or metadata settings")
      eventually("final gallery contains only actual generated images") { fx(window.galleryVM.items.size == galleryExpected) }
      systemTab()
      fx {
        find[TitledPane]("system-info-metadata").setExpanded(false)
        find[TitledPane]("hardware-info-settings").setExpanded(true)
        window.stage.setWidth(1280); window.stage.setHeight(800)
      }
      snapshot("hardware-metadata-1280x800.png")
      fx { window.stage.setWidth(1600); window.stage.setHeight(1000) }
      snapshot("hardware-metadata-expanded.png")
      Files.writeString(output.resolve("hardware-info-report.json"), ujson.write(ujson.Obj("status" -> "passed",
        "mode" -> (if real then "real_model" else "cpu_dry_run"), "model" -> loaded.map(_.name).getOrElse("dry-run"),
        "home" -> home.root.toString, "hardware" -> profiles.head, "checks" -> ujson.Arr.from(checks),
        "gallery_count" -> galleryExpected, "source_sha256" -> sourceHash, "extras" -> extras.toString,
        "enabled_images" -> ujson.Arr.from(enabled.map(_.path.toString)), "remark" -> remark), indent = 2))
      println("HARDWARE INFO VERIFIED: " + output.resolve("hardware-info-report.json"))
    catch
      case error: Throwable => primaryFailure = error; throw error
    finally
      NativeAcceptanceSafety.cleanup(primaryFailure)(
        () => { if window != null then fx { window.dispose() } },
        () => { if window != null then fx { window.stage.close() } },
        () => services.shutdown(), () => Platform.exit())
