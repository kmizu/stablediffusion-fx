package sdfx.ui

import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.*
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.payload.{PayloadSnapshot, PayloadState}
import sdfx.core.presets.{FieldKey, PresetAtom, PresetJson, PresetLimits}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.awt.{BasicStroke, Color}
import java.awt.image.BufferedImage
import java.util.concurrent.{Callable, ConcurrentHashMap, ConcurrentLinkedQueue, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** Root-only manual gate: output-directory cpu|sd15|sdxl model-path-or-empty python-executable
  * [controlnet-asset-root]. The fifth argument is a verified, local, pinned setup directory.
  * Runs the actual native buttons, queue and stdio engine in a fresh AppHome. No clipboard or network.
  * Low-step images here test parity, not image quality. The original 4K widget check is a separate
  * offline gate; this helper uses a small 128px base for hires scale 5.
  */
object RealNativePresetsCheck:
  private final case class MaskAdmission(path: Path, pngSha: String, pixels: Vector[Int])

  private def fx[A](body: => A): A =
    if Platform.isFxApplicationThread then body
    else
      val task = new FutureTask[A](new Callable[A]:
        def call(): A = body)
      Platform.runLater(task)
      task.get(45, TimeUnit.SECONDS)

  private def eventually(label: String, timeout: FiniteDuration = 90.seconds)(condition: => Boolean): Unit =
    val end = System.nanoTime() + timeout.toNanos
    while !condition && System.nanoTime() < end do Thread.sleep(30)
    require(condition, "Timed out: " + label)

  private def sha(bytes: Array[Byte]): String = MessageDigest.getInstance("SHA-256")
    .digest(bytes).map(b => f"${b & 255}%02x").mkString
  private def sha(path: Path): String =
    val digest = MessageDigest.getInstance("SHA-256")
    val in = Files.newInputStream(path)
    try
      val buffer = new Array[Byte](8 * 1024 * 1024)
      var count = in.read(buffer)
      while count != -1 do
        digest.update(buffer, 0, count)
        count = in.read(buffer)
    finally in.close()
    digest.digest().map(b => f"${b & 255}%02x").mkString
  private def shaText(value: String): String = sha(value.getBytes(java.nio.charset.StandardCharsets.UTF_8))
  private def rgb(path: Path): Vector[Int] =
    val image = ImageIO.read(path.toFile)
    require(image != null, s"Cannot decode actual saved image $path")
    image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth).toVector
  private def rgbHash(path: Path): String = sha(rgb(path).flatMap(i =>
    Seq((i >>> 24).toByte, (i >>> 16).toByte, (i >>> 8).toByte, i.toByte)).toArray)

  private final case class ExpectedAsset(name: String, repo: String, revision: String,
      license: String, configBytes: Long, configSha: String, crossAttention: Int,
      weightBytes: Long, weightSha: String)
  private val expectedAssets = Vector(
    ExpectedAsset("sd15-canny", "lllyasviel/control_v11p_sd15_canny",
      "115a470d547982438f70198e353a921996e2e819", "openrail", 996,
      "8f13c37bac4c495785d6b5edcd604c013776690794d2dddff7579bbc70960801", 768,
      1445157124L, "be713fb941fc7c625f0c7d816b6a19115783a665f3049a8974f127e0c075d9a9"),
    ExpectedAsset("sdxl-canny", "diffusers/controlnet-canny-sdxl-1.0",
      "eb115a19a10d14909256db740ed109532ab1483c", "openrail++", 1309,
      "81f9bfed178f666c332892cd568a78ed8918fe08aaee994d05428a75ab44f2ca", 2048,
      5004167864L, "ea99040544a999f814fd854575a3aee069a005d026864c8d321b82576706a221"))

  /** Checks the root-owned acquisition manifest and the exact selected trained weight bytes.
    * Other-family weights are size-checked and retain root's pinned manifest hash evidence.
    */
  private def verifyAssetRoot(root: Path, family: String): String =
    require(Files.isDirectory(root), s"ControlNet asset root missing: $root")
    val manifestPath = root.resolve("manifest.json")
    require(Files.isRegularFile(manifestPath), s"ControlNet manifest missing: $manifestPath")
    val manifest = ujson.read(Files.readString(manifestPath))
    require(manifest("status").str == "verified_downloads_only_not_loaded_or_generated",
      "ControlNet acquisition manifest is not complete")
    val entries = manifest("assets").arr.toVector
    require(entries.map(_("name").str).toSet == expectedAssets.map(_.name).toSet,
      "ControlNet manifest does not contain exactly the SD1.5 and SDXL trained candidates")
    expectedAssets.foreach { expected =>
      val entry = entries.find(_("name").str == expected.name).get
      val child = root.resolve(expected.name).toAbsolutePath.normalize()
      require(Files.isDirectory(child), s"ControlNet directory missing: $child")
      require(Paths.get(entry("directory").str).toAbsolutePath.normalize() == child,
        s"ControlNet manifest points outside selected root: ${expected.name}")
      require(entry("repo").str == expected.repo && entry("revision").str == expected.revision &&
        entry("license").str == expected.license, s"ControlNet provenance mismatch: ${expected.name}")
      val files = entry("files").arr.toVector.map(v => v("name").str -> v).toMap
      require(files.keySet == Set("config.json", "README.md", "diffusion_pytorch_model.safetensors"),
        s"Unexpected ControlNet file manifest: ${expected.name}")
      files.foreach { (name, metadata) =>
        val file = child.resolve(name)
        require(Files.isRegularFile(file) && Files.size(file) == metadata("bytes").num.toLong,
          s"ControlNet file absent or changed size: $file")
      }
      val configFile = child.resolve("config.json")
      require(Files.size(configFile) == expected.configBytes &&
        files("config.json")("sha256").str == expected.configSha &&
        sha(configFile) == expected.configSha,
        s"ControlNet config bytes/hash mismatch: ${expected.name}")
      val config = ujson.read(Files.readString(configFile))
      require(config("_class_name").str == "ControlNetModel" &&
        config("cross_attention_dim").num.toInt == expected.crossAttention,
        s"ControlNet config family mismatch: ${expected.name}")
      val weight = child.resolve("diffusion_pytorch_model.safetensors")
      require(Files.size(weight) == expected.weightBytes &&
        files("diffusion_pytorch_model.safetensors")("sha256").str == expected.weightSha,
        s"ControlNet weight size/manifest hash mismatch: ${expected.name}")
      if family + "-canny" == expected.name then
        require(sha(weight) == expected.weightSha,
          s"Selected ControlNet actual weight hash mismatch: ${expected.name}")
    }
    family + "-canny"

  private def writeEdgeImage(path: Path, size: Int): Unit =
    val image = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB)
    val brush = image.createGraphics()
    try
      brush.setColor(Color.BLACK)
      brush.fillRect(0, 0, size, size)
      brush.setColor(Color.WHITE)
      brush.setStroke(new BasicStroke(3.0f))
      brush.drawRect(size / 5, size / 5, size * 3 / 5, size * 3 / 5)
      brush.drawLine(size / 5, size * 4 / 5, size / 2, size / 3)
      brush.drawLine(size / 2, size / 3, size * 4 / 5, size * 4 / 5)
    finally brush.dispose()
    require(ImageIO.write(image, "png", path.toFile), s"Cannot save authored Canny edges: $path")

  private def fakeCheckpoint(dir: Path): String =
    Files.createDirectories(dir)
    val name = "native-presets-control"
    val header = ujson.Obj(
      "model.diffusion_model.input_blocks.0.0.weight" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(320, 4, 3, 3), "data_offsets" -> ujson.Arr(0, 2)),
      "cond_stage_model.transformer.text_model.x" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(1), "data_offsets" -> ujson.Arr(2, 4)))
      .render().getBytes(java.nio.charset.StandardCharsets.UTF_8)
    Files.write(dir.resolve(s"$name.safetensors"), ByteBuffer.allocate(8 + header.length + 4)
      .order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).put(header).putInt(0).array())
    name

  /** Only holds the queue worker. The parity script below is the registered XYZ plugin. */
  private final class QueueGate(val id: String, val name: String) extends Plugin:
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    def kind = PluginKind.Script
    override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
      entered.countDown()
      require(release.await(90, TimeUnit.SECONDS), "Queue gate was not released")
      JobOutput(ctx.generate(p))

  def main(args: Array[String]): Unit =
    require(args.length == 4 || args.length == 5,
      "output-directory cpu|sd15|sdxl model-path-or-empty python-executable [controlnet-asset-root]")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    val family = args(1).toLowerCase(java.util.Locale.ROOT)
    require(Set("cpu", "sd15", "sdxl")(family), "mode must be cpu, sd15 or sdxl")
    val real = family != "cpu"
    val modelPath = Option.when(real)(Paths.get(args(2)).toAbsolutePath.normalize())
    modelPath.foreach(path => require(Files.isRegularFile(path), s"Local checkpoint missing: $path"))
    val python = Paths.get(args(3)).toAbsolutePath.normalize()
    require(Files.isRegularFile(python), s"Python executable missing: $python")
    val assetRoot = Option.when(args.length == 5)(Paths.get(args(4)).toAbsolutePath.normalize())
    val controlnetName = if real then assetRoot.map(verifyAssetRoot(_, family)) else None
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "native-presets-home-"))
    val modelDir = modelPath.map(_.getParent).getOrElse(home.root.resolve("models/checkpoints"))
    val modelName = modelPath.map(_.getFileName.toString.stripSuffix(".safetensors"))
      .getOrElse(fakeCheckpoint(modelDir))
    val engineDir = AppHome.detect().pythonDir
    val worker = Seq(python.toString, "-m", "sdfx") ++
      (if real then Nil else Seq("--dry-run", "--dry-run-delay-ms", "45"))
    val gate = new QueueGate("native_presets_queue_gate", "Native presets XYZ queue gate")
    val controlGate = new QueueGate("native_presets_controlnet_queue_gate", "Native presets ControlNet queue gate")
    val services = new AppServices(home,
      Settings(pythonExe = Some(python.toString), outputDir = output.resolve("images").toString,
        previewEvery = 0, modelDirs = ModelDirs(checkpoints = Seq(modelDir.toString),
          controlnet = assetRoot.map(_.toString).toSeq)),
      dryRun = !real, builtins = StandardPlugins.withOwnAxes ++ Seq(gate, controlGate),
      launchOverride = Some(() => EngineLaunch(worker, engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = true, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    val events = new ConcurrentLinkedQueue[JobEvent]()
    val masks = new ConcurrentHashMap[String, MaskAdmission]()
    val remove = services.queue.addListener { e =>
      e match
        case queued: JobEvent.JobQueued => queued.job.params.img2img.flatMap(_.mask).foreach { raw =>
          val path = Paths.get(raw)
          require(Files.isRegularFile(path), s"Missing native mask at admission: $path")
          masks.put(queued.job.id, MaskAdmission(path.toAbsolutePath.normalize(), sha(path), rgb(path)))
        }
        case _ => ()
      events.add(e)
      ()
    }
    val library = home.configDir.resolve("native-presets/library-v1.json")
    var window: MainWindow = null
    var primary: Throwable = null
    var checks = Vector.empty[String]
    var runs = Vector.empty[ujson.Value]
    var source: Option[Path] = None
    var sourceHash: Option[String] = None
    var activeCnVerified = false
    var edgeHash: Option[String] = None
    def checked(name: String): Unit = { checks :+= name; println("NATIVE PRESETS CHECK: " + name) }
    def eventList: Vector[JobEvent] = events.iterator().asScala.toVector
    def presetNode[A <: Node](id: String): A =
      val stage = window.presetsInspectorStage.getOrElse(sys.error("Preset stage not created"))
      val found = stage.getScene.lookup("#" + id)
      require(found != null, s"Missing native preset control #$id")
      found.asInstanceOf[A]
    def choose(combo: ComboBox[String], name: String): Unit =
      require(combo.getItems.asScala.contains(name), s"Preset '$name' missing in ${combo.getId}")
      combo.getSelectionModel.select(name)
    def open(mode: Mode, profile: String = "Utilities"): Unit = fx {
      window.select(mode)
      val entry = window.scene.lookup(s"#presets-entry-${mode.key}").asInstanceOf[TitledPane]
      require(entry != null, s"Missing ${mode.key} native preset entry")
      entry.setExpanded(true)
      window.scene.lookup(s"#presets-open-${mode.key}").asInstanceOf[Button].fire()
      require(window.presetsInspectorStage.exists(_.isShowing))
      presetNode[ComboBox[String]]("presets-profile").getSelectionModel.select(profile)
      require(presetNode[Label]("presets-destination").getText.contains(mode.key))
    }
    def saved(name: String): Unit = eventually("native library saved " + name) {
      Files.isRegularFile(library) && scala.util.Try {
        PresetJson.readNative(Files.readAllBytes(library), PresetLimits()).toOption.exists(
          _.collections.exists(_.presets.exists(_.name == name)))
      }.getOrElse(false)
    }
    def quietBeforeEngine(label: String): Unit =
      require(services.supervisor.pid.isEmpty && services.supervisor.loadedModel.isEmpty,
        s"$label started engine/model")
      require(services.queue.isIdle && services.payloads.snapshots.isEmpty && events.isEmpty,
        s"$label made request/queue work")
    def noNewGeneration(label: String, before: (Option[Long], Option[String], Int, Int)): Unit =
      require(services.supervisor.pid == before._1 &&
        services.supervisor.loadedModel.map(_.name) == before._2 &&
        eventList.count(_.isInstanceOf[JobEvent.JobQueued]) == before._3 &&
        services.payloads.snapshots.size == before._4,
        s"$label changed PID, loaded model, queued jobs or captured requests")
    def workCounters: (Option[Long], Option[String], Int, Int) =
      (services.supervisor.pid, services.supervisor.loadedModel.map(_.name),
        eventList.count(_.isInstanceOf[JobEvent.JobQueued]), services.payloads.snapshots.size)
    def descendants(root: Node): Vector[Node] = root match
      case parent: Parent => Vector(root) ++ parent.getChildrenUnmodifiable.asScala.toVector.flatMap(descendants)
      case _ => Vector(root)
    def fieldCheck(key: String): CheckBox =
      val table = fx { presetNode[TableView[?]]("presets-fields") }
      var found: Option[CheckBox] = None
      for row <- 0 until table.getItems.size if found.isEmpty do
        fx { table.scrollTo(row) }
        found = fx {
          table.applyCss(); table.layout()
          descendants(table).collectFirst { case box: CheckBox if box.getId == "presets-field-check-" + key => box }
        }
      found.getOrElse(sys.error(s"No real Detailed field checkbox: $key"))
    def requireRecord(name: String, key: String, expected: PresetAtom): Unit =
      val lib = PresetJson.readNative(Files.readAllBytes(library), PresetLimits()).toOption.get
      val record = lib.collections.flatMap(_.presets).find(_.name == name).get
      require(record.fields.get(FieldKey(key.split("\\.").toVector)).contains(expected),
        s"Saved $name did not retain $key=$expected")
    def saveQuick(mode: Mode, name: String): Unit =
      open(mode)
      val before = workCounters
      fx {
        presetNode[TabPane]("presets-tabs").getSelectionModel.select(0)
        presetNode[TextField]("presets-quick-name").setText(name)
        presetNode[Button]("presets-quick-save").fire()
      }
      saved(name)
      eventually("Quick selector updated after Save") { fx {
        presetNode[ComboBox[String]]("presets-utilities-quick-select").getItems.asScala.contains(name)
      }}
      fx { presetNode[Button]("presets-reload-tools").fire() }
      eventually("native Save/Reload completed") { fx {
        presetNode[Label]("presets-status").getText.startsWith("Loaded") &&
          presetNode[ComboBox[String]]("presets-utilities-quick-select").getItems.asScala.contains(name)
      }}
      noNewGeneration("Save all tracked", before)
    def applyQuick(mode: Mode, name: String): Unit =
      open(mode)
      val before = workCounters
      fx {
        presetNode[TabPane]("presets-tabs").getSelectionModel.select(0)
        val selector = presetNode[ComboBox[String]]("presets-utilities-quick-select")
        if selector.getValue == name then presetNode[Button]("presets-reapply").fire()
        else choose(selector, name)
        require(presetNode[Label]("presets-status").getText.contains("Applied"),
          s"Native preset did not apply: $name")
      }
      noNewGeneration("Quick Apply", before)
    def selectDetailed(keys: Seq[String]): Unit =
      fx { presetNode[TabPane]("presets-tabs").getSelectionModel.select(1) }
      keys.foreach(key => fx { val box = fieldCheck(key); if !box.isSelected then box.fire() })
    def saveDetailed(mode: Mode, name: String, keys: Seq[String]): Unit =
      open(mode)
      selectDetailed(keys)
      val before = workCounters
      fx {
        presetNode[TextField]("presets-name").setText(name)
        presetNode[Button]("presets-save").fire()
      }
      saved(name)
      eventually("Detailed selector updated after Save") { fx {
        presetNode[ComboBox[String]]("presets-utilities-detailed-select").getItems.asScala.contains(name)
      }}
      fx { presetNode[Button]("presets-reload-tools").fire() }
      eventually("native Detailed Save/Reload completed") { fx {
        presetNode[Label]("presets-status").getText.startsWith("Loaded") &&
          presetNode[ComboBox[String]]("presets-utilities-detailed-select").getItems.asScala.contains(name)
      }}
      noNewGeneration("Detailed Save", before)
    def applyDetailed(mode: Mode, name: String): Unit =
      open(mode)
      val before = workCounters
      fx {
        presetNode[TabPane]("presets-tabs").getSelectionModel.select(1)
        val selector = presetNode[ComboBox[String]]("presets-utilities-detailed-select")
        if selector.getValue == name then presetNode[Button]("presets-reapply").fire()
        else choose(selector, name)
        require(presetNode[Label]("presets-status").getText.contains("Applied"))
      }
      noNewGeneration("Detailed Apply", before)
    def submit(mode: Mode, prompt: String, queued: Boolean = false): String =
      if !queued then eventually("Generate ready", 15.minutes) { fx {
        !window.progress.get.running && window.statusBar.generateBtn.getText.contains("Generate")
      }}
      val beforeIds = eventList.collect { case e: JobEvent.JobQueued => e.job.id }.toSet
      fx {
        window.select(mode)
        window.vms(mode).prompt.set(prompt)
        if queued then window.statusBar.queueBtn.fire() else window.statusBar.generateBtn.fire()
      }
      eventually("native job admitted " + prompt) {
        eventList.exists {
          case e: JobEvent.JobQueued if !beforeIds(e.job.id) && e.job.params.prompt == prompt => true
          case _ => false
        }
      }
      val newIds = eventList.collect {
        case e: JobEvent.JobQueued if !beforeIds(e.job.id) && e.job.params.prompt == prompt => e.job.id
      }.distinct
      require(newIds.size == 1, s"Expected one newly admitted native job for '$prompt', got $newIds")
      newIds.head
    def waitJob(id: String): Vector[PayloadSnapshot] =
      eventually("native job finished " + id, 15.minutes) {
        eventList.exists {
          case e: JobEvent.JobFinished if e.jobId == id => true
          case e: JobEvent.JobFailed if e.jobId == id => true
          case _ => false
        }
      }
      eventually("terminal payload " + id) {
        services.payloads.snapshots.exists(s => s.key.jobId == id && s.state.isTerminal)
      }
      require(!eventList.exists { case e: JobEvent.JobFailed if e.jobId == id => true; case _ => false },
        s"Native job failed: $id")
      val captures = services.payloads.snapshots.filter(_.key.jobId == id)
      require(captures.nonEmpty && captures.forall(_.state == PayloadState.Succeeded), s"Bad payload state: $id")
      captures
    def images(id: String): Vector[Path] = eventList.collect {
      case e: JobEvent.ImageSaved if e.jobId == id && !e.grid => e.image.path
    }
    def record(label: String, id: String): Vector[PayloadSnapshot] =
      val snapshots = waitJob(id)
      val paths = images(id)
      require(paths.nonEmpty, s"Actual native job saved no non-grid image: $label/$id")
      val captureDir = output.resolve("captures")
      Files.createDirectories(captureDir)
      snapshots.foreach { s =>
        val raw = s.bodyJson.getOrElse(sys.error(s"Missing submitted body ${s.key}"))
        Files.writeString(captureDir.resolve(s"${label}-${s.key.sequence}.json"), raw)
        s.contextJson.foreach(context => Files.writeString(
          captureDir.resolve(s"${label}-${s.key.sequence}-context.json"), context))
      }
      val maskInfo = Option(masks.get(id)).map { admission =>
        ujson.Obj("admitted_path" -> admission.path.toString,
          "png_sha256" -> admission.pngSha, "pixel_count" -> admission.pixels.size)
      }.getOrElse(ujson.Null)
      runs :+= ujson.Obj("label" -> label, "job_id" -> id,
        "sequence" -> ujson.Arr.from(snapshots.map(_.key.sequence)),
        "body_sha256" -> ujson.Arr.from(snapshots.map(s => shaText(s.bodyJson.get))),
        "context_sha256" -> ujson.Arr.from(snapshots.map(s => s.contextJson.map(shaText).getOrElse(""))),
        "mask" -> maskInfo,
        "images" -> ujson.Arr.from(paths.map(p => ujson.Obj("path" -> p.toString,
          "file_sha256" -> sha(p), "rgb_sha256" -> rgbHash(p)))))
      snapshots
    def comparableBody(id: String, snapshot: PayloadSnapshot): String =
      val raw = snapshot.bodyJson.getOrElse(sys.error(s"Missing body ${snapshot.key}"))
      Option(masks.get(id)) match
        case None => raw
        case Some(admission) =>
          require(admission.pixels.nonEmpty, s"Empty native mask pixels: $id")
          val body = ujson.read(raw)
          val actual = body("params")("img2img")("mask").str
          require(actual.nonEmpty, s"Missing captured mask path: $id")
          require(Paths.get(actual).toAbsolutePath.normalize() == admission.path,
            s"Dispatched mask path does not identify admission file: $id")
          body("params")("img2img").obj.update("mask", ujson.Str("native-mask-sha256:" + admission.pngSha))
          ujson.write(body)
    def parity(label: String, mode: Mode, configure: () => Unit): Unit =
      fx { configure() }
      val before = fx { window.vms(mode).snapshot }
      val directId = submit(mode, s"preset $label")
      val direct = record(label + "-direct", directId)
      saveQuick(mode, label)
      fx {
        window.vms(mode).steps.set(before.steps + 1)
        window.vms(mode).seed.set(777)
        window.vms(mode).prompt.set("edited before apply")
      }
      applyQuick(mode, label)
      require(fx(window.vms(mode).seed.get) == before.seed)
      val presetId = submit(mode, s"preset $label")
      require(directId != presetId, s"Direct and applied $label reused the same job")
      val applied = record(label + "-applied", presetId)
      if mode == Mode.Inpaint then
        val left = Option(masks.get(directId)).getOrElse(sys.error("Missing direct mask evidence"))
        val right = Option(masks.get(presetId)).getOrElse(sys.error("Missing applied mask evidence"))
        require(left.pngSha == right.pngSha && left.pixels == right.pixels,
          "Native Inpaint masks differ in PNG hash or decoded pixels")
      require(direct.map(comparableBody(directId, _)) == applied.map(comparableBody(presetId, _)),
        s"Submitted body mismatch: $label")
      require(images(directId).map(rgb) == images(presetId).map(rgb), s"RGB mismatch: $label")
      checked(s"$label direct/native Save-Reload-Apply ${if mode == Mode.Inpaint then "mask-path-normalized" else "exact"} body and RGB parity")
    try
      window = fx { val w = new MainWindow(new Stage(), services, UiState(), CliOptions()); w.show(); w }
      quietBeforeEngine("Initial MainWindow")
      open(Mode.Txt2Img, "Config Presets")
      eventually("seeded native defaults") { Files.isRegularFile(library) && fx {
        presetNode[ComboBox[String]]("presets-select").getItems.size >= 14
      }}
      fx {
        choose(presetNode[ComboBox[String]]("presets-select"),
          presetNode[ComboBox[String]]("presets-select").getItems.asScala.find(_.toLowerCase.contains("4k")).get)
        val vm = window.vms(Mode.Txt2Img)
        require(vm.width.get == 768 && vm.height.get == 432 && vm.hiresScale.get == 5.0)
        require(vm.hiresTarget == (3840, 2160))
      }
      quietBeforeEngine("Original 4K preset selection")
      checked("original 4K values retained without attempting 4K inference")
      fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.seed.set(-1); vm.subseed.set(99); vm.subseedStrength.set(.4); vm.variationEnabled.set(false)
        window.lastSeed = -1
      }
      val lastSeedBefore = window.lastSeed
      saveQuick(Mode.Txt2Img, "offline random seed")
      fx {
        presetNode[Button]("presets-reload-tools").fire()
        presetNode[TabPane]("presets-tabs").getSelectionModel.select(2)
        presetNode[Button]("presets-inspect").fire()
      }
      eventually("offline reload") { fx { presetNode[Label]("presets-status").getText.nonEmpty } }
      applyQuick(Mode.Txt2Img, "offline random seed")
      requireRecord("offline random seed", "generation.seed", PresetAtom.Integer(-1))
      require(fx { window.vms(Mode.Txt2Img).seed.get == -1 &&
        window.vms(Mode.Txt2Img).subseed.get == 99 &&
        window.vms(Mode.Txt2Img).subseedStrength.get == .4 &&
        !window.vms(Mode.Txt2Img).variationEnabled.get })
      require(window.lastSeed == lastSeedBefore)
      quietBeforeEngine("save/reload/inspect/apply")
      checked("offline save, reload, inspect and apply retain -1/draft variation without job, model or PID")
      fx {
        presetNode[ComboBox[String]]("presets-profile").getSelectionModel.select("Config Presets")
        presetNode[ComboBox[String]]("presets-profile").getSelectionModel.select("Utilities")
        window.vms(Mode.Txt2Img).steps.set(11)
      }
      saveDetailed(Mode.Txt2Img, "offline steps only", Seq("generation.steps"))
      requireRecord("offline steps only", "generation.steps", PresetAtom.Integer(11))
      fx { window.vms(Mode.Txt2Img).steps.set(25); window.vms(Mode.Txt2Img).cfgScale.set(11.0) }
      applyDetailed(Mode.Txt2Img, "offline steps only")
      require(fx { window.vms(Mode.Txt2Img).steps.get == 11 && window.vms(Mode.Txt2Img).cfgScale.get == 11.0 })
      fx { presetNode[CheckBox]("presets-utilities-detailed-stackable").fire() }
      applyDetailed(Mode.Txt2Img, "offline steps only")
      require(fx { window.vms(Mode.Txt2Img).steps.get == 11 && window.vms(Mode.Txt2Img).cfgScale.get == 7.0 })
      fx { presetNode[CheckBox]("presets-utilities-detailed-stackable").fire() }
      quietBeforeEngine("current/default Detailed Apply")
      checked("Detailed selected field applies over Current and Defaults through distinct real Stackable control")
      val hello = services.start()
      require(hello.dryRun == !real)
      val loaded = services.loadCheckpoint(modelName, remember = false)
      if real then require(loaded.kind == family, s"Expected $family, loaded ${loaded.kind}")
      fx { window.model.checkpoint.set(loaded.name) }
      controlnetName.foreach { selected =>
        val expectedNames = expectedAssets.map(_.name).toSet
        require(services.catalog.controlnets.map(_.name).toSet == expectedNames,
          "Actual model catalog does not expose both pinned ControlNet folders")
        require(services.catalog.findControlNet(selected).exists(_.path == assetRoot.get.resolve(selected).toString),
          s"Selected $family ControlNet does not resolve to the pinned local directory")
        eventually("native ControlNet choices follow refreshed actual catalog") { fx {
          window.catalogVM.choices(ChoiceSource.ControlNets).asScala.toSet == expectedNames
        }}
        checked("actual catalog exposes both local trained ControlNet folders and native model choices")
      }
      val size = if real && family == "sdxl" then 384 else 256
      def base(mode: Mode): Unit =
        val vm = window.vms(mode)
        vm.seed.set(12345); vm.subseed.set(99); vm.subseedStrength.set(.4)
        vm.variationEnabled.set(false); vm.steps.set(if real then 3 else 2)
        vm.width.set(size); vm.height.set(size); vm.batchCount.set(1); vm.batchSize.set(1)
        vm.sampler.set("Euler"); vm.scheduler.set("Automatic")
        vm.cfgScale.set(5); vm.negativePrompt.set("blurry")
        vm.hiresEnabled.set(false)
        window.plugins(mode).selectedScript.set("")
      fx { Mode.values.foreach(base) }
      parity("txt2img", Mode.Txt2Img, () => ())
      source = Some(images(eventList.collect { case e: JobEvent.JobQueued if e.job.params.prompt == "preset txt2img" => e.job.id }.head).head)
      sourceHash = source.map(sha)
      fx {
        window.vms(Mode.Img2Img).initImage.set(source.get.toString)
        window.vms(Mode.Img2Img).denoise.set(.45)
      }
      parity("img2img", Mode.Img2Img, () => ())
      fx {
        window.vms(Mode.Inpaint).initImage.set(source.get.toString)
        window.vms(Mode.Inpaint).denoise.set(.45)
      }
      eventually("actual native mask painted") { fx {
        window.panels(Mode.Inpaint).mask.get.paintDemo()
        window.vms(Mode.Inpaint).hasMask.get
      }}
      // Mask temp paths differ between submissions. Raw bodies are retained above. The sole
      // normalization below is the mask path, backed by identical exported PNG pixels.
      parity("inpaint", Mode.Inpaint, () => ())
      require(sha(source.get) == sourceHash.get, "Preset operations changed source image bytes")
      checked("fixed source init image and native canvas mask are unchanged by preset workflow")
      parity("hires", Mode.Txt2Img, () => fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.hiresEnabled.set(true); vm.hiresScale.set(5); vm.hiresSteps.set(if real then 3 else 2)
        vm.hiresUpscaler.set("Latent"); vm.width.set(128); vm.height.set(128)
      })
      fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.hiresEnabled.set(false); vm.width.set(size); vm.height.set(size)
        vm.variationEnabled.set(true)
      }
      parity("variation-positive", Mode.Txt2Img, () => ())
      // Restore source Extra=false with a positive raw draft. Only txt2img pixel effects
      // are asserted; img2img/inpaint get parameter and warning checks below.
      parity("variation-disabled", Mode.Txt2Img, () => fx {
        window.vms(Mode.Txt2Img).variationEnabled.set(false)
      })
      val disabledVariation = runs.find(_("label").str == "variation-disabled-applied").get
      val positiveVariation = runs.find(_("label").str == "variation-positive-applied").get
      val disabledBody = ujson.read(Files.readString(output.resolve("captures/variation-disabled-applied-1.json")))
      val positiveBody = ujson.read(Files.readString(output.resolve("captures/variation-positive-applied-1.json")))
      require(disabledBody("params")("subseed_strength").num == 0.0 &&
        positiveBody("params")("subseed_strength").num == .4)
      require(disabledVariation("body_sha256").arr.nonEmpty && positiveVariation("body_sha256").arr.nonEmpty)
      requireRecord("variation-disabled", "generation.subseedStrength", PresetAtom.Decimal(BigDecimal("0.4")))
      fx {
        window.vms(Mode.Txt2Img).variationEnabled.set(true)
        window.vms(Mode.Txt2Img).subseedStrength.set(0.0)
      }
      val explicitZeroId = submit(Mode.Txt2Img, "preset variation-disabled")
      val explicitZero = record("variation-explicit-zero", explicitZeroId).head
      require(explicitZero.bodyJson.get == Files.readString(
        output.resolve("captures/variation-disabled-applied-1.json")))
      val disabledId = disabledVariation("job_id").str
      require(images(explicitZeroId).map(rgb) == images(disabledId).map(rgb),
        "Source Extra=false differs from native explicit strength zero")
      fx { window.vms(Mode.Txt2Img).subseedStrength.set(.4); window.vms(Mode.Txt2Img).variationEnabled.set(false) }
      checked("txt2img Extra off preserves .4 draft yet matches explicit zero body/RGB; Extra on submits .4")
      // The same gate is inspectable in both image-input modes, but the Python kernel
      // explicitly warns that variation image effects are txt2img-only.
      for mode <- Seq(Mode.Img2Img, Mode.Inpaint) do
        fx { window.vms(mode).variationEnabled.set(true); window.vms(mode).subseedStrength.set(.4) }
        val positiveId = submit(mode, s"preset ${mode.key} variation warning")
        val positive = record(s"${mode.key}-variation-positive", positiveId).head
        require(ujson.read(positive.bodyJson.get)("params")("subseed_strength").num == .4)
        if real then require(eventList.exists {
          case e: JobEvent.Log if e.jobId == positiveId =>
            e.message.contains("variation seed is only applied in txt2img")
          case _ => false
        }, s"Missing engine variation warning for ${mode.key}")
        fx { window.vms(mode).variationEnabled.set(false) }
        val offId = submit(mode, s"preset ${mode.key} variation disabled")
        val off = record(s"${mode.key}-variation-disabled", offId).head
        require(ujson.read(off.bodyJson.get)("params")("subseed_strength").num == 0.0)
      checked("Img2Img/Inpaint submit zero versus positive strength and real-engine unsupported-effect warnings only")
      fx {
        val pv = window.plugins(Mode.Txt2Img)
        pv.form("grid_numbers").get.prop("enabled").get.set("true")
        pv.form("grid_numbers").get.prop("image_numbers").get.set("false")
        pv.form("asymmetric_tiling").get.prop("enabled").get.set("false")
        pv.form("asymmetric_tiling").get.prop("tile_x").get.set("false")
        window.vms(Mode.Txt2Img).controlNet.zipWithIndex.foreach { (slot, index) =>
          slot.enabled.set(false); slot.model.set("None"); slot.image.set(source.get.toString)
          slot.preprocessor.set("none"); slot.weight.set(.7 + index * .1)
          slot.guidanceStart.set(index * .1); slot.guidanceEnd.set(1.0 - index * .1)
        }
      }
      val controlKeys = (0 until 3).flatMap(i => Vector("enabled", "model", "image",
        "preprocessor", "weight", "guidanceStart", "guidanceEnd")
        .map(field => s"controlnet.$i.$field"))
      val directPluginId = submit(Mode.Txt2Img, "preset active always on")
      val directPlugin = record("plugin-direct", directPluginId).head
      saveDetailed(Mode.Txt2Img, "plugin and three disabled slots",
        Seq("plugin.grid_numbers.enabled", "plugin.grid_numbers.image_numbers",
          "plugin.asymmetric_tiling.enabled", "plugin.asymmetric_tiling.tile_x") ++ controlKeys)
      fx {
        val pv = window.plugins(Mode.Txt2Img)
        pv.form("grid_numbers").get.prop("enabled").get.set("false")
        pv.form("asymmetric_tiling").get.prop("tile_x").get.set("true")
        window.vms(Mode.Txt2Img).controlNet.foreach(slot => slot.weight.set(1.0))
      }
      applyDetailed(Mode.Txt2Img, "plugin and three disabled slots")
      require(fx { window.plugins(Mode.Txt2Img).form("grid_numbers").get.prop("enabled").get.get == "true" &&
        window.plugins(Mode.Txt2Img).form("asymmetric_tiling").get.prop("tile_x").get.get == "false" })
      (0 until 3).foreach { i =>
        requireRecord("plugin and three disabled slots", s"controlnet.$i.image", PresetAtom.Text(source.get.toString))
        require(fx { !window.vms(Mode.Txt2Img).controlNet(i).enabled.get &&
          window.vms(Mode.Txt2Img).controlNet(i).weight.get == .7 + i * .1 })
      }
      val pluginId = submit(Mode.Txt2Img, "preset active always on")
      val pluginCapture = record("plugin-active", pluginId).head
      require(directPluginId != pluginId, "Active AlwaysOn direct/applied jobs reused an ID")
      require(directPlugin.bodyJson.get == pluginCapture.bodyJson.get &&
        images(directPluginId).map(rgb) == images(pluginId).map(rgb),
        "Active AlwaysOn direct/native preset body or RGB parity differs")
      val pluginBody = ujson.read(pluginCapture.bodyJson.get)
      require(pluginBody("params")("extra")("Grid image numbers").str == "False")
      require(!pluginBody("params")("hooks").obj.contains("tiling"),
        "Inactive asymmetric tiling draft entered active request hooks")
      for capture <- Seq(directPlugin, pluginCapture) do
        val forms = ujson.read(capture.contextJson.get)("submitted")("configuration")("always_on").arr
        require(forms.exists(x => x("id").str == "grid_numbers" && x("active").bool &&
          x("values")("enabled").str == "true" && x("values")("image_numbers").str == "false"),
          "Frozen configuration lacks active GridNumbers values")
        require(forms.exists(x => x("id").str == "asymmetric_tiling" && !x("active").bool &&
          x("values")("enabled").str == "false" && x("values")("tile_x").str == "false"),
          "Frozen configuration lost inactive asymmetric tiling draft")
      checked("native Detailed restores active AlwaysOn and inactive draft with exact direct/applied body and RGB parity")
      fx {
        val pv = window.plugins(Mode.Txt2Img)
        pv.selectedScript.set("xyz_grid")
        pv.form("xyz_grid").get.prop("x_type").get.set("Seed")
        pv.form("xyz_grid").get.prop("x_values").get.set("12345,12346")
      }
      val directXyzId = submit(Mode.Txt2Img, "preset xyz")
      val directXyz = record("xyz-direct", directXyzId)
      saveDetailed(Mode.Txt2Img, "xyz native controls", Seq("script.selected",
        "plugin.xyz_grid.x_type", "plugin.xyz_grid.x_values"))
      fx {
        window.plugins(Mode.Txt2Img).selectedScript.set("")
        window.plugins(Mode.Txt2Img).form("xyz_grid").get.prop("x_values").get.set("9")
      }
      applyDetailed(Mode.Txt2Img, "xyz native controls")
      require(fx { window.plugins(Mode.Txt2Img).selectedScript.get == "xyz_grid" &&
        window.plugins(Mode.Txt2Img).form("xyz_grid").get.prop("x_values").get.get == "12345,12346" })
      val xyzId = submit(Mode.Txt2Img, "preset xyz")
      val xyz = record("xyz", xyzId)
      require(directXyzId != xyzId, "XYZ direct/applied jobs reused an ID")
      require(xyz.map(_.key.sequence) == Vector(1, 2), "XYZ did not make two actual child requests")
      require(xyz.map(_.key).distinct.size == 2)
      require(directXyz.map(_.bodyJson) == xyz.map(_.bodyJson) &&
        images(directXyzId).map(rgb) == images(xyzId).map(rgb), "XYZ direct/preset parity differs")
      checked("native Detailed script fields select registered XYZ; both child bodies and RGB match direct entry")
      // Hold a real queue worker while a later native Queue submission is frozen.
      fx { window.plugins(Mode.Txt2Img).selectedScript.set(gate.id) }
      val selectedGate = fx { window.plugins(Mode.Txt2Img).selectedScript.get }
      require(selectedGate == gate.id,
        s"Native script selector resolved XYZ queue gate as '$selectedGate', expected '${gate.id}'")
      val gateId = submit(Mode.Txt2Img, "preset queue gate")
      val admittedGate = eventList.collectFirst { case e: JobEvent.JobQueued if e.job.id == gateId => e.job }.get
      val admittedGateScript = admittedGate.script.map(_._1.id).getOrElse("<none>")
      require(admittedGateScript == gate.id,
        s"XYZ queue gate job admitted script '$admittedGateScript', expected '${gate.id}'")
      require(gate.entered.await(60, TimeUnit.SECONDS),
        s"XYZ queue worker did not enter gate: expected ${gate.id}, admitted $admittedGateScript")
      fx {
        window.plugins(Mode.Txt2Img).selectedScript.set("xyz_grid")
        window.vms(Mode.Txt2Img).subseed.set(99)
        window.vms(Mode.Txt2Img).subseedStrength.set(.4)
        window.vms(Mode.Txt2Img).variationEnabled.set(false)
      }
      val queuedId = submit(Mode.Txt2Img, "preset frozen queue", queued = true)
      val frozen = eventList.collectFirst { case e: JobEvent.JobQueued if e.job.id == queuedId => e.job }.get
      val frozenParams = frozen.params
      val frozenSubmitted = frozen.submittedContext.getOrElse(sys.error("Queued UI context missing"))
      val frozenConfiguration = frozenSubmitted.configurationJson.getOrElse(sys.error("Queued UI forms missing"))
      val frozenSteps = frozenParams.steps
      require(frozenSteps == (if real then 3 else 2) &&
        frozenParams.seed == 12345 && frozenParams.subseed == 99 &&
        frozenParams.subseedStrength == 0.0 && !frozenSubmitted.uiVariationEnabled.getOrElse(true),
        "Queued source did not freeze disabled variation with seed 12345 and subseed 99")
      require(frozen.script.exists(_._1.id == "xyz_grid"))
      require(frozen.script.exists(_._2.str("x_values") == "12345,12346"))
      require(frozen.alwaysOn.exists { (plugin, values) =>
        plugin.id == "grid_numbers" && values.bool("enabled") && !values.bool("image_numbers", true)
      })
      fx {
        window.vms(Mode.Txt2Img).steps.set(5)
        window.plugins(Mode.Txt2Img).selectedScript.set("")
        window.plugins(Mode.Txt2Img).form("xyz_grid").get.prop("x_values").get.set("9")
        window.plugins(Mode.Txt2Img).form("grid_numbers").get.prop("enabled").get.set("false")
        window.vms(Mode.Txt2Img).subseed.set(888)
        window.vms(Mode.Txt2Img).subseedStrength.set(.8)
        window.vms(Mode.Txt2Img).variationEnabled.set(true)
      }
      saveQuick(Mode.Txt2Img, "later edit")
      applyQuick(Mode.Txt2Img, "offline random seed")
      // The older preset restores the admitted variation values. Re-establish a
      // different live state after Apply so queued dispatch must use its snapshot.
      fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.subseed.set(888)
        vm.subseedStrength.set(.8)
        vm.variationEnabled.set(true)
      }
      require(fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.variationEnabled.get && vm.snapshot.subseed == 888 &&
          vm.snapshot.subseedStrength == .8 && vm.toParams.subseedStrength == .8
      }, "Live variation no longer differs from the admitted queued draft before release")
      require(frozen.params == frozenParams && frozen.params.prompt == "preset frozen queue" &&
        frozen.params.steps == frozenSteps && frozen.params.subseed == 99 &&
        frozen.params.subseedStrength == 0.0 && frozen.script.exists(_._1.id == "xyz_grid") &&
        frozen.submittedContext.exists(c => c.configurationJson.contains(frozenConfiguration) &&
          c.uiVariationEnabled.contains(false)), "Later native edits changed queued params or source variation")
      gate.release.countDown()
      record("gate", gateId)
      val frozenCaptures = record("queued-frozen", queuedId).sortBy(_.key.sequence)
      require(frozenCaptures.map(_.key.sequence) == Vector(1, 2),
        "Queued XYZ did not dispatch two distinct ordered child requests")
      frozenCaptures.zip(Vector(12345L, 12346L)).foreach { (capture, seed) =>
        val body = ujson.read(capture.bodyJson.get)("params")
        val context = ujson.read(capture.contextJson.get)
        val submitted = context("submitted")
        val forms = submitted("configuration")
        require(body("prompt").str == "preset frozen queue" && body("seed").num.toLong == seed &&
          body("steps").num.toInt == frozenSteps && body("subseed").num.toLong == 99 &&
          body("subseed_strength").num == 0.0 &&
          body("extra")("Grid image numbers").str == "False",
          s"Queued XYZ child $seed did not retain admitted generation/AlwaysOn values")
        require(submitted("ui_variation_enabled").bool == false &&
          forms == ujson.read(frozenConfiguration) &&
          forms("selected_script")("id").str == "xyz_grid" &&
          forms("selected_script")("values")("x_values").str == "12345,12346" &&
          forms("always_on").arr.exists(x => x("id").str == "grid_numbers" && x("active").bool &&
            x("values")("enabled").str == "true" && x("values")("image_numbers").str == "false") &&
          context("requested_seed_options")("seed").num.toLong == seed &&
          context("requested_seed_options")("subseed").num.toLong == 99 &&
          context("requested_seed_options")("subseed_strength").num == 0.0,
          s"Queued XYZ child $seed context did not retain admitted script/plugin/variation values")
      }
      checked("actual queued XYZ job freezes params, plugin/script settings and child bodies before later Save/Apply")
      if real then controlnetName.foreach { selected =>
        val edge = output.resolve(s"controlnet-$family-edges.png")
        writeEdgeImage(edge, size)
        edgeHash = Some(sha(edge))
        fx {
          base(Mode.Txt2Img)
          window.plugins(Mode.Txt2Img).form("grid_numbers").get.prop("enabled").get.set("false")
          val slot = window.vms(Mode.Txt2Img).controlNet(0)
          slot.enabled.set(false); slot.model.set(selected); slot.image.set(edge.toString)
          slot.preprocessor.set("none"); slot.weight.set(1.0)
          slot.guidanceStart.set(0.0); slot.guidanceEnd.set(1.0)
          window.vms(Mode.Txt2Img).controlNet.drop(1).foreach(_.enabled.set(false))
        }
        val disabledId = submit(Mode.Txt2Img, "preset trained canny edge")
        val disabled = record("controlnet-disabled", disabledId).head
        require(ujson.read(disabled.bodyJson.get)("params")("controlnet").arr.isEmpty,
          "Disabled ControlNet draft entered submitted body")
        fx { window.vms(Mode.Txt2Img).controlNet(0).enabled.set(true) }
        val directId = submit(Mode.Txt2Img, "preset trained canny edge")
        val direct = record("controlnet-direct", directId).head
        def requireActive(body: ujson.Value): Unit =
          val units = body("params")("controlnet").arr
          require(units.size == 1, "Exactly one trained ControlNet unit must be submitted")
          val unit = units.head
          require(unit("enabled").bool && unit("model").str == selected &&
            unit("image").str == edge.toString && unit("preprocessor").str == "none" &&
            unit("weight").num == 1.0 && unit("guidance_start").num == 0.0 &&
            unit("guidance_end").num == 1.0, "Submitted trained ControlNet fields differ")
          require(body("controlnet_paths")(selected).str == assetRoot.get.resolve(selected).toString,
            "Submitted trained ControlNet path is not the pinned catalog directory")
        requireActive(ujson.read(direct.bodyJson.get))
        saveDetailed(Mode.Txt2Img, s"trained canny $family", Seq(
          "controlnet.0.enabled", "controlnet.0.model", "controlnet.0.image",
          "controlnet.0.preprocessor", "controlnet.0.weight",
          "controlnet.0.guidanceStart", "controlnet.0.guidanceEnd"))
        requireRecord(s"trained canny $family", "controlnet.0.enabled", PresetAtom.Flag(true))
        requireRecord(s"trained canny $family", "controlnet.0.model", PresetAtom.Text(selected))
        requireRecord(s"trained canny $family", "controlnet.0.image", PresetAtom.Text(edge.toString))
        fx {
          val slot = window.vms(Mode.Txt2Img).controlNet(0)
          slot.enabled.set(false); slot.model.set("None"); slot.image.set("")
          slot.preprocessor.set("canny"); slot.weight.set(.2)
          slot.guidanceStart.set(.2); slot.guidanceEnd.set(.8)
        }
        applyDetailed(Mode.Txt2Img, s"trained canny $family")
        require(fx {
          val slot = window.vms(Mode.Txt2Img).controlNet(0)
          slot.enabled.get && slot.model.get == selected && slot.image.get == edge.toString &&
            slot.preprocessor.get == "none" && slot.weight.get == 1.0 &&
            slot.guidanceStart.get == 0.0 && slot.guidanceEnd.get == 1.0
        }, "Native Detailed Apply did not restore all active trained ControlNet fields")
        val appliedId = submit(Mode.Txt2Img, "preset trained canny edge")
        val applied = record("controlnet-applied", appliedId).head
        require(directId != appliedId && disabledId != directId && disabledId != appliedId,
          "Trained ControlNet disabled/direct/applied jobs reused an ID")
        requireActive(ujson.read(applied.bodyJson.get))
        require(direct.bodyJson.get == applied.bodyJson.get,
          "Trained ControlNet manual/native preset submitted bodies differ")
        require(images(directId).map(rgb) == images(appliedId).map(rgb),
          "Trained ControlNet manual/native preset RGB pixels differ")
        require(images(disabledId).map(rgb) != images(directId).map(rgb),
          "Trained ControlNet active/disabled RGB pixels did not differ at fixed seed")
        require(sha(edge) == edgeHash.get, "Native preset operations changed authored edge PNG bytes")
        checked("real trained ControlNet via native Save/Apply has exact body/RGB parity and a visible fixed-seed effect")
        // The queued generation is already assembled before subsequent native edits.
        fx {
          window.vms(Mode.Txt2Img).controlNet(0).enabled.set(false)
          window.plugins(Mode.Txt2Img).selectedScript.set(controlGate.id)
        }
        val selectedControlGate = fx { window.plugins(Mode.Txt2Img).selectedScript.get }
        require(selectedControlGate == controlGate.id,
          s"Native script selector resolved ControlNet queue gate as '$selectedControlGate', expected '${controlGate.id}'")
        val controlGateId = submit(Mode.Txt2Img, "preset controlnet queue gate")
        val admittedControlGate = eventList.collectFirst {
          case e: JobEvent.JobQueued if e.job.id == controlGateId => e.job
        }.get
        val admittedControlScript = admittedControlGate.script.map(_._1.id).getOrElse("<none>")
        require(admittedControlScript == controlGate.id,
          s"ControlNet queue gate job admitted script '$admittedControlScript', expected '${controlGate.id}'")
        require(controlGate.entered.await(60, TimeUnit.SECONDS),
          s"ControlNet queue worker did not enter gate: expected ${controlGate.id}, admitted $admittedControlScript")
        fx {
          window.vms(Mode.Txt2Img).controlNet(0).enabled.set(true)
          window.plugins(Mode.Txt2Img).selectedScript.set("")
        }
        val queuedControlId = submit(Mode.Txt2Img, "preset controlnet frozen queue", queued = true)
        val queuedControl = eventList.collectFirst {
          case e: JobEvent.JobQueued if e.job.id == queuedControlId => e.job
        }.get
        require(queuedControl.params.controlnet.size == 1 &&
          queuedControl.params.controlnet.head.model == selected &&
          queuedControl.params.controlnet.head.image.contains(edge.toString) &&
          queuedControl.params.controlnet.head.weight == 1.0 &&
          queuedControl.params.controlnet.head.guidanceStart == 0.0 &&
          queuedControl.params.controlnet.head.guidanceEnd == 1.0,
          "Queued trained ControlNet unit was not frozen at admission")
        fx {
          val slot = window.vms(Mode.Txt2Img).controlNet(0)
          slot.enabled.set(false); slot.model.set("None"); slot.image.set("")
          slot.weight.set(.3)
        }
        saveQuick(Mode.Txt2Img, "later trained controlnet edit")
        applyQuick(Mode.Txt2Img, "offline random seed")
        require(queuedControl.params.controlnet.size == 1 &&
          queuedControl.params.controlnet.head.model == selected &&
          queuedControl.params.controlnet.head.image.contains(edge.toString) &&
          queuedControl.params.controlnet.head.weight == 1.0,
          "Later native Save/Apply changed an admitted ControlNet job")
        controlGate.release.countDown()
        record("controlnet-gate", controlGateId)
        val queuedControlCapture = record("controlnet-queued-frozen", queuedControlId).head
        requireActive(ujson.read(queuedControlCapture.bodyJson.get))
        require(sha(edge) == edgeHash.get, "Queued ControlNet changed authored edge PNG bytes")
        activeCnVerified = true
        checked("trained ControlNet job retains its active image/model/weight/path after later native Save/Apply")
        applyDetailed(Mode.Txt2Img, "plugin and three disabled slots")
      }
      fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.subseed.set(99)
        vm.subseedStrength.set(.4)
        vm.variationEnabled.set(false)
      }
      require(fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.snapshot.subseed == 99 && vm.snapshot.subseedStrength == .4 &&
          !vm.variationEnabled.get && vm.toParams.subseedStrength == 0.0
      }, "Pre-restart variation draft/effective gate differs")
      val beforeRestart = workCounters
      fx { window.persist(); window.dispose(); window.stage.close() }
      val restored = UiState.load(UiState.file(home))
      window = fx { val w = new MainWindow(new Stage(), services, restored, CliOptions()); w.show(); w }
      require(fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.snapshot.subseed == 99 && vm.snapshot.subseedStrength == .4 &&
          !vm.variationEnabled.get && vm.toParams.subseedStrength == 0.0
      }, "Fresh MainWindow lost disabled raw variation draft before any preset apply")
      open(Mode.Txt2Img)
      eventually("fresh native pane reloads stored library") { fx {
        presetNode[ComboBox[String]]("presets-utilities-detailed-select").getItems.asScala
          .contains("plugin and three disabled slots")
      }}
      applyQuick(Mode.Txt2Img, "offline random seed")
      require(fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.snapshot.subseed == 99 && vm.snapshot.subseedStrength == .4 &&
          !vm.variationEnabled.get && vm.toParams.subseedStrength == 0.0
      }, "Persisted native variation preset lost disabled raw draft or effective zero")
      applyDetailed(Mode.Txt2Img, "plugin and three disabled slots")
      require(fx { window.vms(Mode.Txt2Img).controlNet.zipWithIndex.forall { (slot, index) =>
        !slot.enabled.get && slot.image.get == source.get.toString &&
          slot.weight.get == .7 + index * .1
      } })
      noNewGeneration("new window/persisted library reapply", beforeRestart)
      require(sha(source.get) == sourceHash.get, "Restart/preset apply changed source image bytes")
      checked("fresh MainWindow restores disabled raw variation before Apply, then native preset retains effective zero and three disabled ControlNet drafts without new work")
      val limits = Vector(
        "Low-step images test parity and a fixed-seed conditioning effect, not artistic quality",
        "Request captures are typed dispatch bodies, not independent stdio byte sniffing",
        "No queue events or seed changes were observed during offline preset operations; internal RNG state is not directly exposed",
        "Inpaint temporary mask path differs between submissions and needs pixel/hash-backed normalization") ++
        (if real && !activeCnVerified then Vector(
          "No trained local ControlNet asset was exercised; active ControlNet generation remains unverified and promotion is blocked")
        else Vector.empty)
      Files.writeString(output.resolve("native-presets-report.json"), ujson.write(ujson.Obj(
        "status" -> (if real && activeCnVerified then "passed_real_model_scope"
          else if real then "incomplete_active_controlnet_gate" else "passed_cpu_scope"),
        "family" -> family, "model_path" -> modelPath.map(_.toString).getOrElse(""),
        "model_sha256" -> modelPath.map(sha).getOrElse(""), "home" -> home.root.toString,
        "source_sha256" -> sourceHash.getOrElse(""),
        "controlnet_asset_root" -> assetRoot.map(_.toString).getOrElse(""),
        "controlnet_selected_model" -> controlnetName.getOrElse(""),
        "controlnet_manifest_sha256" -> assetRoot.map(p => sha(p.resolve("manifest.json"))).getOrElse(""),
        "controlnet_edge_sha256" -> edgeHash.getOrElse(""),
        "active_trained_controlnet_verified" -> activeCnVerified,
        "checks" -> ujson.Arr.from(checks), "runs" -> ujson.Arr.from(runs),
        "limits" -> ujson.Arr.from(limits)), indent = 2))
      println("NATIVE PRESETS REPORT: " + output.resolve("native-presets-report.json"))
    catch
      case error: Throwable =>
        primary = error
        try Files.writeString(output.resolve("native-presets-report.json"), ujson.write(ujson.Obj(
          "status" -> "failed", "family" -> family, "checks" -> ujson.Arr.from(checks),
          "runs" -> ujson.Arr.from(runs), "error" -> error.toString), indent = 2))
        catch case NonFatal(reportError) => error.addSuppressed(reportError)
        throw error
    finally
      gate.release.countDown()
      controlGate.release.countDown()
      NativeAcceptanceSafety.cleanup(primary)(
        () => remove(),
        () => { if window != null then fx { window.dispose() } },
        () => { if window != null then fx { window.stage.close() } },
        () => services.shutdown(), () => Platform.exit())
