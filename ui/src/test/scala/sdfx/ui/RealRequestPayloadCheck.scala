package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.stage.{Stage, Window}
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.GenerationParams
import sdfx.core.payload.{PayloadSnapshot, PayloadState}
import sdfx.core.plugins.api.*
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{PayloadDocument, PayloadJsonTree}
import sdfx.ui.vm.{Mode, UiState}

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, ConcurrentHashMap, ConcurrentLinkedQueue, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** ROOT-ONLY manual native gate: python output-directory [checkpoint-directory checkpoint-name].
  * Uses the actual sdfx stdio worker and native Generate/Queue buttons. The no-checkpoint run uses
  * --dry-run with a tiny synthetic safetensors header; it never imports or starts parent reForge.
  * Captures are compared to submitted UI configuration and output facts. Engine fake get_state wire
  * equality is covered by engine RequestPayloadSuite; this helper cannot independently sniff stdio.
  */
object RealRequestPayloadCheck:
  private def fx[A](body: => A): A =
    if Platform.isFxApplicationThread then body
    else
      val task = new FutureTask[A](new Callable[A]:
        def call(): A = body
      )
      Platform.runLater(task)
      task.get(45, TimeUnit.SECONDS)

  private def eventually(label: String, timeout: FiniteDuration = 90.seconds)(body: => Boolean): Unit =
    val end = System.nanoTime() + timeout.toNanos
    while !body && System.nanoTime() < end do Thread.sleep(30)
    require(body, s"Timed out: $label")

  private def sha(path: Path): String = MessageDigest.getInstance("SHA-256")
    .digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString

  private def shaText(text: String): String = MessageDigest.getInstance("SHA-256")
    .digest(text.getBytes(java.nio.charset.StandardCharsets.UTF_8)).map(b => f"${b & 255}%02x").mkString

  private def pixels(path: Path): Vector[Int] =
    val image = ImageIO.read(path.toFile)
    require(image != null, s"Cannot decode $path")
    image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth).toVector

  private def fakeCheckpoint(dir: Path): String =
    Files.createDirectories(dir)
    val name = "api-payload-control"
    val header = ujson.Obj(
      "model.diffusion_model.input_blocks.0.0.weight" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(320, 4, 3, 3), "data_offsets" -> ujson.Arr(0, 2)),
      "cond_stage_model.transformer.text_model.x" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(1), "data_offsets" -> ujson.Arr(2, 4)))
      .render().getBytes(java.nio.charset.StandardCharsets.UTF_8)
    Files.write(dir.resolve(s"$name.safetensors"), ByteBuffer.allocate(8 + header.length + 4)
      .order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).put(header).putInt(0).array())
    name

  /** A test-only native script. The real UI selects it; every call still goes through the real queue,
    * Generator and Python worker. The gate makes Queue plus later form edits deterministic.
    */
  private final class AcceptanceScript extends Plugin:
    @volatile var entered = new CountDownLatch(1)
    @volatile var release = new CountDownLatch(1)
    def resetGate(): Unit = { entered = new CountDownLatch(1); release = new CountDownLatch(1) }
    def id = "acceptance_payload_script"
    def name = "Acceptance payload script"
    def kind = PluginKind.Script
    override def controls: Seq[sdfx.core.plugins.api.Control] = Seq(sdfx.core.plugins.api.Control.Choice("scenario", "Scenario",
      Seq("repeat", "gate", "fail_after_first", "fail_subrequest"), "repeat"))
    override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
      v.str("scenario", "repeat") match
        case "gate" =>
          entered.countDown()
          require(release.await(90, TimeUnit.SECONDS), "Acceptance gate was not released")
          JobOutput(ctx.generate(p))
        case "fail_after_first" =>
          ctx.generateTransient(p)
          throw new IllegalStateException("test-only script failure after successful transient generation")
        case "fail_subrequest" =>
          ctx.generateTransient(p)
          JobOutput(ctx.generate(p.copy(samplerName = "unsupported acceptance sampler")))
        case _ =>
          ctx.generateTransient(p)
          JobOutput(ctx.generate(p))

  private final class AcceptanceAlwaysOn extends Plugin:
    def id = "acceptance_payload_always_on"
    def name = "Acceptance payload AlwaysOn"
    def kind = PluginKind.AlwaysOn
    override def controls: Seq[sdfx.core.plugins.api.Control] = Seq(sdfx.core.plugins.api.Control.Check("enabled", "Enabled"),
      sdfx.core.plugins.api.Control.Text("tag", "Tag", "default tag"))
    override def process(p: GenerationParams, v: Values): GenerationParams =
      p.copy(extra = p.extra.updated("Acceptance payload tag", v.str("tag")))

  def main(args: Array[String]): Unit =
    require(args.length == 2 || args.length == 4, "python output-directory [checkpoint-directory checkpoint-name]")
    val python = Paths.get(args(0)).toAbsolutePath.normalize()
    val output = Paths.get(args(1)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val real = args.length == 4
    val home = AppHome(Files.createTempDirectory(output, "api-payload-home-"))
    val checkpointDir = if real then Paths.get(args(2)).toAbsolutePath.normalize() else home.root.resolve("models/checkpoints")
    val checkpoint = if real then args(3) else fakeCheckpoint(checkpointDir)
    val engineDir = AppHome.detect().pythonDir
    val command = Seq(python.toString, "-m", "sdfx") ++ (if real then Nil else Seq("--dry-run", "--dry-run-delay-ms", "45"))
    val script = new AcceptanceScript
    val alwaysOn = new AcceptanceAlwaysOn
    val services = new AppServices(home,
      Settings(pythonExe = Some(python.toString), outputDir = output.resolve("images").toString,
        previewEvery = 0, modelDirs = ModelDirs(checkpoints = Seq(checkpointDir.toString))),
      dryRun = !real, builtins = StandardPlugins.withOwnAxes ++ Seq(script, alwaysOn),
      launchOverride = Some(() => EngineLaunch(command, engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = true, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    val events = new ConcurrentLinkedQueue[JobEvent]()
    val maskAtSubmission = new ConcurrentHashMap[String, (String, Vector[Int])]()
    val remove = services.queue.addListener { e =>
      e match
        case q: JobEvent.JobQueued => q.job.params.img2img.flatMap(_.mask).foreach { p =>
          val path = Paths.get(p)
          require(Files.isRegularFile(path), "Native inpaint mask absent at queue submission")
          maskAtSubmission.put(q.job.id, sha(path) -> pixels(path))
        }
        case _ => ()
      events.add(e)
      ()
    }
    var window: MainWindow = null
    var primary: Throwable = null
    var checks = Vector.empty[String]
    var images = Vector.empty[Path]
    var screenshots = Vector.empty[Path]
    var clipboardResult = "not_attempted"
    var clipboardSaved: Option[NativeAcceptanceSafety.Snapshot] = None
    def checked(label: String): Unit = { checks :+= label; println("PAYLOAD CHECK: " + label) }
    def eventList: Vector[JobEvent] = events.iterator().asScala.toVector
    def latestJob(prompt: String): String = eventList.collect {
      case e: JobEvent.JobQueued if e.job.params.prompt == prompt => e.job.id
    }.lastOption.getOrElse(throw new IllegalStateException(s"No queued native job for $prompt"))
    def waitJob(id: String): Vector[PayloadSnapshot] =
      eventually("native job result " + id, 15.minutes) {
        eventList.exists {
          case e: JobEvent.JobFinished if e.jobId == id => true
          case e: JobEvent.JobFailed if e.jobId == id => true
          case e: JobEvent.JobCancelled if e.jobId == id => true
          case _ => false
        }
      }
      eventually("payload terminal state " + id) {
        services.payloads.snapshots.exists(s => s.key.jobId == id && s.state.isTerminal)
      }
      services.payloads.snapshots.filter(_.key.jobId == id)
    def savedImages(id: String): Vector[Path] = eventList.collect {
      case e: JobEvent.ImageSaved if e.jobId == id && !e.grid => e.image.path
    }
    def submit(mode: Mode, prompt: String, queue: Boolean = false): String =
      if !queue then eventually("native Generate returns to idle before " + prompt) { fx {
        !window.progress.get.running && window.statusBar.generateBtn.getText.contains("Generate")
      }}
      fx {
        window.select(mode)
        window.vms(mode).prompt.set(prompt)
        if queue then window.statusBar.queueBtn.fire() else window.statusBar.generateBtn.fire()
      }
      eventually("native UI admission " + prompt) {
        eventList.exists { case e: JobEvent.JobQueued if e.job.params.prompt == prompt => true; case _ => false }
      }
      latestJob(prompt)
    def body(s: PayloadSnapshot): ujson.Value = ujson.read(s.bodyJson.getOrElse(sys.error("No body " + s.key)))
    def context(s: PayloadSnapshot): ujson.Value = ujson.read(s.contextJson.getOrElse(sys.error("No context " + s.key)))
    def assets(s: PayloadSnapshot): ujson.Value = ujson.read(s.assetsJson.getOrElse(sys.error("No assets " + s.key)))
    def assertCapture(s: PayloadSnapshot, mode: Mode, prompt: String): Unit =
      require(s.sourceMode.contains(mode.key) && s.requestType == (if mode == Mode.Txt2Img then "txt2img" else "img2img"))
      require(body(s)("params")("prompt").str == prompt)
      require(context(s)("submitted")("source_mode").str == mode.key)
      require(context(s)("reported_at_capture").obj.contains("captured_at_millis"))
      require(s.state == PayloadState.Succeeded, s"Capture ${s.key} state ${s.state}")
    def configure(mode: Mode, seed: Long, size: Int): Unit = fx {
      val vm = window.vms(mode)
      vm.seed.set(seed); vm.subseed.set(18); vm.steps.set(if real then 3 else 2)
      vm.width.set(size); vm.height.set(size); vm.batchCount.set(1); vm.batchSize.set(1)
      vm.sampler.set("Euler"); vm.scheduler.set("Automatic")
      vm.negativePrompt.set("blurry")
      window.plugins(mode).selectedScript.set("")
    }
    def captureScene(stage: Stage, name: String, width: Int, height: Int): Unit =
      fx {
        val scene = stage.getScene
        require(stage.isShowing && scene != null, s"Stage not showing for $name")
        // Stage dimensions include window decorations. Target the actual scene client area.
        stage.setWidth(width + math.max(0.0, stage.getWidth - scene.getWidth))
        stage.setHeight(height + math.max(0.0, stage.getHeight - scene.getHeight))
      }
      eventually(s"scene client area ${width}x$height for $name", 30.seconds) { fx {
        val scene = stage.getScene
        math.round(scene.getWidth).toInt == width && math.round(scene.getHeight).toInt == height
      }}
      fx {
        val scene = stage.getScene
        scene.getRoot.applyCss(); scene.getRoot.layout()
        val path = output.resolve(name)
        val png = SwingFXUtils.fromFXImage(scene.snapshot(null), null)
        require(png.getWidth == width && png.getHeight == height,
          s"Scene snapshot dimensions ${png.getWidth}x${png.getHeight}, expected ${width}x$height")
        require(ImageIO.write(png, "png", path.toFile))
        val saved = ImageIO.read(path.toFile)
        require(saved != null && saved.getWidth == width && saved.getHeight == height,
          s"Saved PNG dimensions differ from ${width}x$height")
        screenshots :+= path
      }
    def screenshot(name: String, width: Int, height: Int): Unit = captureScene(window.stage, name, width, height)
    def inspector: Stage = fx {
      Window.getWindows.asScala.collectFirst {
        case stage: Stage if stage.isShowing && stage.getScene != null &&
          stage.getScene.getRoot.getId == "payload-inspector" => stage
      }.getOrElse(throw new IllegalStateException("Shared payload inspector stage is not showing"))
    }
    def inInspector[A <: Node](id: String): A = fx {
      val node = inspector.getScene.lookup("#" + id)
      require(node != null, "Missing inspector control #" + id)
      node.asInstanceOf[A]
    }
    def openInspector(mode: Mode): Unit =
      fx {
        window.select(mode)
        val entry = window.scene.lookup("#payload-entry-" + mode.key).asInstanceOf[TitledPane]
        require(entry != null && !entry.isExpanded, s"${mode.key} payload entry must begin collapsed")
        entry.setExpanded(true)
        val open = window.scene.lookup("#payload-open-" + mode.key).asInstanceOf[Button]
        require(open != null)
        open.fire()
      }
      eventually("payload inspector stage") { fx {
        Window.getWindows.asScala.exists(w => w.isShowing && w.getScene != null && w.getScene.getRoot.getId == "payload-inspector")
      }}
    def historySelect(key: sdfx.core.payload.PayloadKey): Unit = fx {
      val combo = inInspector[ComboBox[Object]]("payload-history")
      val at = combo.getItems.asScala.indexWhere { item =>
        try item.getClass.getMethod("key").invoke(item) == key
        catch case NonFatal(_) => false
      }
      require(at >= 0, s"History missing ${key.jobId}/${key.sequence}")
      combo.getSelectionModel.select(at)
    }
    def chooseDocument(document: PayloadDocument): Unit = fx {
      inInspector[ComboBox[PayloadDocument]]("payload-document").getSelectionModel.select(document)
    }
    def inspectorScreenshot(name: String, width: Int, height: Int): Unit =
      captureScene(inspector, name, width, height)
    try
      window = fx { val w = new MainWindow(new Stage(), services, UiState(), CliOptions()); w.show(); w }
      openInspector(Mode.Txt2Img)
      require(fx { inInspector[Label]("payload-status").getText.contains("No payload") })
      require(services.supervisor.pid.isEmpty, "Opening payload inspector started offline engine")
      screenshot("api-payload-form-1280x800.png", 1280, 800)
      checked("all native mode entries begin collapsed; opening empty inspector leaves engine offline")
      val payloadStage = inspector
      fx { payloadStage.hide(); require(!payloadStage.isShowing) }
      val hello = services.start()
      require(hello.dryRun == !real)
      val loaded = services.loadCheckpoint(checkpoint, remember = false)
      require(!real || Set("sd15", "sdxl")(loaded.kind), s"Unexpected real model kind: ${loaded.kind}")
      fx { window.model.checkpoint.set(loaded.name) }
      val size = if real && loaded.kind == "sdxl" then 384 else 256
      Mode.values.foreach(mode => configure(mode, 81402, size))
      val baselineForm = fx { window.vms(Mode.Txt2Img).snapshot }
      // Three matching fixed-seed outputs: inspector closed, open, then closed again.
      val baselineId = submit(Mode.Txt2Img, "payload fixed lighthouse")
      val baseline = waitJob(baselineId).head
      require(fx { !payloadStage.isShowing }, "Inspector opened during the closed baseline generation")
      assertCapture(baseline, Mode.Txt2Img, "payload fixed lighthouse")
      val captureFacts = context(baseline)
      require(captureFacts("reported_at_capture")("model")("name").str == loaded.name,
        "Capture-time model fact did not match the native loaded-model report")
      require(captureFacts("reported_at_completion")("model_name").str == loaded.name,
        "Completion-time model fact is missing or differs from the actual result")
      fx { window.openPayload(Mode.Txt2Img); require(payloadStage.isShowing) }
      eventually("inspector follows actual first capture") { fx {
        inInspector[Label]("payload-status").getText.contains(baselineId)
      }}
      val baselineImage = savedImages(baselineId).head
      val baselinePixels = pixels(baselineImage)
      images :+= baselineImage
      checked("actual native Generate produces exact frozen sdfx body, model capture context and baseline image")

      val repeatId = submit(Mode.Txt2Img, "payload fixed lighthouse")
      val repeated = waitJob(repeatId).head
      require(fx { payloadStage.isShowing }, "Inspector closed during the open-state parity generation")
      require(repeatId != baselineId && repeated.bodyJson == baseline.bodyJson)
      require(pixels(savedImages(repeatId).head) == baselinePixels)
      images ++= savedImages(repeatId)
      checked("two identical native submissions keep distinct job identities and identical body/pixels")

      // Test-only script deliberately emits an identical transient and final request.
      fx {
        val plugins = window.plugins(Mode.Txt2Img)
        plugins.selectedScript.set(script.id)
        plugins.form(script.id).get.prop("scenario").get.set("repeat")
      }
      val scriptId = submit(Mode.Txt2Img, "payload script repeat")
      val repeatedScript = waitJob(scriptId)
      require(repeatedScript.map(_.key.sequence) == Vector(1, 2))
      require(repeatedScript.forall(_.state == PayloadState.Succeeded) && repeatedScript(0).bodyJson == repeatedScript(1).bodyJson)
      require(savedImages(scriptId).size == 1, "Transient request escaped into gallery")
      val config = context(repeatedScript.head)("submitted")("configuration")
      require(config("selected_script")("id").str == script.id)
      require(config("always_on").arr.exists(x => x("id").str == alwaysOn.id &&
        !x("active").bool && x("values")("tag").str == "default tag"))
      images ++= savedImages(scriptId)
      checked("selected script emits distinct transient/final captures; inactive AlwaysOn form is submitted metadata")

      fx {
        window.plugins(Mode.Txt2Img).selectedScript.set("")
        val form = window.plugins(Mode.Txt2Img).form(alwaysOn.id).get
        form.prop("enabled").get.set("true")
        form.prop("tag").get.set("active marker")
      }
      val activeId = submit(Mode.Txt2Img, "payload active always on")
      val active = waitJob(activeId).head
      require(body(active)("params")("extra")("Acceptance payload tag").str == "active marker")
      require(context(active)("submitted")("configuration")("always_on").arr.exists(x => x("id").str == alwaysOn.id && x("active").bool))
      images ++= savedImages(activeId)
      checked("active AlwaysOn changes actual body while inactive form remains context only")

      // The first job is held on the queue worker. The second comes from native Queue, then its form is edited.
      fx {
        window.plugins(Mode.Txt2Img).form(alwaysOn.id).get.prop("enabled").get.set("false")
        window.plugins(Mode.Txt2Img).selectedScript.set(script.id)
        window.plugins(Mode.Txt2Img).form(script.id).get.prop("scenario").get.set("gate")
      }
      val gateId = submit(Mode.Txt2Img, "payload gate")
      require(script.entered.await(60, TimeUnit.SECONDS), "Test-only gate script was not entered")
      fx { window.plugins(Mode.Txt2Img).selectedScript.set("") }
      val queuedId = submit(Mode.Txt2Img, "queued original 猫 <b>", queue = true)
      fx { window.vms(Mode.Txt2Img).prompt.set("edited after queue"); window.select(Mode.Img2Img) }
      script.release.countDown()
      waitJob(gateId); val queued = waitJob(queuedId).head
      assertCapture(queued, Mode.Txt2Img, "queued original 猫 <b>")
      require(!queued.bodyJson.get.contains("edited after queue"))
      require(fx { inInspector[Label]("payload-status").getText.contains(queuedId) },
        "Old-mode completion was lost when Img2Img tab became active")
      require(savedImages(queuedId).nonEmpty)
      images ++= savedImages(gateId) ++ savedImages(queuedId)
      checked("native Queue snapshots form before later edits and old-mode completion retains source mode")

      fx { window.select(Mode.Txt2Img); window.vms(Mode.Txt2Img).hiresEnabled.set(true)
        window.vms(Mode.Txt2Img).hiresScale.set(1.5); window.vms(Mode.Txt2Img).hiresSteps.set(3) }
      val hiresId = submit(Mode.Txt2Img, "payload hires")
      val hires = waitJob(hiresId).head
      require(body(hires)("params")("hires")("enabled").bool)
      require(services.payloads.snapshots.count(_.key.jobId == hiresId) == 1, "Hires created a false second request")
      images ++= savedImages(hiresId)
      fx { window.vms(Mode.Txt2Img).hiresEnabled.set(false) }
      checked("hires is one txt2img body with nested native hires fields")

      // Use the actual generated PNG as the init input, then record default/opt-in asset policy.
      val sourceHash = sha(baselineImage)
      fx { window.vms(Mode.Img2Img).initImage.set(baselineImage.toString) }
      val imageId = submit(Mode.Img2Img, "payload image input")
      val imageCapture = waitJob(imageId).head
      assertCapture(imageCapture, Mode.Img2Img, "payload image input")
      require(body(imageCapture)("params")("img2img")("init_image").str == baselineImage.toString)
      require(!imageCapture.assetsJson.get.contains("data:image/png;base64,"))
      require(assets(imageCapture)("assets").arr.exists(x => x("role").str == "init_image" && x("status").str == "omitted_source_init_images_policy"))
      images ++= savedImages(imageId)
      checked("img2img body retains native source path and default diagnostic omits image bytes")

      // Shared preference changes only future captures. Inpaint exports a temp mask through the native canvas.
      fx { window.select(Mode.Inpaint); window.vms(Mode.Inpaint).initImage.set(baselineImage.toString) }
      eventually("inpaint canvas image loaded and native mask painted") { fx {
        window.panels(Mode.Inpaint).mask.get.paintDemo()
        window.vms(Mode.Inpaint).hasMask.get
      }}
      fx {
        val include = inInspector[CheckBox]("payload-include-images")
        require(!include.isSelected && !services.settings.apiPayloadIncludeImageContents)
        include.fire()
      }
      eventually("native image diagnostic preference On persisted") {
        services.settings.apiPayloadIncludeImageContents &&
          Settings.load(home.settingsFile).apiPayloadIncludeImageContents
      }
      val inpaintId = submit(Mode.Inpaint, "payload inpaint mask")
      val inpaint = waitJob(inpaintId).head
      assertCapture(inpaint, Mode.Inpaint, "payload inpaint mask")
      val maskPath = Paths.get(body(inpaint)("params")("img2img")("mask").str)
      eventually("native temporary mask cleanup") { !Files.exists(maskPath) }
      require(assets(inpaint)("assets").arr.exists(x => x("role").str == "init_image" && x("status").str == "omitted_source_init_images_policy"))
      val maskAsset = assets(inpaint)("assets").arr.find(x => x("role").str == "inpaint_mask").get
      require(maskAsset("png_data_url").str.startsWith("data:image/png;base64,"))
      val decodedMask = java.util.Base64.getDecoder.decode(maskAsset("png_data_url").str.stripPrefix("data:image/png;base64,"))
      val decoded = ImageIO.read(new java.io.ByteArrayInputStream(decodedMask))
      require(decoded != null && decoded.getWidth == size && decoded.getHeight == size)
      val submittedMask = Option(maskAtSubmission.get(inpaintId)).getOrElse(sys.error("No native mask submission sample"))
      require(maskAsset("source_sha256").str == submittedMask._1)
      require(decoded.getRGB(0, 0, size, size, null, 0, size).toVector == submittedMask._2,
        "Diagnostic PNG changed native mask pixels")
      require(sha(baselineImage) == sourceHash)
      images ++= savedImages(inpaintId)
      checked("opt-in mask PNG remains decodable after temp cleanup; init image stays omitted and source is unchanged")

      // Fixed seed image output remains pixel-identical with inspector activity and option changes.
      fx { window.vms(Mode.Txt2Img).load(baselineForm) }
      fx { payloadStage.hide(); require(!payloadStage.isShowing) }
      val afterId = submit(Mode.Txt2Img, "payload fixed lighthouse")
      val after = waitJob(afterId).head
      require(fx { !payloadStage.isShowing }, "Inspector opened during the final closed-state generation")
      val beforeParams = body(baseline)("params").obj
      val afterParams = body(after)("params").obj
      val differingFields = (beforeParams.keySet ++ afterParams.keySet)
        .filter(key => beforeParams.get(key) != afterParams.get(key)).toVector.sorted
      require(after.bodyJson == baseline.bodyJson,
        s"Fixed-seed exact body changed after restoring the baseline form; differing params: ${differingFields.mkString(", ")}")
      require(pixels(savedImages(afterId).head) == baselinePixels)
      images ++= savedImages(afterId)
      checked("same-seed RGB and exact body remain equal after inspector/diagnostic preference changes")
      checked("fixed-seed RGB parity spans closed-before, open-during and closed-after inspector states")

      // Inspect all three documents from one selected native capture; raw preview is bounded but
      // the exact retained String remains available through the explicit Copy action.
      fx { window.openPayload(Mode.Txt2Img); require(payloadStage.isShowing) }
      fx { inInspector[ComboBox[String]]("payload-scope").getSelectionModel.select("All captures") }
      historySelect(inpaint.key)
      chooseDocument(PayloadDocument.Body)
      eventually("inpaint body shown in inspector") { fx {
        inInspector[Label]("payload-status").getText.contains(inpaintId) &&
          inInspector[TextArea]("payload-raw-preview").getText.contains("payload inpaint mask")
      }}
      val raw = fx { inInspector[TextArea]("payload-raw-preview") }
      require(fx { !raw.isEditable && raw.getText.nonEmpty })
      eventually("literal native JSON tree") { fx {
        val tree = inInspector[TreeView[PayloadJsonTree.Entry]]("payload-json-tree")
        tree.getRoot != null && !tree.getRoot.getChildren.isEmpty
      }}
      fx {
        val tree = inInspector[TreeView[PayloadJsonTree.Entry]]("payload-json-tree")
        tree.getSelectionModel.select(tree.getRoot.getChildren.get(0))
        inInspector[Button]("payload-tree-mark").fire()
        inInspector[Button]("payload-tree-show-path").fire()
        require(inInspector[Label]("payload-tree-path").getText.contains("$["))
      }
      inspectorScreenshot("api-payload-tree-marked-1280x800.png", 1280, 800)
      fx { inInspector[TabPane]("payload-views").getSelectionModel.select(1) }
      inspectorScreenshot("api-payload-raw-1600x1000.png", 1600, 1000)
      require(fx { inInspector[Label]("payload-context-facts") != null })
      chooseDocument(PayloadDocument.Context)
      eventually("context document selection") { fx { raw.getText.contains("reported_at_capture") } }
      chooseDocument(PayloadDocument.ImageContents)
      eventually("image diagnostic selection") { fx { raw.getText.contains("native_input_diagnostic") } }
      inspectorScreenshot("api-payload-image-diagnostic-1280x800.png", 1280, 800)
      screenshot("api-payload-form-1600x1000.png", 1600, 1000)
      checked("shared inspector selects exact body/context/image documents and captures both native layouts")

      fx {
        val include = inInspector[CheckBox]("payload-include-images")
        require(include.isSelected && services.settings.apiPayloadIncludeImageContents)
        include.fire()
      }
      eventually("native image diagnostic preference Off persisted") {
        !services.settings.apiPayloadIncludeImageContents &&
          !Settings.load(home.settingsFile).apiPayloadIncludeImageContents &&
          fx { !inInspector[CheckBox]("payload-include-images").isSelected }
      }
      require(services.payloads.snapshots.find(_.key == inpaint.key).flatMap(_.assetsJson) == inpaint.assetsJson,
        "Changing preference retroactively erased a retained capture")
      checked("shared opt-in control persists only future-capture preference")

      // Preflight every advertised clipboard format. If any format is unsafe, explicitly skip
      // without firing Copy or touching clipboard. Restoration is retried by final cleanup.
      fx { NativeAcceptanceSafety.captureClipboard() } match
        case Left(reason) => clipboardResult = "skipped_unsafe_original: " + reason
        case Right(saved) =>
          clipboardSaved = Some(saved)
          val documents = Seq(PayloadDocument.Body -> inpaint.bodyJson.get,
            PayloadDocument.Context -> inpaint.contextJson.get,
            PayloadDocument.ImageContents -> inpaint.assetsJson.get)
          documents.foreach { (document, expected) =>
            chooseDocument(document)
            fx { inInspector[Button]("payload-copy").fire() }
            eventually("native exact Copy " + document.label) { fx {
              javafx.scene.input.Clipboard.getSystemClipboard.getString == expected &&
                inInspector[Label]("payload-copy-feedback").getText.contains("Copied")
            }}
          }
          fx { NativeAcceptanceSafety.restore(saved) }
          clipboardSaved = None
          clipboardResult = "passed_exact_body_context_image_and_restored"
      checked("clipboard action is exact only when original formats passed safety preflight; skip is recorded")

      if !real then
        val longPrompt = "猫<literal>" * 9000
        val longId = submit(Mode.Txt2Img, longPrompt)
        val longCapture = waitJob(longId).head
        historySelect(longCapture.key)
        chooseDocument(PayloadDocument.Body)
        eventually("bounded raw preview") { fx {
          inInspector[Label]("payload-raw-notice").getText.contains("Copy includes complete")
        }}
        require(longCapture.bodyJson.get.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 65536)
        require(fx { inInspector[TextArea]("payload-raw-preview").getText.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 65536 })
        fx { NativeAcceptanceSafety.captureClipboard() } match
          case Left(reason) => clipboardResult += "; long_copy_skipped_unsafe_original: " + reason
          case Right(saved) =>
            clipboardSaved = Some(saved)
            try
              fx { inInspector[Button]("payload-copy").fire() }
              eventually("untruncated native Copy") { fx {
                javafx.scene.input.Clipboard.getSystemClipboard.getString == longCapture.bodyJson.get
              }}
              clipboardResult += "; long_copy_exact_passed"
            finally
              fx { NativeAcceptanceSafety.restore(saved) }
              clipboardSaved = None
        images ++= savedImages(longId)
        checked("over-64KiB actual request keeps full retained body while raw TextArea displays bounded preview")

      // Deterministic script-level failure: the prior transient succeeds, but no second dispatch is claimed.
      if !real then
        fx { window.plugins(Mode.Txt2Img).selectedScript.set(script.id)
          window.plugins(Mode.Txt2Img).form(script.id).get.prop("scenario").get.set("fail_after_first") }
        val failId = submit(Mode.Txt2Img, "payload script failure")
        val failed = waitJob(failId)
        require(eventList.exists { case e: JobEvent.JobFailed if e.jobId == failId => true; case _ => false })
        require(failed.size == 1 && failed.head.state == PayloadState.Succeeded)
        require(eventList.exists { case e: JobEvent.PayloadLifecycle if e.jobId == failId && e.state == PayloadState.Succeeded => true; case _ => false })
        checked("test-only script failure preserves earlier successful subrequest; no false failed dispatch claimed")

        fx { window.plugins(Mode.Txt2Img).form(script.id).get.prop("scenario").get.set("fail_subrequest") }
        val subrequestFailId = submit(Mode.Txt2Img, "payload actual subrequest failure")
        val subrequestFailed = waitJob(subrequestFailId)
        require(eventList.exists { case e: JobEvent.JobFailed if e.jobId == subrequestFailId => true; case _ => false })
        require(subrequestFailed.map(_.key.sequence) == Vector(1, 2))
        require(subrequestFailed.map(_.state) == Vector(PayloadState.Succeeded, PayloadState.Failed))
        require(subrequestFailed.forall(_.bodyJson.nonEmpty))
        require(body(subrequestFailed.last)("params")("sampler_name").str == "unsupported acceptance sampler")
        checked("actual dry-run stdio rejects second dispatched subrequest while prior successful capture remains succeeded")

        script.resetGate()
        fx { window.plugins(Mode.Txt2Img).form(script.id).get.prop("scenario").get.set("gate") }
        val cancellationGate = submit(Mode.Txt2Img, "payload cancellation gate")
        require(script.entered.await(60, TimeUnit.SECONDS))
        fx { window.plugins(Mode.Txt2Img).selectedScript.set("") }
        val cancelledId = submit(Mode.Txt2Img, "payload pending cancelled", queue = true)
        require(services.queue.cancel(cancelledId), "Pending native Queue job was not cancelled")
        val absent = waitJob(cancelledId)
        require(absent.size == 1 && absent.head.state == PayloadState.Cancelled && absent.head.bodyJson.isEmpty)
        script.release.countDown()
        waitJob(cancellationGate)
        images ++= savedImages(cancellationGate)
        checked("cancelled pending native Queue job has a distinct absent body and cannot reuse prior capture")

      val report = ujson.Obj("status" -> "passed", "mode" -> (if real then "real_model" else "cpu_dry_run"),
        "model" -> loaded.name, "model_kind" -> loaded.kind, "home" -> home.root.toString,
        "worker_command" -> ujson.Arr.from(command), "source_checkout" -> "sdfx native stdio only",
        "checks" -> ujson.Arr.from(checks), "images" -> ujson.Arr.from(images.map(p => ujson.Obj(
          "path" -> p.toString, "sha256" -> sha(p)))), "source_sha256" -> sourceHash,
        "captures" -> ujson.Arr.from(services.payloads.snapshots.map(s => ujson.Obj(
          "job_id" -> s.key.jobId, "sequence" -> s.key.sequence,
          "source_mode" -> s.sourceMode.getOrElse(""), "request_type" -> s.requestType,
          "state" -> s.state.wireName, "body_sha256" -> s.bodyJson.map(shaText).getOrElse(""),
          "context_sha256" -> s.contextJson.map(shaText).getOrElse(""),
          "assets_sha256" -> s.assetsJson.map(shaText).getOrElse(""),
          "body_utf8_bytes" -> s.bodyJson.map(_.getBytes(java.nio.charset.StandardCharsets.UTF_8).length).getOrElse(0),
          "warnings" -> ujson.Arr.from(s.warnings)))),
        "screenshots" -> ujson.Arr.from(screenshots.map(_.toString)), "clipboard" -> clipboardResult,
        "limits" -> ujson.Arr("Captured body is the typed dispatch params, not independent transport byte sniffing",
          "Independent fake-engine get_state wire oracle belongs to engine RequestPayloadSuite and must be run by root",
          "Test-only script failure is not an engine subrequest failure"))
      Files.writeString(output.resolve("api-payload-report.json"), ujson.write(report, indent = 2))
      println("API PAYLOAD ACCEPTANCE: " + output.resolve("api-payload-report.json"))
    catch
      case error: Throwable =>
        primary = error
        try Files.writeString(output.resolve("api-payload-report.json"), ujson.write(ujson.Obj(
          "status" -> "failed", "mode" -> (if real then "real_model" else "cpu_dry_run"),
          "checks" -> ujson.Arr.from(checks), "error" -> error.toString,
          "clipboard" -> clipboardResult), indent = 2))
        catch case NonFatal(reportError) => error.addSuppressed(reportError)
        throw error
    finally
      script.release.countDown()
      NativeAcceptanceSafety.cleanup(primary)(
        () => clipboardSaved.foreach(saved => fx { NativeAcceptanceSafety.restore(saved) }),
        () => remove(),
        () => { if window != null then fx { window.dispose() } },
        () => { if window != null then fx { window.stage.close() } },
        () => services.shutdown(), () => Platform.exit())
