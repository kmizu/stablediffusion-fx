package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import java.security.MessageDigest
import java.util.concurrent.{Callable, ConcurrentHashMap, ConcurrentLinkedQueue, CountDownLatch, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Node, Parent}
import javafx.scene.control.*
import javafx.scene.effect.GaussianBlur
import javafx.scene.image.ImageView
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.GenerationParams
import sdfx.core.modelcards.{CardId, CardPreviewMode, ModelCardRecord}
import sdfx.core.payload.{PayloadState, SubmittedPayloadContext}
import sdfx.core.plugins.api.{Control as PluginControl, *}
import sdfx.engine.*
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.*
import sdfx.ui.vm.{Mode, UiState}

/** Manual ROOT-owned CORE acceptance, outside ordinary munit execution.
  * User explicitly reduced the exhaustive Task7 matrix to core native behaviour, three
  * open/close memory observations and one fixed-seed queue path per installed real model.
  * <offline|cpu|sd15|sdxl> <python-exe> <output-dir> [<checkpoint-dir> <checkpoint-name>]
  */
object RealModelCardPreviewCheck:
  private val MiB = 1024L * 1024L
  private def fx[A](body: => A): A =
    if Platform.isFxApplicationThread then body
    else
      val task = new FutureTask[A](new Callable[A] { def call(): A = body })
      Platform.runLater(task); task.get(45, TimeUnit.SECONDS)
  private def sha(path: Path): String =
    val digest = MessageDigest.getInstance("SHA-256")
    val stream = Files.newInputStream(path)
    try
      val buffer = new Array[Byte](1024 * 1024)
      var n = stream.read(buffer)
      while n >= 0 do
        if n > 0 then digest.update(buffer, 0, n)
        n = stream.read(buffer)
    finally stream.close()
    digest.digest().map(b => f"${b & 255}%02x").mkString
  private def textHash(text: String): String = MessageDigest.getInstance("SHA-256")
    .digest(text.getBytes(UTF_8)).map(b => f"${b & 255}%02x").mkString
  private def rgbHash(path: Path): String =
    val image = ImageIO.read(path.toFile)
    require(image != null, s"Cannot decode actual generation: $path")
    try
      val digest = MessageDigest.getInstance("SHA-256")
      val row = new Array[Int](image.getWidth)
      for y <- 0 until image.getHeight do
        image.getRGB(0, y, image.getWidth, 1, row, 0, image.getWidth)
        row.foreach { rgb => digest.update((rgb >>> 16).toByte); digest.update((rgb >>> 8).toByte); digest.update(rgb.toByte) }
      digest.digest().map(b => f"${b & 255}%02x").mkString
    finally image.flush()
  private def descendants(node: Node): Vector[Node] = node match
    case p: Parent => Vector(node) ++ p.getChildrenUnmodifiable.asScala.toVector.flatMap(descendants)
    case _ => Vector(node)
  private def reflected[A](instance: AnyRef, name: String): A =
    val fields = instance.getClass.getDeclaredFields.filter(f => f.getName == name || f.getName.endsWith("$$" + name))
    require(fields.length == 1, s"Missing/ambiguous actual field ${instance.getClass.getName}.$name")
    fields.head.setAccessible(true); fields.head.get(instance).asInstanceOf[A]
  private def gitHead(root: Path): String =
    val git = root.resolve(".git")
    val dir = if Files.isDirectory(git) then git else root.resolve(Files.readString(git).stripPrefix("gitdir:").trim).normalize()
    val head = Files.readString(dir.resolve("HEAD")).trim
    if !head.startsWith("ref: ") then head
    else
      val common = if Files.exists(dir.resolve("commondir")) then dir.resolve(Files.readString(dir.resolve("commondir")).trim).normalize() else dir
      val ref = head.stripPrefix("ref: ")
      val loose = common.resolve(ref)
      if Files.exists(loose) then Files.readString(loose).trim
      else Files.readAllLines(common.resolve("packed-refs")).asScala.find(_.endsWith(" " + ref)).get.takeWhile(_ != ' ')

  private final class PassThrough extends Plugin:
    val id = "card-preview-pass-through"
    val name = "Card preview pass-through (helper only)"
    def kind = PluginKind.Script
    override def controls = Seq(PluginControl.Text("evidence", "Evidence", "retained"))
    @volatile var hold = false
    @volatile var entered = new CountDownLatch(1)
    @volatile var release = new CountDownLatch(1)
    val calls = new AtomicInteger()
    def arm(): Unit = { entered = new CountDownLatch(1); release = new CountDownLatch(1); hold = true }
    override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
      require(v.map == Map("evidence" -> "retained"))
      if hold then
        hold = false; entered.countDown()
        require(release.await(45, TimeUnit.SECONDS), "Finite queue fixture hold timed out")
      calls.incrementAndGet(); JobOutput(ctx.generate(p))
  private final class Marker extends Plugin:
    val id = "card-preview-marker"
    val name = "Card preview marker (helper only)"
    def kind = PluginKind.AlwaysOn
    override def controls = Seq(PluginControl.Text("evidence", "Evidence", "retained"))
    override def isActive(v: Values) = true
    override def process(p: GenerationParams, v: Values): GenerationParams =
      require(v.map == Map("evidence" -> "retained")); p

  // Fixed command text; PID values arrive on structured stdin. No Python observer, shell
  // path interpolation, network operation or per-process GPU attribution is involved.
  private val memoryScript = """
    $ErrorActionPreference = 'Stop'
    $inputData = [Console]::In.ReadLine() | ConvertFrom-Json
    $rows = @()
    foreach ($requestedPid in @($inputData.pids)) {
      $p = Get-Process -Id ([int]$requestedPid) -ErrorAction Stop
      $rows += [pscustomobject]@{pid=$p.Id; workingSet=$p.WorkingSet64; privateBytes=$p.PrivateMemorySize64; peakWorkingSet=$p.PeakWorkingSet64}
    }
    $gpuLines = @(& nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits)
    if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
    $gpus = @()
    foreach ($line in $gpuLines) {
      $v = @($line.Split(',') | ForEach-Object { $_.Trim() })
      if ($v.Length -ne 4) { throw 'Unexpected GPU query shape' }
      $gpus += [pscustomobject]@{index=[int]$v[0]; name=$v[1]; totalMiB=[long]$v[2]; usedMiB=[long]$v[3]}
    }
    [pscustomobject]@{utc=[DateTime]::UtcNow.ToString('o'); processes=$rows; gpuDevices=$gpus} | ConvertTo-Json -Compress -Depth 5
    """

  private final class Run(args: Array[String]):
    require(args.nonEmpty && Set("offline", "cpu", "sd15", "sdxl")(args(0)))
    val mode = args(0)
    val real = mode == "sd15" || mode == "sdxl"
    require(args.length == (if real then 5 else 3), "<offline|cpu|sd15|sdxl> <python-exe> <output-dir> [<checkpoint-dir> <checkpoint-name>]")
    val python = Paths.get(args(1)).toAbsolutePath.normalize()
    val output = Paths.get(args(2)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val root = Paths.get("").toAbsolutePath.normalize()
    val runDir = Files.createTempDirectory(output, "card-preview-core-").toRealPath()
    val home = AppHome(Files.createDirectories(runDir.resolve("home")))
    val sources = Files.createDirectories(runDir.resolve("sources"))
    val start = System.nanoTime()
    val deadline = start + 15.minutes.toNanos
    val checks = ArrayBuffer.empty[ujson.Value]
    val failures = ArrayBuffer.empty[String]
    val cleanupErrors = ArrayBuffer.empty[String]
    val screenshots = ArrayBuffer.empty[ujson.Value]
    val hashes = ArrayBuffer.empty[(Path, String, String)]
    val memory = ArrayBuffer.empty[ujson.Value]
    val runs = ArrayBuffer.empty[ujson.Value]
    val jobRaw = new ConcurrentLinkedQueue[String]()
    val events = new ConcurrentLinkedQueue[JobEvent]()
    val listenerErrors = new ConcurrentLinkedQueue[String]()
    val frozen = new ConcurrentHashMap[String, FrozenJobEvidence]()
    val transport = new ConcurrentHashMap[String, java.util.Set[String]]()
    val windows = ArrayBuffer.empty[MainWindow]
    val script = new PassThrough
    val marker = new Marker
    val starts = new AtomicInteger()
    var services: AppServices = null
    var one: MainWindow = null
    var two: MainWindow = null
    var startedFx = false
    var removeEvents: () => Unit = () => ()
    var worker: Option[ProcessHandle] = None
    var observedWorkerPid: Option[Long] = None
    var observer: Process = null
    var model: Path = null
    var modelName = ""
    var modelHash = ""
    var modelKind = ""
    var pythonVersion = "not launched (offline)"
    var preview: Path = null
    var cardModel: Path = null

    def elapsed: Long = (System.nanoTime() - start) / 1000000L
    def eventually(label: String, timeout: FiniteDuration = 45.seconds)(condition: => Boolean): Unit =
      val end = math.min(deadline, System.nanoTime() + timeout.toNanos)
      var done = condition
      while !done && System.nanoTime() < end do { Thread.sleep(20); done = condition }
      require(done, s"Observation timed out: $label; timeout is not terminal-process evidence or retry permission")
    def check(name: String)(body: => Unit): Unit =
      val began = elapsed
      try
        require(System.nanoTime() < deadline, "Core observation budget exhausted")
        body
        checks += ujson.Obj("name" -> name, "expected" -> "assertions pass", "observed" -> "passed", "elapsedMs" -> (elapsed - began))
      catch case e: Throwable =>
        failures += s"$name: $e"
        checks += ujson.Obj("name" -> name, "expected" -> "assertions pass", "observed" -> "failed", "error" -> e.toString, "elapsedMs" -> (elapsed - began))
        throw e
    def cleanup(name: String)(body: => Unit): Unit =
      try body catch case e: Throwable => cleanupErrors += s"$name: $e"
    def remember(path: Path, provenance: String): Unit = hashes += ((path, sha(path), provenance))
    def sourceInvariant(): Unit = hashes.foreach((p, hash, _) => require(sha(p) == hash, s"Source changed: $p"))
    def eventList: Vector[JobEvent] = events.iterator().asScala.toVector
    def observeWorker(): Unit =
      if services != null && worker.isEmpty then
        services.supervisor.pid.foreach { pid =>
          observedWorkerPid = Some(pid)
          val handle = ProcessHandle.of(pid)
          if handle.isPresent then worker = Some(handle.get())
        }

    def sampleMemory(label: String): Unit =
      require(observer == null || !observer.isAlive, "Previous observer still alive; refusing duplicate")
      val ps = Paths.get(System.getenv("SystemRoot"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
      val file = runDir.resolve(s"memory-${memory.size}-$label.json")
      val pids = Vector(ProcessHandle.current().pid()) ++ worker.filter(_.isAlive).map(_.pid()).toVector
      observer = new ProcessBuilder(ps.toString, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", memoryScript)
        .redirectErrorStream(true).redirectOutput(file.toFile).start()
      try
        val writer = new java.io.OutputStreamWriter(observer.getOutputStream, UTF_8)
        try { writer.write(ujson.write(ujson.Obj("pids" -> ujson.Arr.from(pids)))); writer.write("\n"); writer.flush() }
        finally writer.close()
        require(observer.waitFor(12, TimeUnit.SECONDS), "Bounded core memory snapshot timed out")
        require(observer.exitValue() == 0 && Files.size(file) <= 65536, s"Memory query failed: $file")
        val value = ujson.read(Files.readString(file, UTF_8).stripPrefix("\ufeff"))
        require(value("processes").arr.map(_("pid").num.toLong).toSet == pids.toSet, "Missing actual PID memory")
        require(value("gpuDevices").arr.nonEmpty, "Device-total VRAM query unavailable")
        val m = PreviewImageLoader.metrics
        memory += ujson.Obj("phase" -> label, "elapsedMs" -> elapsed, "observed" -> value,
          "helperRuntimeMaxHeap" -> Runtime.getRuntime.maxMemory(), "owner" -> ujson.Obj("workers" -> m.activeWorkers,
            "sourceBuffers" -> m.admittedBuffers, "sourceBytes" -> m.sourceBytes, "images" -> m.ownedImages,
            "imageBytes" -> m.ownedImageBytes, "mailboxResults" -> m.mailboxItems, "visibleConsumers" -> m.visibleConsumers, "queued" -> m.queued))
      finally
        if observer.isAlive then
          val children = observer.descendants().toList.asScala.toVector
          children.foreach(_.destroyForcibly()); observer.destroyForcibly()
          require(observer.waitFor(2, TimeUnit.SECONDS) && children.forall(h => !h.isAlive), "Memory observer cleanup still alive")

    def stage(window: MainWindow): Stage = fx {
      Window.getWindows.asScala.collectFirst {
        case s: Stage if s.isShowing && s.getOwner == window.stage && s.getTitle == "Models and networks" => s
      }.getOrElse(sys.error("Actual Models and networks stage absent"))
    }
    def list(window: MainWindow): ListView[ModelCardRecord] = fx {
      descendants(stage(window).getScene.getRoot).collectFirst { case l: ListView[?] => l.asInstanceOf[ListView[ModelCardRecord]] }.get
    }
    def menu(window: MainWindow): MenuButton = fx {
      descendants(stage(window).getScene.getRoot).collectFirst { case m: MenuButton if m.getText.startsWith("Named NSFW previews:") => m }.get
    }
    def open(window: MainWindow, name: String): Unit =
      fx { window.ctx.openModelBrowser("checkpoint", None) }
      eventually("native catalog contains requested card") { fx { list(window).getItems.asScala.exists(_.name.contains(name)) } }
      fx { descendants(stage(window).getScene.getRoot).collectFirst {
        case f: TextField if f.getPromptText == "Search name or path" => f
      }.get.setText(name) }
      eventually("one filtered native card") { fx { list(window).getItems.size() == 1 } }
      fx { stage(window).getScene.getRoot.applyCss(); stage(window).getScene.getRoot.layout() }
    def pixels(window: MainWindow): ImageView = fx {
      val expected = list(window).getItems.get(0).name
      val card = stage(window).getScene.getRoot.lookupAll(".model-preview-card").asScala.find(n => descendants(n).exists {
        case l: Label => l.getStyleClass.contains("model-preview-name") && l.getText == expected
        case _ => false
      }).get
      descendants(card).collectFirst { case image: ImageView => image }.get
    }
    def radius(window: MainWindow): Double = fx { pixels(window).getEffect match
      case blur: GaussianBlur => blur.getRadius
      case null => 0.0
      case other => sys.error(s"Unexpected preview effect: $other")
    }
    def bothPreviewsMatch(value: CardPreviewMode): Boolean = fx {
      Seq(one, two).forall { window =>
        val image = pixels(window)
        value match
          case CardPreviewMode.Hide => image.getImage == null && !image.isVisible && !image.isManaged
          case CardPreviewMode.Show => image.getImage != null && image.isVisible && image.isManaged && radius(window) == 0
          case CardPreviewMode.Blur => image.getImage != null && image.isVisible && image.isManaged && radius(window) == 20
      }
    }
    def choose(value: CardPreviewMode): Unit =
      fx { if services.cardPreviews.current != value then menu(one).getItems.asScala.find(_.getText == value.toString).get.fire() }
      eventually("shared native mode labels") { fx {
        services.cardPreviews.current == value && menu(one).getText.endsWith(value.toString) && menu(two).getText.endsWith(value.toString)
      }}
    def screenshot(label: String): Unit =
      val (image, sx, sy) = fx { (stage(one).getScene.snapshot(null), stage(one).getOutputScaleX, stage(one).getOutputScaleY) }
      val path = runDir.resolve(label + ".png")
      require(ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", path.toFile))
      screenshots += ujson.Obj("path" -> path.toString, "sha256" -> sha(path), "actualOutputScaleX" -> sx, "actualOutputScaleY" -> sy)
    def closeBrowsers(): Unit = fx {
      Window.getWindows.asScala.toVector.collect { case s: Stage if s.getTitle == "Models and networks" => s }.foreach(_.close())
    }

    def freeze(phase: String, job: Job, expected: Option[FrozenJobEvidence] = None): FrozenJobEvidence =
      val evidence = FrozenJobEvidence.freeze(job)
      expected.foreach(before => require(before.sameJobAs(evidence), s"Complete Job/plugin reference changed: $phase/${job.id}"))
      // Exact immutable JSON is embedded without a lossy Double-backed ujson round trip.
      jobRaw.add("{\"phase\":" + ujson.write(phase) + ",\"elapsedMs\":" + elapsed + ",\"canonicalHash\":" +
        ujson.write(evidence.canonicalHash) + ",\"snapshot\":" + evidence.snapshotJson + "}")
      evidence
    def listen(): Unit =
      removeEvents = services.queue.addListener { event =>
        events.add(event)
        try event match
          case e: JobEvent.JobQueued => frozen.put(e.job.id, freeze("queued", e.job))
          case e: JobEvent.JobStarted => freeze("dispatch", e.job, Option(frozen.get(e.job.id)))
          case e: JobEvent.JobFinished => services.queue.current.filter(_.id == e.jobId).foreach(j => freeze("terminal", j, Option(frozen.get(j.id))))
          case e: JobEvent.Progress =>
            reflected[Option[EngineProcess]](services.supervisor, "proc").foreach(_.pendingIds
              .filter((_, kind) => kind == "txt2img" || kind == "img2img").foreach { (id, _) =>
                transport.computeIfAbsent(e.jobId, _ => ConcurrentHashMap.newKeySet[String]()).add(id)
              })
          case _ => ()
        catch case error: Throwable => listenerErrors.add(error.toString)
      }
    def fixtureJob(): Job = Job(Job.newId(), GenerationParams(prompt = "a small blue ceramic teapot, studio photograph", negativePrompt = "text, watermark",
      seed = 12345L, subseed = 99L, steps = 2, cfgScale = 5.0, width = 256, height = 256, samplerName = "Euler", scheduler = "Uniform",
      checkpoint = Some(modelName), hooks = Map("freeu" -> ujson.Obj("enabled" -> false))),
      script = Some(script -> Values(Map("evidence" -> "retained"))), alwaysOn = Seq(marker -> Values(Map("evidence" -> "retained"))),
      origin = "card-preview-core", limits = JobLimits(1024, 2, 3, 1024L * 1024L),
      submittedContext = Some(SubmittedPayloadContext("card-preview-core", Some("txt2img"), Some(" {\"evidence\":\"retained\"}\r\n"), Some(false))),
      captureOptions = Some(CaptureOptions(false)))
    def result(label: String, id: String): (String, String, Path) =
      eventually("actual terminal " + id, 5.minutes) { eventList.exists(e => e.jobId == id &&
        (e.isInstanceOf[JobEvent.JobFinished] || e.isInstanceOf[JobEvent.JobFailed] || e.isInstanceOf[JobEvent.JobCancelled])) }
      val terminals = eventList.filter(e => e.jobId == id &&
        (e.isInstanceOf[JobEvent.JobFinished] || e.isInstanceOf[JobEvent.JobFailed] || e.isInstanceOf[JobEvent.JobCancelled]))
      require(terminals.size == 1 && terminals.head.isInstanceOf[JobEvent.JobFinished], s"Unexpected terminal: $terminals")
      eventually("retained successful payload") { services.payloads.snapshots.exists(s => s.key.jobId == id && s.state == PayloadState.Succeeded) }
      val capture = services.payloads.snapshots.filter(_.key.jobId == id)
      require(capture.size == 1 && capture.head.bodyJson.nonEmpty)
      val body = capture.head.bodyJson.get
      val images = eventList.collect { case e: JobEvent.ImageSaved if e.jobId == id && !e.grid => e.image.path }
      require(images.size == 1)
      val dispatch = eventList.collect { case e: JobEvent.JobStarted if e.job.id == id => e.job }
      require(dispatch.size == 1 && frozen.containsKey(id) && listenerErrors.isEmpty)
      freeze("dispatch-compared", dispatch.head, Some(frozen.get(id)))
      Files.writeString(runDir.resolve(label + "-body.json"), body)
      capture.head.contextJson.foreach(text => Files.writeString(runDir.resolve(label + "-context.json"), text))
      val rgb = rgbHash(images.head)
      runs += ujson.Obj("label" -> label, "jobId" -> id, "bodySha256" -> textHash(body), "bodyFile" -> runDir.resolve(label + "-body.json").toString,
        "bodyExclusions" -> ujson.Arr(), "rgbSha256" -> rgb, "image" -> images.head.toString, "imageSha256" -> sha(images.head),
        "observedPendingTransportIds" -> ujson.Arr.from(Option(transport.get(id)).map(_.asScala.toVector.sorted).getOrElse(Vector.empty)),
        "transportScope" -> "actual pending generation IDs observed during Job progress; separate from exact Job identity")
      (body, rgb, images.head)
    def activeSnapshot(phase: String): (String, Vector[String]) =
      val current = services.queue.current.getOrElse(sys.error("No running fixture"))
      val pending = services.queue.pending.toVector
      require(pending.nonEmpty)
      (Vector(current) ++ pending).foreach(j => freeze(phase, j, Some(frozen.get(j.id))))
      jobRaw.add("{\"phase\":" + ujson.write(phase + "-order") + ",\"runningId\":" + ujson.write(current.id) +
        ",\"orderedPendingIds\":" + ujson.write(ujson.Arr.from(pending.map(_.id))) + "}")
      (current.id, pending.map(_.id))

    def setup(): Unit =
      require(Runtime.version().feature() == 21, "Java 21 required")
      require(Files.isRegularFile(python), s"Installed Python absent: $python")
      require(runDir.startsWith(output.toRealPath()))
      require(Job(GenerationParams()).productElementNames.toSet == Set("id", "params", "script", "alwaysOn", "origin", "limits", "submittedContext", "captureOptions"))
      require(GenerationParams().productElementNames.toSet == Set("prompt", "negativePrompt", "seed", "subseed", "subseedStrength", "samplerName", "scheduler", "steps", "cfgScale", "width", "height", "batchSize", "nIter", "clipSkip", "eta", "hires", "img2img", "loras", "controlnet", "hooks", "styles", "checkpoint", "vae", "extra"))
      require(SubmittedPayloadContext("fixture").productElementNames.toSet == Set("origin", "sourceMode", "configurationJson", "uiVariationEnabled"))
      Seq("ui/src/test/scala/sdfx/ui/RealModelCardPreviewCheck.scala", "ui/src/test/scala/sdfx/ui/FrozenJobEvidence.scala",
        "ui/src/main/scala/sdfx/ui/MainWindow.scala", "ui/src/main/scala/sdfx/ui/view/ModelBrowser.scala",
        "ui/src/main/scala/sdfx/ui/view/ModelPreviewCard.scala", "ui/src/main/scala/sdfx/ui/view/PreviewImageLoader.scala").foreach(p => remember(root.resolve(p), "tested source"))
      val models = Files.createDirectories(sources.resolve("models"))
      val header = ujson.write(ujson.Obj("model.diffusion_model.input_blocks.0.0.weight" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(320, 4, 3, 3), "data_offsets" -> ujson.Arr(0, 2)),
        "cond_stage_model.transformer.text_model.x" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(1), "data_offsets" -> ujson.Arr(2, 4)))).getBytes(UTF_8)
      cardModel = Files.write(models.resolve("core-card.safetensors"), ByteBuffer.allocate(8 + header.length + 4)
        .order(ByteOrder.LITTLE_ENDIAN).putLong(header.length.toLong).put(header).putInt(0).array())
      preview = sources.resolve("NSFW-preview.png")
      val image = new BufferedImage(192, 192, BufferedImage.TYPE_INT_RGB)
      try
        for y <- 0 until 192; x <- 0 until 192 do image.setRGB(x, y, if ((x / 8 + y / 8) & 1) == 0 then 0x235ed0 else 0xffcf36)
        require(ImageIO.write(image, "png", preview.toFile))
      finally image.flush()
      remember(cardModel, "synthetic safetensors metadata; CPU/offline only"); remember(preview, "authored isolated card-policy fixture")
      model = if real then
        require(args(4) == (if mode == "sd15" then "animekawa_v10" else "waiIllustriousSDXL_v170"))
        Paths.get(args(3)).toAbsolutePath.normalize().resolve(args(4) + ".safetensors")
      else cardModel
      require(Files.isRegularFile(model), s"Required installed checkpoint missing; no download fallback: $model")
      modelName = model.getFileName.toString.stripSuffix(".safetensors"); modelHash = sha(model)
      if real then remember(model, s"installed $mode checkpoint")
      val dirs = ModelDirs(checkpoints = Seq(models.toString) ++ Option.when(real)(model.getParent.toString),
        vae = Nil, lora = Nil, controlnet = Nil, embeddings = Nil, upscalers = Nil, faceRestore = Nil, svd = Nil, zero123 = Nil, photomaker = Nil)
      val engineDir = root.resolve("python")
      val command = Seq(python.toString, "-m", "sdfx") ++ (if real then Nil else Seq("--dry-run", "--dry-run-delay-ms", "300"))
      services = new AppServices(home, Settings(modelDirs = dirs, outputDir = runDir.resolve("generated-nsfw-gallery").toString,
        pythonExe = Some(python.toString), hfOffline = true, previewEvery = 0,
        modelPreviewOverrides = Map(CardId("checkpoint", cardModel.toString).storageKey -> preview.toString)),
        dryRun = !real, builtins = StandardPlugins.withOwnAxes ++ Seq(script, marker),
        launchOverride = Some(() => {
          val n = starts.incrementAndGet()
          require(mode != "offline", "Offline browser may not start inference")
          require(n == 1, "No retry/duplicate inference worker is permitted by this helper")
          EngineLaunch(command, engineDir, Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0", "HF_HUB_OFFLINE" -> "1", "TRANSFORMERS_OFFLINE" -> "1"))
        }), persistSettings = true, timeouts = EngineTimeouts(load = 5.minutes, generationInactivity = 15.minutes), installShutdownHook = false)
      listen()
      val ready = new CountDownLatch(1)
      Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
      require(ready.await(45, TimeUnit.SECONDS)); startedFx = true
      one = fx { val w = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => sys.error("Clipboard forbidden")); w.show(); w }
      windows += one
      two = fx { val w = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => sys.error("Clipboard forbidden")); w.show(); w }
      windows += two

    def nativeCore(): Unit = check("native cards, shared Show/Blur/Hide and three close/reopen cycles") {
      sampleMemory("before-browser")
      for cycle <- 1 to 3 do
        open(one, "core-card"); open(two, "core-card")
        require((reflected[ModelCardCatalog](reflected[ModelBrowser](one, "browser"), "catalog") eq services.modelCards) &&
          (reflected[ModelCardCatalog](reflected[ModelBrowser](two, "browser"), "catalog") eq services.modelCards))
        if cycle > 1 then check(s"cycle-$cycle reopen preserves shared Hide before any mode action") {
          require(fx {
            services.cardPreviews.current == CardPreviewMode.Hide && services.settings.nsfwCardBlurDefault == "Blur" &&
              Seq(one, two).forall(window => menu(window).getText.endsWith(CardPreviewMode.Hide.toString))
          }, "Reopen changed temporary Hide, shared labels or saved Blur")
          require(bothPreviewsMatch(CardPreviewMode.Hide), "Reopen attached hidden pixels in a realized window")
        }
        choose(CardPreviewMode.Show)
        eventually("Show attaches original preview in both windows") { bothPreviewsMatch(CardPreviewMode.Show) }
        require(fx { Seq(one, two).forall(window => pixels(window).getUserData == preview.toUri.toASCIIString) }, "Chosen literal source URI differs")
        if cycle == 1 then screenshot("core-show")
        choose(CardPreviewMode.Blur)
        eventually("Blur conceals preview in both windows") { bothPreviewsMatch(CardPreviewMode.Blur) }
        if cycle == 1 then screenshot("core-blur")
        sampleMemory(s"cycle-$cycle-open")
        choose(CardPreviewMode.Hide)
        require(bothPreviewsMatch(CardPreviewMode.Hide), "Hide did not immediately detach pixels in both windows")
        require(fx { descendants(stage(one).getScene.getRoot).exists { case b: Button => b.getText == "Choose preview…" && b.isVisible; case _ => false } })
        if cycle == 1 then screenshot("core-hide")
        closeBrowsers()
        eventually("closed browsers release owned previews") {
          val m = PreviewImageLoader.metrics
          m.activeWorkers == 0 && m.admittedBuffers == 0 && m.ownedImages == 0 && m.visibleConsumers == 0 && m.mailboxItems == 0
        }
        sampleMemory(s"cycle-$cycle-closed")
        require(services.cardPreviews.current == CardPreviewMode.Hide && services.settings.nsfwCardBlurDefault == "Blur")
      require(starts.get() == 0 && services.supervisor.pid.isEmpty && services.queue.isIdle)
      val closed = memory.filter(_("phase").str.endsWith("-closed"))
      def ws(s: ujson.Value): Long = s("observed")("processes").arr.find(_("pid").num.toLong == ProcessHandle.current().pid()).get("workingSet").num.toLong
      def gpu(s: ujson.Value): Long = s("observed")("gpuDevices").arr.map(_("usedMiB").num.toLong).sum
      require(ws(closed.last) - ws(closed.head) <= 1024L * MiB, "Three browser cycles grew helper working set by over 1 GiB")
      require(gpu(closed.last) - gpu(closed.head) <= 512L, "Three browser cycles grew device-total VRAM by over 512 MiB")
    }
    def loadModel(): Unit = check("one actual worker and native checkpoint admission") {
      try Await.result(services.startAsync(), 5.minutes)
      finally observeWorker()
      require(worker.nonEmpty, "Started worker has no observed actual process handle")
      pythonVersion = services.supervisor.hello.get.python
      require(pythonVersion.nonEmpty && services.supervisor.hello.get.dryRun == !real)
      if real then require(services.supervisor.hello.get.cuda)
      val id = CardId("checkpoint", model.toString)
      require(Await.result(fx { one.admitCardCheckpoint(id) }, 45.seconds).isRight)
      eventually("checkpoint loaded and native admission released", 5.minutes) {
        services.supervisor.loadedModel.exists(_.path == model.toString) && fx { one.cardCheckpointAdmission(id).allowed }
      }
      modelKind = services.supervisor.loadedModel.get.kind
      require(starts.get() == 1)
    }
    def cpuButtons(): Unit = check("CPU actual native Generate and Queue") {
      fx {
        one.select(Mode.Txt2Img)
        val vm = one.vms(Mode.Txt2Img)
        vm.prompt.set("native core card preview control"); vm.seed.set(12345); vm.subseed.set(99)
        vm.steps.set(2); vm.width.set(256); vm.height.set(256); vm.cfgScale.set(5)
        vm.sampler.set("Euler"); vm.scheduler.set("Uniform"); vm.variationEnabled.set(false)
        one.plugins(Mode.Txt2Img).selectedScript.set(script.id)
      }
      def submit(queued: Boolean): String =
        val before = eventList.collect { case e: JobEvent.JobQueued => e.job.id }.toSet
        fx { if queued then one.statusBar.queueBtn.fire() else one.statusBar.generateBtn.fire() }
        eventually("native button admitted job") { eventList.exists { case e: JobEvent.JobQueued => !before(e.job.id); case _ => false } }
        val ids = eventList.collect { case e: JobEvent.JobQueued if !before(e.job.id) => e.job.id }
        require(ids.size == 1); ids.head
      script.arm()
      try
        val first = submit(false); require(script.entered.await(45, TimeUnit.SECONDS))
        val second = submit(true)
        require(services.queue.current.exists(_.id == first) && services.queue.pending.map(_.id) == Seq(second))
        script.release.countDown()
        val a = result("native-generate", first); val b = result("native-queue", second)
        require(a._1 == b._1 && a._2 == b._2)
        eventually("native queue idle") { services.queue.isIdle }
      finally script.release.countDown()
    }
    def modelCore(): Unit = check("fixed-seed control, view actions with running/waiting Jobs, exact body/RGB") {
      val control = fixtureJob(); services.queue.submit(control)
      val expected = result("model-control", control.id)
      val realPreview = sources.resolve("NSFW-model-control.png")
      Files.copy(expected._3, realPreview, StandardCopyOption.REPLACE_EXISTING)
      remember(realPreview, s"$mode Job ${control.id}, model $modelHash, body ${textHash(expected._1)}, RGB ${expected._2}")
      val settings = new ModelBrowserSettings(services)
      require(settings.saveAssociation(CardId("checkpoint", model.toString), realPreview.toString).isRight)
      open(one, modelName); open(two, modelName); choose(CardPreviewMode.Show)
      eventually("real model preview visible in both windows") { bothPreviewsMatch(CardPreviewMode.Show) }
      screenshot("actual-model-show")
      val galleryBefore = services.gallery.scan().map(i => i.path -> sha(i.path)).toMap
      script.arm()
      try
        val running = fixtureJob(); val waiting = fixtureJob()
        services.queue.submit(running); require(script.entered.await(45, TimeUnit.SECONDS))
        services.queue.submit(waiting)
        eventually("post-submit evidence for both Jobs") { frozen.containsKey(running.id) && frozen.containsKey(waiting.id) }
        val before = activeSnapshot("before-view-actions")
        require(before == ((running.id, Vector(waiting.id))))
        require(!fx { one.cardCheckpointAdmission(CardId("checkpoint", model.toString)).allowed })
        for value <- Seq(CardPreviewMode.Blur, CardPreviewMode.Hide, CardPreviewMode.Show) do
          choose(value)
          value match
            case CardPreviewMode.Hide => require(bothPreviewsMatch(value), "Actual model Hide left pixels attached in a window")
            case _ => eventually(s"actual model preview $value in both windows") { bothPreviewsMatch(value) }
          require(activeSnapshot("after-" + value.toString) == before)
          screenshot("actual-model-" + value.toString.toLowerCase)
        require(settings.saveDefault("Hide").isRight); require(activeSnapshot("after-default") == before)
        require(settings.saveAssociation(CardId("checkpoint", model.toString), preview.toString).isRight)
        require(activeSnapshot("after-association") == before)
        require(services.gallery.scan().map(i => i.path -> sha(i.path)).toMap == galleryBefore, "Viewing changed generated gallery")
        script.release.countDown()
        val a = result("model-running", running.id); val b = result("model-waiting", waiting.id)
        require(a._1 == expected._1 && b._1 == expected._1 && a._2 == expected._2 && b._2 == expected._2,
          "Display actions changed actual model request or RGB")
        require(eventList.collect { case e: JobEvent.JobStarted if Set(running.id, waiting.id)(e.job.id) => e.job.id } == Vector(running.id, waiting.id))
        eventually("terminal queue actually idle") { services.queue.isIdle }
        require(starts.get() == 1 && worker.exists(_.isAlive))
        require(fx { one.galleryVM.items.size() >= 3 }, "Actual gallery lost generated output")
        sampleMemory("after-model-generation")
      finally script.release.countDown()
    }
    def finish(): Unit =
      script.release.countDown()
      cleanup("observe worker before cleanup") { observeWorker() }
      cleanup("source invariance") { sourceInvariant() }
      windows.reverse.foreach(w => cleanup("window close") { fx { w.persist(); w.dispose(); w.stage.close() } })
      cleanup("queue listener") { removeEvents() }
      cleanup("observe worker before supervisor shutdown") { observeWorker() }
      cleanup("services shutdown") { if services != null then services.shutdown() }
      cleanup("actual inference worker termination") {
        require(starts.get() == 0 || worker.nonEmpty,
          "Worker launch attempted but no actual handle observed; termination is unknown/unverified, no retry authorized")
        worker.foreach { h =>
          val end = System.nanoTime() + 45.seconds.toNanos
          while h.isAlive && System.nanoTime() < end do Thread.sleep(50)
          require(!h.isAlive, "Worker remains alive; no duplicate/retry authorized by timeout")
        }
      }
      cleanup("preview owner shutdown") {
        PreviewImageLoader.shutdownProcess()
        val end = System.nanoTime() + 45.seconds.toNanos
        def released: Boolean =
          val m = PreviewImageLoader.metrics
          m.activeWorkers == 0 && m.admittedBuffers == 0 && m.ownedImages == 0 && m.visibleConsumers == 0 && m.mailboxItems == 0 && m.queued == 0
        while !released && System.nanoTime() < end do Thread.sleep(20)
        require(released, "Preview worker/ownership cleanup incomplete")
      }
      cleanup("post-cleanup RAM/VRAM observation") { sampleMemory("after-cleanup") }
      if startedFx then cleanup("JavaFX exit") { Platform.exit() }
      if !listenerErrors.isEmpty then failures ++= listenerErrors.iterator().asScala
      val report = ujson.Obj("status" -> (if failures.isEmpty && cleanupErrors.isEmpty then "passed" else "failed"),
        "scope" -> "core_user_reduced", "testedCommit" -> gitHead(root), "sourcePin" -> "b2cdcebf97d1074278fb84e0a6eec3006dab0fe1", "mode" -> mode,
        "java" -> ujson.Obj("home" -> System.getProperty("java.home"), "version" -> System.getProperty("java.version")),
        "python" -> ujson.Obj("path" -> python.toString, "version" -> pythonVersion, "engineStarts" -> starts.get()),
        "model" -> ujson.Obj("path" -> Option(model).fold("")(_.toString), "sha256" -> modelHash, "observedKind" -> modelKind, "syntheticMetadata" -> !real),
        "checks" -> ujson.Arr.from(checks), "runs" -> ujson.Arr.from(runs), "screenshots" -> ujson.Arr.from(screenshots),
        "sourceHashes" -> ujson.Arr.from(hashes.map((p, hash, provenance) => ujson.Obj("path" -> p.toString, "sha256" -> hash, "provenance" -> provenance))),
        "orderedQueueEvents" -> ujson.Arr.from(eventList.map(e => ujson.Obj("type" -> e.getClass.getSimpleName, "jobId" -> e.jobId))),
        "memory" -> ujson.Obj("samples" -> ujson.Arr.from(memory), "sampling" -> "coarse baseline/three open-close cycles/generation/end; not continuous peak sampling",
          "trendGate" -> "closed cycle 3 minus cycle 1: helper working set <= +1 GiB; device-total VRAM <= +512 MiB; coarse runaway check only",
          "workerTerminated" -> worker.fold[ujson.Value](ujson.Null)(h => ujson.Bool(!h.isAlive)),
          "workerObservation" -> ujson.Obj(
            "supervisorPid" -> observedWorkerPid.fold[ujson.Value](ujson.Null)(pid => ujson.Num(pid.toDouble)),
            "handleObserved" -> worker.nonEmpty,
            "status" -> (worker match
              case Some(h) => if h.isAlive then "still_alive" else "termination_observed"
              case None => if starts.get() == 0 then "no_worker_launched" else "termination_unknown_unverified"))), "elapsedMs" -> elapsed,
        "cleanupErrors" -> ujson.Arr.from(cleanupErrors), "failures" -> ujson.Arr.from(failures),
        "unmeasured" -> ujson.Arr("per-allocation GPU alloc/free", "WDDM per-process GPU attribution", "sub-sample peaks", "codec/native texture allocation separation"),
        "notExecutedByUserScope" -> ujson.Arr("exhaustive themes/DPI/keyboard matrix", "32MP/metadata/configuration stress", "full hardware/heap/250ms threshold proof", "adapter/hires/all-mode inference matrix"),
        "limits" -> ujson.Arr("Native GaussianBlur is an adaptation, not CSS pixel equality", "Device-total VRAM can include unrelated processes",
          "Three fixed-seed generations per real model cover the core queue/view path, not all inference features",
          "Timeout is observation failure, not proof of termination or restart permission"))
      val rendered = ujson.write(report, indent = 2)
      Files.writeString(output.resolve("card-preview-report.json"), rendered.dropRight(1) + ",\n\"jobComparisons\":[" + jobRaw.iterator().asScala.mkString(",") + "]\n}")
      require(failures.isEmpty && cleanupErrors.isEmpty, (failures ++ cleanupErrors).mkString("Core acceptance failed: ", "; ", ""))
    def execute(): Unit =
      try
        check("isolated core setup") { setup() }
        nativeCore()
        if mode != "offline" then
          loadModel()
          if mode == "cpu" then cpuButtons()
          modelCore()
          check("one successful terminal per core Job") {
            val expected = if mode == "cpu" then 5 else 3
            val queued = eventList.collect { case e: JobEvent.JobQueued => e.job.id }
            val finished = eventList.collect { case e: JobEvent.JobFinished => e.jobId }
            require(queued.size == expected && finished.size == expected && queued.toSet == finished.toSet && queued.distinct.size == expected)
            require(script.calls.get() == expected)
          }
        else check("offline zero starts") { require(starts.get() == 0 && services.supervisor.pid.isEmpty && events.isEmpty) }
      catch case e: Throwable => if failures.isEmpty then failures += e.toString
      finally finish()

  def main(args: Array[String]): Unit = new Run(args).execute()
