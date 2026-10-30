package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.gallery.OutputNaming
import sdfx.core.infotext.{HardwareInfo, ImageFacts, Infotext, Img2ImgHiresMetadata}
import sdfx.core.model.*
import sdfx.core.plugins.api.*
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.*
import sdfx.core.payload.*

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import scala.concurrent.duration.*
import scala.concurrent.{Await, Future}
import scala.util.Random
import scala.util.control.NonFatal

/** Receives everything a single job produces. All callbacks run on the job thread or the engine reader thread. */
trait GenerationListener:
  /** Backend interruption is authoritative even before its admission reply arrives. */
  private[engine] def onEngineInterrupted(): Unit = ()
  /** Private transport identity for queued work; never part of GenerationParams or image metadata. */
  def interruptionOwner: Option[String] = None
  private[engine] def imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None
  private[engine] def wdBatchControl: Option[ExtrasBatchControl] = None
  private[engine] def payloadJobId: Option[String] = None
  private[engine] def submittedPayloadContext: Option[SubmittedPayloadContext] = None
  private[engine] def payloadCaptureOptions: Option[CaptureOptions] = None
  private[engine] def nextPayloadKey(): PayloadKey = PayloadKey(s"direct-${java.util.UUID.randomUUID()}", 1)
  def onPreparedRequest(snapshot: PayloadSnapshot): Unit = ()
  def onPayloadLifecycle(key: PayloadKey, state: PayloadState, completionJson: Option[String]): Unit = ()
  def onProgress(p: EngineEvent.Progress): Unit = ()
  def onPreview(p: EngineEvent.Preview): Unit = ()
  def onLog(message: String): Unit = ()
  def onStatus(text: String, done: Int, total: Int): Unit = ()
  def onImageSaved(image: OutputImage, grid: Boolean): Unit = ()
  /** A temp file was created on behalf of the job (engine output, transient generation, upscale, tempFile).
    * JobQueue deletes these when the job ends, except files that are part of the job's output.
    */
  def onTempFile(path: Path): Unit = ()
  def isInterrupted: Boolean = false

object GenerationListener:
  val none: GenerationListener = new GenerationListener {}

/** Finite upper bounds for blocking engine calls. A generation fails (and the engine is restarted) when no
  * progress/preview/log event arrives for `generationInactivity`.
  */
final case class EngineTimeouts(
    load: FiniteDuration = 30.minutes,
    generationInactivity: FiniteDuration = 10.minutes,
    upscale: FiniteDuration = 30.minutes
)

/** Resource limits enforced by the job-scoped context for every generation of a job (script sub-generations
  * included), checked before anything is sent to the engine.
  */
final case class JobLimits(
    maxSide: Int = 8192,
    maxImagesPerGeneration: Int = 64,
    maxGenerations: Int = 10000,
    maxPixelsPerGeneration: Long = 64L * 1024 * 1024
):
  /** Throws [[JobLimitExceeded]] when `p` (as the `n`-th generation of the job, 1-based) breaks a limit. */
  def check(p: GenerationParams, n: Int): Unit =
    def fail(msg: String) = throw JobLimitExceeded(msg)
    if n > maxGenerations then fail(s"job exceeds the limit of $maxGenerations generations")
    if p.width > maxSide || p.height > maxSide then fail(s"size ${p.width}x${p.height} exceeds the maximum side $maxSide")
    val images = p.batchSize.toLong * p.nIter
    if images > maxImagesPerGeneration then fail(s"batch size x count = $images exceeds the limit of $maxImagesPerGeneration images")
    val px = p.width.toLong * p.height * p.batchSize
    if px > maxPixelsPerGeneration then fail(s"${p.width}x${p.height} x batch ${p.batchSize} = $px pixels exceeds the limit of $maxPixelsPerGeneration")
    if p.hires.enabled && !p.isImg2Img then
      val (hw, hh) = p.hires.targetSize(p.width, p.height)
      if hw > maxSide || hh > maxSide then fail(s"hires target ${hw}x$hh exceeds the maximum side $maxSide")
      val hpx = hw.toLong * hh * p.batchSize
      if hpx > maxPixelsPerGeneration then fail(s"hires ${hw}x$hh x batch ${p.batchSize} = $hpx pixels exceeds the limit of $maxPixelsPerGeneration")
    Img2ImgHiresMetadata.active(p.hooks).foreach { config =>
      val source = if p.hires.enabled && !p.isImg2Img then p.hires.targetSize(p.width, p.height) else (p.width, p.height)
      val (rw, rh) = Img2ImgHiresMetadata.targetSize(config, source._1, source._2)
      if rw > maxSide || rh > maxSide then fail(s"img2img Hires target ${rw}x$rh exceeds the maximum side $maxSide")
      // Refinement runs one completed image at a time.
      if rw.toLong * rh > maxPixelsPerGeneration then fail(s"img2img Hires ${rw}x$rh exceeds the pixel limit $maxPixelsPerGeneration")
    }

/** A generation request violates the job's [[JobLimits]]. */
final case class JobLimitExceeded(message: String) extends IllegalArgumentException(message)

/** What [[Generator.prepare]] produces: the params recorded in infotext and the request sent to the engine. */
final case class PreparedGeneration(recorded: GenerationParams, request: GenerationRequest, mode: String, warnings: Seq[String],
    finalOutputDir: Option[Path] = None)

/** The single-generation pipeline (seed resolution, styles, LoRA syntax, model switching, engine call, PNG saving).
  * The Generator itself is a [[PluginContext]] without job scope; [[forJob]] gives a job-scoped context whose
  * `generate` also applies the job's AlwaysOn plugins.
  */
final class Generator(
    client: EngineClient,
    catalog: ModelCatalog,
    home: AppHome,
    settings: () => Settings,
    styles: () => Seq[PromptStyle] = null,
    random: Random = new Random(),
    timeouts: EngineTimeouts = EngineTimeouts(),
    val payloads: RequestPayloadStore = new RequestPayloadStore()
) extends PluginContext:

  private val styleSource: () => Seq[PromptStyle] = Option(styles).getOrElse(() => Styles.load(home.stylesFile))

  def outputsDir: Path = home.outputs(settings())
  private[engine] def defaultCaptureOptions: CaptureOptions = CaptureOptions(settings().apiPayloadIncludeImageContents)

  // ---- PluginContext (unscoped) ---------------------------------------------------------------------------
  def generate(p: GenerationParams): Seq[OutputImage] = generateWith(p, GenerationListener.none)
  def upscale(image: Path, upscaler: String, scale: Double): Path = upscaleImage(image, upscaler, scale)
  def saveImage(img: BufferedImage, kind: String, params: GenerationParams, infotext: String): OutputImage =
    saveWith(img, kind, params, infotext, GenerationListener.none)
  def tempFile(suffix: String): Path = Files.createTempFile(home.tmpDir(settings()), "sdfx-", suffix)
  def choices(source: ChoiceSource): Seq[String] = catalog.choices(source)
  override def modelPath(source: ChoiceSource, name: String): Option[String] = catalog.pathOf(source, name)
  override def engineCall(kind: String, params: ujson.Obj): ujson.Value = engineCallWith(kind, params, GenerationListener.none)
  def log(message: String): Unit = System.err.println(s"[sdfx] $message")
  def status(text: String, done: Int, total: Int): Unit = ()
  def isInterrupted: Boolean = false

  /** Non-queue tool context bound to a snapshot; secrets are never added to request metadata. */
  def forImageSession(secret: Option[sdfx.core.png.ImageScrambleSecret], warn: String => Unit = _ => ()): PluginContext =
    forJob(new GenerationListener {
      override private[engine] def imageSecret = secret
      override def onLog(message: String): Unit = warn(message)
    })

  /** A context bound to one job: events go to `listener`; every `generate` applies the active AlwaysOn plugins. */
  def forJob(listener: GenerationListener, alwaysOn: Seq[(Plugin, Values)] = Nil, limits: JobLimits = JobLimits()): PluginContext =
    val self = this
    val generations = new java.util.concurrent.atomic.AtomicInteger(0)
    def limited(p: GenerationParams): GenerationParams =
      val q = applyAlwaysOn(p, alwaysOn)
      limits.check(q, generations.incrementAndGet())
      q
    // Set when an engine call ends with an `interrupted` error, so scripts that catch the error see
    // `isInterrupted == true` and can assemble partial output (grids) before rethrowing.
    val engineInterrupted = new java.util.concurrent.atomic.AtomicBoolean(false)
    def guard[T](body: => T): T =
      try body
      catch
        case e: EngineError if e.isInterrupted =>
          engineInterrupted.set(true)
          listener.onEngineInterrupted()
          throw e
    val scopedListener: GenerationListener = new GenerationListener:
      override def interruptionOwner: Option[String] = listener.interruptionOwner
      override private[engine] def imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = listener.imageSecret
      override private[engine] def payloadJobId: Option[String] = listener.payloadJobId
      override private[engine] def submittedPayloadContext: Option[SubmittedPayloadContext] = listener.submittedPayloadContext
      override private[engine] def payloadCaptureOptions: Option[CaptureOptions] = listener.payloadCaptureOptions
      override private[engine] def nextPayloadKey(): PayloadKey = listener.nextPayloadKey()
      override def onPreparedRequest(snapshot: PayloadSnapshot): Unit = listener.onPreparedRequest(snapshot)
      override def onPayloadLifecycle(key: PayloadKey, state: PayloadState, completionJson: Option[String]): Unit =
        listener.onPayloadLifecycle(key, state, completionJson)
      override def onProgress(p: EngineEvent.Progress): Unit = listener.onProgress(p)
      override def onPreview(p: EngineEvent.Preview): Unit = listener.onPreview(p)
      override def onLog(m: String): Unit = listener.onLog(m)
      override def onStatus(t: String, d: Int, n: Int): Unit = listener.onStatus(t, d, n)
      override def onImageSaved(i: OutputImage, g: Boolean): Unit = listener.onImageSaved(i, g)
      override def onTempFile(path: Path): Unit = listener.onTempFile(path)
      override def isInterrupted: Boolean = listener.isInterrupted || engineInterrupted.get()
    new PluginContext:
      override def imageScramblingEnabled: Boolean = listener.imageSecret.nonEmpty
      override def plaintextVideoAllowed: Boolean = listener.imageSecret.forall(_.plaintextVideoAllowed)
      override def resourceHashesEnabled: Boolean = alwaysOn.exists((plugin, values) =>
        plugin.id == "pnginfo_injection" && plugin.isActive(values))
      def generate(p: GenerationParams): Seq[OutputImage] = guard(self.generateWith(limited(p), scopedListener))
      override def generateTransient(p: GenerationParams): Seq[OutputImage] =
        guard(self.generateTransientWith(limited(p), scopedListener))
      def upscale(image: Path, upscaler: String, scale: Double): Path = guard {
        if isInterrupted then throw interruptedError
        val out = self.upscaleImage(image, upscaler, scale, listener.interruptionOwner, listener.imageSecret, listener.onLog)
        listener.onTempFile(out)
        out
      }
      def saveImage(img: BufferedImage, kind: String, params: GenerationParams, infotext: String): OutputImage =
        self.saveWith(img, kind, params, infotext, listener)
      override def readImage(path: Path): BufferedImage =
        val decoded = sdfx.core.png.ScrambledPng.read(path, listener.imageSecret.map(_.use(identity)))
        if decoded.scrambled && decoded.passwordVerified.isEmpty then listener.onLog(ScrambledInput.LegacyWarning)
        decoded.image
      def tempFile(suffix: String): Path =
        val t = self.tempFile(suffix)
        listener.onTempFile(t)
        t
      def choices(source: ChoiceSource): Seq[String] = self.choices(source)
      override def modelPath(source: ChoiceSource, name: String): Option[String] = self.modelPath(source, name)
      override def engineCall(kind: String, params: ujson.Obj): ujson.Value = guard {
        if isInterrupted then throw interruptedError
        if kind == "img2vid" then requirePlaintextVideoAllowed()
        val fields = kind match
          case "caption" | "restore_faces" | "novel_view" | "img2vid" => Seq("image_path")
          case "image_tool" => Seq("image_path", "reference_path")
          case _ => Nil
        ScrambledInput.withObject(params, fields, home.tmpDir(settings()), listener.imageSecret, listener.onLog) { resolved =>
          self.engineCallWith(kind, resolved, scopedListener)
        }
      }
      def log(message: String): Unit = listener.onLog(message)
      def status(text: String, done: Int, total: Int): Unit = listener.onStatus(text, done, total)
      def isInterrupted: Boolean = scopedListener.isInterrupted

  def applyAlwaysOn(p: GenerationParams, alwaysOn: Seq[(Plugin, Values)]): GenerationParams =
    alwaysOn.foldLeft(p) { case (acc, (pl, v)) =>
      if pl.kind == PluginKind.AlwaysOn && pl.isActive(v) then pl.process(acc, v) else acc
    }

  private def interruptedError = EngineError(Protocol.ErrorCodes.Interrupted, "interrupted")

  // ---- pipeline ---------------------------------------------------------------------------------------------

  /** A1111 style random seed (0 until 2^32-1). */
  def randomSeed(): Long = (random.nextLong() & 0xffffffffL) % 4294967294L

  /** Seeds, styles, LoRA extraction and path resolution; no engine calls. */
  /** Hooks whose `model` field names a file from a model list; the engine gets the resolved path (infotext keeps
    * the name).
    */
  private val hookModelSources: Map[String, ChoiceSource] = Map(
    "ip_adapter" -> ChoiceSource.ControlNets, "controllllite" -> ChoiceSource.ControlNets,
    "fooocus_inpaint" -> ChoiceSource.ControlNets, "face_restore" -> ChoiceSource.FaceRestorers,
    "photomaker" -> ChoiceSource.PhotoMakerModels)

  def resolveHookModels(hooks: Map[String, ujson.Value], warn: String => Unit): Map[String, ujson.Value] =
    hooks.map {
      case (k, o: ujson.Obj) if hookModelSources.contains(k) =>
        o.value.get("model") match
          case Some(ujson.Str(name)) if name.nonEmpty && name != "None" =>
            catalog.pathOf(hookModelSources(k), name) match
              case Some(path) => k -> ujson.Obj.from(o.value.toSeq.map((f, v) => f -> (if f == "model" then ujson.Str(path) else v)))
              case None =>
                warn(s"$k model not found: $name")
                k -> o
          case _ => k -> o
      case other => other
    }

  def prepare(p0: GenerationParams): PreparedGeneration =
    val problems = p0.validate
    if problems.nonEmpty then throw EngineError(Protocol.ErrorCodes.InvalidParams, problems.mkString("; "))
    val seed = if p0.seed < 0 then randomSeed() else p0.seed
    val subseed = if p0.subseed < 0 then randomSeed() else p0.subseed
    val composed = HiresPreparation.compose(p0, styleSource(), settings().hiresFixTweaks.saveTemplate)
    val prompt = composed.prompt
    val negative = composed.negativePrompt
    val refinement = Img2ImgHiresMetadata.active(composed.hooks)
    val recorded = composed.copy(seed = seed, subseed = subseed,
      extra = p0.extra.filterNot((key, _) => HardwareInfo.fieldNames(key) ||
        Set("Hires prompt mode", "NUI Hires template", "NUI Manual LoRAs", Img2ImgHiresMetadata.key)(key)) ++
        refinement.map(h => Img2ImgHiresMetadata.key -> Img2ImgHiresMetadata.encode(h)))

    val ex = LoraSyntax.extract(prompt)
    val hrEx = LoraSyntax.extract(recorded.hires.prompt)
    val refinePrompt = refinement.map(_("prompt").str).filter(_.nonEmpty).getOrElse(prompt)
    val refineEx = LoraSyntax.extract(refinePrompt)
    val composable = recorded.hooks.get("composable_lora").exists {
      case o: ujson.Obj => o.value.get("enabled").forall {
        case ujson.Bool(value) => value
        case ujson.Num(value) => value != 0
        case ujson.Str(value) => !Set("disable", "disabled", "false", "off", "no", "0", "")(value.trim.toLowerCase(java.util.Locale.ROOT))
        case _ => true
      }
      case ujson.Bool(value) => value
      case ujson.Str(value) => Set("enable", "enabled", "true", "on", "yes", "1")(value.trim.toLowerCase(java.util.Locale.ROOT))
      case _ => false
    }
    val allLoras =
      if composable then
        // Placement and repeated tag weights belong to the composable engine. Resolve
        // each file once; retain its first native TE/UNet settings instead of summing tags.
        val tags = """<(?:lora|lyco):[^>]+>""".r.findAllIn(prompt + " " + recorded.hires.prompt +
          refinement.fold("")(h => " " + h("prompt").str + " " + h("negative_prompt").str))
          .flatMap(t => LoraSyntax.extract(t).loras).toSeq
        (recorded.loras ++ tags).foldLeft(Vector.empty[LoraRef])((found, lr) =>
          if found.exists(_.name == lr.name) then found else found :+ lr)
      else recorded.loras ++ ex.loras ++ hrEx.loras.filterNot(l => ex.loras.exists(_.name == l.name))
    val separateHiresLoras = HiresPreparation.active(recorded).isDefined && !composable
    val warnings = Seq.newBuilder[String]
    val refinementLoras = refinement.map(_ => recorded.loras ++ refineEx.loras)
    val loraPaths = (allLoras ++ refinementLoras.getOrElse(Nil)).map(_.name).distinct.flatMap { n =>
      catalog.findLora(n).map(e => n -> e.path).orElse {
        val f = home.resolve(n)
        if Files.isRegularFile(f) && withinDirs(f, settings().modelDirs.lora) then Some(n -> f.toString)
        else
          warnings += s"LoRA not found: $n"
          None
      }
    }.toMap
    val cnPaths = recorded.controlnet.filter(u => u.enabled && u.model.nonEmpty && u.model != "None").flatMap { u =>
      catalog.findControlNet(u.model).map(e => u.model -> e.path).orElse {
        warnings += s"ControlNet model not found: ${u.model}"
        None
      }
    }.toMap
    val upPaths = Seq(Option.when(recorded.hires.enabled)(recorded.hires.upscaler), refinement.map(_("upscaler").str)).flatten
      .flatMap(u => catalog.findUpscaler(u).map(e => u -> e.path))
      .toMap
    val s = settings()
    val resolvedHooks = resolveHookModels(recorded.hooks, msg => warnings += msg).map {
      case ("img2img_hires_fix", _) if refinement.isDefined =>
        val config = ujson.Obj.from(refinement.get.value.toSeq)
        if !composable && config("prompt").str.nonEmpty then config("prompt") = refineEx.prompt
        "img2img_hires_fix" -> config
      case (id @ ("save_intermediate" | "composable_lora"), o: ujson.Obj) =>
        val persistent = ujson.Obj.from(o.value.toSeq)
        persistent("output_dir") = home.outputs(s).toAbsolutePath.toString
        id -> persistent
      case other => other
    }
    val sent = recorded.copy(
      prompt = if composable then prompt else ex.prompt,
      hires = recorded.hires.copy(prompt = if composable then recorded.hires.prompt else hrEx.prompt),
      loras = (if separateHiresLoras then recorded.loras ++ ex.loras else allLoras).filter(l => loraPaths.contains(l.name)),
      hooks = resolvedHooks
    )
    val req = GenerationRequest(
      params = sent,
      outputDir = home.tmpDir(s).toAbsolutePath.toString,
      loraPaths = loraPaths,
      manualLoras = Option.when(composable)(recorded.loras.filter(l => loraPaths.contains(l.name))),
      hiresLoras = Option.when(separateHiresLoras)((recorded.loras ++ hrEx.loras).filter(l => loraPaths.contains(l.name))),
      img2imgHiresLoras = refinementLoras.map(_.filter(l => loraPaths.contains(l.name))),
      embeddingPaths = catalog.embeddings.map(_.path),
      controlnetPaths = cnPaths,
      upscalerPaths = upPaths,
      previewEvery = s.previewEvery,
      previewMethod = s.previewMethod,
      systemInfoFields = s.systemInfoMetadataFields,
      hardwareInfo = HardwareInfo.Options(s.hardwareInfoEnabled, s.hardwareInfoRemark),
      vaeRecoveryDir = home.outputs(s).toAbsolutePath.toString,
      lbwPresets = if (sent.loras ++ Option.when(separateHiresLoras)(recorded.loras ++ hrEx.loras).getOrElse(Nil) ++ refinementLoras.getOrElse(Nil))
        .exists(_.lbw.nonEmpty) then Generator.lbwPresets(home.configDir) else Map.empty
    )
    val globalOutputs = home.outputs(s).toAbsolutePath.normalize()
    val defaultOutputs = home.outputs(Settings()).toAbsolutePath.normalize()
    val hiresDestination = HiresPreparation.active(recorded).flatMap(_.value.get("output_dir"))
      .collect { case ujson.Str(path) if path.trim.nonEmpty => path.trim }
      .orElse(Option.when(recorded.hires.enabled && !recorded.isImg2Img)(s.hiresFixTweaks.outputDir.trim).filter(_.nonEmpty))
    // Native outputDir always has a default; a non-default global location takes
    // precedence, matching upstream's explicit outdir_samples override.
    val finalOutputs = if globalOutputs != defaultOutputs then globalOutputs else
      hiresDestination.map(path => home.resolve(path).toAbsolutePath.normalize()).getOrElse(globalOutputs)
    PreparedGeneration(recorded, req, if recorded.isImg2Img then GenerationMode.Img2Img else GenerationMode.Txt2Img,
      warnings.result(), Some(finalOutputs))

  private def await[T](f: Future[T], timeout: FiniteDuration, what: String): T =
    try Await.result(f, timeout)
    catch
      case _: java.util.concurrent.TimeoutException =>
        val err = EngineError(EngineProcess.EngineTimeout, s"$what did not finish within $timeout")
        client.supervisor.forceRestart(err)
        throw err

  /** Waits for a generation, failing (and restarting the engine) after `timeouts.generationInactivity` without events. */
  private def awaitActive[T](f: Future[T], lastActivity: java.util.concurrent.atomic.AtomicLong): T =
    var res: Option[T] = None
    while res.isEmpty do
      try res = Some(Await.result(f, 1.second))
      catch
        case _: java.util.concurrent.TimeoutException =>
          val idle = (System.nanoTime() - lastActivity.get()).nanos
          if idle > timeouts.generationInactivity then
            val err = EngineError(EngineProcess.EngineTimeout, s"no progress from the engine for ${timeouts.generationInactivity}")
            client.supervisor.forceRestart(err)
            throw err
    res.get

  /** True when `p` lies inside one of the configured model directories (resolved against the app home). */
  def withinDirs(p: Path, dirs: Seq[String]): Boolean =
    val n = p.toAbsolutePath.normalize()
    dirs.exists(d => n.startsWith(home.resolve(d).toAbsolutePath.normalize()))

  private def samePath(a: String, b: String): Boolean =
    try Paths.get(a).toAbsolutePath.normalize() == Paths.get(b).toAbsolutePath.normalize()
    catch case NonFatal(_) => a == b

  /** Deletes leftover files in outputs/.tmp older than `olderThan` (call at startup). Files of another process
    * sharing the same app home (GUI + headless MCP) are younger and survive. Returns the count.
    */
  def cleanTempDir(olderThan: FiniteDuration = 6.hours): Int =
    val cutoff = System.currentTimeMillis() - olderThan.toMillis
    val dir = home.tmpDir(settings())
    var n = 0
    try
      val st = Files.list(dir)
      try
        st.iterator().forEachRemaining { f =>
          try
            if Files.isRegularFile(f) && Files.getLastModifiedTime(f).toMillis < cutoff && Files.deleteIfExists(f) then n += 1
          catch case NonFatal(_) => ()
        }
      finally st.close()
    catch case NonFatal(_) => ()
    n

  /** Loads the checkpoint / VAE requested by `p` when they differ from what is loaded. */
  def ensureModel(p: GenerationParams, listener: GenerationListener = GenerationListener.none): Unit =
    val s = settings()
    p.checkpoint.map(_.trim).filter(n => n.nonEmpty && n != "None").foreach { name =>
      val path = catalog.findCheckpoint(name).map(_.path)
        .orElse(Some(home.resolve(name)).filter(f => Files.isRegularFile(f) && withinDirs(f, s.modelDirs.checkpoints)).map(_.toString))
        .getOrElse(throw EngineError(Protocol.ErrorCodes.InvalidParams, s"checkpoint not found: $name"))
      if !client.supervisor.loadedModel.exists(m => samePath(m.path, path)) then
        listener.onLog(s"loading checkpoint $name")
        val m = await(client.loadModelOwned(LoadModelRequest(path, s.dtype, s.vramMode, None, s.vaeTiling), listener.interruptionOwner), timeouts.load, "load_model")
        catalog.rememberHash(m)
    }
    p.vae.map(_.trim).filter(_.nonEmpty).foreach { v =>
      val loaded = client.supervisor.loadedModel
      if v == "Automatic" || v == "None" then
        if loaded.exists(_.vaeName.isDefined) then await(client.setVaeOwned(None, listener.interruptionOwner), timeouts.load, "set_vae")
      else
        val e = catalog.findVae(v).getOrElse(throw EngineError(Protocol.ErrorCodes.InvalidParams, s"VAE not found: $v"))
        def stem(x: String) = Paths.get(x.replace('\\', '/')).getFileName.toString.replaceFirst("\\.[^.]+$", "")
        if !loaded.flatMap(_.vaeName).exists(n => n == e.name || stem(n) == stem(e.name) || stem(n) == stem(e.path)) then
          listener.onLog(s"loading VAE ${e.name}")
          await(client.setVaeOwned(Some(e.path), listener.interruptionOwner), timeouts.load, "set_vae")
    }

  /** Runs one generation and saves every image under outputs/<txt2img|img2img>/<date>/. */
  private[engine] def runWdBatch(p: GenerationParams, spec: WdBatchSpec, listener: GenerationListener, limits: JobLimits): JobOutput =
    if listener.isInterrupted then throw interruptedError
    limits.check(p, 1)
    WdBatchSafety.validateResolved(p)
    ensureModel(p, listener)
    if listener.isInterrupted then throw interruptedError
    val token = client.supervisor.loadedModel.flatMap(_.modelToken).filter(_.nonEmpty)
      .getOrElse(throw EngineError(Protocol.ErrorCodes.Unsupported, "WD batch requires a loaded checkpoint identity"))
    val owner = listener.interruptionOwner.getOrElse(throw new IllegalStateException("WD batch requires queue ownership"))
    val control = listener.wdBatchControl.getOrElse(throw new IllegalStateException("WD batch cancellation gate unavailable"))
    val prepared = prepare(p)
    WdBatchSafety.validateResolved(prepared.request.params)
    if LoraSyntax.extract(prepared.recorded.prompt).loras.nonEmpty then
      throw EngineError(Protocol.ErrorCodes.Unsupported, "WD batch does not support prompt LoRA tags")
    prepared.warnings.foreach(listener.onLog)
    val body = WdBatchPrepareRequest(spec.imagesRoot, prepared.request.copy(vaeRecoveryDir = ""),
      spec.outputParent, spec.outputName, token)
    listener.onStatus("WD v3 batch: preparing images", 0, 0)
    val activity = new java.util.concurrent.atomic.AtomicLong(System.nanoTime())
    val handle = client.wdBatchOwned(body, owner, control, { event =>
      activity.set(System.nanoTime())
      event match
        case value: EngineEvent.Progress => listener.onProgress(value)
        case value: EngineEvent.Preview => listener.onPreview(value)
        case value: EngineEvent.Log => listener.onLog(value.message)
    }, listener.imageSecret)
    val result = WdBatchSafety.awaitActive(handle.result, activity, timeouts.generationInactivity, () => {
      WdBatchSafety.timeoutAction(handle, control, client.supervisor)
    })
    val root = Paths.get(spec.outputParent).resolve(spec.outputName).toAbsolutePath.normalize()
    val validated = WdBatchSafety.validatePaths(result, root)
    val output = validated.map { (row, image, path) =>
      val text = PngMetadata.readParameters(path).getOrElse(
        throw EngineError(Protocol.ErrorCodes.Internal, "WD batch PNG is missing persisted generation parameters"))
      val base = prepared.recorded.copy(img2img = prepared.recorded.img2img.map(_.copy(initImage = row.input)))
      val effective = Infotext.toParams(Infotext.parse(text), base)
      OutputImage(path, effective, text, image.seed)
    }
    result.rows.filter(_.status == "error").foreach(row => listener.onLog(s"WD batch failed for ${row.input}: ${row.error.get}"))
    output.foreach(saved => listener.onImageSaved(saved, grid = false))
    JobOutput(output)

  def generateWith(p: GenerationParams, listener: GenerationListener): Seq[OutputImage] =
    val (prep, result) = runEngine(p, listener)
    val saved = saveResult(prep, result, listener)
    if result.interrupted then
      listener.onEngineInterrupted()
      throw interruptedError
    saved

  /** Same pipeline as [[generateWith]] (events forwarded, model ensured) but the PNGs stay in outputs/.tmp:
    * no output naming, no `onImageSaved`. The returned paths are engine temp files the caller must delete.
    * `infotext` is filled in but not embedded in the file.
    */
  def generateTransientWith(p: GenerationParams, listener: GenerationListener): Seq[OutputImage] =
    val (prep, result) = runEngine(p, listener)
    if result.interrupted then
      // A caller cannot receive transient paths through an exception. Preserve
      // completed images through the normal gallery path before reporting cancellation.
      saveResult(prep, result, listener)
      listener.onEngineInterrupted()
      throw interruptedError
    val loaded = client.supervisor.loadedModel
    result.images.map { img =>
      val (params, text) = describe(prep, result, img, loaded)
      OutputImage(Paths.get(img.path), params, text, img.seed)
    }

  override def generateTransient(p: GenerationParams): Seq[OutputImage] = generateTransientWith(p, GenerationListener.none)

  private def describe(prep: PreparedGeneration, result: GenerationResult, img: GeneratedImage, loaded: Option[LoadedModel]): (GenerationParams, String) =
    val params = prep.recorded.copy(seed = img.seed, subseed = if img.subseed >= 0 then img.subseed else prep.recorded.subseed)
    val hiresFields = img.hiresSeed.toSeq.map { seed =>
      val info = ujson.Obj("Seed" -> sdfx.core.infotext.HiresSeedMetadata.encode(seed))
      img.hiresSubseed.foreach(s => info("Subseed") = sdfx.core.infotext.HiresSeedMetadata.encode(s))
      img.hiresSubseedStrength.foreach(s => info("Strength") = ujson.Num(s))
      for
        w <- img.hiresSeedResizeFromW.filter(_ > 0)
        h <- img.hiresSeedResizeFromH.filter(_ > 0)
      do info("Resize") = ujson.Arr(w, h)
      // Upstream swaps single/double quotes so its JSON survives infotext CSV parsing.
      "Hires seed" -> info.render().map { case '"' => '\''; case '\'' => '"'; case c => c }
    }
    val templateFields = HiresPreparation.active(params).toSeq.flatMap(_.value.get("prompt_template")).flatMap { t =>
      Seq("NUI Hires template" -> t.render()) ++
        t.obj.get("mode").map(_.str).filter(_.nonEmpty).map("Hires prompt mode" -> _)
    }
    val resourceFields = Option.when(params.hooks.get("pnginfo_injection").exists(o =>
      !o.obj.get("enabled").contains(ujson.Bool(false)))) {
      "Hashes" -> sdfx.core.infotext.ResourceHashes.merge(params.extra.get("Hashes"), result.resourceHashes.toSeq)
    }.toSeq
    val facts = ImageFacts(
      seed = img.seed,
      subseed = params.subseed,
      modelName = result.modelName.orElse(loaded.map(_.name)),
      modelHash = result.modelHash.orElse(loaded.map(_.hash).filter(_.nonEmpty)).map(h => h.take(10)),
      vaeName = result.vaeName.orElse(loaded.flatMap(_.vaeName)),
      vaeHash = result.vaeHash.orElse(loaded.flatMap(_.vaeHash)).map(_.take(10)),
      loraHashes = result.loraHashes.toSeq.sortBy(_._1),
      scheduler = result.scheduler,
      extra = scala.collection.immutable.ListMap.from(result.systemInfo.toSeq.map(v => "System Info" -> v.render()) ++
        Option.when(result.vaeRecovery.nonEmpty)("VAE recovery" -> ujson.Arr.from(result.vaeRecovery).render()) ++ resourceFields ++ hiresFields ++ templateFields) ++ HardwareInfo.fields(img.hardwareInfo)
    )
    (params, Infotext.write(params, facts))

  private def runEngine(p: GenerationParams, listener: GenerationListener): (PreparedGeneration, GenerationResult) =
    if listener.isInterrupted then throw interruptedError
    val original = prepare(p)
    ScrambledInput.withParams(original.request.params, home.tmpDir(settings()), listener.imageSecret, listener.onLog) { params =>
      runEnginePrepared(p, listener, original.copy(request = original.request.copy(params = params)))
    }

  private def runEnginePrepared(p: GenerationParams, listener: GenerationListener, prep: PreparedGeneration): (PreparedGeneration, GenerationResult) =
    prep.warnings.foreach(listener.onLog)
    ensureModel(prep.recorded, listener)
    if listener.isInterrupted then throw interruptedError
    val activity = new java.util.concurrent.atomic.AtomicLong(System.nanoTime())
    val jsonLimits = PayloadJsonBudget.Limits(maxBytes = math.min(payloads.limits.maxEntryBytes, Int.MaxValue.toLong - 1).toInt)
    var captureKey: Option[PayloadKey] = None
    def diagnostic(action: => Unit): Unit =
      try action
      catch
        case error: InterruptedException => System.err.println(s"[sdfx] Payload diagnostic failed: ${error.getClass.getSimpleName}")
        case NonFatal(error) => System.err.println(s"[sdfx] Payload diagnostic failed: ${error.getClass.getSimpleName}")
    def lifecycle(state: PayloadState, completion: Option[String] = None): Unit = captureKey.foreach { key =>
      diagnostic(payloads.transition(key, state, completion))
      diagnostic(listener.onPayloadLifecycle(key, state, completion))
    }
    val result = try
      val handle = client.generateOwned(prep.request, prep.mode, { ev =>
        activity.set(System.nanoTime())
        ev match
          case e: EngineEvent.Progress => listener.onProgress(e)
          case e: EngineEvent.Preview  => listener.onPreview(e)
          case EngineEvent.Log(_, m)   => listener.onLog(m)
      }, listener.interruptionOwner, body => {
        val key = listener.nextPayloadKey()
        captureKey = Some(key)
        val submitted = listener.submittedPayloadContext.getOrElse(SubmittedPayloadContext("direct"))
        val model = client.supervisor.loadedModel
        val options = listener.payloadCaptureOptions.getOrElse(defaultCaptureOptions)
        var warnings = Vector.empty[String]
        val context = try
          PayloadCaptureContext.captured(submitted, p, body.capturedAtMillis, model, jsonLimits,
            prep.finalOutputDir.map(_.toString)) match
            case Right(json) => Some(json)
            case Left(reason) =>
              warnings :+= s"Submitted context unavailable: $reason"
              None
          catch
            case NonFatal(_) =>
              warnings :+= "Submitted context unavailable: invalid diagnostic JSON"
              None
        val assets =
          if RequestPayloadStore.utf8Size(body.bodyJson, payloads.limits.maxBodyBytes) > payloads.limits.maxBodyBytes then
            warnings :+= "Image diagnostic unavailable: exact body exceeds capture byte limit"
            None
          else
            try
              val result = PayloadAssets.collect(body.bodyJson, options.includeImageContents && listener.imageSecret.isEmpty)
              warnings ++= result.warnings
              Some(result.assetsJson)
            catch
              case NonFatal(_) =>
                warnings :+= "Image diagnostic unavailable: input inspection failed"
                None
        val snapshot = PayloadSnapshot(key, body.requestType, Some(body.bodyJson), body.capturedAtMillis,
          context, assets, PayloadState.Prepared, warnings, submitted.sourceMode)
        payloads.put(snapshot)
        payloads.snapshots.find(_.key == key).foreach(s => diagnostic(listener.onPreparedRequest(s)))
      }, () => if listener.isInterrupted then throw interruptedError, listener.imageSecret)
      lifecycle(PayloadState.Submitted)
      val completed = awaitActive(handle.result, activity)
      val completion = PayloadCaptureContext.completion(completed, jsonLimits) match
        case Right(json) => json
        case Left(reason) =>
          captureKey.foreach(key => diagnostic(payloads.warn(key, s"Completion context unavailable: $reason")))
          None
      lifecycle(if completed.interrupted then PayloadState.Cancelled else PayloadState.Succeeded, completion)
      completed
    catch
      case error: InterruptedException => lifecycle(PayloadState.Cancelled); throw error
      case NonFatal(error) =>
        lifecycle(if JobRunner.isInterruption(error) then PayloadState.Cancelled else PayloadState.Failed)
        throw error
    result.images.foreach(i => listener.onTempFile(Paths.get(i.path)))
    result.warnings.foreach(listener.onLog)
    result.vaeRecovery.foreach(a => listener.onLog(s"VAE decode recovery: ${a("latent_path").str}"))
    result.composableLoraTrace.foreach(t => listener.onLog(s"Composable LoRA trace: ${t.path}"))
    result.intermediateArtifacts.filter(a => a.`type` == "manifest" || a.`type` == "video")
      .foreach(a => listener.onLog(s"Intermediate ${a.`type`}: ${a.path}"))
    (prep, result)

  private def saveResult(prep: PreparedGeneration, result: GenerationResult, listener: GenerationListener): Seq[OutputImage] =
    val loaded = client.supervisor.loadedModel
    val kind = if prep.mode == GenerationMode.Img2Img then "img2img" else "txt2img"
    val outputs = prep.finalOutputDir.getOrElse(outputsDir)
    try saveAll(prep, result, listener, loaded, kind, outputs)
    finally
      // Engine temp files are never needed after this point, whether saving succeeded or not.
      result.images.foreach(img =>
        try Files.deleteIfExists(Paths.get(img.path))
        catch case NonFatal(_) => ()
      )

  private def saveAll(prep: PreparedGeneration, result: GenerationResult, listener: GenerationListener, loaded: Option[LoadedModel],
      kind: String, outputs: Path): Seq[OutputImage] =
    result.images.map { img =>
      val (params, text) = describe(prep, result, img, loaded)
      val src = Paths.get(img.path)
      val dst = OutputNaming.nextFile(outputs, kind, img.seed)
      try
        val pixels = PngMetadata.readImage(src)
        sdfx.core.png.ScrambledPng.write(pixels, dst, Seq(PngMetadata.ParametersKey -> text), listener.imageSecret.map(_.use(identity)))
      catch
        case NonFatal(e) =>
          Files.deleteIfExists(dst)
          throw e
      try Files.deleteIfExists(src)
      catch case NonFatal(_) => ()
      val out = OutputImage(dst, params, text, img.seed)
      listener.onImageSaved(out, grid = false)
      out
    }

  /** Non-generation engine request. Fills `output_dir` with the temp dir and registers returned image paths as job
    * temp files (callers save what they keep). Progress events are forwarded to the listener.
    */
  def engineCallWith(kind: String, params: ujson.Obj, listener: GenerationListener): ujson.Value =
    val p = ujson.Obj.from(params.value)
    if !p.value.contains("output_dir") then p("output_dir") = home.tmpDir(settings()).toAbsolutePath.toString
    val r = await(client.rawOwned(kind, p, listener.interruptionOwner, {
      case e: EngineEvent.Progress => listener.onProgress(e)
      case e: EngineEvent.Preview  => listener.onPreview(e)
      case EngineEvent.Log(_, m)   => listener.onLog(m)
    }), timeouts.upscale, kind)
    r match
      case o: ujson.Obj =>
        Seq("path", "preview").flatMap(k => o.value.get(k)).collect { case ujson.Str(s) => s }.foreach(s => listener.onTempFile(Paths.get(s)))
        o.value.get("images").collect { case a: ujson.Arr => a.value.collect { case ujson.Str(s) => s } }
          .getOrElse(Nil).foreach(s => listener.onTempFile(Paths.get(s)))
        o.value.get("outputs").collect { case m: ujson.Obj => m.value.values.collect { case ujson.Str(s) => s } }
          .getOrElse(Nil).foreach(s => listener.onTempFile(Paths.get(s)))
      case _ => ()
    r

  def upscaleImage(image: Path, upscaler: String, scale: Double, owner: Option[String] = None,
      secret: Option[sdfx.core.png.ImageScrambleSecret] = None, warn: String => Unit = _ => ()): Path =
    ScrambledInput.withPlain(image, home.tmpDir(settings()), secret, warn) { plain =>
      protectToolOutput(upscalePlain(plain, upscaler, scale, owner), plain, secret)
    }

  /** Worker tools create a plaintext temporary first; do not leave it readable after success. */
  private def protectToolOutput(output: Path, input: Path, secret: Option[sdfx.core.png.ImageScrambleSecret]): Path =
    secret.foreach { value =>
      require(!Files.isSameFile(input, output), "Tool output cannot overwrite its input")
      try
        val pixels = PngMetadata.readImage(output)
        val texts = PngMetadata.readTexts(output).toSeq
        value.use(pass => sdfx.core.png.ScrambledPng.write(pixels, output, texts, Some(pass)))
      catch
        case NonFatal(error) =>
          try Files.deleteIfExists(output) catch case NonFatal(cleanup) => error.addSuppressed(cleanup)
          throw error
    }
    output

  private def upscalePlain(image: Path, upscaler: String, scale: Double, owner: Option[String]): Path =
    val s = settings()
    val paths = catalog.findUpscaler(upscaler).map(e => upscaler -> e.path).toMap
    val req = UpscaleRequest(image.toAbsolutePath.toString, upscaler, scale, home.tmpDir(s).toAbsolutePath.toString, paths,
      HardwareInfo.Options(s.hardwareInfoEnabled, s.hardwareInfoRemark))
    val r = Json.fromJs[ImagePathResult](await(client.rawOwned(Protocol.Types.Upscale, Json.toJs(req), owner), timeouts.upscale, "upscale"))
    val output = Paths.get(r.path)
    r.hardwareInfo.foreach { record =>
      require(!Files.isSameFile(image, output), "Upscale metadata cannot overwrite the original input image")
      val pixels = PngMetadata.readImage(output)
      val existing = PngMetadata.readTexts(output)
      val processing = scala.collection.immutable.ListMap("Postprocess upscaler" -> upscaler,
        "Postprocess upscale by" -> scale.toString, "Size" -> s"${pixels.getWidth}x${pixels.getHeight}") ++ HardwareInfo.fields(Some(record))
      val extra = existing.toSeq.filterNot(_._1 == PngMetadata.ParametersKey) ++
        existing.get(PngMetadata.ParametersKey).filter(_.nonEmpty).filter(_ => !existing.contains("Source parameters"))
          .map("Source parameters" -> _)
      PngMetadata.write(pixels, output, (PngMetadata.ParametersKey -> Infotext.render("", "", processing)) +: extra)
    }
    output

  def preprocess(image: Path, preprocessor: String, args: Map[String, Double] = Map.empty,
      secret: Option[sdfx.core.png.ImageScrambleSecret] = None, warn: String => Unit = _ => ()): Path =
    val s = settings()
    ScrambledInput.withPlain(image, home.tmpDir(s), secret, warn) { plain =>
      val output = Paths.get(await(client.preprocess(PreprocessRequest(preprocessor, plain.toAbsolutePath.toString, home.tmpDir(s).toAbsolutePath.toString, args)), timeouts.upscale, "preprocess").path)
      protectToolOutput(output, plain, secret)
    }

  /** Saves an image to outputs/<kind>/<date>/ (`grid`/`grids` -> outputs/grids). */
  def saveWith(img: BufferedImage, kind: String, params: GenerationParams, infotext: String, listener: GenerationListener): OutputImage =
    val k = if kind == "grid" then "grids" else kind
    val dst = OutputNaming.nextFile(outputsDir, k, params.seed, if k.contains("grid") then "-grid" else "")
    try sdfx.core.png.ScrambledPng.write(img, dst, Seq(PngMetadata.ParametersKey -> infotext), listener.imageSecret.map(_.use(identity)))
    catch
      case NonFatal(e) =>
        Files.deleteIfExists(dst)
        throw e
    val out = OutputImage(dst, params, infotext, params.seed)
    listener.onImageSaved(out, grid = k.contains("grid"))
    out

object Generator:
  /** LoRA Block Weight user presets, `config/lbwpresets.txt`: one `NAME:w1,w2,...` per line (`#` comments). */
  def lbwPresets(configDir: java.nio.file.Path): Map[String, String] =
    val f = configDir.resolve("lbwpresets.txt")
    if !java.nio.file.Files.isRegularFile(f) then Map.empty
    else
      import scala.jdk.CollectionConverters.*
      java.nio.file.Files.readAllLines(f, java.nio.charset.StandardCharsets.UTF_8).asScala.toSeq
        .map(_.trim).filter(l => l.nonEmpty && !l.startsWith("#") && l.contains(':'))
        .map { l =>
          val i = l.indexOf(':')
          l.take(i).trim -> l.drop(i + 1).trim
        }.filter(_._1.nonEmpty).toMap
