package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.gallery.OutputNaming
import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.*
import sdfx.core.plugins.api.*
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.*

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import scala.concurrent.duration.*
import scala.concurrent.{Await, Future}
import scala.util.Random
import scala.util.control.NonFatal

/** Receives everything a single job produces. All callbacks run on the job thread or the engine reader thread. */
trait GenerationListener:
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

/** A generation request violates the job's [[JobLimits]]. */
final case class JobLimitExceeded(message: String) extends IllegalArgumentException(message)

/** What [[Generator.prepare]] produces: the params recorded in infotext and the request sent to the engine. */
final case class PreparedGeneration(recorded: GenerationParams, request: GenerationRequest, mode: String, warnings: Seq[String])

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
    timeouts: EngineTimeouts = EngineTimeouts()
) extends PluginContext:

  private val styleSource: () => Seq[PromptStyle] = Option(styles).getOrElse(() => Styles.load(home.stylesFile))

  def outputsDir: Path = home.outputs(settings())

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
          throw e
    val scopedListener: GenerationListener = new GenerationListener:
      override def onProgress(p: EngineEvent.Progress): Unit = listener.onProgress(p)
      override def onPreview(p: EngineEvent.Preview): Unit = listener.onPreview(p)
      override def onLog(m: String): Unit = listener.onLog(m)
      override def onStatus(t: String, d: Int, n: Int): Unit = listener.onStatus(t, d, n)
      override def onImageSaved(i: OutputImage, g: Boolean): Unit = listener.onImageSaved(i, g)
      override def onTempFile(path: Path): Unit = listener.onTempFile(path)
      override def isInterrupted: Boolean = listener.isInterrupted || engineInterrupted.get()
    new PluginContext:
      def generate(p: GenerationParams): Seq[OutputImage] = guard(self.generateWith(limited(p), scopedListener))
      override def generateTransient(p: GenerationParams): Seq[OutputImage] =
        guard(self.generateTransientWith(limited(p), scopedListener))
      def upscale(image: Path, upscaler: String, scale: Double): Path = guard {
        if isInterrupted then throw interruptedError
        val out = self.upscaleImage(image, upscaler, scale)
        listener.onTempFile(out)
        out
      }
      def saveImage(img: BufferedImage, kind: String, params: GenerationParams, infotext: String): OutputImage =
        self.saveWith(img, kind, params, infotext, listener)
      def tempFile(suffix: String): Path =
        val t = self.tempFile(suffix)
        listener.onTempFile(t)
        t
      def choices(source: ChoiceSource): Seq[String] = self.choices(source)
      override def modelPath(source: ChoiceSource, name: String): Option[String] = self.modelPath(source, name)
      override def engineCall(kind: String, params: ujson.Obj): ujson.Value = guard {
        if isInterrupted then throw interruptedError
        self.engineCallWith(kind, params, scopedListener)
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
    val (prompt, negative) =
      if p0.styles.isEmpty then (p0.prompt, p0.negativePrompt)
      else Styles.apply(styleSource(), p0.styles, p0.prompt, p0.negativePrompt)
    val recorded = p0.copy(prompt = prompt, negativePrompt = negative, styles = Nil, seed = seed, subseed = subseed)

    val ex = LoraSyntax.extract(prompt)
    val hrEx = LoraSyntax.extract(recorded.hires.prompt)
    val allLoras = (recorded.loras ++ ex.loras ++ hrEx.loras.filterNot(l => ex.loras.exists(_.name == l.name)))
    val warnings = Seq.newBuilder[String]
    val loraPaths = allLoras.map(_.name).distinct.flatMap { n =>
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
    val upPaths = Seq(Option.when(recorded.hires.enabled)(recorded.hires.upscaler)).flatten
      .flatMap(u => catalog.findUpscaler(u).map(e => u -> e.path))
      .toMap
    val s = settings()
    val sent = recorded.copy(
      prompt = ex.prompt,
      hires = recorded.hires.copy(prompt = hrEx.prompt),
      loras = allLoras.filter(l => loraPaths.contains(l.name)),
      hooks = resolveHookModels(recorded.hooks, msg => warnings += msg)
    )
    val req = GenerationRequest(
      params = sent,
      outputDir = home.tmpDir(s).toAbsolutePath.toString,
      loraPaths = loraPaths,
      embeddingPaths = catalog.embeddings.map(_.path),
      controlnetPaths = cnPaths,
      upscalerPaths = upPaths,
      previewEvery = s.previewEvery,
      previewMethod = s.previewMethod
    )
    PreparedGeneration(recorded, req, if recorded.isImg2Img then GenerationMode.Img2Img else GenerationMode.Txt2Img, warnings.result())

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
        val m = await(client.loadModel(LoadModelRequest(path, s.dtype, s.vramMode, None, s.vaeTiling)), timeouts.load, "load_model")
        catalog.rememberHash(m)
    }
    p.vae.map(_.trim).filter(_.nonEmpty).foreach { v =>
      val loaded = client.supervisor.loadedModel
      if v == "Automatic" || v == "None" then
        if loaded.exists(_.vaeName.isDefined) then await(client.setVae(None), timeouts.load, "set_vae")
      else
        val e = catalog.findVae(v).getOrElse(throw EngineError(Protocol.ErrorCodes.InvalidParams, s"VAE not found: $v"))
        def stem(x: String) = Paths.get(x.replace('\\', '/')).getFileName.toString.replaceFirst("\\.[^.]+$", "")
        if !loaded.flatMap(_.vaeName).exists(n => n == e.name || stem(n) == stem(e.name) || stem(n) == stem(e.path)) then
          listener.onLog(s"loading VAE ${e.name}")
          await(client.setVae(Some(e.path)), timeouts.load, "set_vae")
    }

  /** Runs one generation and saves every image under outputs/<txt2img|img2img>/<date>/. */
  def generateWith(p: GenerationParams, listener: GenerationListener): Seq[OutputImage] =
    val (prep, result) = runEngine(p, listener)
    saveResult(prep, result, listener)

  /** Same pipeline as [[generateWith]] (events forwarded, model ensured) but the PNGs stay in outputs/.tmp:
    * no output naming, no `onImageSaved`. The returned paths are engine temp files the caller must delete.
    * `infotext` is filled in but not embedded in the file.
    */
  def generateTransientWith(p: GenerationParams, listener: GenerationListener): Seq[OutputImage] =
    val (prep, result) = runEngine(p, listener)
    val loaded = client.supervisor.loadedModel
    result.images.map { img =>
      val (params, text) = describe(prep, result, img, loaded)
      OutputImage(Paths.get(img.path), params, text, img.seed)
    }

  override def generateTransient(p: GenerationParams): Seq[OutputImage] = generateTransientWith(p, GenerationListener.none)

  private def describe(prep: PreparedGeneration, result: GenerationResult, img: GeneratedImage, loaded: Option[LoadedModel]): (GenerationParams, String) =
    val params = prep.recorded.copy(seed = img.seed, subseed = if img.subseed >= 0 then img.subseed else prep.recorded.subseed)
    val facts = ImageFacts(
      seed = img.seed,
      subseed = params.subseed,
      modelName = result.modelName.orElse(loaded.map(_.name)),
      modelHash = result.modelHash.orElse(loaded.map(_.hash).filter(_.nonEmpty)).map(h => h.take(10)),
      vaeName = result.vaeName.orElse(loaded.flatMap(_.vaeName)),
      vaeHash = result.vaeHash.orElse(loaded.flatMap(_.vaeHash)).map(_.take(10)),
      loraHashes = result.loraHashes.toSeq.sortBy(_._1),
      scheduler = result.scheduler
    )
    (params, Infotext.write(params, facts))

  private def runEngine(p: GenerationParams, listener: GenerationListener): (PreparedGeneration, GenerationResult) =
    if listener.isInterrupted then throw interruptedError
    val prep = prepare(p)
    prep.warnings.foreach(listener.onLog)
    ensureModel(prep.recorded, listener)
    if listener.isInterrupted then throw interruptedError
    val activity = new java.util.concurrent.atomic.AtomicLong(System.nanoTime())
    val handle = client.generate(prep.request, prep.mode, { ev =>
      activity.set(System.nanoTime())
      ev match
        case e: EngineEvent.Progress => listener.onProgress(e)
        case e: EngineEvent.Preview  => listener.onPreview(e)
        case EngineEvent.Log(_, m)   => listener.onLog(m)
    })
    val result = awaitActive(handle.result, activity)
    result.images.foreach(i => listener.onTempFile(Paths.get(i.path)))
    result.warnings.foreach(listener.onLog)
    (prep, result)

  private def saveResult(prep: PreparedGeneration, result: GenerationResult, listener: GenerationListener): Seq[OutputImage] =
    val loaded = client.supervisor.loadedModel
    val kind = if prep.mode == GenerationMode.Img2Img then "img2img" else "txt2img"
    val outputs = outputsDir
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
      try PngMetadata.copyWithParameters(src, dst, text)
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
    val r = await(client.raw(kind, p, {
      case e: EngineEvent.Progress => listener.onProgress(e)
      case e: EngineEvent.Preview  => listener.onPreview(e)
      case EngineEvent.Log(_, m)   => listener.onLog(m)
    }), timeouts.upscale, kind)
    r match
      case o: ujson.Obj =>
        Seq("path", "preview").flatMap(k => o.value.get(k)).collect { case ujson.Str(s) => s }.foreach(s => listener.onTempFile(Paths.get(s)))
        o.value.get("images").collect { case a: ujson.Arr => a.value.collect { case ujson.Str(s) => s } }
          .getOrElse(Nil).foreach(s => listener.onTempFile(Paths.get(s)))
      case _ => ()
    r

  def upscaleImage(image: Path, upscaler: String, scale: Double): Path =
    val s = settings()
    val paths = catalog.findUpscaler(upscaler).map(e => upscaler -> e.path).toMap
    val r = await(client.upscale(UpscaleRequest(image.toAbsolutePath.toString, upscaler, scale, home.tmpDir(s).toAbsolutePath.toString, paths)), timeouts.upscale, "upscale")
    Paths.get(r.path)

  def preprocess(image: Path, preprocessor: String, args: Map[String, Double] = Map.empty): Path =
    val s = settings()
    Paths.get(await(client.preprocess(PreprocessRequest(preprocessor, image.toAbsolutePath.toString, home.tmpDir(s).toAbsolutePath.toString, args)), timeouts.upscale, "preprocess").path)

  /** Saves an image to outputs/<kind>/<date>/ (`grid`/`grids` -> outputs/grids). */
  def saveWith(img: BufferedImage, kind: String, params: GenerationParams, infotext: String, listener: GenerationListener): OutputImage =
    val k = if kind == "grid" then "grids" else kind
    val dst = OutputNaming.nextFile(outputsDir, k, params.seed, if k.contains("grid") then "-grid" else "")
    try PngMetadata.write(img, dst, Seq(PngMetadata.ParametersKey -> infotext))
    catch
      case NonFatal(e) =>
        Files.deleteIfExists(dst)
        throw e
    val out = OutputImage(dst, params, infotext, params.seed)
    listener.onImageSaved(out, grid = k.contains("grid"))
    out
