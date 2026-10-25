package sdfx.mcp

import sdfx.core.model.*
import sdfx.core.plugins.api.*
import sdfx.core.protocol.*
import sdfx.engine.*

import java.nio.file.{Files, Path}
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap
import scala.concurrent.duration.*
import scala.concurrent.{Await, Future, Promise, TimeoutException}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal
import scala.util.{Failure, Success}

/** The MCP tool set over [[AppServices]]. All generations go through `services.queue` with origin "mcp".
  *
  * @param awaitReady blocks up to the given time until the engine can take work (TimeoutException if not yet)
  * @param allowedDirs folders besides the outputs folder whose images may be named by path
  */
final class McpTools(
    services: AppServices,
    awaitReady: FiniteDuration => Unit,
    readyTimeout: FiniteDuration = 10.minutes,
    allowedDirs: Seq[Path] = PathPolicy.fromEnv(),
    limits: JobLimits = McpTools.StrictLimits
):
  import McpTools.*

  private given scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global

  def this(services: AppServices) = this(services, McpTools.waitForEngine(services))

  // ---- helpers ---------------------------------------------------------------------------------------------

  /** Waits for the engine in short slices so a cancelled call stops waiting. */
  private def ready(ctx: ToolContext = ToolContext.none): Unit =
    val end = System.nanoTime() + readyTimeout.toNanos
    var done = false
    while !done do
      if ctx.isCancelled then throw ToolError("cancelled")
      val left = end - System.nanoTime()
      if left <= 0 then throw ToolError(s"the engine did not become ready within $readyTimeout")
      try
        awaitReady(math.min(left, 500.millis.toNanos).nanos)
        done = true
      catch
        case _: TimeoutException => ()
        case e: ToolError        => throw e
        case NonFatal(e)         => throw ToolError(s"the engine is not available: ${msg(e)}")

  private def engineUsable: Boolean = services.supervisor.state.isUsable

  private def catalog = services.catalog
  private def outputsDir: Path = services.generator.outputsDir

  /** Image path inputs: outputs folder (incl. the engine tmp dir below it) plus the configured extra folders. */
  val pathPolicy: PathPolicy = new PathPolicy(() => Seq(outputsDir, services.home.tmpDir(services.settings)) ++ allowedDirs)

  private def tempFile(suffix: String): Path = services.generator.tempFile(suffix)

  /** Refreshes an empty catalog once the engine is up (never triggers / waits for an engine start). */
  private def ensureCatalog(): Unit =
    if engineUsable && catalog.checkpoints.isEmpty && catalog.samplers.samplers.isEmpty then
      try catalog.refresh()
      catch case NonFatal(_) => ()

  private def modelKind(checkpoint: Option[String]): String =
    checkpoint.flatMap(catalog.findCheckpoint).map(_.kind)
      .orElse(services.supervisor.loadedModel.map(_.kind))
      .getOrElse("")
      .toLowerCase

  private def defaultSide(checkpoint: Option[String]): Int =
    val k = modelKind(checkpoint)
    if k.contains("xl") || k.contains("sd3") || k.contains("flux") then 1024 else 512

  private def matchName(kind: String, value: String, options: Seq[String]): String =
    if options.isEmpty then value
    else
      options.find(_ == value).orElse(options.find(_.equalsIgnoreCase(value))).getOrElse(
        throw ToolError(s"unknown $kind '$value'. Available: ${options.mkString(", ")}")
      )

  // ---- generation params -------------------------------------------------------------------------------------

  /** Parsed generation request plus output options. Temp files are deleted after the job. */
  private final case class GenRequest(params: GenerationParams, returnImages: Boolean, maxSide: Int, temps: Seq[Path])

  private def parseGen(a: Args, img2img: Boolean, requirePrompt: Boolean = true): GenRequest =
    val temps = Seq.newBuilder[Path]
    try
      ensureCatalog()
      val prompt = if requirePrompt then a.reqStr("prompt") else a.str("prompt").getOrElse("")
      val checkpoint = a.str("checkpoint").map(_.trim).filter(_.nonEmpty)
      checkpoint.foreach { c =>
        if catalog.checkpoints.nonEmpty && catalog.findCheckpoint(c).isEmpty then
          throw ToolError(s"checkpoint not found: $c. Available: ${catalog.checkpoints.map(_.name).mkString(", ")}")
      }
      val vae = a.str("vae").map(_.trim).filter(_.nonEmpty)
      vae.filterNot(v => v == "Automatic" || v == "None").foreach { v =>
        if catalog.findVae(v).isEmpty then
          throw ToolError(s"VAE not found: $v. Available: Automatic, None${catalog.vaes.map(", " + _.name).mkString}")
      }
      val samplerNames = catalog.samplers.samplers.filter(_.supported).map(_.name)
      val sampler = a.str("sampler_name").map(s => matchName("sampler", s, samplerNames)).getOrElse(
        if samplerNames.isEmpty || samplerNames.contains("DPM++ 2M") then "DPM++ 2M" else samplerNames.head
      )
      val schedulers = catalog.samplers.schedulers
      val scheduler = a.str("scheduler").map(s =>
        if s.equalsIgnoreCase("Automatic") then "Automatic" else matchName("scheduler", s, schedulers)
      ).getOrElse("Automatic")
      val batch = a.int("batch_size", 1, MaxImages).getOrElse(1)
      val nIter = a.int("n_iter", 1, MaxImages).getOrElse(1)
      if batch * nIter > MaxImages then throw ToolError(s"batch_size * n_iter must be at most $MaxImages (got ${batch * nIter})")

      val loraTags = a.arr("loras").getOrElse(Nil).map {
        case o: ujson.Obj =>
          val la = Args(o, Seq("name", "weight"))
          val n = la.reqStr("name")
          val w = la.num("weight", -10, 10).getOrElse(1.0)
          val e = catalog.findLora(n).getOrElse(
            throw ToolError(s"LoRA not found: $n. Available: ${if catalog.loras.isEmpty then "(none)" else catalog.loras.map(_.name).mkString(", ")}")
          )
          LoraSyntax.tag(e.name, w)
        case ujson.Str(n) =>
          val e = catalog.findLora(n).getOrElse(throw ToolError(s"LoRA not found: $n"))
          LoraSyntax.tag(e.name)
        case _ => throw ToolError("loras must be an array of {name, weight} objects")
      }
      val fullPrompt =
        if loraTags.isEmpty then prompt
        else if prompt.trim.isEmpty then loraTags.mkString(", ")
        else s"$prompt, ${loraTags.mkString(", ")}"

      val hires = a.objOpt("hires").map { o =>
        val h = Args(o, HiresKeys)
        val up = h.str("upscaler").map(u => matchName("hires upscaler", u, catalog.choices(ChoiceSource.HiresUpscalers))).getOrElse("Latent")
        HiresParams(
          enabled = h.bool("enabled").getOrElse(true),
          scale = h.num("scale", 1.0, 4.0).getOrElse(2.0),
          upscaler = up,
          secondPassSteps = h.int("steps", 0, 150).getOrElse(0),
          denoisingStrength = h.num("denoising_strength", 0, 1).getOrElse(0.55),
          resizeX = h.int("resize_x", 0, MaxSide).getOrElse(0),
          resizeY = h.int("resize_y", 0, MaxSide).getOrElse(0)
        )
      }.getOrElse(HiresParams())

      val base = outputsDir.toAbsolutePath.normalize()
      val i2i = Option.when(img2img) {
        val (init, t1) = Images.materialize(a.reqStr("init_image"), "init_image", pathPolicy, base, tempFile)
        t1.foreach(temps += _)
        val mask = a.str("mask").filter(_.trim.nonEmpty).map { m =>
          val (mp, t2) = Images.materialize(m, "mask", pathPolicy, base, tempFile)
          t2.foreach(temps += _)
          mp
        }
        Img2ImgParams(
          initImage = init.toString,
          mask = mask.map(_.toString),
          denoisingStrength = a.num("denoising_strength", 0, 1).getOrElse(0.75),
          resizeMode = enumOrInt(a, "resize_mode", ResizeModes),
          maskBlur = a.int("mask_blur", 0, 64).getOrElse(4),
          inpaintingFill = enumOrInt(a, "inpainting_fill", FillModes, default = 1),
          inpaintFullRes = a.bool("inpaint_full_res").getOrElse(false),
          inpaintFullResPadding = a.int("inpaint_full_res_padding", 0, 256).getOrElse(32),
          inpaintingMaskInvert = a.bool("inpainting_mask_invert").getOrElse(false)
        )
      }

      val (dw, dh) = i2i match
        case Some(i) if !a.has("width") || !a.has("height") =>
          val (w, h) = Images.size(java.nio.file.Paths.get(i.initImage))
          def f8(v: Int) = math.max(64, math.min(2048, v) / 8 * 8)
          (f8(w), f8(h))
        case _ =>
          val s = defaultSide(checkpoint)
          (s, s)

      val p = GenerationParams(
        prompt = fullPrompt,
        negativePrompt = a.str("negative_prompt").getOrElse(""),
        seed = a.long("seed", -1L, 4294967295L).getOrElse(-1L),
        samplerName = sampler,
        scheduler = scheduler,
        steps = a.int("steps", 1, 150).getOrElse(20),
        cfgScale = a.num("cfg_scale", 0, 30).getOrElse(7.0),
        width = a.int("width", 64, MaxSide).getOrElse(dw),
        height = a.int("height", 64, MaxSide).getOrElse(dh),
        batchSize = batch,
        nIter = nIter,
        clipSkip = a.int("clip_skip", 1, 12).getOrElse(1),
        hires = hires,
        img2img = i2i,
        styles = a.strList("styles").getOrElse(Nil),
        checkpoint = checkpoint,
        vae = vae,
        extra = a.stringMap("extra").getOrElse(Map.empty)
      )
      val problems = p.validate
      if problems.nonEmpty then throw ToolError(problems.mkString("; "))
      checkWork(p)
      GenRequest(p, a.bool("return_images").getOrElse(true), a.int("max_image_side", 0, MaxSide).getOrElse(1024), temps.result())
    catch
      case e: Throwable =>
        temps.result().foreach(Images.deleteQuietly)
        throw e

  /** Bounds the work of one job: per-batch pixels, hires target size and total output pixels. */
  private def checkWork(p: GenerationParams): Unit =
    val batchPx = p.width.toLong * p.height * p.batchSize
    if batchPx > MaxBatchPixels then
      throw ToolError(s"width * height * batch_size is too large ($batchPx px, max $MaxBatchPixels); lower the size or batch_size")
    val (tw, th) = if p.hires.enabled && !p.isImg2Img then p.hires.targetSize(p.width, p.height) else (p.width, p.height)
    if tw > MaxSide || th > MaxSide then throw ToolError(s"hires target ${tw}x$th exceeds ${MaxSide}x$MaxSide; lower hires.scale or resize_x/resize_y")
    val total = tw.toLong * th * p.batchSize * p.nIter
    if total > MaxJobPixels then
      throw ToolError(s"this job would produce $total px in total (max $MaxJobPixels); lower the size, hires scale or image count")

  private def mcpPending: Int =
    services.queue.pending.count(_.origin == Origin) + asyncJobs.values().asScala.count(_.status == "waiting_for_engine")

  private def checkPendingCap(): Unit =
    if mcpPending >= MaxPendingJobs then
      throw ToolError(s"too many queued MCP jobs (max $MaxPendingJobs); wait for some to finish (get_job) or cancel them")

  /** Check-and-submit under one lock so concurrent calls cannot overshoot [[MaxPendingJobs]]. */
  private val pendingLock = new Object
  private def submitCapped(job: Job): Future[JobOutput] = pendingLock.synchronized {
    checkPendingCap()
    services.queue.submit(job)
  }

  /** Finds a [[JobLimitExceeded]] in the cause chain. */
  private def limitError(e: Throwable): Option[JobLimitExceeded] = e match
    case l: JobLimitExceeded => Some(l)
    case null                => None
    case _                   => Option(e.getCause).filter(_ ne e).flatMap(limitError)

  private def enumOrInt(a: Args, k: String, names: Seq[String], default: Int = 0): Int =
    a.obj.value.get(k) match
      case Some(ujson.Str(s)) if s.trim.toIntOption.isEmpty =>
        val i = names.indexWhere(_.equalsIgnoreCase(s.trim))
        if i < 0 then throw ToolError(s"$k must be one of ${names.mkString(", ")} or 0..${names.size - 1}") else i
      case _ => a.int(k, 0, names.size - 1).getOrElse(default)

  // ---- running jobs ------------------------------------------------------------------------------------------

  /** Maps queue events of one job to MCP progress notifications (percent, 0..100). */
  private def progressListener(jobId: String, p: GenerationParams, scripted: Boolean, ctx: ToolContext): JobEvent => Unit =
    val hiresSplit = p.hires.enabled && !p.isImg2Img && !scripted
    {
      case JobEvent.JobQueued(j, pos) if j.id == jobId =>
        ctx.progress(0, Some(100), if pos > 1 then s"queued (position $pos)" else "queued")
      case JobEvent.JobStarted(j) if j.id == jobId =>
        ctx.progress(0.1, Some(100), "started")
      case JobEvent.Progress(id, e, text, done, total) if id == jobId =>
        val steps = math.max(1, e.total)
        val imgs = math.max(1, e.images)
        var frac = (math.min(e.image, imgs - 1) + math.min(e.step, steps).toDouble / steps) / imgs
        if hiresSplit then frac = if e.pass == "base" then frac * 0.5 else 0.5 + frac * 0.5
        if total > 0 then frac = (math.min(done, total) + frac) / total
        val m = Seq(
          Option.when(text.nonEmpty)(text),
          Some(s"${e.pass} step ${e.step}/${e.total}"),
          Option.when(e.images > 1)(s"image ${e.image + 1}/${e.images}")
        ).flatten.mkString(", ")
        ctx.progress(math.min(99.9, 0.1 + frac * 99.8), Some(100), m)
      case JobEvent.Status(id, text, done, total) if id == jobId && total > 0 =>
        ctx.progress(math.min(99.9, 0.1 + done.toDouble / total * 99.8), Some(100), text)
      case _ => ()
    }

  private enum Outcome:
    case Done(out: JobOutput)
    case Cancelled(partial: JobOutput)

  private def runBlocking(job: Job, ctx: ToolContext, temps: Seq[Path]): Outcome =
    try
      if ctx.isCancelled then Outcome.Cancelled(JobOutput(Nil))
      else
        ready(ctx)
        val remove = services.queue.addListener(progressListener(job.id, job.params, job.script.isDefined, ctx))
        try
          val f = submitCapped(job)
          ctx.onCancel { () =>
            // JobQueue.cancel misses a job between dequeue and start; retry once shortly after
            if !services.queue.cancel(job.id) then
              Future {
                Thread.sleep(150)
                if !f.isCompleted then services.queue.cancel(job.id)
              }
            ()
          }
          try
            val out = Await.result(f, Duration.Inf)
            ctx.progress(100, Some(100), "done")
            Outcome.Done(out)
          catch
            case e: JobCancelledException => Outcome.Cancelled(e.partial)
            case e: InterruptedException =>
              services.queue.cancel(job.id)
              throw e
            case NonFatal(e) =>
              limitError(e) match
                case Some(l) => throw ToolError(s"job limit exceeded: ${l.message}")
                case None    => throw ToolError(s"generation failed: ${msg(e)}")
        finally remove()
    finally temps.foreach(Images.deleteQuietly)

  private def imagesJson(out: JobOutput): ujson.Arr =
    ujson.Arr.from(out.grids.map(imageJson(_, grid = true)) ++ out.images.map(imageJson(_, grid = false)))

  private def imageJson(o: OutputImage, grid: Boolean): ujson.Value =
    ujson.Obj("path" -> o.path.toAbsolutePath.toString, "seed" -> o.seed.toDouble, "infotext" -> o.infotext, "grid" -> grid)

  /** Inline previews within [[MaxPreviews]] images and [[MaxInlineBytes]] of base64; the rest are listed by path. */
  private def previews(out: JobOutput, returnImages: Boolean, maxSide: Int): (Seq[ujson.Value], Seq[String]) =
    if !returnImages then (Nil, Nil)
    else
      val warn = Seq.newBuilder[String]
      val all = out.all
      if all.size > MaxPreviews then warn += s"only the first $MaxPreviews of ${all.size} images are returned inline"
      var budget = MaxInlineBytes.toLong
      var omitted = 0
      val imgs = all.take(MaxPreviews).flatMap { o =>
        if budget <= 0 then
          omitted += 1
          None
        else
          try
            val e =
              if Images.estimateBase64(o.path, maxSide) > budget then null
              else Images.encode(o.path, maxSide)
            if e == null || e.base64.length > budget then
              omitted += 1
              budget = 0
              None
            else
              budget -= e.base64.length
              Some(Content.image(e.base64, e.mimeType))
          catch
            case NonFatal(ex) =>
              warn += s"cannot encode ${o.path}: ${msg(ex)}"
              None
      }
      if omitted > 0 then
        warn += s"$omitted preview(s) omitted (inline limit ${MaxInlineBytes / 1024 / 1024} MB per response); use get_image or a smaller max_image_side"
      (imgs, warn.result())

  private def outputResult(jobId: String, out: JobOutput, returnImages: Boolean, maxSide: Int, header: Option[String] = None, isError: Boolean = false): ToolResult =
    val (imgs, warnings) = previews(out, returnImages, maxSide)
    val text = (header.toSeq ++ out.all.map(o => s"${o.path.toAbsolutePath}\n${o.infotext}") ++ warnings).mkString("\n\n")
    val structured = ujson.Obj("job_id" -> jobId, "images" -> imagesJson(out))
    ToolResult(imgs :+ Content.text(if text.isEmpty then "no images were produced" else text) :+ Content.text(ujson.write(structured)), Some(structured), isError)

  private def generationTool(img2img: Boolean, args: ujson.Obj, ctx: ToolContext): ToolResult =
    ready(ctx) // name checks need the catalog
    val req = parseGen(Args(args, genKeys(img2img)), img2img)
    val job = Job(Job.newId(), req.params, origin = Origin, limits = limits)
    runBlocking(job, ctx, req.temps) match
      case Outcome.Done(out) => outputResult(job.id, out, req.returnImages, req.maxSide)
      case Outcome.Cancelled(partial) =>
        outputResult(job.id, partial, req.returnImages, req.maxSide, Some(s"job ${job.id} was interrupted; ${partial.all.size} image(s) saved"), isError = true)

  // ---- async jobs -------------------------------------------------------------------------------------------

  private final class AsyncJob(val job: Job, val kind: String, val returnImages: Boolean, val maxSide: Int, val created: Instant):
    @volatile var status: String = "queued"
    @volatile var progress: Double = 0
    @volatile var message: String = ""
    @volatile var output: Option[JobOutput] = None
    @volatile var error: Option[String] = None
    @volatile var cancelRequested = false
    val result: Promise[JobOutput] = Promise()
    def future: Future[JobOutput] = result.future
    def active: Boolean = status == "queued" || status == "running" || status == "waiting_for_engine"
    val tracker: ToolContext = new ToolContext:
      def progress(p: Double, total: Option[Double], m: String): Unit =
        if p >= AsyncJob.this.progress then AsyncJob.this.progress = p
        if m.nonEmpty then message = m
        if status == "queued" && p > 0 then status = "running"
      def onCancel(f: () => Unit): Unit = ()
      def isCancelled: Boolean = false

  private val asyncJobs = new ConcurrentHashMap[String, AsyncJob]()

  private def pruneAsync(): Unit =
    if asyncJobs.size > MaxAsyncJobs then
      asyncJobs.values().asScala.toSeq.filterNot(_.active)
        .sortBy(_.created.toEpochMilli).take(asyncJobs.size - MaxAsyncJobs)
        .foreach(j => asyncJobs.remove(j.job.id))

  /** Registers the job and returns at once; waiting for the engine and queue submission happen in the background
    * (status `waiting_for_engine` until then). Any failure is recorded on the job and its temp files are freed.
    */
  private def submitAsync(job: Job, kind: String, req: GenRequest): ToolResult =
    val aj = new AsyncJob(job, kind, req.returnImages, req.maxSide, Instant.now())
    val usable = engineUsable
    try
      pendingLock.synchronized {
        checkPendingCap()
        if !usable then aj.status = "waiting_for_engine" // counts as pending until submitted
        asyncJobs.put(job.id, aj)
      }
      pruneAsync()
    catch
      case e: Throwable =>
        asyncJobs.remove(job.id)
        req.temps.foreach(Images.deleteQuietly)
        throw e
    aj.future.onComplete { r =>
      req.temps.foreach(Images.deleteQuietly)
      r match
        case Success(out) =>
          aj.output = Some(out)
          aj.progress = 100
          aj.status = "done"
        case Failure(e: JobCancelledException) =>
          aj.output = Some(e.partial)
          aj.status = "cancelled"
        case Failure(e) =>
          aj.error = Some(limitError(e).map(l => s"job limit exceeded: ${l.message}").getOrElse(msg(e)))
          aj.status = "failed"
    }
    def launch(): Unit =
      try
        if !engineUsable then ready(new ToolContext:
          def progress(p: Double, t: Option[Double], m: String): Unit = ()
          def onCancel(f: () => Unit): Unit = ()
          def isCancelled: Boolean = aj.cancelRequested
        )
        if aj.cancelRequested then throw JobCancelledException(job.id, JobOutput(Nil))
        ensureCatalog()
        val remove = services.queue.addListener(progressListener(job.id, job.params, job.script.isDefined, aj.tracker))
        try
          // a waiting job already holds a pending slot; an immediate one takes it now
          val f = if aj.status == "waiting_for_engine" then services.queue.submit(job) else submitCapped(job)
          if aj.status == "waiting_for_engine" then aj.status = "queued"
          f.onComplete { r =>
            remove()
            aj.result.tryComplete(r)
          }
        catch
          case e: Throwable =>
            remove()
            throw e
      catch
        case e: ToolError if aj.cancelRequested => aj.result.tryFailure(JobCancelledException(job.id, JobOutput(Nil)))
        case e: Throwable                       => aj.result.tryFailure(e)
    if !usable then Future(launch()) else launch()
    val pos = services.queue.pending.indexWhere(_.id == job.id) + 1
    ToolResult.json(
      ujson.Obj("job_id" -> job.id, "status" -> aj.status, "queue_position" -> pos),
      Some(s"submitted ${job.id}${if aj.status == "waiting_for_engine" then " (waiting for the engine to start)" else ""}. Poll get_job with job_id=\"${job.id}\" and wait_seconds.")
    )

  /** cancel_job for async jobs that have not reached the queue yet. */
  private def cancelWaiting(id: String): Boolean =
    Option(asyncJobs.get(id)).filter(_.status == "waiting_for_engine") match
      case Some(aj) =>
        aj.cancelRequested = true
        true
      case None => false

  private def getJob(a: Args, ctx: ToolContext): ToolResult =
    val id = a.reqStr("job_id")
    val aj = Option(asyncJobs.get(id)).getOrElse(throw ToolError(s"unknown job_id: $id (only jobs submitted with *_async are tracked)"))
    val wait = a.num("wait_seconds", 0, 300).getOrElse(0.0)
    val end = System.nanoTime() + (wait * 1e9).toLong
    // wait in slices so a cancelled get_job returns promptly
    while !aj.future.isCompleted && !ctx.isCancelled && System.nanoTime() < end do
      val slice = math.min(500L, math.max(1L, (end - System.nanoTime()) / 1000000L))
      try Await.ready(aj.future, slice.millis)
      catch case _: TimeoutException => ()
    // onComplete runs asynchronously; give the callback a moment to record the outcome
    if aj.future.isCompleted then
      val stop = System.nanoTime() + 2.seconds.toNanos
      while aj.active && System.nanoTime() < stop do Thread.sleep(5)
    val pos = services.queue.pending.indexWhere(_.id == id) + 1
    val returnImages = a.bool("return_images").getOrElse(aj.returnImages)
    val maxSide = a.int("max_image_side", 0, MaxSide).getOrElse(aj.maxSide)
    val base = ujson.Obj(
      "job_id" -> id,
      "kind" -> aj.kind,
      "status" -> aj.status,
      "progress" -> math.min(1.0, aj.progress / 100.0),
      "message" -> aj.message,
      "queue_position" -> pos
    )
    aj.error.foreach(e => base("error") = e)
    aj.output match
      case Some(out) =>
        base("images") = imagesJson(out)
        val (imgs, warnings) = previews(out, returnImages, maxSide)
        val head = s"job $id: ${aj.status}"
        val text = (Seq(head) ++ out.all.map(o => s"${o.path.toAbsolutePath}\n${o.infotext}") ++ warnings).mkString("\n\n")
        ToolResult(imgs :+ Content.text(text) :+ Content.text(ujson.write(base)), Some(base), isError = aj.status == "cancelled")
      case None =>
        base("images") = ujson.Arr()
        val text = aj.status match
          case "failed" => s"job $id failed: ${aj.error.getOrElse("")}"
          case s        => f"job $id: $s, ${aj.progress}%.0f%%${if aj.message.nonEmpty then s" (${aj.message})" else ""}${if pos > 0 then s", queue position $pos" else ""}"
        ToolResult(Seq(Content.text(text), Content.text(ujson.write(base))), Some(base), isError = aj.status == "failed")

  // ---- scripts ---------------------------------------------------------------------------------------------

  private def scriptPlugins: Seq[Plugin] =
    (services.registry.scripts(Tab.Txt2Img) ++ services.registry.scripts(Tab.Img2Img)).distinctBy(_.id)

  private def controlJson(c: Control): ujson.Value =
    val o = ujson.Obj("key" -> c.key, "label" -> c.label)
    c match
      case Control.Slider(_, _, min, max, step, d) =>
        o("type") = "number"; o("min") = min; o("max") = max; o("step") = step; o("default") = d
      case Control.IntField(_, _, min, max, d) =>
        o("type") = "integer"; o("min") = min; o("max") = max; o("default") = d
      case Control.Check(_, _, d) =>
        o("type") = "boolean"; o("default") = d
      case Control.Choice(_, _, opts, d) =>
        o("type") = "choice"; o("options") = ujson.Arr.from(opts); o("default") = d
      case Control.DynamicChoice(_, _, src, d, allowNone) =>
        val opts = (if allowNone then Seq("None") else Nil) ++ (try catalog.choices(src) catch case NonFatal(_) => Nil)
        o("type") = "choice"; o("options") = ujson.Arr.from(opts); o("default") = d
      case Control.Text(_, _, d, multiline, placeholder) =>
        o("type") = "string"; o("default") = d
        if placeholder.nonEmpty then o("placeholder") = placeholder
        if multiline then o("multiline") = true
      case Control.ImageFile(_, _, multiple) =>
        o("type") = if multiple then "image_paths" else "image_path"
        o("description") = if multiple then "newline separated absolute image paths (allowed dirs only)" else "absolute image path (allowed dirs only)"
      case Control.Note(_, _) =>
        o("type") = "note"
    o

  private def scriptJson(p: Plugin): ujson.Value =
    ujson.Obj(
      "id" -> p.id,
      "name" -> p.name,
      "description" -> p.description,
      "tabs" -> ujson.Arr.from(p.tabs.toSeq.map(_.toString.toLowerCase).sorted),
      "controls" -> ujson.Arr.from(p.controls.map(controlJson))
    )

  private def runScript(args: ujson.Obj, ctx: ToolContext): ToolResult =
    val a = Args(args, Seq("script_id", "values", "params", "return_images", "max_image_side", "async"))
    val id = a.reqStr("script_id")
    val async = a.bool("async").getOrElse(false)
    val plugin = scriptPlugins.find(_.id == id).orElse(scriptPlugins.find(_.name.equalsIgnoreCase(id))).getOrElse(
      throw ToolError(s"unknown script: $id. Available: ${scriptPlugins.map(_.id).mkString(", ")}")
    )
    val values = a.stringMap("values").getOrElse(Map.empty)
    val known = plugin.controls.map(_.key).toSet
    val unknown = values.keySet.filterNot(known.contains)
    if unknown.nonEmpty then throw ToolError(s"unknown control(s) for ${plugin.id}: ${unknown.mkString(", ")}. Controls: ${known.toSeq.sorted.mkString(", ")}")
    if plugin.id == "custom_code" then throw ToolError("custom_code cannot be run over MCP")
    // image controls are file paths read by the engine: same allowed-directory policy as init_image / get_image
    val imageKeys = plugin.controls.collect { case c: Control.ImageFile => c.key }.toSet
    val checkedValues = values.map { (k, v) =>
      if !imageKeys.contains(k) then k -> v
      else k -> v.split("[\\r\\n]+").map(_.trim).filter(_.nonEmpty)
        .map(spec => pathPolicy.resolve(spec, s"values.$k", outputsDir).toString).mkString("\n")
    }
    val pObj = a.objOpt("params").getOrElse(ujson.Obj())
    val img2img = pObj.value.get("init_image").exists(_ != ujson.Null)
    if img2img && !plugin.tabs.contains(Tab.Img2Img) then throw ToolError(s"${plugin.id} does not support img2img")
    if !img2img && !plugin.tabs.contains(Tab.Txt2Img) then throw ToolError(s"${plugin.id} needs params.init_image (img2img only)")
    val returnImages = a.bool("return_images").getOrElse(true)
    val maxSide = a.int("max_image_side", 0, MaxSide).getOrElse(1024)
    if !async then ready(ctx)
    val req = parseGen(Args(pObj, scriptParamKeys(img2img)), img2img, requirePrompt = false).copy(returnImages = returnImages, maxSide = maxSide)
    val job =
      try Job(Job.newId(), req.params, script = Some(plugin -> Values(services.registry.values(plugin).map ++ checkedValues)), origin = Origin, limits = limits)
      catch
        case e: Throwable =>
          req.temps.foreach(Images.deleteQuietly)
          throw e
    if async then submitAsync(job, s"script:${plugin.id}", req)
    else
      runBlocking(job, ctx, req.temps) match
        case Outcome.Done(out) => outputResult(job.id, out, req.returnImages, req.maxSide, Some(s"${plugin.name}: ${out.all.size} image(s)"))
        case Outcome.Cancelled(partial) =>
          outputResult(job.id, partial, req.returnImages, req.maxSide, Some(s"job ${job.id} was interrupted; ${partial.all.size} image(s) saved"), isError = true)

  // ---- tool definitions ------------------------------------------------------------------------------------

  private def tool(n: String, t: String, d: String, schema: ujson.Obj, out: Option[ujson.Obj] = None, ro: Boolean = false,
      destructiveHint: Boolean = false, idempotentHint: Boolean = false)(f: (ujson.Obj, ToolContext) => ToolResult): McpTool =
    new McpTool:
      def name = n
      override def title = t
      def description = d
      def inputSchema = schema
      override def outputSchema = out
      override def readOnly = ro
      override def destructive = destructiveHint
      override def idempotent = idempotentHint || ro
      def call(args: ujson.Obj, ctx: ToolContext) = f(args, ctx)

  private def genSchema(img2img: Boolean): ujson.Obj =
    val common: Seq[(String, ujson.Value)] = Seq(
      "prompt" -> Schema.str("Positive prompt. A1111 syntax works: (emphasis:1.2), [from:to:when], <lora:name:weight>."),
      "negative_prompt" -> Schema.str("Negative prompt (things to avoid)."),
      "width" -> Schema.int("Image width in px, multiple of 8. Default 512 (SD1.5) / 1024 (SDXL)" + (if img2img then "; img2img defaults to the init image size." else "."), Some(64), Some(4096)),
      "height" -> Schema.int("Image height in px, multiple of 8. Same defaults as width.", Some(64), Some(4096)),
      "steps" -> Schema.int("Sampling steps.", Some(1), Some(150), Some(20)),
      "cfg_scale" -> Schema.num("Classifier-free guidance scale.", Some(0.0), Some(30.0), Some(7.0)),
      "sampler_name" -> Schema.str("Sampler (see list_samplers), e.g. 'Euler a', 'DPM++ 2M'.", default = Some("DPM++ 2M")),
      "scheduler" -> Schema.str("Schedule type (see list_samplers), e.g. Automatic, Karras.", default = Some("Automatic")),
      "seed" -> Schema.int("Seed; -1 = random. The actual seed is returned for each image.", Some(-1), Some(4294967295L), Some(-1)),
      "batch_size" -> Schema.int(s"Images per batch. batch_size * n_iter <= $MaxImages.", Some(1), Some(MaxImages), Some(1)),
      "n_iter" -> Schema.int(s"Number of batches. batch_size * n_iter <= $MaxImages.", Some(1), Some(MaxImages), Some(1)),
      "clip_skip" -> Schema.int("CLIP skip.", Some(1), Some(12), Some(1)),
      "checkpoint" -> Schema.str("Checkpoint name (see list_models). Loaded before generating if different; omitted = current model."),
      "vae" -> Schema.str("VAE name, 'Automatic' (checkpoint default) or 'None'. Omitted = keep current."),
      "hires" -> Schema.obj(
        Seq(
          "enabled" -> Schema.bool("Enable the hires fix second pass.", Some(true)),
          "scale" -> Schema.num("Upscale factor.", Some(1.0), Some(4.0), Some(2.0)),
          "upscaler" -> Schema.str("Upscaler (see list_samplers hires_upscalers), e.g. Latent.", default = Some("Latent")),
          "steps" -> Schema.int("Second pass steps; 0 = same as steps.", Some(0), Some(150), Some(0)),
          "denoising_strength" -> Schema.num("Second pass denoising strength.", Some(0.0), Some(1.0), Some(0.55)),
          "resize_x" -> Schema.int("Explicit target width (0 = use scale).", Some(0), Some(4096), Some(0)),
          "resize_y" -> Schema.int("Explicit target height (0 = use scale).", Some(0), Some(4096), Some(0))
        )
      ).tap(_("description") = "Hires fix (txt2img only)."),
      "loras" -> Schema.arr(
        Schema.obj(Seq("name" -> Schema.str("LoRA name (see list_loras)."), "weight" -> Schema.num("Weight.", Some(-10.0), Some(10.0), Some(1.0))), Seq("name")),
        "LoRAs to apply; appended to the prompt as <lora:name:weight> tags."
      ),
      "styles" -> Schema.arr(Schema.str(), "Saved prompt style names (see list_samplers styles)."),
      "return_images" -> Schema.bool("Return base64 previews inline. Files are always saved.", Some(true)),
      "max_image_side" -> Schema.int("Downscale returned previews so the longest side is at most this (0 = full size). The saved file stays full size.", Some(0), Some(MaxSide.toLong), Some(1024)),
      "extra" -> Schema.scalarMap("Advanced: extra key/value parameters recorded in the infotext and passed to the engine.")
    )
    val i2i: Seq[(String, ujson.Value)] =
      if !img2img then Nil
      else
        Seq(
          "init_image" -> Schema.str("Source image: absolute file path, base64 PNG/JPEG, or data URI."),
          "mask" -> Schema.str("Inpainting mask (white = repaint): absolute path, base64 or data URI."),
          "denoising_strength" -> Schema.num("How much to change the image (0 = none, 1 = everything).", Some(0.0), Some(1.0), Some(0.75)),
          "resize_mode" -> ujson.Obj(
            "description" -> s"0 just resize, 1 crop and resize, 2 resize and fill (or the name).",
            "oneOf" -> ujson.Arr(Schema.int("", Some(0), Some(2)), Schema.str("", ResizeModes)),
            "default" -> 0
          ),
          "inpainting_fill" -> ujson.Obj(
            "description" -> "Masked content: 0 fill, 1 original, 2 latent noise, 3 latent nothing (or the name).",
            "oneOf" -> ujson.Arr(Schema.int("", Some(0), Some(3)), Schema.str("", FillModes)),
            "default" -> 1
          ),
          "inpaint_full_res" -> Schema.bool("Inpaint only the masked area at full resolution.", Some(false)),
          "inpaint_full_res_padding" -> Schema.int("Padding around the masked area (px).", Some(0), Some(256), Some(32)),
          "inpainting_mask_invert" -> Schema.bool("Invert the mask (repaint the black area).", Some(false)),
          "mask_blur" -> Schema.int("Mask blur radius.", Some(0), Some(64), Some(4))
        )
    val props = if img2img then (common.filterNot(_._1 == "hires") ++ i2i) else common
    Schema.obj(props, if img2img then Seq("prompt", "init_image") else Seq("prompt"))

  private val imagesOutputSchema: ujson.Obj = Schema.obj(
    Seq(
      "job_id" -> Schema.str("Queue job id."),
      "images" -> Schema.arr(
        Schema.obj(
          Seq(
            "path" -> Schema.str("Absolute path of the saved full-size PNG."),
            "seed" -> Schema.int("Seed of this image."),
            "infotext" -> Schema.str("A1111-style generation parameters embedded in the PNG."),
            "grid" -> Schema.bool("Whether this is a grid image.")
          ),
          Seq("path", "seed", "infotext"),
          additional = true
        )
      )
    ),
    Seq("images"),
    additional = true
  )

  private def engineStateName(s: EngineState): String = s match
    case EngineState.Stopped    => "stopped"
    case EngineState.Starting   => "starting"
    case EngineState.Ready(_)   => "ready"
    case EngineState.Busy       => "busy"
    case EngineState.Crashed(m) => s"crashed: $m"

  private def loadedJson: ujson.Value =
    services.supervisor.loadedModel match
      case Some(m) =>
        val o = ujson.Obj("name" -> m.name, "path" -> m.path, "kind" -> m.kind, "hash" -> m.hash.take(10))
        m.vaeName.foreach(v => o("vae") = v)
        o
      case None => ujson.Null

  private def samePath(a: String, b: String): Boolean =
    try java.nio.file.Paths.get(a).toAbsolutePath.normalize() == java.nio.file.Paths.get(b).toAbsolutePath.normalize()
    catch case NonFatal(_) => a == b

  /** txt2img + img2img parameters for run_script (hires only applies without init_image). */
  private def scriptParamsSchema: ujson.Obj =
    val props = (genSchema(false)("properties").obj.toSeq ++ genSchema(true)("properties").obj.toSeq)
      .filterNot((k, _) => k == "return_images" || k == "max_image_side")
      .distinctBy(_._1)
    val o = Schema.obj(props)
    o("description") = "Base generation parameters (as txt2img; include init_image for img2img scripts, hires only without it)."
    o

  val all: Seq[McpTool] = Seq(
    tool("txt2img", "Text to image",
      "Generate image(s) from a text prompt and wait for the result. Returns downscaled previews, the saved file paths, seeds and infotext. Long jobs: consider txt2img_async.",
      genSchema(false), Some(imagesOutputSchema))((a, ctx) => generationTool(false, a, ctx)),
    tool("img2img", "Image to image / inpaint",
      "Transform an existing image guided by a prompt (init_image), or inpaint with a mask. Waits for the result.",
      genSchema(true), Some(imagesOutputSchema))((a, ctx) => generationTool(true, a, ctx)),
    tool("txt2img_async", "Text to image (async)",
      "Queue a txt2img job and return its job_id immediately. Poll get_job for progress and results.",
      genSchema(false))((a, _) => {
        checkPendingCap()
        val req = parseGen(Args(a, genKeys(false)), img2img = false)
        submitAsync(Job(Job.newId(), req.params, origin = Origin, limits = limits), "txt2img", req)
      }),
    tool("img2img_async", "Image to image (async)",
      "Queue an img2img/inpaint job and return its job_id immediately. Poll get_job for progress and results.",
      genSchema(true))((a, _) => {
        checkPendingCap()
        val req = parseGen(Args(a, genKeys(true)), img2img = true)
        submitAsync(Job(Job.newId(), req.params, origin = Origin, limits = limits), "img2img", req)
      }),
    tool("get_job", "Get async job",
      "Status, progress (0..1) and, when finished, the images of a job started with *_async or run_script async. wait_seconds blocks until the job finishes or the time passes.",
      Schema.obj(
        Seq(
          "job_id" -> Schema.str("Job id returned by *_async."),
          "wait_seconds" -> Schema.num("Wait up to this many seconds for completion.", Some(0.0), Some(300.0), Some(0.0)),
          "return_images" -> Schema.bool("Override return_images of the submission."),
          "max_image_side" -> Schema.int("Override max_image_side of the submission.", Some(0), Some(MaxSide.toLong))
        ),
        Seq("job_id")
      ), ro = true)((a, ctx) => getJob(Args(a, Seq("job_id", "wait_seconds", "return_images", "max_image_side")), ctx)),
    tool("cancel_job", "Cancel job", "Cancel a queued MCP job or interrupt it if it is running.",
      Schema.obj(Seq("job_id" -> Schema.str("Job id.")), Seq("job_id")), destructiveHint = true, idempotentHint = true)((a, _) => {
        val id = Args(a, Seq("job_id")).reqStr("job_id")
        val q = services.queue
        val isMcp = asyncJobs.containsKey(id) || (q.current.toSeq ++ q.pending).exists(j => j.id == id && j.origin == Origin)
        if !isMcp then throw ToolError(s"no queued or running MCP job with id $id")
        val ok = cancelWaiting(id) || q.cancel(id)
        ToolResult.json(ujson.Obj("job_id" -> id, "cancelled" -> ok), Some(if ok then s"cancel requested for $id" else s"$id is no longer queued or running"))
      }),
    tool("list_models", "List checkpoints",
      "Checkpoints (with kind: sd15 / sdxl ...) and VAEs available to the engine, marking the loaded one.",
      Schema.obj(Seq("refresh" -> Schema.bool("Rescan the model folders first.", Some(false)))), ro = true)((a, ctx) => {
        val args = Args(a, Seq("refresh"))
        ready(ctx)
        if args.bool("refresh").getOrElse(false) then
          try catalog.refresh()
          catch case NonFatal(e) => throw ToolError(s"refresh failed: ${msg(e)}")
        else ensureCatalog()
        val loaded = services.supervisor.loadedModel
        val cks = catalog.checkpoints.map { c =>
          ujson.Obj("name" -> c.name, "kind" -> c.kind, "path" -> c.path, "size_mb" -> c.sizeMb, "loaded" -> loaded.exists(m => samePath(m.path, c.path)))
        }
        val v = ujson.Obj("checkpoints" -> ujson.Arr.from(cks), "vaes" -> ujson.Arr.from(catalog.vaes.map(_.name)), "loaded" -> loadedJson)
        val lines = catalog.checkpoints.map(c => s"${if loaded.exists(m => samePath(m.path, c.path)) then "* " else "  "}${c.name} [${c.kind}]")
        ToolResult.json(v, Some((s"Checkpoints (${cks.size}, * = loaded):" +: lines :+ s"VAEs: ${if catalog.vaes.isEmpty then "(none)" else catalog.vaes.map(_.name).mkString(", ")}").mkString("\n")))
      }),
    tool("set_model", "Load checkpoint",
      "Load a checkpoint (and optionally a VAE) and make it the default for later generations.",
      Schema.obj(Seq("name" -> Schema.str("Checkpoint name, title, path or hash prefix (see list_models)."), "vae" -> Schema.str("VAE name, 'Automatic' or 'None'.")), Seq("name")),
      destructiveHint = true, idempotentHint = true)((a, ctx) => {
        val args = Args(a, Seq("name", "vae"))
        val name = args.reqStr("name")
        ready(ctx)
        ensureCatalog()
        // only catalog entries (inside the configured checkpoint folders) can be loaded through MCP
        if catalog.findCheckpoint(name).isEmpty then
          throw ToolError(s"checkpoint not found: $name. Available: ${catalog.checkpoints.map(_.name).mkString(", ")}")
        try
          services.loadCheckpoint(name)
          args.str("vae").foreach(v => services.setVae(Some(v)))
        catch case e: EngineError => throw ToolError(e.message)
        ToolResult.json(ujson.Obj("loaded" -> loadedJson), Some(s"loaded ${services.supervisor.loadedModel.map(m => s"${m.name} [${m.kind}]").getOrElse(name)}"))
      }),
    tool("list_loras", "List LoRAs", "LoRA networks available for loras=[{name, weight}] or <lora:name:weight> prompt tags.",
      Schema.obj(Seq("query" -> Schema.str("Case-insensitive substring filter."))), ro = true)((a, ctx) => {
        val q = Args(a, Seq("query")).str("query").getOrElse("").toLowerCase
        ready(ctx)
        ensureCatalog()
        val ls = catalog.loras.filter(l => q.isEmpty || l.name.toLowerCase.contains(q))
        val v = ujson.Obj("loras" -> ujson.Arr.from(ls.map(l => ujson.Obj("name" -> l.name, "path" -> l.path, "tag" -> LoraSyntax.tag(l.name)))))
        ToolResult.json(v, Some(if ls.isEmpty then "no LoRAs found" else ls.map(l => s"${l.name}  ${LoraSyntax.tag(l.name)}").mkString("\n")))
      }),
    tool("list_samplers", "List samplers",
      "Samplers, schedule types, hires-fix upscalers, upscalers and saved styles accepted by txt2img/img2img.",
      Schema.empty, ro = true)((a, ctx) => {
        Args(a, Nil)
        ready(ctx)
        ensureCatalog()
        val s = catalog.samplers
        val styles = try Styles.load(services.home.stylesFile).map(_.name) catch case NonFatal(_) => Vector.empty
        val v = ujson.Obj(
          "samplers" -> ujson.Arr.from(s.samplers.filter(_.supported).map(x => ujson.Obj("name" -> x.name, "default_scheduler" -> x.defaultScheduler))),
          "schedulers" -> ujson.Arr.from(s.schedulers),
          "hires_upscalers" -> ujson.Arr.from(catalog.choices(ChoiceSource.HiresUpscalers)),
          "upscalers" -> ujson.Arr.from(catalog.choices(ChoiceSource.Upscalers)),
          "styles" -> ujson.Arr.from(styles)
        )
        ToolResult.json(v, Some(Seq(
          s"Samplers: ${s.samplers.filter(_.supported).map(_.name).mkString(", ")}",
          s"Schedulers: ${s.schedulers.mkString(", ")}",
          s"Hires upscalers: ${catalog.choices(ChoiceSource.HiresUpscalers).mkString(", ")}",
          s"Upscalers: ${catalog.choices(ChoiceSource.Upscalers).mkString(", ")}",
          s"Styles: ${if styles.isEmpty then "(none)" else styles.mkString(", ")}"
        ).mkString("\n")))
      }),
    tool("list_recent", "Recent images", "Most recent images in the outputs folder (newest first) with their infotext; query filters by file name or parameters.",
      Schema.obj(Seq("limit" -> Schema.int("Max images.", Some(1), Some(200), Some(10)), "query" -> Schema.str("Case-insensitive filter on file name / infotext."))), ro = true)((a, _) => {
        val args = Args(a, Seq("limit", "query"))
        val limit = args.int("limit", 1, 200).getOrElse(10)
        val items = services.gallery.search(args.str("query").getOrElse(""), limit)
        val v = ujson.Obj("images" -> ujson.Arr.from(items.map(i =>
          ujson.Obj("path" -> i.path.toAbsolutePath.toString, "modified" -> Instant.ofEpochMilli(i.modified).toString, "infotext" -> i.parameters.map(ujson.Str(_)).getOrElse(ujson.Null))
        )))
        ToolResult.json(v, Some(if items.isEmpty then s"no images under ${services.gallery.rootDir}" else items.map(i => s"${i.path.toAbsolutePath}\n${i.parameters.getOrElse("(no parameters)")}").mkString("\n\n")))
      }),
    tool("get_image", "Get image", "Return an image from the outputs folder (or another allowed folder) with its infotext.",
      Schema.obj(Seq("path" -> Schema.str("Absolute path, or path relative to the outputs folder."), "max_side" -> Schema.int("Downscale so the longest side is at most this (0 = original).", Some(0), Some(MaxSide.toLong), Some(1024))), Seq("path")),
      ro = true)((a, _) => {
        val args = Args(a, Seq("path", "max_side"))
        val raw = args.reqStr("path")
        if raw.trim.isEmpty then throw ToolError("path is empty")
        val p = pathPolicy.resolve(raw, "path", outputsDir.toAbsolutePath.normalize())
        val n = p.getFileName.toString.toLowerCase
        if !Seq(".png", ".jpg", ".jpeg", ".webp").exists(n.endsWith) then throw ToolError("only .png / .jpg / .webp files can be read")
        val side = args.int("max_side", 0, MaxSide).getOrElse(1024)
        if Images.estimateBase64(p, side) > MaxInlineBytes then throw ToolError("image too large to return inline; use a smaller max_side")
        val e = Images.encode(p, side)
        if e.base64.length > MaxInlineBytes then throw ToolError(s"image too large to return inline (${e.width}x${e.height}); use a smaller max_side")
        val info = services.gallery.item(p).flatMap(_.parameters)
        val structured = ujson.Obj("path" -> p.toString, "infotext" -> info.map(ujson.Str(_)).getOrElse(ujson.Null))
        ToolResult(
          Seq(Content.image(e.base64, e.mimeType), Content.text(s"$p\n${info.getOrElse("(no parameters)")}"), Content.text(ujson.write(structured))),
          Some(structured)
        )
      }),
    tool("get_status", "Engine status", "Engine state, device, loaded model, current job and queue length.", Schema.empty, ro = true)((a, _) => {
      Args(a, Nil)
      val sup = services.supervisor
      val q = services.queue
      val v = ujson.Obj(
        "engine_state" -> engineStateName(sup.state),
        "dry_run" -> services.dryRun,
        "loaded_model" -> loadedJson,
        "current_job" -> q.current.map(j => ujson.Obj("job_id" -> j.id, "origin" -> j.origin): ujson.Value).getOrElse(ujson.Null),
        "queue_length" -> q.pending.size,
        "outputs_dir" -> outputsDir.toAbsolutePath.toString
      )
      sup.hello.foreach { h =>
        v("engine") = ujson.Obj("version" -> h.engineVersion, "device" -> h.device, "cuda" -> h.cuda, "vram_mb" -> h.vramMb, "python" -> h.python,
          "torch" -> h.torch.map(ujson.Str(_)).getOrElse(ujson.Null), "dry_run" -> h.dryRun, "features" -> ujson.Arr.from(h.features))
      }
      ToolResult.json(v)
    }),
    tool("interrupt", "Interrupt", "Stop the currently running generation (any origin). clear_queue also drops all pending jobs.",
      Schema.obj(Seq("clear_queue" -> Schema.bool("Also remove pending jobs.", Some(false)))), destructiveHint = true)((a, _) => {
        val args = Args(a, Seq("clear_queue"))
        val cleared = if args.bool("clear_queue").getOrElse(false) then services.queue.clear() else 0
        val stopped = services.queue.interruptCurrent()
        ToolResult.json(ujson.Obj("interrupted" -> stopped, "cleared" -> cleared),
          Some(s"${if stopped then "interrupt requested" else "nothing running"}${if cleared > 0 then s"; removed $cleared pending job(s)" else ""}"))
      }),
    tool("list_scripts", "List scripts", "Scripts usable with run_script (X/Y/Z plot, prompt matrix, loopback, SD upscale ...) with their controls (key, type, options, default).",
      Schema.empty, ro = true)((a, _) => {
        Args(a, Nil)
        ensureCatalog()
        ToolResult.json(ujson.Obj("scripts" -> ujson.Arr.from(scriptPlugins.map(scriptJson))))
      }),
    tool("run_script", "Run script",
      "Run a script plugin (see list_scripts). values = control values by key (e.g. xyz_grid: {x_type: 'Seed', x_values: '1-4'}); params = the txt2img arguments (add init_image for img2img scripts). Waits for the result unless async=true (then poll get_job).",
      Schema.obj(
        Seq(
          "script_id" -> Schema.str("Script id from list_scripts (e.g. xyz_grid)."),
          "values" -> Schema.scalarMap("Control values keyed by control key; missing keys use their defaults."),
          "params" -> scriptParamsSchema,
          "return_images" -> Schema.bool("Return base64 previews inline.", Some(true)),
          "max_image_side" -> Schema.int("Downscale returned previews (0 = full size).", Some(0), Some(MaxSide.toLong), Some(1024)),
          "async" -> Schema.bool("Queue and return {job_id} immediately; poll get_job.", Some(false))
        ),
        Seq("script_id")
      ))((a, ctx) => runScript(a, ctx))
  )

object McpTools:
  val Origin = "mcp"
  val MaxImages = 16
  val MaxPreviews = 16
  val MaxAsyncJobs = 100
  val MaxPendingJobs = 32
  /** Max width/height of any image (base, hires target, previews). */
  val MaxSide = 4096
  val MaxBatchPixels: Long = 4096L * 4096
  val MaxJobPixels: Long = 64L * 1024 * 1024
  /** Max total base64 of inline images in one response. */
  val MaxInlineBytes: Int = 20 * 1024 * 1024
  /** Engine-enforced limits for every MCP job, script sub-generations (X/Y/Z cells, Size / Hires axes) included. */
  val StrictLimits: JobLimits = JobLimits(maxSide = 4096, maxImagesPerGeneration = 16, maxGenerations = 64, maxPixelsPerGeneration = 4096L * 4096)
  val ResizeModes: Seq[String] = Seq("just resize", "crop and resize", "resize and fill")
  val FillModes: Seq[String] = Seq("fill", "original", "latent noise", "latent nothing")
  val HiresKeys: Seq[String] = Seq("enabled", "scale", "upscaler", "steps", "denoising_strength", "resize_x", "resize_y")

  private val CommonKeys = Seq("prompt", "negative_prompt", "width", "height", "steps", "cfg_scale", "sampler_name", "scheduler", "seed",
    "batch_size", "n_iter", "clip_skip", "checkpoint", "vae", "loras", "styles", "return_images", "max_image_side", "extra")
  private val Img2ImgKeys = Seq("init_image", "mask", "denoising_strength", "resize_mode", "inpainting_fill", "inpaint_full_res",
    "inpaint_full_res_padding", "inpainting_mask_invert", "mask_blur")

  def genKeys(img2img: Boolean): Seq[String] = if img2img then CommonKeys ++ Img2ImgKeys else CommonKeys :+ "hires"
  def scriptParamKeys(img2img: Boolean): Seq[String] = genKeys(img2img).filterNot(k => k == "return_images" || k == "max_image_side")

  def msg(e: Throwable): String = e match
    case ee: EngineError => ee.message
    case _               => Option(e.getMessage).getOrElse(e.toString)

  /** Readiness for the GUI case: waits while the engine is starting; a stopped/crashed engine is left to the queue. */
  def waitForEngine(services: AppServices): FiniteDuration => Unit = timeout =>
    val end = System.nanoTime() + timeout.toNanos
    while services.supervisor.state == EngineState.Starting && System.nanoTime() < end do Thread.sleep(100)
    if services.supervisor.state == EngineState.Starting then throw new TimeoutException("engine still starting")

  /** Readiness for headless mode: waits on the engine start future. */
  def awaitStart(start: Future[?]): FiniteDuration => Unit = timeout =>
    Await.result(start, timeout)
    ()

  extension (o: ujson.Obj) private def tap(f: ujson.Obj => Unit): ujson.Obj = { f(o); o }
