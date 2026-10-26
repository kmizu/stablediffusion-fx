package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.model.*
import sdfx.core.payload.*
import sdfx.core.plugins.api.*
import sdfx.core.protocol.*
import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, TimeUnit}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class PayloadCaptureContextSuite extends munit.FunSuite:
  import FakeEngine.*
  override val munitTimeout: Duration = 120.seconds
  private val base = GenerationParams(prompt = "<literal 猫>", seed = 9, subseed = 0, width = 64, height = 64, steps = 1)

  private def withServices(test: AppServices => Unit): Unit =
    val home = tempHome()
    val services = new AppServices(home, Settings(lastModel = Some("foo")), true, Nil,
      launchOverride = Some(() => launch()), persistSettings = false, installShutdownHook = false)
    try
      services.start()
      test(services)
    finally
      services.shutdown()
      deleteTree(home.root)

  private def captures(s: AppServices, id: String): Vector[PayloadSnapshot] = s.payloads.snapshots.filter(_.key.jobId == id)
  private def context(s: PayloadSnapshot): ujson.Value = ujson.read(s.contextJson.get)
  private def uiContext(mode: String) = Some(SubmittedPayloadContext("ui", Some(mode), Some("{\"ui_forms\":\"available\"}")))

  test("disabled UI gate and native seed-options inference have separate provenance") {
    withServices { s =>
      val frozen = SubmittedPayloadContext("ui", Some("txt2img"), None, Some(false))
      val job = Job(Job.newId(), base.copy(subseed = 99L, subseedStrength = 0.0), submittedContext = Some(frozen))
      await(s.queue.submit(job))
      val saved = captures(s, job.id).head
      val diagnostic = context(saved)
      assertEquals(diagnostic("submitted")("ui_variation_enabled").bool, false)
      assertEquals(diagnostic("submitted")("configuration"), ujson.Null)
      val requested = diagnostic("requested_seed_options")
      assertEquals(requested("subseed_strength").num, 0.0)
      assertEquals(requested("extras_requested").bool, true)
      assertEquals(requested("extras_requested_basis").str, "native_params_inference")
      assert(!saved.bodyJson.get.contains("ui_variation_enabled"))
      assertEquals(ujson.read(saved.bodyJson.get)("params")("subseed_strength").num, 0.0)
    }
  }

  test("script override changes native inference but never frozen UI gate") {
    val overrideScript = new Plugin:
      def id = "variation-override"
      def name = "Variation override"
      def kind = PluginKind.Script
      override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
        JobOutput(ctx.generate(p.copy(subseedStrength = 0.7)))
    withServices { s =>
      val job = Job(Job.newId(), base.copy(subseed = 99L, subseedStrength = 0.0),
        script = Some(overrideScript -> Values.empty),
        submittedContext = Some(SubmittedPayloadContext("ui", Some("txt2img"), None, Some(false))))
      await(s.queue.submit(job))
      val saved = captures(s, job.id).head
      val diagnostic = context(saved)
      assertEquals(diagnostic("submitted")("ui_variation_enabled").bool, false)
      val requested = diagnostic("requested_seed_options")
      assertEquals(requested("subseed_strength").num, 0.7)
      assertEquals(requested("extras_requested").bool, true)
      assertEquals(requested("extras_requested_basis").str, "native_params_inference")
      assertEquals(ujson.read(saved.bodyJson.get)("params")("subseed_strength").num, 0.7)
      assert(!saved.bodyJson.get.contains("ui_variation_enabled"))
    }
  }

  test("MCP and direct origins keep UI gate unknown without changing legacy native inference") {
    withServices { s =>
      val mcp = Job(Job.newId(), base.copy(subseed = 99L, subseedStrength = 0.0), origin = "mcp")
      await(s.queue.submit(mcp))
      s.generator.generate(base.copy(subseed = 99L, subseedStrength = 0.0))
      val found = captures(s, mcp.id) ++ s.payloads.snapshots.filter(_.key.jobId.startsWith("direct-"))
      assertEquals(found.size, 2)
      found.foreach { saved =>
        val diagnostic = context(saved)
        assertEquals(diagnostic("submitted")("ui_variation_enabled"), ujson.Null)
        assertEquals(diagnostic("requested_seed_options")("extras_requested").bool, true)
        assertEquals(diagnostic("requested_seed_options")("extras_requested_basis").str, "native_params_inference")
        assert(!saved.bodyJson.get.contains("ui_variation_enabled"))
      }
    }
  }

  private class LoopScript(failure: Option[String] = None) extends Plugin:
    def id = "payload-loop"
    def name = "Payload loop"
    def kind = PluginKind.Script
    override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
      ctx.generateTransient(p)
      val second = failure.fold(p)(code => p.copy(extra = p.extra.updated("force_error", code)))
      JobOutput(ctx.generate(second))

  test("each identical transient/final script body has its own identity and terminal event") {
    withServices { s =>
      val events = new ConcurrentLinkedQueue[JobEvent]()
      val remove = s.queue.addListener(e => { events.add(e); () })
      val job = Job(Job.newId(), base, script = Some(new LoopScript -> Values.empty), submittedContext = uiContext("txt2img"))
      try
        assertEquals(await(s.queue.submit(job)).images.size, 1)
        val saved = captures(s, job.id)
        assertEquals(saved.map(_.key.sequence), Vector(1, 2))
        assertEquals(saved.map(_.state), Vector(PayloadState.Succeeded, PayloadState.Succeeded))
        assertEquals(saved(0).bodyJson, saved(1).bodyJson)
        assertEquals(events.asScala.collect { case e: JobEvent.PayloadCaptured => e.snapshot.key }.toVector, saved.map(_.key))
        assertEquals(ujson.read(saved.last.bodyJson.get), await(s.client.raw("get_state"))("last_generation")("request"))
      finally remove()
    }
  }

  for (code, expected) <- Seq("invalid_params" -> PayloadState.Failed, "interrupted" -> PayloadState.Cancelled) do
    test(s"earlier success stays succeeded when later sub-generation ends $code") {
      withServices { s =>
        val job = Job(Job.newId(), base, script = Some(new LoopScript(Some(code)) -> Values.empty), submittedContext = uiContext("txt2img"))
        intercept[Exception](await(s.queue.submit(job)))
        val saved = captures(s, job.id)
        assertEquals(saved.map(_.key.sequence), Vector(1, 2))
        assertEquals(saved.map(_.state), Vector(PayloadState.Succeeded, expected))
        assert(saved.forall(_.bodyJson.nonEmpty))
      }
    }

  test("queued cancellation has its own absent body and cannot borrow an earlier successful body") {
    withServices { s =>
      await(s.queue.submit(Job(base)))
      val gate = Job(base.copy(seed = 10))
      val entered = new CountDownLatch(1)
      val release = new CountDownLatch(1)
      val remove = s.queue.addListener {
        case JobEvent.JobStarted(j) if j.id == gate.id => entered.countDown(); release.await(20, TimeUnit.SECONDS); ()
        case _ => ()
      }
      try
        val running = s.queue.submit(gate)
        assert(entered.await(20, TimeUnit.SECONDS))
        val cancelled = Job(Job.newId(), base, submittedContext = uiContext("inpaint"))
        val result = s.queue.submit(cancelled)
        assert(s.queue.cancel(cancelled.id))
        intercept[JobCancelledException](await(result))
        val own = captures(s, cancelled.id)
        assertEquals(own.size, 1)
        assertEquals(own.head.bodyJson, None)
        assertEquals(own.head.state, PayloadState.Cancelled)
        assertEquals(s.payloads.latestForMode("inpaint").map(_.key.jobId), Some(cancelled.id))
        release.countDown()
        await(running)
      finally
        release.countDown()
        remove()
    }
  }

  test("mutable nested hooks freeze on queue submission, while actual dispatch remains the sole body") {
    withServices { s =>
      val gate = Job(base)
      val entered = new CountDownLatch(1)
      val release = new CountDownLatch(1)
      val remove = s.queue.addListener {
        case JobEvent.JobStarted(j) if j.id == gate.id => entered.countDown(); release.await(20, TimeUnit.SECONDS); ()
        case _ => ()
      }
      try
        val running = s.queue.submit(gate)
        assert(entered.await(20, TimeUnit.SECONDS))
        val nested = ujson.Obj("array" -> ujson.Arr("submitted"))
        val job = Job(Job.newId(), base.copy(hooks = Map("test" -> nested)), submittedContext = uiContext("txt2img"))
        val queued = s.queue.submit(job)
        nested("array")(0) = "edited after queue"
        release.countDown()
        await(running)
        await(queued)
        val body = ujson.read(captures(s, job.id).head.bodyJson.get)
        assertEquals(body("params")("hooks")("test")("array")(0).str, "submitted")
        assertEquals(body, await(s.client.raw("get_state"))("last_generation")("request"))
      finally
        release.countDown()
        remove()
    }
  }

  test("actual cancellation during optional capture work prevents later transport dispatch") {
    withServices { s =>
      await(s.queue.submit(Job(base)))
      val baseline = await(s.client.raw("get_state"))("last_generation")
      val job = Job(Job.newId(), base.copy(seed = 88), submittedContext = uiContext("txt2img"))
      val cancelled = new java.util.concurrent.atomic.AtomicBoolean(false)
      val remove = s.payloads.subscribe { all =>
        if all.exists(c => c.key.jobId == job.id && c.state == PayloadState.Prepared) && cancelled.compareAndSet(false, true) then
          val outcome = await(s.queue.cancelResult(job.id))
          assert(outcome.cancelled && !outcome.blocked)
      }
      try
        intercept[JobCancelledException](await(s.queue.submit(job)))
        assert(cancelled.get(), "required cancellation gate was not exercised")
        assertEquals(await(s.client.raw("get_state"))("last_generation"), baseline)
        val own = captures(s, job.id)
        assertEquals(own.size, 1)
        assertEquals(own.head.state, PayloadState.Cancelled)
        // A prepared body is observable, but its existence is not evidence it was sent.
        assertEquals(Json.fromJs[Long](ujson.read(own.head.bodyJson.get)("params")("seed")), 88L)
      finally remove()
    }
  }

  test("capture preference freezes once at submission throughout a script") {
    withServices { s =>
      val path = s.home.root.resolve("mask.png")
      javax.imageio.ImageIO.write(new java.awt.image.BufferedImage(8, 8, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", path.toFile)
      val script = new LoopScript:
        override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
          ctx.generateTransient(p)
          s.updateSettings(_.copy(apiPayloadIncludeImageContents = false))
          JobOutput(ctx.generate(p))
      s.updateSettings(_.copy(apiPayloadIncludeImageContents = true))
      val job = Job(Job.newId(), base.copy(img2img = Some(Img2ImgParams(path.toString, Some(path.toString)))),
        script = Some(script -> Values.empty), submittedContext = uiContext("inpaint"))
      await(s.queue.submit(job))
      assertEquals(captures(s, job.id).size, 2)
      assert(captures(s, job.id).forall(_.assetsJson.exists(_.contains("data:image/png;base64,"))))
      val explicitOff = Job(Job.newId(), job.params, captureOptions = Some(CaptureOptions(false)))
      s.updateSettings(_.copy(apiPayloadIncludeImageContents = true))
      await(s.queue.submit(explicitOff))
      assert(!captures(s, explicitOff.id).head.assetsJson.get.contains("data:image/png;base64,"))
    }
  }

  test("direct Generator calls allocate distinct direct identities without a fabricated UI mode") {
    withServices { s =>
      s.generator.generateTransient(base)
      s.generator.generate(base)
      val saved = s.payloads.snapshots
      assertEquals(saved.size, 2)
      assert(saved.forall(_.key.jobId.startsWith("direct-")))
      assertEquals(saved.map(_.key.jobId).distinct.size, 2)
      saved.foreach { c =>
        assertEquals(context(c)("submitted")("origin").str, "direct")
        assertEquals(context(c)("submitted")("source_mode"), ujson.Null)
      }
    }
  }

  test("requested seed extras remain separate from resolved native seeds without invented resize fields") {
    withServices { s =>
      Seq((-1L, -1L, 0.0, false), (-1L, 17L, 0.0, true), (-1L, -1L, 0.25, true),
        (Long.MaxValue, Long.MaxValue, 0.0, true)).foreach { (seed, subseed, strength, extras) =>
        val job = Job(Job.newId(), base.copy(seed = seed, subseed = subseed, subseedStrength = strength),
          submittedContext = uiContext("txt2img"))
        await(s.queue.submit(job))
        val saved = captures(s, job.id).head
        val requested = context(saved)("requested_seed_options")
        assertEquals(Json.fromJs[Long](requested("seed")), seed)
        assertEquals(Json.fromJs[Long](requested("subseed")), subseed)
        assertEquals(requested("subseed_strength").num, strength)
        assertEquals(requested("extras_requested").bool, extras)
        assertEquals(requested("seed_resize").str, "not_in_native_schema")
        val body = ujson.read(saved.bodyJson.get)("params")
        assert(Json.fromJs[Long](body("seed")) >= 0)
        assert(Json.fromJs[Long](body("subseed")) >= 0)
        if seed == Long.MaxValue then
          assertEquals(Json.fromJs[Long](body("seed")), Long.MaxValue)
          assertEquals(Json.fromJs[Long](body("subseed")), Long.MaxValue)
        assert(!body.obj.contains("seed_enable_extras"))
        assert(!body.obj.contains("seed_resize_from_w"))
        assert(!body.obj.contains("seed_resize_from_h"))
      }
    }
  }

  test("model facts reported at capture stay separate from subsequent reported completion facts") {
    withServices { s =>
      val job = Job(Job.newId(), base, submittedContext = uiContext("txt2img"))
      val remove = s.queue.addListener {
        case e: JobEvent.PayloadCaptured if e.jobId == job.id =>
          // Mutates only the supervisor's later reported model through the real typed command.
          await(s.client.loadModel(LoadModelRequest(s.home.root.resolve("models/checkpoints/sdxl/Bar_XL.safetensors").toString)))
        case _ => ()
      }
      try
        await(s.queue.submit(job))
        val c = context(captures(s, job.id).head)
        assert(c("reported_at_capture").obj.contains("captured_at_millis"))
        assertEquals(c("reported_at_capture")("model")("name").str, "foo")
        assertEquals(c("reported_at_completion")("model_name").str, "Bar_XL")
        assertEquals(c("reported_at_capture")("model")("name").str, "foo")
      finally remove()
    }
  }

  test("throwing optional prepared/lifecycle listener cannot prevent a valid generation") {
    withServices { s =>
      val listener = new GenerationListener:
        override def onPreparedRequest(snapshot: PayloadSnapshot): Unit = throw new IllegalStateException("closed")
        override def onPayloadLifecycle(key: PayloadKey, state: PayloadState, completionJson: Option[String]): Unit =
          throw new IllegalStateException("closed lifecycle")
      assertEquals(s.generator.generateWith(base, listener).size, 1)
      assertEquals(Json.fromJs[Long](await(s.client.raw("get_state"))("last_generation")("request")("params")("seed")), 9L)
    }
  }
