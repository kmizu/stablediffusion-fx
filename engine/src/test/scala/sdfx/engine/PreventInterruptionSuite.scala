package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.{EngineError, GenerationRequest, InterruptionDecision, InterruptionProtection, Json, LoadModelRequest, Protocol}
import java.util.concurrent.ConcurrentLinkedQueue
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Contracts precede implementation. Fixture must implement the same reader policy as real Python tests. */
class PreventInterruptionSuite extends munit.FunSuite:
  import FakeEngine.*
  override val munitTimeout: Duration = 90.seconds
  private def withServices(body: AppServices => Unit): Unit =
    val home = tempHome()
    val services = new AppServices(home, Settings(lastModel = Some("foo")), dryRun = true, builtins = Nil,
      launchOverride = Some(() => launch()), persistSettings = false, installShutdownHook = false)
    try
      services.start()
      body(services)
    finally
      services.shutdown()
      deleteTree(home.root)

  // Legacy-route fallbacks make these regression tests execute before the new
  // member APIs exist. Scala selects the real member methods once implemented.
  extension (supervisor: EngineSupervisor)
    private def captureInterrupt(owner: Option[String] = None): () => scala.concurrent.Future[InterruptionDecision] =
      () => supervisor.requestInterrupt(owner)

  test("protected active cancellation cannot set native job flags or discard later batch images") {
    withServices { s =>
      val events = new ConcurrentLinkedQueue[JobEvent]()
      val remove = s.queue.addListener(events.add(_))
      try
        await(s.client.raw("interruption_protection", ujson.Obj("enabled" -> true)))
        val job = Job(GenerationParams(seed = 4, steps = 12, width = 64, height = 64, batchSize = 2,
          extra = Map("step_delay" -> "0.03")))
        val finished = s.queue.submit(job)
        assert(eventually()(events.asScala.exists(_.isInstanceOf[JobEvent.Progress])))
        assertEquals(s.queue.interruptCurrent(), false)
        assertEquals(s.queue.cancel(job.id), false)
        assertEquals(await(finished).images.size, 2)
        assert(!events.asScala.exists(_.isInstanceOf[JobEvent.JobCancelled]))
      finally remove()
    }
  }
  test("protection leaves pending-only queue cancellation and clearing unchanged") {
    withServices { s =>
      await(s.client.raw("interruption_protection", ujson.Obj("enabled" -> true)))
      val first = Job(GenerationParams(steps = 20, width = 64, height = 64, extra = Map("step_delay" -> ".03")))
      val running = s.queue.submit(first)
      assert(eventually()(s.queue.current.exists(_.id == first.id)))
      val second = Job(first.params)
      val third = Job(first.params)
      val pending = s.queue.submit(second)
      val cleared = s.queue.submit(third)
      assert(s.queue.cancel(second.id))
      assertEquals(s.queue.clear(), 1)
      intercept[JobCancelledException](await(pending))
      intercept[JobCancelledException](await(cleared))
      assertEquals(await(running).images.size, 1)
    }
  }
  test("blocked supervisor interrupt never arms forced restart escalation") {
    val sup = new EngineSupervisor(() => launch(), interruptGrace = 100.millis)
    val output = java.nio.file.Files.createTempDirectory("protect-images-")
    try
      sup.start()
      await(sup.request("interruption_protection", ujson.Obj("enabled" -> true)))
      await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("models/x.safetensors"))))
      val pid = sup.pid
      val request = GenerationRequest(GenerationParams(steps = 12, width = 64, height = 64,
        extra = Map("step_delay" -> ".03", "ignore_interrupt" -> "true")), output.toString)
      val result = sup.request(Protocol.Types.Txt2Img, Json.toJs(request))
      assert(eventually()(sup.state == EngineState.Busy))
      sup.interrupt()
      assertEquals(await(result)("images").arr.size, 1)
      assertEquals(sup.pid, pid)
    finally
      sup.close()
      deleteTree(output)
  }
  test("explicit user restart is blocked while health recovery and shutdown still work") {
    val sup = new EngineSupervisor(() => launch())
    try
      sup.start()
      await(sup.request("interruption_protection", ujson.Obj("enabled" -> true)))
      val before = sup.pid
      val blocked = intercept[EngineError](sup.restart())
      assertEquals(blocked.code, "interruption_blocked")
      assertEquals(sup.pid, before)
      await(sup.forceRestart())
      assertNotEquals(sup.pid, before)
      assertEquals(await(sup.request("interruption_protection"))("enabled").bool, false)
    finally sup.close()
  }
  test("Off interrupt remains JobCancelled even if backend Interrupted arrives before native flag update") {
    withServices { s =>
      val events = new ConcurrentLinkedQueue[JobEvent]()
      val remove = s.queue.addListener(events.add(_))
      try
        val job = Job(GenerationParams(steps = 50, width = 64, height = 64, extra = Map("step_delay" -> ".02")))
        val result = s.queue.submit(job)
        assert(eventually()(events.asScala.exists(_.isInstanceOf[JobEvent.Progress])))
        assert(s.queue.interruptCurrent())
        intercept[JobCancelledException](await(result))
        assert(!events.asScala.exists(_.isInstanceOf[JobEvent.JobFailed]))
        assert(events.asScala.exists(_.isInstanceOf[JobEvent.JobCancelled]))
      finally remove()
    }
  }
  test("late admission for a completed job cannot set the next job cancellation flag") {
    withServices { s =>
      val admission = scala.concurrent.Promise[InterruptionDecision]()
      val queue = new JobQueue(s.generator, _ => admission.future)
      try
        val first = Job(GenerationParams(steps = 8, width = 64, height = 64, extra = Map("step_delay" -> ".02")))
        val second = Job(first.params.copy(steps = 20))
        val one = queue.submit(first)
        assert(eventually()(queue.current.exists(_.id == first.id)))
        val pending = queue.interruptCurrentResult()
        val two = queue.submit(second)
        assertEquals(await(one).images.size, 1)
        assert(eventually()(queue.current.exists(_.id == second.id)))
        admission.success(InterruptionDecision(accepted = true, blocked = false, protection = InterruptionProtection()))
        assertEquals(await(pending).interrupted, false)
        assertEquals(await(two).images.size, 1)
      finally queue.shutdown()
    }
  }
  test("delayed backend dispatch for job A cannot interrupt job B after queue handover") {
    withServices { s =>
      val gate = scala.concurrent.Promise[Unit]()
      val queue = new JobQueue(s.generator, owner => gate.future.flatMap(_ => s.supervisor.requestInterrupt(owner))(
        using scala.concurrent.ExecutionContext.parasitic))
      val progress = new ConcurrentLinkedQueue[JobEvent]()
      val remove = queue.addListener(progress.add(_))
      try
        val first = Job(GenerationParams(steps = 8, width = 64, height = 64, extra = Map("step_delay" -> ".02")))
        val second = Job(first.params.copy(steps = 35))
        val one = queue.submit(first)
        assert(eventually()(progress.asScala.exists {
          case p: JobEvent.Progress => p.jobId == first.id
          case _ => false
        }))
        val admission = queue.interruptCurrentResult()
        val two = queue.submit(second)
        assertEquals(await(one).images.size, 1)
        assert(eventually()(progress.asScala.exists {
          case p: JobEvent.Progress => p.jobId == second.id
          case _ => false
        }))
        gate.success(()) // Invoke the actual supervisor only after B has begun sampling.
        await(admission)
        assertEquals(await(two).images.size, 1)
        assert(!progress.asScala.exists {
          case p: JobEvent.JobCancelled => p.jobId == second.id
          case _ => false
        })
      finally
        remove()
        queue.shutdown()
    }
  }
  test("typed protection query and toggle bypass startup model reload work") {
    val sup = new EngineSupervisor(() => launch())
    val client = new EngineClient(sup)
    try
      sup.start()
      await(sup.request(Protocol.Types.LoadModel, ujson.Obj("path" -> "slow.safetensors", "delay" -> 2.0)))
      val oldPid = sup.pid
      val restarted = sup.forceRestart()
      assert(eventually()(sup.pid.nonEmpty && sup.pid != oldPid && !restarted.isCompleted), "Did not observe live reader during startup reload")
      val state = await(client.interruptionProtection(), 300.millis)
      assertEquals(state, InterruptionProtection())
      assert(!restarted.isCompleted, "Query waited for the reload instead of using the live reader")
      val enabled = await(client.interruptionProtection(Some(true)), 300.millis)
      assert(enabled.enabled)
      assert(!restarted.isCompleted, "Toggle waited for model work")
      await(restarted)
      assert(await(client.interruptionProtection()).enabled, "Completing the existing reload reset reader policy")
    finally sup.close()
  }
  test("stop during authorized restart close cannot launch a late replacement engine") {
    val dir = java.nio.file.Files.createTempDirectory("restart-stop-gate-")
    val gate = dir.resolve("shutdown-gate")
    java.nio.file.Files.writeString(gate, "block the old process after shutdown acknowledgement")
    val sup = new EngineSupervisor(() => launch("--shutdown-gate", gate.toString))
    try
      sup.start()
      val restarting = scala.concurrent.Future(sup.restart())(using scala.concurrent.ExecutionContext.global)
      assert(eventually()(java.nio.file.Files.exists(dir.resolve("shutdown-gate.entered"))), "Restart never reached the old-process close gate")
      assertEquals(sup.pid, None)
      sup.close() // Actual teardown wins while the authorized restart worker waits for old.close().
      java.nio.file.Files.delete(gate)
      scala.concurrent.Await.ready(restarting, 10.seconds)
      assert(restarting.value.exists(_.isFailure), "A restart superseded by stop must not report a replacement engine")
      assertEquals(sup.state, EngineState.Stopped)
      assertEquals(sup.pid, None)
      assert(!sup.isRunning)
    finally
      java.nio.file.Files.deleteIfExists(gate)
      sup.close()
      deleteTree(dir)
  }
  test("explicit restart from Stopped still starts an engine without reviving a later stop") {
    val sup = new EngineSupervisor(() => launch())
    try
      val hello = sup.restart()
      assertEquals(hello.engineVersion, "fake-0")
      assert(sup.state.isInstanceOf[EngineState.Ready])
      sup.stop()
      assertEquals(sup.restart().engineVersion, "fake-0")
    finally sup.close()
  }

  test("idle capture ownership cannot cancel a job that starts before dispatch") {
    withServices { s =>
      val gate = scala.concurrent.Promise[Unit]()
      val queue = new JobQueue(s.generator, owner => gate.future.flatMap(_ => s.supervisor.requestInterrupt(owner))(
        using scala.concurrent.ExecutionContext.parasitic))
      val progress = new ConcurrentLinkedQueue[JobEvent]()
      val remove = queue.addListener(progress.add(_))
      try
        assert(queue.isIdle)
        val pending = queue.interruptCurrentResult()
        val job = Job(GenerationParams(seed = 12, steps = 35, width = 64, height = 64,
          extra = Map("step_delay" -> ".02")))
        val result = queue.submit(job)
        assert(eventually()(progress.asScala.exists(_.isInstanceOf[JobEvent.Progress])))
        gate.success(())
        assertEquals(await(pending).interrupted, false)
        assertEquals(await(result).images.size, 1)
      finally
        remove()
        queue.shutdown()
    }
  }
  test("standalone generation handle ownership cannot interrupt a subsequent generation") {
    val sup = new EngineSupervisor(() => launch())
    val client = new EngineClient(sup)
    val output = java.nio.file.Files.createTempDirectory("handle-owner-")
    try
      sup.start()
      await(client.loadModel(LoadModelRequest("model.safetensors")))
      val request = GenerationRequest(GenerationParams(seed = 11, steps = 1, width = 64, height = 64), output.toString)
      val first = client.generate(request, GenerationMode.Txt2Img)
      assertEquals(await(first.result).images.size, 1)
      val progressing = new java.util.concurrent.atomic.AtomicBoolean(false)
      val second = client.generate(request.copy(params = request.params.copy(steps = 35,
        extra = Map("step_delay" -> ".02"))), GenerationMode.Txt2Img, {
        case _: sdfx.core.protocol.EngineEvent.Progress => progressing.set(true)
        case _ => ()
      })
      assert(eventually()(progressing.get()))
      first.interrupt()
      assertEquals(await(second.result).images.size, 1)
    finally
      sup.close()
      deleteTree(output)
  }
  test("queued cancellation ownership cannot escalate into restarting unrelated active work") {
    val sup = new EngineSupervisor(() => launch(), interruptGrace = 100.millis)
    val output = java.nio.file.Files.createTempDirectory("queued-owner-")
    try
      sup.start()
      val pid = sup.pid
      val busy = sup.requestOwned(Protocol.Types.LoadModel,
        ujson.Obj("path" -> "slow.safetensors", "delay" -> 1.5), _ => (), Some("owner-b"))
      assert(eventually()(sup.state == EngineState.Busy))
      val request = GenerationRequest(GenerationParams(seed = 9, steps = 1, width = 64, height = 64), output.toString)
      val queued = sup.requestOwned(Protocol.Types.Txt2Img, Json.toJs(request), _ => (), Some("owner-a"))
      assert(await(sup.requestInterrupt(Some("owner-a"))).affected)
      assertEquals(await(busy)("name").str, "slow")
      assertEquals(intercept[EngineError](await(queued)).code, Protocol.ErrorCodes.Interrupted)
      assertEquals(sup.pid, pid)
    finally
      sup.close()
      deleteTree(output)
  }

  test("runtime interrupt reaches nonqueue work and still obeys shared protection") {
    withServices { s =>
      val progressing = new java.util.concurrent.atomic.AtomicBoolean(false)
      val request = GenerationRequest(GenerationParams(seed = 18, steps = 100, width = 64, height = 64,
        extra = Map("step_delay" -> ".02")), s.home.tmpDir(s.settings).toString)
      await(s.client.interruptionProtection(Some(true)))
      val running = s.client.generate(request, GenerationMode.Txt2Img, {
        case _: sdfx.core.protocol.EngineEvent.Progress => progressing.set(true)
        case _ => ()
      })
      assert(eventually()(progressing.get()))
      assert(s.queue.isIdle, "This must exercise work outside the native queue")
      val blocked = await(s.interruptRuntime())
      assert(blocked.blocked && !blocked.interrupted)
      assert(!running.result.isCompleted)
      await(s.client.interruptionProtection(Some(false)))
      assert(await(s.interruptRuntime()).interrupted)
      assertEquals(intercept[EngineError](await(running.result)).code, Protocol.ErrorCodes.Interrupted)
    }
  }
  test("runtime snapshot captured while empty cannot cancel later nonqueue work") {
    withServices { s =>
      val captured = s.supervisor.captureInterrupt()
      val progressing = new java.util.concurrent.atomic.AtomicBoolean(false)
      val request = GenerationRequest(GenerationParams(seed = 18, steps = 35, width = 64, height = 64,
        extra = Map("step_delay" -> ".02")), s.home.tmpDir(s.settings).toString)
      val running = s.client.generate(request, GenerationMode.Txt2Img, {
        case _: sdfx.core.protocol.EngineEvent.Progress => progressing.set(true)
        case _ => ()
      })
      assert(eventually()(progressing.get()))
      assert(!await(captured()).affected)
      assertEquals(await(running.result).images.size, 1)
    }
  }
  test("runtime snapshot for completed nonqueue A cannot cancel later B") {
    withServices { s =>
      val progressing = new java.util.concurrent.atomic.AtomicBoolean(false)
      val request = GenerationRequest(GenerationParams(seed = 18, steps = 15, width = 64, height = 64,
        extra = Map("step_delay" -> ".02")), s.home.tmpDir(s.settings).toString)
      def run() = s.client.generate(request, GenerationMode.Txt2Img, {
        case _: sdfx.core.protocol.EngineEvent.Progress => progressing.set(true)
        case _ => ()
      })
      val first = run()
      assert(eventually()(progressing.get()))
      val captured = s.supervisor.captureInterrupt()
      assertEquals(await(first.result).images.size, 1)
      progressing.set(false)
      val second = run()
      assert(eventually()(progressing.get()))
      assert(!await(captured()).affected)
      assertEquals(await(second.result).images.size, 1)
    }
  }

  test("runtime interrupt reaches active nonqueue work when a native job waits behind it") {
    withServices { s =>
      val backendProgress = new java.util.concurrent.atomic.AtomicBoolean(false)
      val request = GenerationRequest(GenerationParams(seed = 21, steps = 100, width = 64, height = 64,
        extra = Map("step_delay" -> ".02")), s.home.tmpDir(s.settings).toString)
      val backend = s.client.generate(request, GenerationMode.Txt2Img, {
        case _: sdfx.core.protocol.EngineEvent.Progress => backendProgress.set(true)
        case _ => ()
      })
      assert(eventually()(backendProgress.get()))
      val native = Job(GenerationParams(seed = 22, steps = 15, width = 64, height = 64,
        extra = Map("step_delay" -> ".02")))
      val nativeResult = s.queue.submit(native)
      assert(eventually()(s.queue.current.exists(_.id == native.id)))
      assert(await(s.interruptRuntime()).interrupted)
      assertEquals(intercept[EngineError](await(backend.result)).code, Protocol.ErrorCodes.Interrupted)
      intercept[JobCancelledException](await(nativeResult))
    }
  }

  test("explicit Off restart during remembered startup cannot reuse the detached startup future") {
    val entered = new java.util.concurrent.CountDownLatch(1)
    val release = new java.util.concurrent.CountDownLatch(1)
    val gateNextReload = new java.util.concurrent.atomic.AtomicBoolean(false)
    val sup = new EngineSupervisor(() => launch(), log = line => {
      if line.contains("reloading last model after restart") && gateNextReload.compareAndSet(true, false) then
        entered.countDown()
        require(release.await(20, java.util.concurrent.TimeUnit.SECONDS), "Startup gate was not released")
    })
    var oldStartup: scala.concurrent.Future[?] = null
    var explicitRestart: scala.concurrent.Future[?] = null
    try
      sup.start()
      await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("models/x.safetensors"))))
      gateNextReload.set(true)
      oldStartup = sup.forceRestart()
      assert(entered.await(10, java.util.concurrent.TimeUnit.SECONDS), "Did not reach live remembered-model startup")
      val detachedPid = sup.pid
      assert(detachedPid.nonEmpty)
      assert(!await(new EngineClient(sup).interruptionProtection()).enabled)
      explicitRestart = scala.concurrent.Future(sup.restart())(using scala.concurrent.ExecutionContext.global)
      // The old initializer deliberately remains suspended. A replacement must
      // own a new attempt instead of awaiting that now-detached process's future.
      await(explicitRestart, 3.seconds)
      assert(sup.pid.nonEmpty && sup.pid != detachedPid, "Explicit restart did not install a replacement")
      assert(sup.state.isUsable)
      val replacementPid = sup.pid
      val replacementModel = sup.loadedModel
      assert(replacementModel.nonEmpty)
      release.countDown()
      scala.concurrent.Await.ready(oldStartup, 10.seconds)
      assert(oldStartup.value.exists(_.isFailure), "Detached startup must not report a usable old process")
      assertEquals(sup.pid, replacementPid)
      assert(sup.state.isUsable, "Old startup completion polluted the replacement state")
      assertEquals(sup.loadedModel, replacementModel)
      sup.restart()
      assert(sup.pid.nonEmpty && sup.pid != replacementPid)
      assertEquals(sup.loadedModel.map(_.path), replacementModel.map(_.path), "Remembered model was lost after stale completion")
    finally
      release.countDown()
      Seq(oldStartup, explicitRestart).filter(_ != null).foreach { f =>
        try scala.concurrent.Await.ready(f, 10.seconds)
        catch case _: Throwable => ()
      }
      sup.close()
  }

  test("runtime interrupt follows actual wire order when earlier request serialization is delayed") {
    withServices { s =>
      val entered = new java.util.concurrent.CountDownLatch(1)
      val release = new java.util.concurrent.CountDownLatch(1)
      val gated = new scala.collection.mutable.ArrayBuffer[ujson.Value]:
        override def foreach[U](f: ujson.Value => U): Unit =
          entered.countDown()
          require(release.await(20, java.util.concurrent.TimeUnit.SECONDS), "Serialization gate was not released")
          super.foreach(f)
      gated += ujson.Str("gate")
      // Read the actual owned transport, without replacing or changing its
      // state. requestWithId returns only after the wire write has finished.
      val field = classOf[EngineSupervisor].getDeclaredFields.find(_.getName == "proc").get
      field.setAccessible(true)
      val process = field.get(s.supervisor).asInstanceOf[Option[EngineProcess]].get
      val earlierParams = Json.toJs(GenerationRequest(GenerationParams(seed = 26, steps = 1, width = 64, height = 64),
        s.home.tmpDir(s.settings).toString))
      earlierParams("gate") = new ujson.Arr(gated)
      val earlier = scala.concurrent.Future {
        process.requestWithId(Protocol.Types.Txt2Img, earlierParams)
      }(using scala.concurrent.ExecutionContext.global)
      var earlierResult: scala.concurrent.Future[ujson.Value] = null
      try
        assert(entered.await(10, java.util.concurrent.TimeUnit.SECONDS), "Earlier request did not enter serialization")
        val progressing = new java.util.concurrent.CountDownLatch(1)
        val request = GenerationRequest(GenerationParams(seed = 25, steps = 300, width = 64, height = 64,
          extra = Map("step_delay" -> ".02")), s.home.tmpDir(s.settings).toString)
        val running = s.client.generate(request, GenerationMode.Txt2Img, {
          case _: sdfx.core.protocol.EngineEvent.Progress => progressing.countDown()
          case _ => ()
        })
        assert(progressing.await(10, java.util.concurrent.TimeUnit.SECONDS), "Actual B never began sampling")
        assert(!running.result.isCompleted)
        release.countDown()
        earlierResult = await(earlier)._2 // A is now sent behind the already-active B.
        assert(await(s.interruptRuntime()).interrupted)
        assertEquals(intercept[EngineError](await(running.result)).code, Protocol.ErrorCodes.Interrupted)
        assertEquals(await(earlierResult)("images").arr.size, 1)
      finally
        release.countDown()
        try scala.concurrent.Await.ready(earlier, 10.seconds)
        catch case _: Throwable => ()
    }
  }
