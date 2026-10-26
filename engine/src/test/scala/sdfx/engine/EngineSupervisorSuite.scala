package sdfx.engine

import sdfx.core.model.GenerationParams
import sdfx.core.protocol.*

import java.nio.file.Files
import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import scala.concurrent.ExecutionContext
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class EngineSupervisorSuite extends munit.FunSuite:
  import FakeEngine.*

  override val munitTimeout: Duration = 120.seconds

  private def supervisor(maxRestarts: Int = 3, grace: FiniteDuration = 10.seconds) =
    new EngineSupervisor(() => launch(), _ => (), maxRestarts = maxRestarts, interruptGrace = grace)

  private def enginePid(sup: EngineSupervisor): Long = await(sup.request("get_state"))("pid").num.toLong

  private def slowGen(sup: EngineSupervisor, extra: Map[String, String]) =
    val dir = Files.createTempDirectory("sdfx-sup-")
    val req = GenerationRequest(GenerationParams(seed = 3, steps = 100, width = 64, height = 64, extra = extra), dir.toString)
    sup.request(Protocol.Types.Txt2Img, Json.toJs(req))

  test("crash -> automatic restart -> last model re-issued") {
    val sup = supervisor()
    val states = new ConcurrentLinkedQueue[EngineState]()
    sup.addStateListener(states.add(_))
    try
      sup.start()
      assert(sup.state.isInstanceOf[EngineState.Ready])
      val m = Json.fromJs[LoadedModel](await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("models/x.safetensors")))))
      assertEquals(m.name, "x")
      assertEquals(sup.loadedModel.map(_.path), Some("models/x.safetensors"))
      val pid1 = enginePid(sup)
      val pending = sup.request("echo", ujson.Obj("delay_ms" -> 10000))
      intercept[EngineError](await(sup.request("crash")))
      val pe = intercept[EngineError](await(pending))
      assertEquals(pe.code, EngineProcess.EngineCrashed)
      assert(eventually()(sup.state.isInstanceOf[EngineState.Ready] && sup.pid.exists(_ != pid1)), s"state ${sup.state}")
      val st = await(sup.request("get_state"))
      assertEquals(st("loaded").str, "models/x.safetensors")
      assertNotEquals(st("pid").num.toLong, pid1)
      val seen = states.asScala.toList
      assert(seen.exists(_.isInstanceOf[EngineState.Crashed]), seen.toString)
      assert(seen.contains(EngineState.Starting), seen.toString)
    finally sup.close()
  }

  test("gives up after too many crashes") {
    val sup = supervisor(maxRestarts = 1)
    try
      sup.start()
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(sup.state.isInstanceOf[EngineState.Ready]))
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(sup.state.isInstanceOf[EngineState.Crashed]))
      Thread.sleep(1000)
      assert(sup.state.isInstanceOf[EngineState.Crashed], sup.state.toString)
      // An explicit start recovers.
      sup.start()
      assert(sup.state.isInstanceOf[EngineState.Ready])
    finally sup.close()
  }

  test("busy while generating, cooperative interrupt") {
    val sup = supervisor()
    try
      sup.start()
      await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("m.safetensors"))))
      val f = slowGen(sup, Map("step_delay" -> "0.05"))
      assert(eventually()(sup.state == EngineState.Busy))
      Thread.sleep(200)
      sup.interrupt()
      val e = intercept[EngineError](await(f, 5.seconds))
      assert(e.isInterrupted, e.toString)
      assert(eventually()(sup.state.isInstanceOf[EngineState.Ready]))
    finally sup.close()
  }

  test("interrupt force-restarts an engine that does not stop") {
    val sup = supervisor(grace = 1.second)
    try
      sup.start()
      await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("m.safetensors"))))
      val pid1 = enginePid(sup)
      val f = slowGen(sup, Map("step_delay" -> "0.2", "ignore_interrupt" -> "true"))
      Thread.sleep(300)
      val t0 = System.nanoTime()
      sup.interrupt()
      val e = intercept[EngineError](await(f, 10.seconds))
      assert(e.isInterrupted, e.toString)
      assert((System.nanoTime() - t0).nanos < 6.seconds)
      assert(eventually()(sup.state.isInstanceOf[EngineState.Ready] && sup.pid.exists(_ != pid1)))
      assertEquals(await(sup.request("get_state"))("loaded").str, "m.safetensors")
    finally sup.close()
  }

  test("stop is not treated as a crash") {
    val sup = supervisor()
    sup.start()
    sup.stop()
    Thread.sleep(500)
    assertEquals(sup.state, EngineState.Stopped)
    val e = intercept[EngineError](await(sup.request("ping")))
    assertEquals(e.code, EngineProcess.EngineNotRunning)
    sup.close()
  }

  test("default policy: three restarts, then gives up") {
    val sup = new EngineSupervisor(() => launch(), _ => ())
    try
      sup.start()
      for i <- 1 to 3 do
        val before = sup.pid
        intercept[EngineError](await(sup.request("crash")))
        assert(eventually()(sup.state.isInstanceOf[EngineState.Ready] && sup.pid.isDefined && sup.pid != before), s"restart $i: ${sup.state}")
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(sup.state.isInstanceOf[EngineState.Crashed]))
      Thread.sleep(800)
      assert(sup.state.isInstanceOf[EngineState.Crashed], sup.state.toString)
      val e = intercept[EngineError](await(sup.request("ping")))
      assertEquals(e.code, EngineProcess.EngineCrashed)
    finally sup.close()
  }

  test("old crashes expire from the restart window") {
    val sup = new EngineSupervisor(() => launch(), _ => (), maxRestarts = 1, restartWindow = 1500.millis)
    try
      sup.start()
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(sup.state.isInstanceOf[EngineState.Ready]))
      Thread.sleep(2000)
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(sup.state.isInstanceOf[EngineState.Ready] && sup.isRunning), sup.state.toString)
    finally sup.close()
  }

  test("crash while re-loading the model after a restart triggers another restart") {
    val launches = new java.util.concurrent.atomic.AtomicInteger(0)
    // launch #2 dies when the remembered model is re-issued; launch #3 is healthy
    val sup = new EngineSupervisor(() => if launches.incrementAndGet() == 2 then launch("--crash-on-load") else launch(), _ => ())
    try
      sup.start()
      await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("m.safetensors"))))
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(launches.get() >= 3 && sup.state.isInstanceOf[EngineState.Ready] && sup.isRunning), s"${sup.state} launches=${launches.get()}")
      assertEquals(await(sup.request("get_state"))("loaded").str, "m.safetensors")
    finally sup.close()
  }

  test("a stale interrupt timer does not kill the next generation") {
    val sup = supervisor(grace = 1.second)
    try
      sup.start()
      await(sup.request(Protocol.Types.LoadModel, Json.toJs(LoadModelRequest("m.safetensors"))))
      val pid1 = enginePid(sup)
      val a = slowGen(sup, Map("step_delay" -> "0.05"))
      Thread.sleep(150)
      sup.interrupt()
      intercept[EngineError](await(a, 5.seconds))
      // B runs well past A's grace period
      val b = slowGen(sup, Map("step_delay" -> "0.025"))
      val r = Json.fromJs[GenerationResult](await(b, 20.seconds))
      assertEquals(r.images.size, 1)
      assertEquals(enginePid(sup), pid1)
    finally sup.close()
  }

  test("stop during start leaves no engine running") {
    val sup = new EngineSupervisor(() => launch("--hello-delay", "1"), _ => ())
    val f = sup.startAsync()
    Thread.sleep(200)
    sup.stop()
    intercept[Exception](await(f, 20.seconds))
    Thread.sleep(300)
    assert(!sup.isRunning)
    assertEquals(sup.state, EngineState.Stopped)
    sup.close()
  }

  test("stop() then start() never returns the stale start future") {
    val sup = new EngineSupervisor(() => launch("--hello-delay", "1"), _ => ())
    try
      val old = sup.startAsync()
      Thread.sleep(200)
      sup.stop()
      val h = sup.start()
      assertEquals(h.engineVersion, "fake-0")
      assert(sup.isRunning)
      assert(sup.state.isInstanceOf[EngineState.Ready], sup.state.toString)
      intercept[Exception](await(old, 20.seconds))
      assertEquals(await(sup.request("ping"))("pong").bool, true)
    finally sup.close()
  }

  test("a fatal Error while launching fails the start instead of hanging") {
    val sup = new EngineSupervisor(() => throw new NoClassDefFoundError("missing/Thing"), _ => ())
    val t0 = System.nanoTime()
    intercept[Throwable](sup.start())
    assert((System.nanoTime() - t0).nanos < 10.seconds)
    assert(sup.state.isInstanceOf[EngineState.Crashed], sup.state.toString)
    sup.close()
  }

  test("start failure state is published before callers can observe the failure") {
    val readingFailure = new CountDownLatch(1)
    val releaseFailure = new CountDownLatch(1)
    val observedFailure = new CountDownLatch(1)
    val observedState = new AtomicReference[EngineState]()
    val controlled = new RuntimeException("controlled launch failure"):
      override def getMessage: String =
        readingFailure.countDown()
        if !releaseFailure.await(10, TimeUnit.SECONDS) then throw new IllegalStateException("Failure publication gate was not released")
        super.getMessage
    val sup = new EngineSupervisor(() => throw controlled, _ => ())
    try
      val failed = sup.startAsync()
      failed.onComplete { _ =>
        observedState.set(sup.state)
        observedFailure.countDown()
      }(using ExecutionContext.parasitic)
      assert(readingFailure.await(10, TimeUnit.SECONDS), "Launch failure never reached state publication")
      // Hold the exact publication step, rather than hope the caller wins a scheduler race.
      // A start result must remain pending until its failure state can be constructed/published.
      assert(!observedFailure.await(1, TimeUnit.SECONDS), "Start failure escaped while Crashed state publication was blocked")
      releaseFailure.countDown()
      intercept[RuntimeException](await(failed))
      assert(observedFailure.await(10, TimeUnit.SECONDS))
      assert(observedState.get.isInstanceOf[EngineState.Crashed], observedState.get.toString)
      assert(sup.state.isInstanceOf[EngineState.Crashed], sup.state.toString)
    finally
      releaseFailure.countDown()
      sup.close()
  }

  test("model reload timeout after a restart reports Crashed instead of Ready") {
    val sup = new EngineSupervisor(() => launch(), _ => (), reloadTimeout = 1.second)
    try
      sup.start()
      // the fake engine sleeps `delay` seconds in load_model; the first load may take as long as it likes
      await(sup.request(Protocol.Types.LoadModel, ujson.Obj("path" -> "slow.safetensors", "delay" -> 3.0)))
      intercept[EngineError](await(sup.request("crash")))
      assert(eventually()(sup.state match
        case EngineState.Crashed(m) => m.contains("reloading")
        case _                      => false
      ), sup.state.toString)
      assert(!sup.isRunning)
    finally sup.close()
  }

  test("loadedModel is updated before the caller sees the load result") {
    val sup = supervisor()
    val client = new EngineClient(sup)
    try
      sup.start()
      for i <- 1 to 20 do
        val m = await(client.loadModel(LoadModelRequest(s"m$i.safetensors")))
        assertEquals(sup.loadedModel.map(_.path), Some(m.path))
    finally sup.close()
  }
