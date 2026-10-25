package sdfx.engine

import sdfx.core.protocol.*

import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import scala.concurrent.duration.*
import scala.concurrent.{Await, Future}
import scala.jdk.CollectionConverters.*
import scala.util.Try

class EngineProcessSuite extends munit.FunSuite:
  import FakeEngine.*

  override val munitTimeout: Duration = 90.seconds

  private val logs = new ConcurrentLinkedQueue[String]()

  private def withEngine[T](extra: String*)(body: EngineProcess => T): T =
    val ep = EngineProcess.start(launch(extra*), logs.add(_))
    try body(ep)
    finally ep.kill()

  test("handshake and ping") {
    withEngine("--tag", "x") { ep =>
      assertEquals(ep.hello.engineVersion, "fake-x")
      assertEquals(ep.hello.protocol, 1)
      assert(ep.hello.dryRun)
      assertEquals(ep.hello.features, Seq("fake"))
      assertEquals(await(ep.request("ping"))("pong").bool, true)
    }
    assert(logs.asScala.exists(_.contains("fake engine starting")), "stderr lines go to the log listener")
  }

  test("hello timeout") {
    val t0 = System.nanoTime()
    val e = intercept[EngineError](EngineProcess.start(launch("--hello-delay", "15").copy(helloTimeout = 1.second)))
    assertEquals(e.code, EngineProcess.EngineTimeout)
    assert((System.nanoTime() - t0).nanos < 10.seconds)
  }

  test("exit before hello fails the start with stderr detail") {
    val e = intercept[EngineError](EngineProcess.start(launch("--crash-on-start")))
    assertEquals(e.code, EngineProcess.EngineStartFailed)
    assert(e.detail.contains("boom at start"), e.detail)
  }

  test("concurrent requests are correlated by id even when answered out of order") {
    withEngine() { ep =>
      val delays = Seq(400, 50, 250, 0, 120)
      val fs = delays.zipWithIndex.map((d, i) => ep.request("echo", ujson.Obj("delay_ms" -> d, "value" -> s"v$i")))
      fs.zipWithIndex.foreach((f, i) => assertEquals(await(f)("value").str, s"v$i"))
    }
  }

  test("very long lines and unicode survive") {
    withEngine() { ep =>
      val big = "あ" * 300000 + "x" * 1500000
      assertEquals(await(ep.request("echo", ujson.Obj("value" -> big)))("value").str, big)
    }
  }

  test("events are delivered before the result") {
    withEngine() { ep =>
      await(ep.request(Protocol.Types.LoadModel, ujson.Obj("path" -> "m.safetensors")))
      val events = new ConcurrentLinkedQueue[EngineEvent]()
      val dir = java.nio.file.Files.createTempDirectory("sdfx-ev-")
      val req = GenerationRequest(sdfx.core.model.GenerationParams(seed = 1, steps = 3, width = 64, height = 64), dir.toString)
      val r = Json.fromJs[GenerationResult](await(ep.request(Protocol.Types.Txt2Img, Json.toJs(req), events.add(_))))
      assertEquals(r.images.size, 1)
      val evs = events.asScala.toList
      assertEquals(evs.collect { case p: EngineEvent.Progress => p.step }, List(1, 2, 3))
      assert(evs.exists(_.isInstanceOf[EngineEvent.Preview]))
      deleteTree(dir)
    }
  }

  test("error replies map to EngineError") {
    withEngine() { ep =>
      val e = intercept[EngineError](await(ep.request("fail", ujson.Obj("code" -> "oom", "message" -> "out of memory"))))
      assertEquals(e.code, Protocol.ErrorCodes.Oom)
      assertEquals(e.message, "out of memory")
      assertEquals(e.detail, "trace here")
      val u = intercept[EngineError](await(ep.request("no_such_type")))
      assertEquals(u.code, Protocol.ErrorCodes.Unsupported)
    }
  }

  test("non-JSON stdout is logged, not fatal") {
    withEngine() { ep =>
      await(ep.request("stdout_noise"))
      assert(logs.asScala.exists(_.contains("this is not json")))
      assertEquals(await(ep.request("ping"))("pong").bool, true)
    }
  }

  test("crash fails all pending requests and reports the exit") {
    val exit = new AtomicReference[(Int, Boolean)]()
    val latch = new CountDownLatch(1)
    val ep = EngineProcess.start(launch(), _ => (), (c, exp) => { exit.set((c, exp)); latch.countDown() })
    val slow = ep.request("echo", ujson.Obj("delay_ms" -> 10000, "value" -> 1))
    val crash = ep.request("crash")
    val e1 = intercept[EngineError](await(slow))
    val e2 = intercept[EngineError](await(crash))
    assertEquals(e1.code, EngineProcess.EngineCrashed)
    assertEquals(e2.code, EngineProcess.EngineCrashed)
    assert(e1.detail.contains("crashing on purpose"), e1.detail)
    assert(latch.await(10, TimeUnit.SECONDS))
    assertEquals(exit.get(), (3, false))
    assert(!ep.isAlive)
    val after = intercept[EngineError](await(ep.request("ping")))
    assertEquals(after.code, EngineProcess.EngineNotRunning)
  }

  test("close sends shutdown and reports an expected exit") {
    val exit = new AtomicReference[(Int, Boolean)]()
    val ep = EngineProcess.start(launch(), _ => (), (c, exp) => exit.set((c, exp)))
    ep.close(3.seconds)
    assert(eventually()(exit.get() != null))
    assertEquals(exit.get(), (0, true))
  }
