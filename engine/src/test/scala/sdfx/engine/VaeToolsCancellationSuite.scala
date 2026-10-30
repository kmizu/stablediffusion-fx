package sdfx.engine

import munit.FunSuite
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.*
import scala.concurrent.Await
import scala.concurrent.duration.*

class VaeToolsCancellationSuite extends FunSuite:
  test("cancelling before startup finishes prevents VAE dispatch") {
    import FakeEngine.*
    val entered = new java.util.concurrent.CountDownLatch(1)
    val release = new java.util.concurrent.CountDownLatch(1)
    val sup = new EngineSupervisor(() => {
      entered.countDown()
      if !release.await(10, java.util.concurrent.TimeUnit.SECONDS) then throw new IllegalStateException("startup not released")
      launch()
    })
    try
      val startup = sup.startAsync()
      assert(entered.await(3, java.util.concurrent.TimeUnit.SECONDS))
      val vae = new EngineClient(sup).vaeToolsOwned(ujson.Obj())
      val decision = vae.interrupt()
      release.countDown()
      Await.result(startup, 20.seconds)
      assert(Await.result(decision, 20.seconds).affected)
      assert(intercept[EngineError](Await.result(vae.result, 20.seconds)).isInterrupted)
      assert(!Await.result(sup.request("get_state"), 20.seconds)("requests").arr.exists(_.str == "vae_tools"))
    finally
      release.countDown()
      sup.close()
  }

  test("cancelling queued VAE work leaves an unrelated generation running") {
    import FakeEngine.*
    val sup = new EngineSupervisor(() => launch())
    try
      Await.result(sup.startAsync(), 20.seconds)
      val client = new EngineClient(sup)
      val generation = client.generate(GenerationRequest(GenerationParams(steps = 30, width = 64, height = 64,
        extra = Map("step_delay" -> "0.08", "allow_no_model" -> "true")),
        java.nio.file.Files.createTempDirectory("vae-cancel-other").toString), GenerationMode.Txt2Img)
      assert(eventually(5.seconds)(Await.result(sup.request("get_state"), 5.seconds)("requests").arr.exists(_.str == "txt2img")))
      val vae = client.vaeToolsOwned(ujson.Obj())
      assert(eventually(5.seconds)(sup.processSnapshot.exists(_.pendingTypes.contains(Protocol.Types.VaeTools))))
      val decision = Await.result(vae.interrupt(), 5.seconds)
      assert(decision.accepted && decision.affected)
      assert(Await.result(generation.result, 20.seconds).images.nonEmpty)
      assert(intercept[EngineError](Await.result(vae.result, 20.seconds)).isInterrupted)
      assert(!Await.result(vae.interrupt(), 5.seconds).affected)
    finally sup.close()
  }
