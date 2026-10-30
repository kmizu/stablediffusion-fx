package sdfx.engine

import munit.FunSuite
import sdfx.core.protocol.*
import sdfx.core.model.GenerationParams
import scala.concurrent.Await
import scala.concurrent.duration.*

class EmbeddingMergeCancellationSuite extends FunSuite:
  test("cancel before startup prevents embedding merge file operation dispatch") {
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
      val operation = new EngineClient(sup).embeddingMergeOwned(EmbeddingMergeRequest(text = "cat"))
      val decision = operation.interrupt()
      release.countDown()
      Await.result(startup, 20.seconds)
      assert(Await.result(decision, 20.seconds).affected)
      assert(intercept[EngineError](Await.result(operation.result, 20.seconds)).isInterrupted)
      assert(!Await.result(sup.request("get_state"), 20.seconds)("requests").arr.exists(_.str == "embedding_merge"))
    finally
      release.countDown()
      sup.close()
  }

  test("cancel queued embedding merge copy leaves unrelated generation running") {
    import FakeEngine.*
    val sup = new EngineSupervisor(() => launch())
    val output = java.nio.file.Files.createTempDirectory("embedding merge-other-generation-")
    try
      Await.result(sup.startAsync(), 20.seconds)
      val client = new EngineClient(sup)
      val generation = client.generate(GenerationRequest(GenerationParams(steps = 30, width = 64, height = 64,
        extra = Map("step_delay" -> "0.08", "allow_no_model" -> "true")), output.toString), GenerationMode.Txt2Img)
      assert(eventually(5.seconds)(Await.result(sup.request("get_state"), 5.seconds)("requests").arr.exists(_.str == "txt2img")))
      val copy = client.embeddingMergeOwned(EmbeddingMergeRequest(text = "cat", operation = "save", name = "new"))
      assert(eventually(5.seconds)(sup.processSnapshot.exists(_.pendingTypes.contains(Protocol.Types.EmbeddingMerge))))
      val decision = Await.result(copy.interrupt(), 5.seconds)
      assert(decision.accepted && decision.affected)
      assert(Await.result(generation.result, 20.seconds).images.nonEmpty)
      assert(intercept[EngineError](Await.result(copy.result, 20.seconds)).isInterrupted)
      assert(!Await.result(copy.interrupt(), 5.seconds).affected)
    finally
      sup.close()
      deleteTree(output)
  }
