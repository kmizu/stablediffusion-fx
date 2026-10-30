package sdfx.ui

import sdfx.ui.vm.LatestPreview
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.{ExecutionContext, Promise}
import scala.util.Success

class LatestPreviewSuite extends munit.FunSuite:
  given ExecutionContext = ExecutionContext.parasitic

  test("only one preview runs and intermediate revisions are discarded") {
    val started = ArrayBuffer.empty[(Int, Promise[String])]
    val delivered = ArrayBuffer.empty[(Long, String)]
    val discarded = ArrayBuffer.empty[String]
    val work = new LatestPreview[Int, String](n => {
      val p = Promise[String](); started += ((n, p)); p.future
    }, (revision, result) => result match
      case Success(value) => delivered += ((revision, value))
      case _ => (), discarded += _)
    work.request(1); work.request(2); val latest = work.request(3)
    assertEquals(started.map(_._1).toSeq, Seq(1))
    started.head._2.success("old")
    assertEquals(discarded.toSeq, Seq("old"))
    assertEquals(started.map(_._1).toSeq, Seq(1, 3))
    started.last._2.success("new")
    assertEquals(delivered.toSeq, Seq(latest -> "new"))
    assert(work.isCurrent(latest))
    work.invalidate()
    assert(!work.isCurrent(latest))
  }

  test("closing discards an in-flight result and prevents new work") {
    val promise = Promise[String]()
    val discarded = ArrayBuffer.empty[String]
    val work = new LatestPreview[Int, String](_ => promise.future, (_, _) => fail("unexpected delivery"), discarded += _)
    work.request(1); work.close(); promise.success("result")
    assertEquals(discarded.toSeq, Seq("result"))
    intercept[IllegalStateException](work.request(2))
  }
