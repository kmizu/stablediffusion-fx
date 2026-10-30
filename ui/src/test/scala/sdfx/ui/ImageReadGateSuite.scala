package sdfx.ui

import sdfx.core.png.ImageScrambleSession
import sdfx.ui.vm.ImageReadGate
import java.nio.file.Path
import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.concurrent.ExecutionContext

class ImageReadGateSuite extends munit.FunSuite:
  test("new requests and session changes discard delayed image results") {
    given ExecutionContext = ExecutionContext.global
    val session = new ImageScrambleSession
    session.unlock("first")
    val started = new CountDownLatch(1); val release = new CountDownLatch(1); val done = new CountDownLatch(1); val delivered = new CountDownLatch(2)
    val gate = new ImageReadGate[String](session, run => { run(); delivered.countDown() }, (p, _) => {
      if p.toString == "slow" then { started.countDown(); release.await(10, TimeUnit.SECONDS) }
      p.toString
    })
    val seen = new java.util.concurrent.CopyOnWriteArrayList[String]()
    gate.request(Path.of("slow"))(_.foreach(seen.add))
    assert(started.await(10, TimeUnit.SECONDS))
    gate.request(Path.of("new")) { r => r.foreach(seen.add); done.countDown() }
    assert(done.await(10, TimeUnit.SECONDS)); release.countDown()
    assert(delivered.await(10, TimeUnit.SECONDS))
    assertEquals(seen.toArray.toSeq, Seq("new"))

    val started2 = new CountDownLatch(1); val release2 = new CountDownLatch(1); val finished = new CountDownLatch(1)
    val other = new ImageReadGate[String](session, run => { run(); finished.countDown() }, (_, _) => { started2.countDown(); release2.await(10, TimeUnit.SECONDS); "stale" })
    other.request(Path.of("image"))(_.foreach(seen.add))
    assert(started2.await(10, TimeUnit.SECONDS)); session.lock(); release2.countDown()
    assert(finished.await(10, TimeUnit.SECONDS))
    assertEquals(seen.toArray.toSeq, Seq("new"))
  }
