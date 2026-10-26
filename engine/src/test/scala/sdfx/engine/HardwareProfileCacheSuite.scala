package sdfx.engine

import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.duration.*

/** Deterministic metadata-reply ownership tests; no backend/model is launched. */
class HardwareProfileCacheSuite extends munit.FunSuite:
  test("offline profile lookup performs no request and does not cache invented process identity") {
    var pid: Option[Long] = None
    var calls = 0
    val cache = new HardwareProfileCache(() => pid, _ => { calls += 1; Future.successful("actual profile") })
    assertEquals(Await.result(cache.profile(), 1.second), "unknown")
    assertEquals(calls, 0)
    pid = Some(3L)
    assertEquals(Await.result(cache.profile(), 1.second), "actual profile")
    assertEquals(Await.result(cache.profile(), 1.second), "actual profile")
    assertEquals(calls, 1)
  }

  test("delayed old-process metadata cannot overwrite a newer cached profile or poison its revision-free namespace") {
    var pid: Option[Long] = Some(1L)
    val first = Promise[String]()
    val second = Promise[String]()
    var calls = 0
    val cache = new HardwareProfileCache(() => pid, _ => { calls += 1; if calls == 1 then first.future else second.future })
    val old = cache.profile()
    pid = Some(2L)
    val current = cache.profile()
    second.success("new hardware")
    assertEquals(Await.result(current, 1.second), "new hardware")
    first.success("old hardware")
    intercept[IllegalStateException](Await.result(old, 1.second))
    assertEquals(Await.result(cache.profile(), 1.second), "new hardware")
    assertEquals(calls, 2)
  }

  test("failed profile query may be retried instead of permanently caching transport failure") {
    var calls = 0
    val cache = new HardwareProfileCache(() => Some(1L), _ =>
      calls += 1
      if calls == 1 then Future.failed(new IllegalArgumentException("temporary transport failure"))
      else Future.successful("known hardware"))
    intercept[IllegalArgumentException](Await.result(cache.profile(), 1.second))
    assertEquals(Await.result(cache.profile(), 1.second), "known hardware")
    assertEquals(calls, 2)
  }
