package sdfx.engine

import sdfx.core.protocol.*
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent.duration.*

/** Passive hardware lookups cannot create or migrate engine ownership. */
class HardwareStatusOwnershipSuite extends munit.FunSuite:
  import FakeEngine.*

  private def withSupervisor(check: (EngineSupervisor, AtomicInteger) => Unit): Unit =
    val launches = new AtomicInteger()
    val sup = new EngineSupervisor(() => { launches.incrementAndGet(); launch() }, _ => ())
    try check(sup, launches)
    finally sup.close()

  test("passive hardware status never starts an offline engine") {
    withSupervisor { (sup, launches) =>
      val error = intercept[EngineError](await(sup.systemStatusIfRunning(123L), 5.seconds))
      assertEquals(error.code, EngineProcess.EngineNotRunning)
      assertEquals(launches.get(), 0)
      assertEquals(sup.pid, None)
    }
  }

  test("captured hardware lookup cannot restart a process that stopped before query admission") {
    withSupervisor { (sup, launches) =>
      sup.start()
      val captured = sup.pid.get
      val cache = new HardwareProfileCache(() => sup.pid, expectedPid =>
        assertEquals(expectedPid, captured)
        sup.stop() // Deterministically place shutdown between capture and dispatch.
        sup.systemStatusIfRunning(expectedPid).map(_("hardware_info").str)(using scala.concurrent.ExecutionContext.parasitic))
      intercept[Exception](await(cache.profile(), 5.seconds))
      assertEquals(launches.get(), 1)
      assertEquals(sup.pid, None)
      assertEquals(sup.state, EngineState.Stopped)
    }
  }

  test("stale hardware PID is rejected while current PID can inspect the existing process") {
    withSupervisor { (sup, launches) =>
      sup.start()
      val oldPid = sup.pid.get
      sup.stop()
      sup.start()
      val currentPid = sup.pid.get
      assertNotEquals(currentPid, oldPid)
      val stale = intercept[EngineError](await(sup.systemStatusIfRunning(oldPid), 5.seconds))
      assertEquals(stale.code, EngineProcess.EngineNotRunning)
      val status = await(sup.systemStatusIfRunning(currentPid), 5.seconds)
      val actualChild = await(sup.request("get_state"), 5.seconds)("pid").num.toLong
      assertEquals(status("pid").num.toLong, actualChild)
      assertEquals(status("hardware_info").str, "Fixture GPU, Fixture CPU, 32GB RAM")
      assertEquals(launches.get(), 2)
    }
  }

  test("timed out hardware status releases the cache for a bounded retry on the same process") {
    val launches = new AtomicInteger()
    val sup = new EngineSupervisor(() => { launches.incrementAndGet(); launch("--hardware-status-delay-once", "0.25") }, _ => ())
    try
      sup.start()
      val pid = sup.pid
      var queries = 0
      val cache = new HardwareProfileCache(() => sup.pid, expectedPid =>
        queries += 1
        val timeout = if queries == 1 then 50.millis else 2.seconds
        sup.systemStatusIfRunning(expectedPid, timeout).map(_("hardware_info").str)(using scala.concurrent.ExecutionContext.parasitic))
      val error = intercept[EngineError](await(cache.profile(), 3.seconds))
      assertEquals(error.code, EngineProcess.EngineTimeout)
      assertEquals(await(cache.profile(), 3.seconds), "Fixture GPU, Fixture CPU, 32GB RAM")
      assertEquals(queries, 2)
      assertEquals(launches.get(), 1)
      assertEquals(sup.pid, pid)
    finally sup.close()
  }
