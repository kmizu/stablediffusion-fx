package sdfx.ui

import sdfx.core.config.Settings
import sdfx.ui.vm.{SingleFlight, VramModes}

class Round2Suite extends munit.FunSuite:

  test("SingleFlight releases the gate when start throws synchronously and reports the error") {
    val g = new SingleFlight
    var err: Option[String] = None
    assert(g.run(_ => throw new IllegalStateException("copy failed"), e => err = Some(e.getMessage)))
    assertEquals(err, Some("copy failed"))
    assert(!g.isBusy)
    var ran = false
    assert(g.run(finish => { ran = true; finish() }, _ => ()))
    assert(ran && !g.isBusy)
  }

  test("SingleFlight blocks while in flight and is released by the async callback (also after errors)") {
    val g = new SingleFlight
    var pending: () => Unit = () => ()
    assert(g.run(finish => pending = finish, _ => ()))
    assert(g.isBusy)
    assert(!g.run(_ => fail("must not start while busy"), _ => ()))
    pending() // async error path calls finish too
    assert(!g.isBusy)
    pending() // idempotent
    assert(!g.isBusy)
  }

  test("VRAM modes: engine values only, legacy values mapped") {
    assertEquals(VramModes.Modes.map(_._1), Seq("gpu", "model_offload", "sequential_offload"))
    assertEquals(VramModes.normalize("medvram"), "model_offload")
    assertEquals(VramModes.normalize("lowvram"), "sequential_offload")
    assertEquals(VramModes.normalize("cpu"), "sequential_offload")
    assertEquals(VramModes.normalize("model_offload"), "model_offload")
    assertEquals(VramModes.normalize("bogus"), "gpu")
    assertEquals(VramModes.index("lowvram"), 2)
    assertEquals(VramModes.normalizeSettings(Settings(vramMode = "medvram")).vramMode, "model_offload")
  }
