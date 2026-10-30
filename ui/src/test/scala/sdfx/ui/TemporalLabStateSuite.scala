package sdfx.ui

import java.nio.file.Path
import sdfx.core.protocol.*
import sdfx.ui.vm.{TemporalLabForms, TemporalLabState, TemporalPreviewCache, TemporalPreviewSlots}

class TemporalLabStateSuite extends munit.FunSuite:
  private val dfi = TemporalLabDfi("reference", "generated", "output", "run",
    sourceDenoise = 2, tolerance = 7.5, expand = 3, smooth = 5, interDenoise = 7,
    interDenoiseSize = 11, interDenoiseFrequency = 4, interBlur = 2.0,
    refreshFrequency = 6, preserve = 15, frameLimit = 12)

  test("DFI and preview retain every independent control and exact protocol names") {
    val request = TemporalLabForms.dfi(dfi).toOption.get
    assertEquals(request, dfi)
    val p = request.params
    assertEquals(p("operation").str, "dfi")
    assertEquals(p("source_denoise").num, 2.0)
    assertEquals(p("tolerance").num, 7.5)
    assertEquals(p("expand").num, 3.0)
    assertEquals(p("smooth").num, 5.0)
    assertEquals(p("inter_denoise").num, 7.0)
    assertEquals(p("inter_denoise_size").num, 11.0)
    assertEquals(p("inter_denoise_frequency").num, 4.0)
    assertEquals(p("inter_blur").num, 2.0)
    assertEquals(p("refresh_frequency").num, 6.0)
    assertEquals(p("preserve").num, 15.0)
    assertEquals(p("frame_limit").num, 12.0)
    val preview = TemporalLabForms.preview(dfi, 3).toOption.get.params
    assertEquals(preview("operation").str, "preview")
    assertEquals(preview("frame_index").num, 3.0)
    assertEquals(preview("source_denoise").num, 2.0)
    assertEquals(preview("tolerance").num, 7.5)
    assertEquals(preview("expand").num, 3.0)
    assertEquals(preview("smooth").num, 5.0)
    assert(!preview.obj.contains("output_parent"))
  }

  test("other operations preserve typed options without interference") {
    val deflicker = TemporalLabDeflicker("frames", "parent", "out", 0.3, 0.6, true)
    val fuse = TemporalLabFuse("style", "video", "parent", "out", 0.75)
    val extract = TemporalLabExtract("file.mp4", "parent", "out", 12.5)
    assertEquals(TemporalLabForms.deflicker(deflicker), Right(deflicker))
    assertEquals(TemporalLabForms.fuse(fuse), Right(fuse))
    assertEquals(TemporalLabForms.extract(extract), Right(extract))
    assertEquals(deflicker.params("normalize").bool, true)
    assertEquals(fuse.params("strength").num, 0.75)
    assertEquals(extract.params("fps").num, 12.5)
    assert(TemporalLabForms.deflicker(deflicker.copy(blend = 0, overlay = 0, normalize = false)).isRight)
  }

  test("path and numeric validation rejects invalid requests before starting") {
    assert(TemporalLabForms.dfi(dfi.copy(referenceDir = "  ")).isLeft)
    assert(TemporalLabForms.dfi(dfi.copy(outputName = "../escape")).isLeft)
    assert(TemporalLabForms.dfi(dfi.copy(tolerance = Double.NaN)).isLeft)
    assert(TemporalLabForms.dfi(dfi.copy(smooth = 4)).isLeft)
    assert(TemporalLabForms.dfi(dfi.copy(interDenoiseFrequency = 0)).isLeft)
    assert(TemporalLabForms.preview(dfi, 0).isLeft)
    assert(TemporalLabForms.fuse(TemporalLabFuse("s", "v", "o", "n", 0)).isLeft)
    assert(TemporalLabForms.extract(TemporalLabExtract("v", "o", "n", Double.PositiveInfinity)).isLeft)
    assert(TemporalLabForms.deflicker(TemporalLabDeflicker("f", "o", "n", -0.1)).isLeft)
  }

  test("one active job, progress and stale or disposed results") {
    val state = new TemporalLabState
    val ticket = state.start(dfi).toOption.get
    assert(state.busy)
    assert(state.start(dfi).isLeft)
    state.progress(ticket, 2, 10)
    assertEquals(state.progressFraction, 0.2)
    state.invalidate()
    state.complete(ticket, TemporalLabResult("dfi", outputDir = Some("output/run")))
    assert(!state.busy)
    assertEquals(state.result, None)
    val second = state.start(dfi).toOption.get
    state.dispose()
    state.complete(second, TemporalLabResult("dfi", outputDir = Some("output/run")))
    assertEquals(state.result, None)
    assert(state.start(dfi).isLeft)
  }

  test("cancel decision distinguishes blocked, no matching job, and accepted") {
    val state = new TemporalLabState
    val ticket = state.start(dfi).toOption.get
    state.cancelDecision(ticket, InterruptionDecision(accepted = false, blocked = true, protection = InterruptionProtection(true), affected = false))
    assert(state.busy)
    assert(state.status.toLowerCase.contains("blocked"))
    state.cancelDecision(ticket, InterruptionDecision(accepted = true, blocked = false, protection = InterruptionProtection(), affected = false))
    assert(state.busy)
    assert(state.status.toLowerCase.contains("no"))
    state.cancelDecision(ticket, InterruptionDecision(accepted = true, blocked = false, protection = InterruptionProtection(), affected = true))
    assert(state.busy)
    assert(state.status.toLowerCase.contains("request"))
    state.fail(ticket, "interrupted")
    assert(!state.busy)
    assert(state.status.contains("interrupted"))
  }

  test("playback index and bounded cache preserve seek, pause and stop") {
    val player = new TemporalPreviewCache(maxFrames = 2, maxBytes = 10)
    val frames = Vector(Path.of("1.png"), Path.of("2.png"), Path.of("3.png"))
    player.setFrames(frames)
    assertEquals(player.current, Some(frames.head))
    player.play()
    assert(player.playing)
    player.advance()
    assertEquals(player.current, Some(frames(1)))
    player.pause()
    player.advance()
    assertEquals(player.current, Some(frames(1)))
    player.seek(2)
    assertEquals(player.current, Some(frames(2)))
    player.remember(frames(0), Array.fill[Byte](4)(1))
    player.remember(frames(1), Array.fill[Byte](4)(2))
    player.remember(frames(2), Array.fill[Byte](4)(3))
    assert(player.cachedCount <= 2 && player.cachedBytes <= 10)
    player.remember(frames(2), Array.fill[Byte](11)(3))
    assert(player.cachedCount <= 2 && player.cachedBytes <= 10)
    player.stop()
    assert(!player.playing)
    assertEquals(player.current, Some(frames.head))
    player.dispose()
    assertEquals(player.cachedCount, 0)
    assertEquals(player.current, None)
  }

  test("preview slots coalesce pending loads, ignore stale callbacks, and cancel on dispose") {
    val callbacks = scala.collection.mutable.ArrayBuffer.empty[Either[String, String] => Unit]
    val launched = scala.collection.mutable.ArrayBuffer.empty[Path]
    val shown = scala.collection.mutable.ArrayBuffer.empty[String]
    val errors = scala.collection.mutable.ArrayBuffer.empty[String]
    var cancelled = 0
    val slots = new TemporalPreviewSlots[String]((_, path, done) => {
      launched += path; callbacks += done; () => { cancelled += 1 }
    })
    val one = Path.of("one.png"); val two = Path.of("two.png"); val three = Path.of("three.png")
    slots.show("previous", one, shown += _, errors += _)
    slots.show("previous", two, shown += _, errors += _)
    slots.show("previous", three, shown += _, errors += _)
    assertEquals(launched.toVector, Vector(one))
    callbacks(0)(Right("old"))
    assertEquals(shown.toVector, Vector.empty)
    assertEquals(launched.toVector, Vector(one, three))
    callbacks(1)(Right("new"))
    assertEquals(shown.toVector, Vector("new"))
    slots.show("current", one, shown += _, errors += _)
    slots.clear()
    callbacks(2)(Right("stale after clear"))
    assertEquals(shown.toVector, Vector("new"))
    slots.show("current", two, shown += _, errors += _)
    slots.dispose()
    assertEquals(cancelled, 1)
    callbacks(3)(Right("after dispose"))
    assertEquals(shown.toVector, Vector("new"))
  }

  test("preview slot error is visible and later request recovers") {
    val callbacks = scala.collection.mutable.ArrayBuffer.empty[Either[String, String] => Unit]
    val errors = scala.collection.mutable.ArrayBuffer.empty[String]
    val shown = scala.collection.mutable.ArrayBuffer.empty[String]
    val slots = new TemporalPreviewSlots[String]((_, _, done) => { callbacks += done; () => () })
    slots.show("frame", Path.of("bad.png"), shown += _, errors += _)
    callbacks(0)(Left("unreadable"))
    assertEquals(errors.toVector, Vector("unreadable"))
    slots.show("frame", Path.of("good.png"), shown += _, errors += _)
    callbacks(1)(Right("recovered"))
    assertEquals(shown.toVector, Vector("recovered"))
  }

  test("session reset discards a pending old frame and starts the new one after stale completion") {
    val callbacks = scala.collection.mutable.ArrayBuffer.empty[Either[String, String] => Unit]
    val launched = scala.collection.mutable.ArrayBuffer.empty[Path]
    val shown = scala.collection.mutable.ArrayBuffer.empty[String]
    val slots = new TemporalPreviewSlots[String]((_, path, done) => { launched += path; callbacks += done; () => () })
    val old = Path.of("old.png"); val current = Path.of("current.png")
    slots.show("frame", old, shown += _, _ => ())
    slots.clear()
    slots.show("frame", current, shown += _, _ => ())
    assertEquals(launched.toVector, Vector(old))
    callbacks.head(Left("Image access changed"))
    assertEquals(launched.toVector, Vector(old, current))
    assertEquals(shown.toVector, Vector.empty)
    callbacks(1)(Right("current pixels"))
    assertEquals(shown.toVector, Vector("current pixels"))
  }

  test("slow frame loading still presents completed frames under continuous playback ticks") {
    val callbacks = scala.collection.mutable.ArrayBuffer.empty[Either[String, String] => Unit]
    val started = scala.collection.mutable.ArrayBuffer.empty[Path]
    val shown = scala.collection.mutable.ArrayBuffer.empty[String]
    val errors = scala.collection.mutable.ArrayBuffer.empty[String]
    val slots = new TemporalPreviewSlots[String]((_, path, done) => { started += path; callbacks += done; () => () })
    val frames = (1 to 8).map(i => Path.of(s"$i.png")).toVector
    slots.show("frame", frames(0), shown += _, errors += _)
    slots.show("frame", frames(1), shown += _, errors += _)
    slots.show("frame", frames(2), shown += _, errors += _)
    callbacks(0)(Right("frame 1 decoded"))
    assertEquals(shown.toVector, Vector("frame 1 decoded"))
    assertEquals(started.toVector, Vector(frames(0), frames(2)))
    slots.show("frame", frames(3), shown += _, errors += _)
    slots.show("frame", frames(4), shown += _, errors += _)
    callbacks(1)(Left("frame 3 corrupt"))
    assertEquals(errors.toVector, Vector("frame 3 corrupt"))
    assertEquals(started.toVector, Vector(frames(0), frames(2), frames(4)))
    slots.clear()
    slots.show("frame", frames(5), shown += _, errors += _)
    callbacks(2)(Right("stale old revision"))
    assertEquals(shown.toVector, Vector("frame 1 decoded"))
    callbacks(3)(Right("frame 6 decoded"))
    assertEquals(shown.toVector, Vector("frame 1 decoded", "frame 6 decoded"))
  }
