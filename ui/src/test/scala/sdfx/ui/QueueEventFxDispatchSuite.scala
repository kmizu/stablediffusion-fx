package sdfx.ui

import sdfx.core.model.GenerationParams
import sdfx.core.payload.{PayloadKey, PayloadSnapshot, PayloadState}
import sdfx.core.plugins.api.OutputImage
import sdfx.core.protocol.EngineEvent
import sdfx.engine.{Job, JobEvent}

import java.nio.file.Paths
import scala.collection.mutable.ArrayBuffer

/** A blocked FX scheduler exposes accidental retention of diagnostic bodies in queued closures. */
class QueueEventFxDispatchSuite extends munit.FunSuite:
  test("a burst of capture and lifecycle diagnostics queues no FX closures") {
    val exact = "x" * (1024 * 1024)
    val key = PayloadKey("script-1", 1)
    val captured = JobEvent.PayloadCaptured(PayloadSnapshot(key, "txt2img", Some(exact), 1L,
      state = PayloadState.Submitted, sourceMode = Some("txt2img")))
    val lifecycle = JobEvent.PayloadLifecycle(key, PayloadState.Succeeded, Some(exact))
    val queued = ArrayBuffer.empty[() => Unit]
    val ordinary = ArrayBuffer.empty[JobEvent]
    val images = ArrayBuffer.empty[JobEvent.ImageSaved]
    val post: (() => Unit) => Unit = task => { queued += task; () }
    val onImage: JobEvent.ImageSaved => Unit = e => { images += e; () }
    val onOrdinary: JobEvent => Unit = e => { ordinary += e; () }
    for _ <- 0 until 100 do
      QueueEventFxDispatch.forward(captured, post, onImage, onOrdinary)
      QueueEventFxDispatch.forward(lifecycle, post, onImage, onOrdinary)
    assertEquals(queued.size, 0)
    assertEquals(ordinary.size, 0)
    assertEquals(images.size, 0)
  }

  test("ordinary job and progress events still use FX while images keep their off-FX path") {
    val queued = ArrayBuffer.empty[() => Unit]
    val ordinary = ArrayBuffer.empty[JobEvent]
    val images = ArrayBuffer.empty[JobEvent.ImageSaved]
    val post: (() => Unit) => Unit = task => { queued += task; () }
    val onImage: JobEvent.ImageSaved => Unit = e => { images += e; () }
    val onOrdinary: JobEvent => Unit = e => { ordinary += e; () }
    val events = Vector[JobEvent](
      JobEvent.JobQueued(Job("job-1", GenerationParams()), 1),
      JobEvent.JobStarted(Job("job-1", GenerationParams())),
      JobEvent.Progress("job-1", EngineEvent.Progress(2, 20), "", 0, 0),
      JobEvent.Status("job-1", "script", 1, 2),
      JobEvent.Log("job-1", "ordinary log"))
    events.foreach(e => QueueEventFxDispatch.forward(e, post, onImage, onOrdinary))
    val saved = JobEvent.ImageSaved("job-1",
      OutputImage(Paths.get("out.png"), GenerationParams(), "info", 1L), grid = false)
    QueueEventFxDispatch.forward(saved, post, onImage, onOrdinary)
    assertEquals(queued.size, events.size)
    assertEquals(ordinary.size, 0) // The scheduler is deliberately blocked.
    assertEquals(images.toVector, Vector(saved))
    queued.foreach(_())
    assertEquals(ordinary.toVector, events)
  }
