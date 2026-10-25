package sdfx.ui

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.{JobOutput, OutputImage}
import sdfx.core.protocol.EngineEvent
import sdfx.engine.{Job, JobEvent}
import sdfx.ui.vm.*

import java.nio.file.Paths

class ProgressVMSuite extends munit.FunSuite:
  private val j1 = Job("j1", GenerationParams())
  private val j2 = Job("j2", GenerationParams())
  private def img(n: Int) = OutputImage(Paths.get(s"out-$n.png"), GenerationParams(), s"info $n", n.toLong)
  private def prog(step: Int, total: Int, pass: String = "base", image: Int = 0, images: Int = 1, eta: Double = 0) =
    JobEvent.Progress("j1", EngineEvent.Progress(step, total, pass, eta, image, images), "", 0, 0)

  private def run(events: JobEvent*): ProgressState =
    val vm = new ProgressVM
    events.foreach(vm.accept)
    vm.get

  test("queued and started jobs track pending count") {
    val s = run(JobEvent.JobQueued(j1, 1), JobEvent.JobQueued(j2, 2))
    assertEquals(s.pendingCount, 2)
    assert(!s.running)
    val s2 = ProgressVM.reduce(s, JobEvent.JobStarted(j1))
    assertEquals(s2.current, Some("j1"))
    assertEquals(s2.pending, Vector("j2"))
    assertEquals(s2.message, "Generating…")
  }

  test("base and hires passes, preview, finished") {
    val s = run(
      JobEvent.JobQueued(j1, 1),
      JobEvent.JobStarted(j1),
      prog(5, 20, eta = 12.4),
      JobEvent.Preview("j1", "AAAA", 5),
      prog(20, 20),
      prog(3, 10, "hires", eta = 3)
    )
    assertEquals(s.base, PassProgress(20, 20))
    assertEquals(s.hires, PassProgress(3, 10))
    assert(s.hiresSeen)
    assertEquals(s.preview, Some("AAAA"))
    assertEquals(s.previewSeq, 1)
    assertEquals(s.stepText, "Step 20/20 · Hires 3/10 · ETA 0:03")
    assertEquals(s.hires.fraction, 0.3)

    val s2 = Seq(
      prog(10, 10, "hires"),
      prog(1, 20, image = 1, images = 2) // second image: the finished hires bar resets
    ).foldLeft(s)(ProgressVM.reduce)
    assertEquals(s2.hires, PassProgress(0, 10))
    assertEquals(s2.stepText, "Step 1/20 · Hires 0/10 · image 2/2")

    val done = Seq(
      JobEvent.ImageSaved("j1", img(1), grid = false),
      JobEvent.ImageSaved("j1", img(2), grid = false),
      JobEvent.JobFinished("j1", JobOutput(Seq(img(1), img(2))))
    ).foldLeft(s2)(ProgressVM.reduce)
    assert(!done.running)
    assertEquals(done.saved.size, 2)
    assertEquals(done.outcome, Some(Outcome.Finished))
    assertEquals(done.message, "Done · 2 images")
    assertEquals(done.preview, None)
    assertEquals(done.base.fraction, 1.0)
    assertEquals(done.stepText, "")
  }

  test("script status text") {
    val s = run(
      JobEvent.JobStarted(j1),
      JobEvent.Status("j1", "X/Y/Z", 3, 12),
      JobEvent.Progress("j1", EngineEvent.Progress(2, 20), "X/Y/Z", 4, 12)
    )
    assertEquals(s.statusText, "X/Y/Z 4/12")
  }

  test("failed job keeps the error") {
    val s = run(JobEvent.JobStarted(j1), prog(3, 20), JobEvent.JobFailed("j1", new RuntimeException("[oom] out of memory")))
    assert(!s.running)
    assertEquals(s.outcome, Some(Outcome.Failed))
    assertEquals(s.error, Some("[oom] out of memory"))
    assert(s.message.startsWith("Failed:"))
  }

  test("cancelled: running job and pending job") {
    val running = run(JobEvent.JobQueued(j1, 1), JobEvent.JobStarted(j1), JobEvent.JobQueued(j2, 1), prog(4, 20),
      JobEvent.JobCancelled("j1", JobOutput(Seq(img(1)))))
    assert(!running.running)
    assertEquals(running.outcome, Some(Outcome.Cancelled))
    assertEquals(running.message, "Interrupted · 1 image saved")
    assertEquals(running.pending, Vector("j2"))
    val pend = ProgressVM.reduce(running, JobEvent.JobCancelled("j2", JobOutput(Nil)))
    assertEquals(pend.pending, Vector.empty)
    assertEquals(pend.outcome, Some(Outcome.Cancelled)) // unchanged by the pending cancel
  }

  test("events of other jobs are ignored") {
    val s = run(JobEvent.JobStarted(j1), JobEvent.Progress("other", EngineEvent.Progress(9, 9), "", 0, 0), JobEvent.Preview("other", "x", 1))
    assertEquals(s.base, PassProgress())
    assertEquals(s.preview, None)
  }

  test("eta formatting") {
    assertEquals(ProgressVM.formatEta(0), "0:00")
    assertEquals(ProgressVM.formatEta(65.4), "1:05")
    assertEquals(ProgressVM.formatEta(3725), "1:02:05")
  }
