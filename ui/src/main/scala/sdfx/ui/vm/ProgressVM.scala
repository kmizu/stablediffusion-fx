package sdfx.ui.vm

import javafx.beans.property.SimpleObjectProperty
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.JobEvent

/** Progress of one sampling pass. */
final case class PassProgress(step: Int = 0, total: Int = 0):
  def fraction: Double = if total <= 0 then 0.0 else math.min(1.0, step.toDouble / total)
  def isDone: Boolean = total > 0 && step >= total
  def text: String = if total <= 0 then "" else s"$step/$total"

enum Outcome:
  case Finished, Failed, Cancelled

/** Everything the status bar shows, derived from the job event stream by [[ProgressVM.reduce]]. */
final case class ProgressState(
    current: Option[String] = None,
    pending: Vector[String] = Vector.empty,
    base: PassProgress = PassProgress(),
    hires: PassProgress = PassProgress(),
    hiresSeen: Boolean = false,
    image: Int = 0,
    images: Int = 1,
    batchSize: Int = 1,
    etaS: Double = 0,
    status: String = "",
    done: Int = 0,
    total: Int = 0,
    preview: Option[String] = None,
    previewSeq: Int = 0,
    saved: Vector[OutputImage] = Vector.empty,
    outcome: Option[Outcome] = None,
    error: Option[String] = None,
    message: String = "Idle"
):
  def running: Boolean = current.isDefined
  def pendingCount: Int = pending.size

  /** "Step 12/20 · image 2/4 · ETA 0:05" */
  def stepText: String =
    if !running then ""
    else
      val parts = Seq(
        Option.when(base.total > 0)(s"Step ${base.text}"),
        Option.when(hiresSeen && hires.total > 0)(s"Hires ${hires.text}"),
        Option.when(images > 1)(
          if batchSize > 1 then
            s"Batch ${image / batchSize + 1}/${(images + batchSize - 1) / batchSize} · images ${image + 1}–${math.min(images, image + batchSize)}/$images"
          else s"image ${image + 1}/$images"),
        Option.when(etaS > 0)(s"ETA ${ProgressVM.formatEta(etaS)}")
      ).flatten
      parts.mkString(" · ")

  /** Script status ("X/Y/Z 3/12"). */
  def statusText: String =
    if status.isEmpty then "" else if total > 0 then s"$status $done/$total" else status

object ProgressVM:
  def formatEta(s: Double): String =
    val secs = math.max(0, math.round(s)).toInt
    if secs >= 3600 then f"${secs / 3600}%d:${secs % 3600 / 60}%02d:${secs % 60}%02d" else f"${secs / 60}%d:${secs % 60}%02d"

  /** Pure reducer of the job event stream. */
  def reduce(s: ProgressState, e: JobEvent): ProgressState = e match
    case _: JobEvent.PayloadCaptured | _: JobEvent.PayloadLifecycle => s
    case JobEvent.JobQueued(job, _) =>
      if s.current.contains(job.id) || s.pending.contains(job.id) then s else s.copy(pending = s.pending :+ job.id)
    case JobEvent.JobStarted(job) =>
      ProgressState(current = Some(job.id), pending = s.pending.filterNot(_ == job.id), previewSeq = s.previewSeq, message = "Generating…")
    case JobEvent.Progress(id, p, text, done, total) if s.current.contains(id) =>
      val withPass =
        if p.pass == "hires" then s.copy(hires = PassProgress(p.step, p.total), hiresSeen = true)
        else
          // a new image starts with the base pass again: reset the finished hires bar
          val hires = if s.hires.isDone then PassProgress(0, s.hires.total) else s.hires
          s.copy(base = PassProgress(p.step, p.total), hires = hires)
      val st = if text.nonEmpty then withPass.copy(status = text, done = done, total = total) else withPass
      st.copy(image = p.image, images = math.max(1, p.images), batchSize = math.max(1, p.batchSize), etaS = p.etaS)
    case JobEvent.Status(id, text, done, total) if s.current.contains(id) =>
      s.copy(status = text, done = done, total = total)
    case JobEvent.Preview(id, b64, _) if s.current.contains(id) =>
      s.copy(preview = Some(b64), previewSeq = s.previewSeq + 1)
    case JobEvent.ImageSaved(id, img, _) if s.current.contains(id) =>
      s.copy(saved = s.saved :+ img)
    case JobEvent.JobFinished(id, out) if s.current.contains(id) =>
      val n = out.images.size
      s.copy(
        current = None,
        base = if s.base.total > 0 then s.base.copy(step = s.base.total) else s.base,
        hires = if s.hiresSeen && s.hires.total > 0 then s.hires.copy(step = s.hires.total) else s.hires,
        etaS = 0,
        preview = None,
        outcome = Some(Outcome.Finished),
        message = s"Done · $n image${if n == 1 then "" else "s"}"
      )
    case f @ JobEvent.JobFailed(id, _) if s.current.contains(id) =>
      val msg = f.message
      s.copy(current = None, etaS = 0, preview = None, outcome = Some(Outcome.Failed), error = Some(msg), message = s"Failed: $msg")
    case JobEvent.JobFailed(id, _) =>
      s.copy(pending = s.pending.filterNot(_ == id))
    case JobEvent.JobCancelled(id, partial) if s.current.contains(id) =>
      val n = partial.images.size
      s.copy(current = None, etaS = 0, preview = None, outcome = Some(Outcome.Cancelled),
        message = if n > 0 then s"Interrupted · $n image${if n == 1 then "" else "s"} saved" else "Interrupted")
    case JobEvent.JobCancelled(id, _) =>
      s.copy(pending = s.pending.filterNot(_ == id))
    case _ => s

/** Observable progress state. `accept` must be called on the FX thread when views are bound to it. */
final class ProgressVM:
  val state = new SimpleObjectProperty[ProgressState](ProgressState())
  def accept(e: JobEvent): Unit = state.set(ProgressVM.reduce(state.get, e))
  def get: ProgressState = state.get
