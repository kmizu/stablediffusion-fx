package sdfx.engine

import scala.concurrent.{Await, ExecutionContext, Future, Promise}
import scala.concurrent.duration.Duration
import sdfx.core.protocol.{InterruptionDecision, InterruptionProtection}

/** Batch-owned cancellation; admission waits for an in-flight protection decision. */
final class ExtrasBatchControl(cancelIdle: () => Future[InterruptionDecision],
    readProtection: () => Future[InterruptionProtection] = () => Future.failed(
      new IllegalStateException("Interruption protection state unavailable"))):
  private val lock = new Object
  private var active: Option[(TemporalLabCancelGate, Boolean)] = None
  private var pending: Option[Future[InterruptionDecision]] = None
  private var accepted = false
  private var terminal = false
  private var publishing = false
  def cancellationAccepted: Boolean = lock.synchronized(accepted)
  /** Internal timeout abort: do not admit any later stage even during startup. */
  private[engine] def abortAfterTimeout(): Unit = lock.synchronized { accepted = true }
  def allowDispatch(): Boolean = lock.synchronized(!accepted && !terminal && !publishing && pending.isEmpty)
  private[engine] def attach(gate: TemporalLabCancelGate, finalStage: Boolean = false): Future[Boolean] = lock.synchronized {
    if accepted || terminal || publishing then Future.successful(false)
    else
      active = Some((gate, finalStage))
      pending match
        case None => gate.beforeDispatch()
        case Some(p) => p.flatMap { _ =>
          if cancellationAccepted then Future.successful(false) else gate.beforeDispatch()
        }(using ExecutionContext.parasitic)
  }
  private[engine] def detach(gate: TemporalLabCancelGate): Unit = lock.synchronized {
    active.filter(_._1 eq gate).foreach { (_, finalStage) =>
      active = None
      if finalStage then terminal = true
    }
  }
  def cancel(): Future[InterruptionDecision] = lock.synchronized {
    if terminal then
      Future(readProtection())(using ExecutionContext.global).flatMap(identity)(using ExecutionContext.parasitic)
        .map { protection =>
          if protection.enabled then
            InterruptionDecision(false, true, protection, affected = false,
              message = Some("Interruptions are blocked; turn off Prevent interruption to continue"))
          else
            InterruptionDecision(true, false, protection, affected = false,
              message = Some("Extras batch has already finished"))
        }(using ExecutionContext.parasitic)
    else if publishing then
      Future(readProtection())(using ExecutionContext.global).flatMap(identity)(using ExecutionContext.parasitic)
        .map { protection =>
          if protection.enabled then
            InterruptionDecision(false, true, protection, affected = false,
              message = Some("Interruptions are blocked; turn off Prevent interruption to continue"))
          else
            InterruptionDecision(false, false, protection, affected = false,
              message = Some("Output publication is in progress; retry cancellation after it completes."), tooLate = true)
        }(using ExecutionContext.parasitic)
    else pending match
      case Some(p) => p
      case None =>
        val promise = Promise[InterruptionDecision]()
        pending = Some(promise.future)
        val finalActive = active.exists(_._2)
        val result = active match
          case Some((gate, _)) => gate.cancel()
          case None => Future(cancelIdle())(using ExecutionContext.global).flatMap(identity)(using ExecutionContext.parasitic)
        result.onComplete { value => lock.synchronized {
          value.foreach(d => if d.accepted && !d.blocked && (d.affected || !finalActive) then accepted = true)
          pending = None
          promise.tryComplete(value)
        }}(using ExecutionContext.parasitic)
        promise.future
  }
  /** Claim publication atomically with cancel admission; never hold lock during filesystem I/O. */
  def publishSafely[A](body: => A): A =
    var claimed = false
    while !claimed do
      val waiting = lock.synchronized {
        if accepted then throw new IllegalStateException("Extras cancelled")
        if terminal then throw new IllegalStateException("Extras batch already finished")
        if publishing then throw new IllegalStateException("Extras publication already in progress")
        pending match
          case Some(p) => Some(p)
          case None =>
            publishing = true
            claimed = true
            None
      }
      waiting.foreach(p => Await.result(p, Duration.Inf))
    try body
    finally lock.synchronized { publishing = false }

  def finish(): Unit = lock.synchronized { terminal = true }
