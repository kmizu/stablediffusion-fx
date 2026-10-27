package sdfx.engine

import sdfx.core.protocol.{InterruptionDecision, InterruptionProtection}
import scala.concurrent.{ExecutionContext, Future, Promise}
import scala.util.{Failure, Try}

/** Owns cancellation admission for one request. Never writes to the engine on the caller's thread. */
private[engine] final class TemporalLabCancelGate(
    owner: String,
    capture: String => (() => Future[InterruptionDecision])
):
  private val lock = new Object
  private enum Phase:
    case Before, Admitting, Writing, Sent, Terminal
  private var phase = Phase.Before
  private var pending: Option[Promise[InterruptionDecision]] = None

  private def unaffected = InterruptionDecision(accepted = true, blocked = false,
    protection = InterruptionProtection(), affected = false)

  /** Requests cancellation; blocked/error results are observable and retryable. */
  def cancel(): Future[InterruptionDecision] = lock.synchronized {
    if phase == Phase.Terminal then Future.successful(unaffected)
    else pending match
      case Some(current) => current.future
      case None =>
        val result = Promise[InterruptionDecision]()
        pending = Some(result)
        if phase == Phase.Sent then launch(result, None)
        result.future
  }

  /** Before writing, admit an earlier cancellation against the live engine's
    * protection policy. True reserves dispatch; false means no work may be sent.
    */
  def beforeDispatch(): Future[Boolean] = lock.synchronized {
    if phase != Phase.Before then Future.failed(new IllegalStateException("temporal request already admitted"))
    else pending match
      case None =>
        phase = Phase.Writing
        Future.successful(true)
      case Some(result) =>
        phase = Phase.Admitting
        val admission = Promise[Boolean]()
        launch(result, Some(admission))
        admission.future
  }

  /** Called only after the owned work envelope is written to the process pipe. */
  def dispatched(): Unit = lock.synchronized {
    if phase == Phase.Writing || phase == Phase.Before then
      phase = Phase.Sent
      pending.foreach(p => launch(p, None))
  }

  /** A terminal request cannot cause a later cancellation to target other work. */
  def completed(): Unit = lock.synchronized {
    val old = phase
    phase = Phase.Terminal
    if old == Phase.Before || old == Phase.Writing then
      pending.foreach(_.trySuccess(unaffected))
      pending = None
  }

  // capture snapshots the particular owner/process/request. Calling its returned
  // action can acquire a blocked writer; always perform that action on a worker.
  private def launch(result: Promise[InterruptionDecision], admission: Option[Promise[Boolean]]): Unit =
    Try(capture(owner)) match
      case Failure(error) =>
        pending = None
        admission.foreach { p => phase = Phase.Terminal; p.tryFailure(error) }
        result.tryFailure(error)
      case scala.util.Success(action) =>
        Future(action())(using ExecutionContext.global)
          .flatMap(identity)(using ExecutionContext.parasitic)
          .onComplete { value =>
            lock.synchronized {
              if pending.contains(result) then pending = None
              admission match
                case None => result.tryComplete(value)
                case Some(p) => value match
                  case Failure(error) =>
                    phase = Phase.Terminal
                    p.tryFailure(error)
                    result.tryFailure(error)
                  case scala.util.Success(decision) if decision.blocked =>
                    phase = Phase.Writing
                    result.trySuccess(decision)
                    p.trySuccess(true)
                  case scala.util.Success(decision) if decision.accepted =>
                    phase = Phase.Terminal
                    result.trySuccess(decision.copy(affected = true))
                    p.trySuccess(false)
                  case scala.util.Success(_) =>
                    val error = new IllegalStateException("invalid interruption admission")
                    phase = Phase.Terminal
                    p.tryFailure(error)
                    result.tryFailure(error)
            }
          }(using ExecutionContext.parasitic)
