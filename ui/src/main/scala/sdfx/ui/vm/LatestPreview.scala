package sdfx.ui.vm

import scala.concurrent.{ExecutionContext, Future}
import scala.util.{Failure, Success, Try}
import scala.util.control.NonFatal

/** One running request and at most one replacement; old successful artifacts are discarded. */
final class LatestPreview[A, B](start: A => Future[B], deliver: (Long, Try[B]) => Unit, discard: B => Unit)
    (using ec: ExecutionContext):
  private var revision = 0L
  private var running = false
  private var pending: Option[(Long, A)] = None
  private var closed = false

  def request(input: A): Long = synchronized {
    if closed then throw new IllegalStateException("Preview is closed")
    revision += 1
    val id = revision
    pending = Some(id -> input)
    launchPending()
    id
  }

  def invalidate(): Unit = synchronized { revision += 1; pending = None }
  def close(): Unit = synchronized { closed = true; revision += 1; pending = None }
  def isCurrent(id: Long): Boolean = synchronized { !closed && revision == id }

  private def launchPending(): Unit =
    if !running && !closed then
      pending.foreach { (id, input) =>
        pending = None
        running = true
        val result = try start(input) catch case NonFatal(e) => Future.failed(e)
        result.onComplete { outcome =>
          val current = synchronized { !closed && revision == id }
          try
            if current then deliver(id, outcome)
            else outcome match
              case Success(value) => discard(value)
              case Failure(_) => ()
          finally synchronized { running = false; launchPending() }
        }
      }
