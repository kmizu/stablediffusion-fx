package sdfx.engine

import scala.concurrent.{ExecutionContext, Future, Promise}
import scala.util.{Failure, Try}
import scala.util.control.NonFatal

/** The response belongs to the process queried, never whichever process happens to exist on completion. */
final class HardwareProfileCache(process: () => Option[Long], query: Long => Future[String]):
  private val lock = new Object
  private var cached: Option[(Long, Promise[String])] = None
  def profile(): Future[String] =
    val (future, launch) = lock.synchronized {
      process() match
        case None =>
          cached = None
          (Future.successful("unknown"), None)
        case Some(pid) => cached.filter(_._1 == pid) match
          case Some((_, promise)) => (promise.future, None)
          case None =>
            val promise = Promise[String]()
            cached = Some(pid -> promise)
            (promise.future, Some(pid -> promise))
    }
    launch.foreach { (pid, promise) =>
      val pending = try query(pid) catch case NonFatal(error) => Future.failed(error)
      pending.onComplete { result =>
        val accepted: Try[String] = lock.synchronized {
          val current = cached.exists((_, candidate) => candidate eq promise)
          val value = if current && process().contains(pid) then result
            else Failure(new IllegalStateException("Engine changed during hardware metadata lookup"))
          if value.isFailure && current then cached = None
          value
        }
        promise.tryComplete(accepted)
      }(using ExecutionContext.parasitic)
    }
    future

/** Shared native metadata preview/paste facts. Asking while offline never starts an engine or model. */
final class HardwareInfoService(services: AppServices):
  private val cache = new HardwareProfileCache(() => services.supervisor.pid, pid =>
    services.supervisor.systemStatusIfRunning(pid).map { data =>
      data.obj.get("hardware_info").collect { case ujson.Str(value) if value.nonEmpty => value }.getOrElse("unknown")
    }(using ExecutionContext.parasitic))
  def profile(): Future[String] = cache.profile()
