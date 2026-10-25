package sdfx.ui

import javafx.application.Platform

import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.{ExecutorService, Executors, ThreadFactory}
import scala.concurrent.{ExecutionContext, Future}
import scala.util.{Failure, Success}

/** FX thread helpers: never block the FX thread, deliver results back on it. */
object Fx:
  private val counter = new AtomicInteger()
  private val pool: ExecutorService = Executors.newCachedThreadPool((r => {
    val t = new Thread(r, s"sdfx-ui-bg-${counter.incrementAndGet()}")
    t.setDaemon(true)
    t
  }): ThreadFactory)

  /** Background executor for blocking work started from the UI. */
  given bg: ExecutionContext = ExecutionContext.fromExecutorService(pool)

  /** Runs `f` on the FX thread (immediately when already on it). */
  def run(f: => Unit): Unit =
    if Platform.isFxApplicationThread then f else Platform.runLater(() => f)

  /** Always defers `f` to a later FX pulse. */
  def later(f: => Unit): Unit = Platform.runLater(() => f)

  /** Runs `task` in the background and calls back on the FX thread. */
  def async[T](task: => T)(onSuccess: T => Unit)(onError: Throwable => Unit): Unit =
    onFx(Future(task))(onSuccess)(onError)

  /** Future completion delivered on the FX thread. */
  def onFx[T](f: Future[T])(onSuccess: T => Unit)(onError: Throwable => Unit): Unit =
    f.onComplete {
      case Success(v) => run(onSuccess(v))
      case Failure(e) => run(onError(e))
    }(using ExecutionContext.parasitic)

  /** Root cause message suitable for a notification. */
  def message(e: Throwable): String =
    var c = e
    while (c.getMessage == null || c.isInstanceOf[java.util.concurrent.ExecutionException]) && c.getCause != null && (c.getCause ne c) do
      c = c.getCause
    Option(c.getMessage).filter(_.nonEmpty).getOrElse(c.getClass.getSimpleName)
