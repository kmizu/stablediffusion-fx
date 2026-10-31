package sdfx.ui.presets

import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean
import scala.util.control.NonFatal

/** All callbacks are delivered on the supplied UI dispatcher; late operations cannot replace newer state. */
final class ModelPresetAsync(execute: (() => Unit) => Unit, dispatch: (() => Unit) => Unit,
    busy: Boolean => Unit, failure: Throwable => Unit):
  private var serial = 0L
  private var active = new AtomicBoolean(false)
  private var disposed = false
  def cancel(): Unit = active.set(true)
  def dispose(): Unit =
    active.set(true); disposed = true; serial += 1
  def run[A](work: (() => Unit) => A)(accept: A => Unit): Unit =
    if disposed then return
    active.set(true)
    active = new AtomicBoolean(false)
    val token = active
    serial += 1
    val ticket = serial
    def check(): Unit = if token.get then throw new CancellationException("Operation cancelled")
    busy(true)
    execute(() => {
      val result = try
        check()
        val value = work(() => check())
        // Work checks before publication. A later cancel cannot undo a committed file.
        Right(value)
      catch case NonFatal(e) => Left(e)
      dispatch(() => if !disposed && serial == ticket then
        busy(false)
        result match
          case Right(value) => accept(value)
          case Left(error) => failure(error)
      )
    })
