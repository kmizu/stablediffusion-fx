package sdfx.ui.view

import javafx.animation.{KeyFrame, Timeline}
import javafx.scene.control.CheckBox
import javafx.scene.layout.HBox
import javafx.util.Duration
import sdfx.core.protocol.InterruptionProtection
import sdfx.ui.Fx

/** Runtime-only presentation. Every protected action is admitted by the owned server, never this cache. */
final class InterruptionControls(ctx: UiContext):
  private val toggle = new CheckBox("Prevent interruption")
  toggle.setId("prevent-interruption-toggle")
  toggle.setSelected(false)
  private val status = W.label("Off", "hint")
  status.setId("prevent-interruption-status")
  private var disposed = false
  private var changing = false
  private var polling = false
  private var updating = false
  private var process: Option[Long] = None
  private var current = InterruptionProtection()
  private var revision = -1L
  W.tip(toggle, "Protect active work from Interrupt, API/MCP cancellation and explicit engine restart. Turn off to allow them. Runtime-only; pending queue removal and application shutdown remain available.")
  val node = new HBox(5, toggle, status)
  node.getStyleClass.add("interruption-controls")
  private def render(value: InterruptionProtection): Unit =
    updating = true
    toggle.setSelected(value.enabled)
    toggle.setAccessibleText(s"Prevent interruption. ${if value.enabled then "On" else "Off"}")
    updating = false
    status.setText(if value.enabled then "On" else "Off")
  private def resetStopped(): Unit =
    process = None
    revision = -1
    current = InterruptionProtection()
    render(current)
  private def publish(value: InterruptionProtection, requestedProcess: Option[Long]): Unit =
    if disposed then return
    val now = ctx.services.supervisor.pid
    // A response initiated before a process was known cannot identify its owner after replacement.
    // Never give it the current PID/revision namespace; ask that known process for a fresh snapshot.
    if requestedProcess.isEmpty || requestedProcess != now then
      refresh()
      return
    if now != process then
      process = now
      revision = -1
    if value.revision >= revision then
      current = value
      revision = value.revision
      render(value)
  def refresh(): Unit =
    if !disposed && !changing && !polling then
      if !ctx.services.supervisor.isRunning then resetStopped()
      else
        polling = true
        val pid = ctx.services.supervisor.pid
        Fx.onFx(ctx.services.client.interruptionProtection()) { state =>
          polling = false
          if !changing then publish(state, pid)
        } { _ =>
          polling = false
          if !disposed then status.setText("Unavailable")
        }
  toggle.setOnAction(_ =>
    if !updating && !disposed then
      val value = toggle.isSelected
      val pid = ctx.services.supervisor.pid
      changing = true
      toggle.setDisable(true)
      status.setText("Updating…")
      Fx.onFx(ctx.services.client.interruptionProtection(Some(value))) { state =>
        changing = false
        if !disposed then
          toggle.setDisable(false)
          publish(state, pid)
          refresh()
      } { error =>
        changing = false
        if !disposed then
          toggle.setDisable(false)
          // Rollback is only a display of an already known same-process snapshot, never a publication
          // of cached state as though it belonged to a newly started process.
          if process.nonEmpty && process == ctx.services.supervisor.pid then render(current)
          status.setText("Unavailable")
          ctx.error("Cannot update interruption protection: " + Fx.message(error))
          refresh()
      }
  )
  private val timer = new Timeline(new KeyFrame(Duration.seconds(5), _ => refresh()))
  timer.setCycleCount(javafx.animation.Animation.INDEFINITE)
  timer.play()
  private val removeState = ctx.services.supervisor.addStateListener(_ => Fx.run(refresh()))
  refresh()
  def dispose(): Unit =
    disposed = true
    timer.stop()
    removeState()
