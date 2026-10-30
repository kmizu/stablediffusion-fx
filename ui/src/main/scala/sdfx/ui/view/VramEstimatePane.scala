package sdfx.ui.view

import javafx.animation.{KeyFrame, Timeline}
import javafx.scene.control.Label
import javafx.util.Duration
import sdfx.core.model.GenerationParams
import sdfx.ui.Fx

/** One request at a time, only for a visible generation form and an idle runtime. */
final class VramEstimatePane(ctx: UiContext, params: () => GenerationParams):
  val node = new Label("VRAM estimate: calibrate the loaded model in VRAM Estimator.")
  node.setWrapText(true)
  private var active = false
  private var closed = false
  private var busy = false
  private var epoch = 0L
  private val timer = new Timeline(new KeyFrame(Duration.seconds(2), _ => refresh()))
  timer.setCycleCount(javafx.animation.Animation.INDEFINITE)
  def setActive(value: Boolean): Unit =
    active = value
    epoch += 1
    if value then { timer.play(); refresh() } else timer.stop()
  private def refresh(): Unit =
    if !closed && active && !busy && !ctx.vramEstimator.isBenchmarking && ctx.services.queue.isIdle then
      val generation = params()
      val token = epoch
      busy = true
      import scala.concurrent.ExecutionContext.Implicits.global
      val query = scala.concurrent.Future(ctx.vramEstimator.history()).flatMap { records =>
        ctx.vramEstimator.profile().flatMap { current =>
          val fingerprint = current.obj.get("current_fingerprint").filter(_ != ujson.Null).map(_.str)
          fingerprint.flatMap(value => sdfx.engine.VramEstimatorHistory.newestMeasured(records, value)) match
            case None => scala.concurrent.Future.successful(ujson.Obj("available" -> false, "reason" -> "no_matching_calibration"))
            case Some(record) =>
              val mode = if generation.isImg2Img then "img2img" else if !generation.hires.enabled then "txt2img"
                else if generation.hires.upscaler.toLowerCase.startsWith("latent") then "txt2img_hr_latent" else "txt2img_hr"
              val (width, height) = if generation.hires.enabled && !generation.isImg2Img then generation.hires.targetSize(generation.width, generation.height)
                else (generation.width, generation.height)
              ctx.vramEstimator.estimate(record, mode, width, height, generation.batchSize)
        }
      }
      Fx.onFx(query) { result =>
        busy = false
        if !closed && active && token == epoch && generation == params() then
          if result("available").bool then
            val total = result("estimated_total_mb").num
            val capacity = result("total_mb").num
            val extrapolation = if result("extrapolated").bool then " · extrapolated" else ""
            node.setText(f"Vanilla VRAM estimate: $total%.0f / $capacity%.0f MiB$extrapolation · ${generation.width}×${generation.height}, batch ${generation.batchSize} · extensions excluded; cache may affect baseline")
          else node.setText(s"VRAM estimate unavailable: ${result("reason").str.replace('_', ' ')}")
      } { error =>
        busy = false
        if !closed && active && token == epoch then node.setText(s"VRAM estimate unavailable: ${Fx.message(error)}")
      }
  def dispose(): Unit = { closed = true; epoch += 1; timer.stop() }
