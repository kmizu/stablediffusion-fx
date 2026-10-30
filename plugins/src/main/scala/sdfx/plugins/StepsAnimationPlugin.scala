package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Independent native implementation of documented Steps Animation behavior. */
final class StepsAnimationPlugin extends Plugin:
  val id = "steps_animation"
  val name = "Steps Animation"
  val kind = PluginKind.Script
  override val requiresEngineFeatures = Seq(id)
  override val description = "Create a movie from denoised intermediate steps during one generation."
  override val controls: Seq[Control] = Seq(
    Note("scope", "Requires ffmpeg on PATH and batch size/count of one. Each denoised frame uses the VAE decoder. Cannot run with Save intermediate images."),
    Slider("duration", "Duration (seconds)", 0.1, 120, 0.1, 5),
    IntField("fps", "Output frames per second", 1, 240, 30),
    Choice("codec", "Media codec", Seq("h264", "h265", "vp9", "av1", "prores"), "h264"),
    Choice("interpolation", "Interpolation", Seq("none", "blend", "mci"), "blend"),
    IntField("skip_initial", "Skip initial steps", 0, 10000, 0),
    Check("run_incomplete", "Export captured frames after interruption or failure"),
    Note("partial_help", "Partial export is bounded to 60 seconds. Interrupt again to stop it; the job remains cancelled and captured frames remain available on export failure."),
    Check("debug", "Log each captured step"),
    Check("create_output", "Create animation file", true),
    Check("delete_intermediate", "Delete captured images after successful export"),
    Text("frames_dir", "Frames folder (relative to output)", "steps-animation/frames"),
    Text("output_dir", "Video folder (relative to output)", "steps-animation/videos")
  )
  def settings(v: Values): ujson.Obj = ujson.Obj(
    "duration" -> v.double("duration", 5), "fps" -> v.int("fps", 30),
    "codec" -> v.str("codec", "h264"), "interpolation" -> v.str("interpolation", "blend"),
    "skip_initial" -> v.int("skip_initial"), "create_output" -> v.bool("create_output", true),
    "delete_intermediate" -> v.bool("delete_intermediate"),
    "run_incomplete" -> v.bool("run_incomplete"), "debug" -> v.bool("debug"),
    "frames_dir" -> v.str("frames_dir", "steps-animation/frames"),
    "output_dir" -> v.str("output_dir", "steps-animation/videos"))
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    require(p.batchSize == 1 && p.nIter == 1, "Steps Animation requires batch size/count of one")
    val config = settings(v)
    JobOutput(ctx.generate(p.copy(hooks = p.hooks.updated(id, config),
      extra = p.extra.updated("Steps animation", config.render()))))
