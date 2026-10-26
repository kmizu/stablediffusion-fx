package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Source-resolution VAE encoding and interpolation, from feynlee/latent-upscale (MIT behaviour). */
final class LatentUpscalePlugin extends Plugin:
  val id = "latent_upscale"
  val name = "Latent Upscale"
  val kind = PluginKind.Script
  override val tabs = Set(Tab.Img2Img)
  override val requiresEngineFeatures = Seq(id)
  override val description = "Encode the source image before resizing, interpolate its latents, then denoise for the full sampling-step count."
  override val controls: Seq[Control] = Seq(
    Note("workflow", "Uses the image's original resolution for VAE encoding and the main Width/Height as the latent target. " +
      "Overrides pixel Resize mode. Sampling steps means actual denoising steps, even at lower strength; strength must be above zero. " +
      "Only masked mode keeps the native crop/resize and overlay."),
    Choice("method", "Upscale method", LatentUpscalePlugin.methods, "nearest-exact"),
    Note("method_help", "Nearest-exact and area preserve crisp structures; bilinear/bicubic blend neighbours. " +
      "Antialiasing also helps when reducing a dimension. Upstream linear/trilinear are invalid for 4-D image latents and are not offered."),
    Choice("scheduler", "Noise scheduler", LatentUpscalePlugin.schedules, "simple"),
    Note("schedule_help", "Automatic keeps the main Noise schedule. Simple/Normal and the other named schedules use the model's training sigmas " +
      "with a compatible sampler (Euler and DPM++ support custom sigmas). Other samplers report their native fallback. " +
      "No extra model download is required.")
  )
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p, name)
    val method = v.str("method", "nearest-exact")
    val schedule = v.str("scheduler", "simple")
    require(LatentUpscalePlugin.methods.contains(method), "Choose a 2-D image-latent interpolation method")
    require(LatentUpscalePlugin.schedules.contains(schedule), "Unknown Latent Upscale scheduler")
    require(p.img2img.get.denoisingStrength > 0, "Latent Upscale denoising strength must be above zero")
    val configured = p.copy(hooks = p.hooks.updated(id, ujson.Obj("method" -> method, "scheduler" -> schedule)),
      extra = p.extra ++ Seq("Script" -> name, "Latent upscale method" -> method,
        "Latent upscale scheduler" -> schedule, "Latent upscale fixed steps" -> "True"))
    JobOutput(Support.step(ctx)(ctx.generate(configured)).getOrElse(Nil))

object LatentUpscalePlugin:
  val methods = Seq("bilinear", "bilinear-antialiased", "bicubic", "bicubic-antialiased", "area", "nearest", "nearest-exact")
  val schedules = Seq("simple", "normal", "karras", "exponential", "polyexponential", "automatic")
  def replay(fields: Map[String, String]): Option[Values] =
    for
      method <- fields.get("Latent upscale method").filter(methods.contains)
      schedule <- fields.get("Latent upscale scheduler").filter(schedules.contains)
    yield Values(Map("method" -> method, "scheduler" -> schedule))
