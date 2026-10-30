package sdfx.plugins

import sdfx.core.infotext.Img2ImgHiresMetadata
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object Img2ImgHiresFixPlugin:
  def replay(fields: Map[String, String]): Option[Values] =
    fields.get(Img2ImgHiresMetadata.key) match
      case None => Some(Values(Map("enabled" -> "false")))
      case Some(text) => Img2ImgHiresMetadata.parse(text).map(h => Values(h.value.toMap.map { (k, v) =>
        k -> (v match
          case ujson.Str(s) if (k == "sampler" && s == "Use same sampler") || (k == "scheduler" && s == "Use same scheduler") => "None"
          case ujson.Str(s) => s
          case ujson.Bool(b) => b.toString
          case ujson.Num(n) => Values.fmt(n)
          case _ => throw new IllegalArgumentException("Invalid Hires value"))
      })).orElse(Some(Values(Map("enabled" -> "false"))))

class Img2ImgHiresFixPlugin extends HookPlugin:
  def id = "img2img_hires_fix"
  def name = "img2img Hires Fix"
  def hook = id
  override def description = "Upscale and refine each completed image in txt2img, img2img or inpaint. Pixel upscale and VAE re-encode; separate from native latent Hires."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("method", "Refines the completed image after postprocessing. Existing native Hires can run first. A zero denoise value only resizes pixels."),
    DynamicChoice("upscaler", "Pixel upscaler", ChoiceSource.Upscalers, "R-ESRGAN 4x+", allowNone = false),
    IntField("steps", "Hires steps (0 = inherit)", 0, 25, 0),
    Slider("denoise", "Denoising strength", 0, 1, .01, .33),
    Slider("ratio", "Upscale by", 0, 4, .05, 2),
    IntField("width", "Resize width (both dimensions override scale)", 0, 2048, 0),
    IntField("height", "Resize height", 0, 2048, 0),
    DynamicChoice("sampler", "Sampler (None = inherit)", ChoiceSource.Samplers, "None"),
    DynamicChoice("scheduler", "Schedule (None = inherit)", ChoiceSource.Schedulers, "None"),
    Slider("cfg", "CFG (0 = inherit)", 0, 30, .5, 0),
    Text("prompt", "Refinement prompt", multiline = true, placeholder = "Empty inherits the first-pass prompt"),
    Text("negative_prompt", "Refinement negative prompt", multiline = true, placeholder = "Empty inherits the first-pass prompt"))
  def hookFields(v: Values): ujson.Obj = Img2ImgHiresMetadata.normalize(ujson.Obj(
    "ratio" -> v.double("ratio", 2), "width" -> v.int("width"), "height" -> v.int("height"),
    "steps" -> v.int("steps"), "upscaler" -> v.str("upscaler", "R-ESRGAN 4x+"),
    "denoise" -> v.double("denoise", .33), "sampler" -> (if v.str("sampler", "None") == "None" then "Use same sampler" else v.str("sampler")),
    "scheduler" -> (if v.str("scheduler", "None") == "None" then "Use same scheduler" else v.str("scheduler")), "cfg" -> v.double("cfg"),
    "prompt" -> v.str("prompt"), "negative_prompt" -> v.str("negative_prompt")))
  def infotextKeys(h: ujson.Obj): Seq[(String, String)] = Seq(Img2ImgHiresMetadata.key -> Img2ImgHiresMetadata.encode(h))
