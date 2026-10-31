package sdfx.mcp

import java.nio.file.Path
import sdfx.core.model.{ControlNetUnit, HiresParams, LoraRef}

/** Validated, catalog-bound request fields shared by all generation entry points. */
object ExtendedGeneration:
  final case class Fields(subseed: Long, subseedStrength: Double, eta: Option[Double], loras: Seq[LoraRef], controlnet: Seq[ControlNetUnit])
  val keys = Seq("subseed", "subseed_strength", "eta", "manual_loras", "controlnet")
  val hiresKeys = Seq("enabled", "scale", "upscaler", "steps", "denoising_strength", "resize_x", "resize_y", "prompt", "negative_prompt", "cfg_scale", "sampler_name", "scheduler")

  def parse(a: Args, lora: String => String, control: String => String, preprocessor: String => String,
      image: (String, String) => Path, allowInitImageFallback: Boolean = false): Fields =
    val refs = a.arr("manual_loras").getOrElse(Nil)
    if refs.size > 32 then throw ToolError("manual_loras supports at most 32 entries")
    val loras = refs.map {
      case obj: ujson.Obj =>
        val v = Args(obj, Seq("name", "weight", "unet", "lbw", "start", "stop", "dyn"))
        val start = v.num("start", 0, 1); val stop = v.num("stop", 0, 1)
        if start.isDefined && stop.isDefined && start.get > stop.get then throw ToolError("LoRA start must not exceed stop")
        val lbw = v.str("lbw").map(_.trim).filter(_.nonEmpty)
        if lbw.exists(_.length > 16384) then throw ToolError("LoRA lbw is too long")
        LoraRef(lora(v.reqStr("name")), v.num("weight", -10, 10).getOrElse(1), v.num("unet", -10, 10), lbw,
          start, stop, v.int("dyn", 1, 65536))
      case _ => throw ToolError("manual_loras entries must be objects")
    }
    val units = a.arr("controlnet").getOrElse(Nil)
    if units.size > 8 then throw ToolError("controlnet supports at most 8 entries")
    val controls = units.map {
      case obj: ujson.Obj =>
        val v = Args(obj, Seq("enabled", "model", "image", "preprocessor", "weight", "guidance_start", "guidance_end", "preprocessor_args"))
        val enabled = v.bool("enabled").getOrElse(true)
        val model = v.str("model").filter(_.nonEmpty).map(control).getOrElse(if enabled then throw ToolError("ControlNet model required") else "")
        val input = v.str("image").filter(_.nonEmpty).map(s => image(s, "controlnet.image").toString)
        val start = v.num("guidance_start", 0, 1).getOrElse(0.0); val end = v.num("guidance_end", 0, 1).getOrElse(1.0)
        if start > end then throw ToolError("ControlNet guidance_start must not exceed guidance_end")
        val params = v.objOpt("preprocessor_args").map { obj =>
          if obj.value.size > 32 then throw ToolError("too many preprocessor arguments")
          obj.value.toSeq.map { (k, value) =>
            if k.isEmpty || k.length > 64 then throw ToolError("invalid preprocessor argument name")
            value match
              case ujson.Num(n) if n.isFinite && math.abs(n) <= 1000000 => k -> n
              case _ => throw ToolError(s"preprocessor_args.$k must be a finite number")
          }.toMap
        }.getOrElse(Map.empty[String, Double])
        ControlNetUnit(enabled, model, input, preprocessor(v.str("preprocessor").getOrElse("none")),
          v.num("weight", 0, 2).getOrElse(1), start, end, params)
      case _ => throw ToolError("controlnet entries must be objects")
    }
    if !allowInitImageFallback && controls.exists(c => c.enabled && c.image.isEmpty) then
      throw ToolError("ControlNet image required for txt2img; img2img may use init_image")
    Fields(a.long("subseed", -1, 4294967295L).getOrElse(-1L), a.num("subseed_strength", 0, 1).getOrElse(0),
      a.num("eta", 0, 10), loras, controls)

  def withHires(a: Args, base: HiresParams, sampler: String => String, scheduler: String => String): HiresParams =
    base.copy(prompt = a.str("prompt").getOrElse(""), negativePrompt = a.str("negative_prompt").getOrElse(""),
      cfgScale = a.num("cfg_scale", 0, 30), samplerName = a.str("sampler_name").map(sampler), scheduler = a.str("scheduler").map(scheduler))

  val schema: Seq[(String, ujson.Value)] = Seq(
    "subseed" -> Schema.int("Variation seed (-1 = random)", Some(-1), Some(4294967295L)),
    "subseed_strength" -> Schema.num("Variation strength", Some(0), Some(1)),
    "eta" -> Schema.num("Sampler eta", Some(0), Some(10)),
    "manual_loras" -> Schema.arr(Schema.obj(Seq("name" -> Schema.str(), "weight" -> Schema.num("", Some(-10), Some(10)),
      "unet" -> Schema.num("", Some(-10), Some(10)), "lbw" -> Schema.str(), "start" -> Schema.num("", Some(0), Some(1)),
      "stop" -> Schema.num("", Some(0), Some(1)), "dyn" -> Schema.int("", Some(1), Some(65536))), Seq("name")), maxItems = Some(32)),
    "controlnet" -> Schema.arr(Schema.obj(Seq("enabled" -> Schema.bool(), "model" -> Schema.str(), "image" -> Schema.str(),
      "preprocessor" -> Schema.str(), "weight" -> Schema.num("", Some(0), Some(2)), "guidance_start" -> Schema.num("", Some(0), Some(1)),
      "guidance_end" -> Schema.num("", Some(0), Some(1)), "preprocessor_args" -> ujson.Obj("type" -> "object", "additionalProperties" -> ujson.Obj("type" -> "number")))) , maxItems = Some(8))
  )
