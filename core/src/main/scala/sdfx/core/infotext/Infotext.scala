package sdfx.core.infotext

import sdfx.core.Version
import sdfx.core.model.*

import scala.collection.immutable.ListMap

/** Per-image facts that are only known after generation. */
case class ImageFacts(
    seed: Long,
    subseed: Long = -1L,
    modelName: Option[String] = None,
    modelHash: Option[String] = None,
    vaeName: Option[String] = None,
    vaeHash: Option[String] = None,
    loraHashes: Seq[(String, String)] = Nil,
    scheduler: Option[String] = None,
    extra: ListMap[String, String] = ListMap.empty
)

/** Parsed infotext: prompt, negative prompt, and the ordered `Key: value` fields (values unquoted). */
case class ParsedInfotext(prompt: String, negativePrompt: String, fields: ListMap[String, String]):
  def get(k: String): Option[String] = fields.get(k)

/** A1111 / reForge compatible "parameters" text, the format embedded in PNG tEXt chunks.
  *
  * {{{
  * prompt line(s)
  * Negative prompt: ...
  * Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 1, Size: 512x512, ...
  * }}}
  */
object Infotext:
  // Same expression as modules/infotext_utils.py re_param_code ((?U): Python 3 `\w`/`\s` are Unicode-aware).
  private val ReParam = """(?U)\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)""".r
  private val ReSize = """^(\d+)x(\d+)$""".r

  def quote(v: String): String =
    if !v.contains(',') && !v.contains('\n') && !v.contains(':') then v
    else ujson.write(ujson.Str(v)) // json.dumps(ensure_ascii=False) equivalent

  def unquote(v: String): String =
    if v.length < 2 || v.head != '"' || v.last != '"' then v
    else
      try ujson.read(v).str
      catch case _: Exception => v

  private def num(d: Double): String =
    if d.isNaN || d.isInfinite then d.toString
    else if d == math.rint(d) && math.abs(d) < 1e15 then d.toLong.toString
    else BigDecimal(d).bigDecimal.stripTrailingZeros.toPlainString

  /** Builds the generation fields in A1111 order. `None` values are omitted. */
  def fields(p: GenerationParams, f: ImageFacts): ListMap[String, String] =
    val hr = p.hires
    val hrOn = hr.enabled && !p.isImg2Img
    val schedule = f.scheduler.getOrElse(p.scheduler)
    val inpaint = p.img2img.filter(_.mask.isDefined)
    val entries: Seq[(String, Option[String])] = Seq(
      "Steps" -> Some(p.steps.toString),
      "Sampler" -> Some(p.samplerName),
      "Schedule type" -> Some(schedule),
      "CFG scale" -> Some(num(p.cfgScale)),
      "Seed" -> Some(f.seed.toString),
      "Size" -> Some(s"${p.width}x${p.height}"),
      "Model hash" -> f.modelHash,
      "Model" -> f.modelName,
      "VAE hash" -> f.vaeHash,
      "VAE" -> f.vaeName,
      "Variation seed" -> Option.when(p.subseedStrength > 0)(f.subseed.toString),
      "Variation seed strength" -> Option.when(p.subseedStrength > 0)(num(p.subseedStrength)),
      "Denoising strength" -> (p.img2img.map(i => num(i.denoisingStrength)) orElse Option.when(hrOn)(num(hr.denoisingStrength))),
      "Clip skip" -> Option.when(p.clipSkip > 1)(p.clipSkip.toString),
      // A1111 extra_generation_params, in the order processing.py inserts them (img2img init / txt2img init).
      "Mask mode" -> inpaint.filter(_.inpaintingMaskInvert).map(_ => "Inpaint not masked"),
      "Mask blur" -> inpaint.filter(_.maskBlur > 0).map(_.maskBlur.toString),
      "Inpaint area" -> inpaint.filter(_.inpaintFullRes).map(_ => "Only masked"),
      "Masked area padding" -> inpaint.filter(_.inpaintFullRes).map(_.inpaintFullResPadding.toString),
      "Masked content" -> inpaint.flatMap(i => maskedContent.get(i.inpaintingFill)),
      "Hires sampler" -> Option.when(hrOn)(hr.samplerName).flatten.filter(_ != p.samplerName),
      "Hires prompt" -> Option.when(hrOn && hr.prompt.nonEmpty)(hr.prompt),
      "Hires negative prompt" -> Option.when(hrOn && hr.negativePrompt.nonEmpty)(hr.negativePrompt),
      "Hires CFG Scale" -> Option.when(hrOn)(hr.cfgScale).flatten.map(num),
      "Hires schedule type" -> Option.when(hrOn)(hr.scheduler).flatten.filter(_ != schedule),
      "Hires upscale" -> Option.when(hrOn && hr.resizeX == 0 && hr.resizeY == 0)(num(hr.scale)),
      "Hires resize" -> Option.when(hrOn && (hr.resizeX != 0 || hr.resizeY != 0))(s"${hr.resizeX}x${hr.resizeY}"),
      "Hires steps" -> Option.when(hrOn && hr.secondPassSteps > 0)(hr.secondPassSteps.toString),
      "Hires upscaler" -> Option.when(hrOn)(hr.upscaler),
      "Lora hashes" -> Option.when(f.loraHashes.nonEmpty)(f.loraHashes.map((n, h) => s"$n: $h").mkString(", "))
    )
    val base = ListMap.from(entries.collect { case (k, Some(v)) => k -> v })
    val extras = p.extra ++ f.extra
    (base ++ extras) + ("Version" -> Version.label)

  /** `inpainting_fill` values A1111 records as "Masked content" (1 = original is the default and omitted). */
  private val maskedContent: Map[Int, String] = Map(0 -> "fill", 2 -> "latent noise", 3 -> "latent nothing")

  def write(p: GenerationParams, f: ImageFacts): String =
    render(p.prompt, p.negativePrompt, fields(p, f))

  def render(prompt: String, negative: String, fields: ListMap[String, String]): String =
    val line = fields.map((k, v) => if k == v then k else s"$k: ${quote(v)}").mkString(", ")
    val neg = if negative.nonEmpty then s"\nNegative prompt: $negative" else ""
    s"$prompt$neg\n$line".trim

  def parse(text: String): ParsedInfotext =
    val all = text.trim.split("\n", -1).toList
    val (lines, last) =
      if all.isEmpty then (Nil, "")
      else
        val l = all.last
        if ReParam.findAllMatchIn(l).size < 3 then (all, "") else (all.init, l)
    var prompt = ""
    var negative = ""
    var inNegative = false
    lines.foreach { raw =>
      var line = raw.trim
      if line.startsWith("Negative prompt:") then
        inNegative = true
        line = line.drop(16).trim
      if inNegative then negative = if negative.isEmpty then line else s"$negative\n$line"
      else prompt = if prompt.isEmpty then line else s"$prompt\n$line"
    }
    val fields = ListMap.from(ReParam.findAllMatchIn(last).map { m =>
      val v = m.group(2).trim
      m.group(1).trim -> (if v.startsWith("\"") && v.endsWith("\"") then unquote(v) else v)
    })
    ParsedInfotext(prompt, negative, fields)

  /** Keys consumed by [[toParams]]; anything else is kept in `extra` so plugins can round-trip their values. */
  val knownKeys: Set[String] = Set(
    "Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size", "Model hash", "Model", "VAE hash", "VAE",
    "Variation seed", "Variation seed strength", "Denoising strength", "Mask mode", "Mask blur", "Inpaint area",
    "Masked area padding", "Masked content",
    "Clip skip", "Hires upscale", "Hires resize", "Hires steps", "Hires upscaler", "Hires sampler", "Hires schedule type",
    "Hires prompt", "Hires negative prompt", "Hires CFG Scale", "Lora hashes", "Version", "TI hashes"
  )

  /** Restores generation params from infotext on top of `base` (A1111 "send to txt2img"). */
  def toParams(pi: ParsedInfotext, base: GenerationParams = GenerationParams()): GenerationParams =
    val f = pi.fields
    def i(k: String) = f.get(k).flatMap(_.trim.toIntOption)
    def l(k: String) = f.get(k).flatMap(_.trim.toLongOption)
    def d(k: String) = f.get(k).flatMap(_.trim.toDoubleOption)
    val (w, h) = f.get("Size") match
      case Some(ReSize(a, b)) => (a.toInt, b.toInt)
      case _                  => (base.width, base.height)
    // A1111 historically combined sampler and schedule ("DPM++ 2M Karras").
    val (sampler, sched) = splitSampler(f.getOrElse("Sampler", base.samplerName), f.get("Schedule type"))
    val hrOn = f.contains("Hires upscaler") || f.contains("Hires upscale") || f.contains("Hires resize")
    val hrResize = f.get("Hires resize") match
      case Some(ReSize(a, b)) => (a.toInt, b.toInt)
      case _                  => (0, 0)
    val hires =
      if hrOn then
        base.hires.copy(
          enabled = true,
          scale = d("Hires upscale").getOrElse(base.hires.scale),
          resizeX = hrResize._1,
          resizeY = hrResize._2,
          secondPassSteps = i("Hires steps").getOrElse(0),
          upscaler = f.getOrElse("Hires upscaler", base.hires.upscaler),
          denoisingStrength = d("Denoising strength").getOrElse(base.hires.denoisingStrength),
          samplerName = f.get("Hires sampler"),
          scheduler = f.get("Hires schedule type"),
          prompt = f.getOrElse("Hires prompt", ""),
          negativePrompt = f.getOrElse("Hires negative prompt", ""),
          cfgScale = d("Hires CFG Scale")
        )
      else base.hires.copy(enabled = false)
    val subStrength = d("Variation seed strength").getOrElse(0.0)
    base.copy(
      prompt = pi.prompt,
      negativePrompt = pi.negativePrompt,
      steps = i("Steps").getOrElse(base.steps),
      samplerName = sampler,
      scheduler = sched.getOrElse(base.scheduler),
      cfgScale = d("CFG scale").getOrElse(base.cfgScale),
      seed = l("Seed").getOrElse(base.seed),
      width = w,
      height = h,
      subseed = l("Variation seed").getOrElse(-1L),
      subseedStrength = subStrength,
      clipSkip = i("Clip skip").getOrElse(1),
      hires = hires,
      // Inpaint fields use A1111's paste defaults when absent (Whole picture, padding 32, original, Inpaint masked).
      img2img = base.img2img.map(ii =>
        ii.copy(
          denoisingStrength = if hrOn then ii.denoisingStrength else d("Denoising strength").getOrElse(ii.denoisingStrength),
          maskBlur = i("Mask blur").getOrElse(ii.maskBlur),
          inpaintFullRes = f.get("Inpaint area").contains("Only masked"),
          inpaintFullResPadding = i("Masked area padding").getOrElse(32),
          inpaintingFill = f.get("Masked content").flatMap(v => maskedContent.collectFirst { case (k, `v`) => k }).getOrElse(1),
          inpaintingMaskInvert = f.get("Mask mode").contains("Inpaint not masked")
        )
      ),
      // Kept as a ListMap so plugin keys are written back in their infotext order.
      extra = f.filterNot((k, _) => knownKeys.contains(k))
    )

  // (suffix as written in old infotexts, canonical label). Longer suffixes first ("SGM Uniform" before "Uniform").
  private val legacySchedules = Seq(
    "Karras" -> "Karras", "Exponential" -> "Exponential", "Polyexponential" -> "Polyexponential",
    "SGM Uniform" -> "SGM Uniform", "SGMUniform" -> "SGM Uniform", "Uniform" -> "Uniform"
  )

  /** Splits A1111's pre-1.9 combined names ("DPM++ 2M Karras"). Case-insensitive like `get_sampler_and_scheduler`. */
  def splitSampler(sampler: String, schedule: Option[String]): (String, Option[String]) =
    schedule match
      case Some(s) => (sampler, Some(s))
      case None =>
        val lower = sampler.toLowerCase(java.util.Locale.ROOT)
        legacySchedules.find((suffix, _) => lower.endsWith(" " + suffix.toLowerCase(java.util.Locale.ROOT))) match
          case Some((suffix, label)) => (sampler.dropRight(suffix.length + 1), Some(label))
          case None                  => (sampler, None)
