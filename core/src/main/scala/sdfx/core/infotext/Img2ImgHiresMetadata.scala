package sdfx.core.infotext

import scala.util.Try

/** Source-compatible post-image refinement metadata; independent of native Hires. */
object Img2ImgHiresMetadata:
  val key = "img2img Hires Fix"
  def targetSize(h: ujson.Obj, width: Int, height: Int): (Int, Int) =
    val explicit = h("width").num > 0 && h("height").num > 0
    def side(key: String, size: Int): Int =
      val raw = if explicit then h(key).num else size * h("ratio").num
      math.max(64, raw.toInt / 8 * 8)
    (side("width", width), side("height", height))
  def active(hooks: Map[String, ujson.Value]): Option[ujson.Obj] =
    def enabled(v: ujson.Value, default: Boolean): Boolean = v match
      case ujson.Bool(b) => b
      case ujson.Num(n) => n != 0
      case ujson.Str(s) => s.trim.toLowerCase(java.util.Locale.ROOT) match
        case "enable" | "enabled" | "true" | "on" | "yes" | "1" => true
        case "disable" | "disabled" | "false" | "off" | "no" | "0" | "" => false
        case _ => default
      case _ => default
    hooks.get("img2img_hires_fix").flatMap {
      case o: ujson.Obj if o.value.get("enabled").forall(enabled(_, true)) => Some(normalize(o))
      case v @ (ujson.Bool(_) | ujson.Str(_)) if enabled(v, false) => Some(normalize(ujson.Obj()))
      case _ => None
    }
  def normalize(input: ujson.Obj): ujson.Obj =
    def number(key: String, default: Double, high: Double, integral: Boolean = false): Double =
      val value = input.value.get(key).fold(default)(_.num)
      require(value.isFinite && value >= 0 && value <= high && (!integral || value == math.floor(value)), s"Invalid img2img Hires $key")
      value
    def text(key: String, default: String): String = input.value.get(key).fold(default)(_.str).trim
    val ratio = number("ratio", 2, 4)
    val width = number("width", 0, 2048, true).toInt
    val height = number("height", 0, 2048, true).toInt
    require(width % 8 == 0 && height % 8 == 0, "img2img Hires dimensions must be multiples of 8")
    require(ratio > 0 || (width > 0 && height > 0), "img2img Hires needs a ratio or both dimensions")
    val upscaler = text("upscaler", "R-ESRGAN 4x+")
    require(upscaler.nonEmpty && !upscaler.startsWith("Latent"), "img2img Hires needs a pixel upscaler")
    val sampler = text("sampler", "Use same sampler")
    val scheduler = text("scheduler", "Use same scheduler")
    require(sampler.nonEmpty && scheduler.nonEmpty, "img2img Hires sampler/scheduler required")
    ujson.Obj("enabled" -> true, "ratio" -> ratio, "width" -> width, "height" -> height,
      "steps" -> number("steps", 0, 25, true), "upscaler" -> upscaler,
      "denoise" -> number("denoise", .33, 1), "cfg" -> number("cfg", 0, 30),
      "sampler" -> sampler, "scheduler" -> scheduler, "prompt" -> text("prompt", ""),
      "negative_prompt" -> text("negative_prompt", ""))

  def encode(input: ujson.Obj): String =
    val h = normalize(input)
    val output = ujson.Obj.from(h.value.toSeq.filterNot((key, _) => Set("enabled", "ratio", "width", "height")(key)))
    output("scale") = if h("width").num > 0 && h("height").num > 0 then
      ujson.Str(s"${h("width").num.toInt}x${h("height").num.toInt}") else h("ratio")
    swap(output.render())

  private def swap(text: String): String = text.map { case '\'' => '"'; case '"' => '\''; case c => c }
  def parse(text: String): Option[ujson.Obj] =
    if text.length > 65536 then None
    else
      def decode(text: String): Option[ujson.Obj] = Try {
        val h = ujson.read(text).obj
        val out = ujson.Obj.from(h.toSeq.filterNot(_._1 == "scale"))
        h("scale") match
          case ujson.Str(size) =>
            val dimensions = size.split("x", -1)
            require(dimensions.length == 2)
            out("ratio") = 0
            out("width") = dimensions(0).toInt
            out("height") = dimensions(1).toInt
          case ujson.Num(scale) => out("ratio") = scale
          case _ => throw new IllegalArgumentException("Invalid img2img Hires scale")
        normalize(out)
      }.toOption
      decode(text).orElse(decode(swap(text)))
