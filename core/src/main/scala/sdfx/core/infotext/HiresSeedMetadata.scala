package sdfx.core.infotext

import scala.util.Try

/** Bounded source-compatible Hires Fix Tweaks seed metadata; never evaluates expressions. */
object HiresSeedMetadata:
  private val MaxExact = 9007199254740991L
  def encode(seed: Long): ujson.Value =
    if seed >= -1 && seed <= MaxExact then ujson.Num(seed.toDouble) else ujson.Str(seed.toString)
  def seedValue(value: ujson.Value): Option[Long] = value match
    case ujson.Str(s) => s.toLongOption.filter(_ >= -1)
    case ujson.Num(n) if n.isFinite && n >= -1 && n <= MaxExact.toDouble && n == math.floor(n) => Some(n.toLong)
    case _ => None
  def parse(text: String): Option[ujson.Obj] =
    if text.length > 4096 then None
    else
      // Preserve integer literals before ujson's floating-point parser sees them.
      val integer = """("(?:Seed|Subseed)"\s*:\s*)(-?\d+)(?=\s*[,}])""".r
      def decode(s: String): Option[ujson.Obj] =
        val exact = integer.replaceAllIn(s, m => java.util.regex.Matcher.quoteReplacement(m.group(1) + "\"" + m.group(2) + "\""))
        Try(ujson.read(exact)).toOption.collect { case o: ujson.Obj => o }
      val parsed = decode(text).orElse(decode(text.map { case '\'' => '"'; case '"' => '\''; case c => c }))
      parsed.flatMap { o =>
        def number(key: String, default: Double, low: Double, high: Double, integral: Boolean): Option[Double] =
          o.value.get(key) match
            case None => Some(default)
            case Some(ujson.Num(n)) if n.isFinite && n >= low && n <= high && (!integral || n == math.floor(n)) => Some(n)
            case _ => None
        val resize = o.value.get("Resize") match
          case None => Some((0, 0))
          case Some(a: ujson.Arr) if a.value.size == 2 =>
            val dims = a.value.collect { case ujson.Num(n) if n.isFinite && n >= 0 && n <= 16384 && n % 8 == 0 => n.toInt }
            Option.when(dims.size == 2)((dims.head, dims.last))
          case _ => None
        for
          _ <- Option.when(o.value.keySet.subsetOf(Set("Seed", "Subseed", "Strength", "Resize")))(())
          seed <- seedValue(o.value.getOrElse("Seed", ujson.Num(0)))
          sub <- seedValue(o.value.getOrElse("Subseed", ujson.Num(0)))
          strength <- number("Strength", 0, 0, 1, false)
          size <- resize
        yield ujson.Obj("enabled" -> true, "seed_enabled" -> true, "extras" -> true,
          "seed" -> encode(seed), "subseed" -> encode(sub), "strength" -> strength,
          "resize_width" -> size._1, "resize_height" -> size._2, "batch_count" -> 1)
      }
