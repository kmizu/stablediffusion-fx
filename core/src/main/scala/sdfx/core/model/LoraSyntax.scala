package sdfx.core.model

/** `<lora:name:weight>` / `<lyco:name:weight>` handling, A1111 compatible. */
object LoraSyntax:
  private val Re = """<(?:lora|lyco):([^:>]+)((?::[^:>]*)*)>""".r

  case class Extracted(prompt: String, loras: Seq[LoraRef])

  /** Removes the tags from the prompt and returns them. Weights of the same LoRA are summed (first occurrence
    * order preserved). Separators left dangling by a removed tag (", ," / leading / trailing commas) are cleaned up.
    */
  def extract(prompt: String): Extracted =
    val found = Re.findAllMatchIn(prompt).map(m => parseTag(m.group(1).trim, Option(m.group(2)).getOrElse(""))).toSeq
    // the same LoRA twice: weights add up, the other arguments come from its first tag
    val merged = found.groupBy(_.name).map((n, rs) => n -> rs.head.copy(weight = rs.map(_.weight).sum))
    val order = found.map(_.name).distinct
    val removed = Re.replaceAllIn(prompt, "").replaceAll("[ \t]{2,}", " ").replaceAll(" ,", ",")
    val stripped =
      if found.isEmpty then removed.trim
      else
        removed
          .replaceAll(",[ \t]*(?=,)", "")      // ", ," left where a tag was
          .replaceAll("(?m)^[ \t]*,[ \t]*", "") // line starting with the separator of a removed tag
          .replaceAll("(?m)[ \t]*,[ \t]*$", "") // line ending with one
          .trim
    Extracted(stripped, order.map(merged))

  /** `:te:unet` positional multipliers, `te=` / `unet=` / `lbw=` / `start=` / `stop=` / `step=a-b` named ones. */
  def parseTag(name: String, args: String): LoraRef =
    val parts = args.split(':').toSeq.drop(1).map(_.trim)
    val (named, positional) = parts.partition(_.contains('='))
    val kv = named.map { s =>
      val i = s.indexOf('=')
      s.take(i).trim.toLowerCase -> s.drop(i + 1).trim
    }.toMap
    def num(s: String): Option[Double] = s.trim.toDoubleOption
    val te = kv.get("te").flatMap(num).orElse(positional.headOption.flatMap(num)).getOrElse(1.0)
    val unet = kv.get("unet").flatMap(num).orElse(positional.lift(1).flatMap(num))
    val (stepStart, stepStop) = kv.get("step").map(_.split('-')) match
      case Some(Array(a, b)) => (num(a), num(b))
      case _                 => (None, None)
    LoraRef(name, te, unet, kv.get("lbw").filter(_.nonEmpty),
      kv.get("start").flatMap(num).orElse(stepStart), kv.get("stop").flatMap(num).orElse(stepStop))

  /** Rewrites every `<lora:...>` tag through `f(name, args)` -> new args (the part after the name, ":"-prefixed). */
  def rewriteTags(prompt: String, f: (String, String) => String): String =
    Re.replaceAllIn(prompt, m =>
      val kind = if m.matched.startsWith("<lyco:") then "lyco" else "lora"
      scala.util.matching.Regex.quoteReplacement(s"<$kind:${m.group(1)}${f(m.group(1).trim, Option(m.group(2)).getOrElse(""))}>"))

  /** Sets (or replaces) named arguments of a tag's argument string, e.g. `withNamed(":0.8:lbw=ALL", "lbw" -> "INS")`. */
  def withNamed(args: String, kv: (String, String)*): String =
    val keys = kv.map(_._1.toLowerCase).toSet
    val kept = args.split(':').toSeq.drop(1).filterNot(p => p.contains('=') && keys(p.take(p.indexOf('=')).trim.toLowerCase))
    (kept ++ kv.filter(_._2.nonEmpty).map((k, v) => s"$k=$v")).map(":" + _).mkString

  def tag(name: String, weight: Double = 1.0): String =
    // Locale-independent ("1.0", never "1,0").
    val w = if weight == math.rint(weight) && math.abs(weight) < 1e15 then s"${weight.toLong}.0" else BigDecimal(weight).bigDecimal.stripTrailingZeros.toPlainString
    s"<lora:$name:$w>"
