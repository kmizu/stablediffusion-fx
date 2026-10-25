package sdfx.core.model

/** `<lora:name:weight>` / `<lyco:name:weight>` handling, A1111 compatible. */
object LoraSyntax:
  private val Re = """<(?:lora|lyco):([^:>]+)(?::([^:>]*))?(?::[^>]*)?>""".r

  case class Extracted(prompt: String, loras: Seq[LoraRef])

  /** Removes the tags from the prompt and returns them. Weights of the same LoRA are summed (first occurrence
    * order preserved). Separators left dangling by a removed tag (", ," / leading / trailing commas) are cleaned up.
    */
  def extract(prompt: String): Extracted =
    val found = Re.findAllMatchIn(prompt).map { m =>
      val w = Option(m.group(2)).map(_.trim).filter(_.nonEmpty).flatMap(_.toDoubleOption).getOrElse(1.0)
      LoraRef(m.group(1).trim, w)
    }.toSeq
    val merged = found
      .groupBy(_.name)
      .map((n, rs) => n -> rs.map(_.weight).sum)
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
    Extracted(stripped, order.map(n => LoraRef(n, merged(n))))

  def tag(name: String, weight: Double = 1.0): String =
    // Locale-independent ("1.0", never "1,0").
    val w = if weight == math.rint(weight) && math.abs(weight) < 1e15 then s"${weight.toLong}.0" else BigDecimal(weight).bigDecimal.stripTrailingZeros.toPlainString
    s"<lora:$name:$w>"
