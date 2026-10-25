package sdfx.plugins

import sdfx.core.plugins.api.Values

/** How an X/Y/Z axis interprets its values text. */
enum ValueKind:
  case Str, IntNum, FloatNum, Permutations

/** A1111 X/Y/Z value parsing: CSV with quoting, integer/float ranges, counted ranges, permutations. */
object XyzValues:
  // Same expressions as scripts/xyz_grid.py (the float dot is escaped here on purpose).
  private val ReRange = """\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*""".r
  private val ReRangeFloat =
    """\s*([+-]?\s*\d+(?:\.\d*)?)\s*-\s*([+-]?\s*\d+(?:\.\d*)?)(?:\s*\(([+-]\d+(?:\.\d*)?)\s*\))?\s*""".r
  private val ReRangeCount = """\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)\s*\[(\d+)\s*]\s*""".r
  private val ReRangeCountFloat = """\s*([+-]?\s*\d+(?:\.\d*)?)\s*-\s*([+-]?\s*\d+(?:\.\d*)?)\s*\[(\d+)\s*]\s*""".r

  private def n(s: String): String = s.replace(" ", "")

  /** Splits CSV text: commas (and newlines) separate, double quotes group (`""` is a literal quote), fields are
    * trimmed and empty fields dropped.
    */
  def csv(text: String): Seq[String] =
    val out = scala.collection.mutable.ArrayBuffer[String]()
    val cur = new StringBuilder
    var inQ = false
    var i = 0
    def flush(): Unit =
      out += cur.toString.trim
      cur.clear()
    while i < text.length do
      val c = text.charAt(i)
      if inQ then
        if c == '"' then
          if i + 1 < text.length && text.charAt(i + 1) == '"' then
            cur += '"'
            i += 1
          else inQ = false
        else cur += c
      else if c == '"' && cur.toString.trim.isEmpty then
        cur.clear()
        inQ = true
      else if c == ',' || c == '\n' || c == '\r' then flush()
      else cur += c
      i += 1
    flush()
    out.filter(_.nonEmpty).toSeq

  /** Inverse of [[csv]]: quotes values containing commas, quotes or surrounding spaces. */
  def toCsv(values: Seq[String]): String =
    values
      .map(v =>
        if v.contains(',') || v.contains('"') || v.contains('\n') || v != v.trim then "\"" + v.replace("\"", "\"\"") + "\""
        else v
      )
      .mkString(", ")

  /** Upper bound on the values of one axis (and on prompt-order permutations). */
  val MaxValues = 10000
  val MaxPermutations = 720

  private def tooMany(n: Long): Nothing =
    throw new IllegalArgumentException(s"too many values on one axis ($n, max $MaxValues)")

  /** Python `range(start, end + 1, step)`, exactly like A1111 (so a descending `10-1 (-3)` gives 10, 7, 4). */
  def intRange(start: Long, end: Long, step: Long): Seq[Long] =
    if step == 0 then Seq(start)
    else
      val stop = end + 1
      val count = if step > 0 then math.max(0L, (stop - start + step - 1) / step) else math.max(0L, (start - stop - step - 1) / -step)
      if count > MaxValues then tooMany(count)
      (0L until count).map(i => start + i * step)


  /** numpy.linspace. */
  def linspace(start: Double, end: Double, num: Int): Seq[Double] =
    if num > MaxValues then tooMany(num)
    if num <= 0 then Nil
    else if num == 1 then Seq(start)
    else (0 until num).map(i => if i == num - 1 then end else start + (end - start) * i / (num - 1))

  /** numpy `arange(start, end + step, step)` with decimal arithmetic (avoids 0.30000000000000004). */
  def floatRange(start: BigDecimal, end: BigDecimal, step: BigDecimal): Seq[Double] =
    if step == 0 then Seq(start.toDouble)
    else
      val count = ((end + step - start) / step).setScale(0, BigDecimal.RoundingMode.CEILING).toLong
      if count > MaxValues then tooMany(count)
      (0L until math.max(0L, count)).map(i => (start + step * i).toDouble)

  /** Parses a values text into value strings. Throws IllegalArgumentException for invalid numbers or too many
    * values.
    */
  def parse(text: String, kind: ValueKind): Seq[String] =
    val items = csv(text)
    val out = kind match
      case ValueKind.Str => items
      case ValueKind.Permutations =>
        val total = (1 to items.size).foldLeft(1L)((a, b) => math.min(a * b, Long.MaxValue / 64))
        if total > MaxPermutations then
          throw new IllegalArgumentException(s"too many permutations (${items.size} items, max $MaxPermutations orderings)")
        items.permutations.map(toCsv).toSeq
      case ValueKind.IntNum =>
        items.flatMap {
          case ReRange(a, b, st) =>
            intRange(n(a).toLong, n(b).toLong, Option(st).map(s => n(s).toLong).getOrElse(1L)).map(_.toString)
          case ReRangeCount(a, b, c) => linspace(n(a).toLong.toDouble, n(b).toLong.toDouble, c.toInt).map(_.toLong.toString)
          case v =>
            v.trim.toLongOption.orElse(v.trim.toDoubleOption.map(_.toLong)) match
              case Some(i) => Seq(i.toString)
              case None    => throw new IllegalArgumentException(s"not an integer: '$v'")
        }
      case ValueKind.FloatNum =>
        items.flatMap {
          case ReRangeFloat(a, b, st) =>
            floatRange(BigDecimal(n(a)), BigDecimal(n(b)), Option(st).map(s => BigDecimal(n(s))).getOrElse(BigDecimal(1)))
              .map(Values.fmt)
          case ReRangeCountFloat(a, b, c) => linspace(n(a).toDouble, n(b).toDouble, c.toInt).map(Values.fmt)
          case v =>
            v.trim.toDoubleOption match
              case Some(d) => Seq(Values.fmt(d))
              case None    => throw new IllegalArgumentException(s"not a number: '$v'")
        }
    if out.size > MaxValues then tooMany(out.size)
    out
