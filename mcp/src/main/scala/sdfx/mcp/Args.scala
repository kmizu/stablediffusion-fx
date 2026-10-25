package sdfx.mcp

/** Typed, strict access to tool arguments. Unknown keys and wrong types raise [[ToolError]] (-> isError result). */
final class Args(val obj: ujson.Obj, allowed: Iterable[String]):
  private val m = obj.value
  locally {
    val unknown = m.keySet.toSeq.filterNot(allowed.toSet.contains).sorted
    if unknown.nonEmpty then
      throw ToolError(s"unknown argument(s): ${unknown.mkString(", ")}. Allowed: ${allowed.toSeq.sorted.mkString(", ")}")
  }

  private def present(k: String): Option[ujson.Value] = m.get(k).filterNot(_ == ujson.Null)
  def has(k: String): Boolean = present(k).isDefined

  def str(k: String): Option[String] = present(k).map {
    case ujson.Str(s) => s
    case ujson.Num(n) => if n.isWhole then n.toLong.toString else n.toString
    case _            => throw ToolError(s"$k must be a string")
  }
  def reqStr(k: String): String = str(k).getOrElse(throw ToolError(s"missing required argument: $k"))

  def long(k: String, lo: Long, hi: Long): Option[Long] = present(k).map { v =>
    val n = v match
      case ujson.Num(d) if d.isWhole && math.abs(d) < 9.0e15 => d.toLong
      case ujson.Str(s) if s.trim.toLongOption.isDefined   => s.trim.toLong
      case _                                                => throw ToolError(s"$k must be an integer")
    if n < lo || n > hi then throw ToolError(s"$k must be in $lo..$hi (got $n)")
    n
  }
  def int(k: String, lo: Int, hi: Int): Option[Int] = long(k, lo, hi).map(_.toInt)

  def num(k: String, lo: Double, hi: Double): Option[Double] = present(k).map { v =>
    val d = v match
      case ujson.Num(x)                                       => x
      case ujson.Str(s) if s.trim.toDoubleOption.isDefined => s.trim.toDouble
      case _                                                  => throw ToolError(s"$k must be a number")
    if d.isNaN || d < lo || d > hi then throw ToolError(s"$k must be in $lo..$hi (got $d)")
    d
  }

  def bool(k: String): Option[Boolean] = present(k).map {
    case ujson.Bool(b)                                                  => b
    case ujson.Str(s) if Set("true", "false").contains(s.toLowerCase) => s.toLowerCase == "true"
    case _                                                              => throw ToolError(s"$k must be a boolean")
  }

  def objOpt(k: String): Option[ujson.Obj] = present(k).map {
    case o: ujson.Obj => o
    case _            => throw ToolError(s"$k must be an object")
  }

  def arr(k: String): Option[Seq[ujson.Value]] = present(k).map {
    case ujson.Arr(a) => a.toSeq
    case _            => throw ToolError(s"$k must be an array")
  }

  def strList(k: String): Option[Seq[String]] = present(k).map {
    case ujson.Arr(a) => a.toSeq.map {
        case ujson.Str(s) => s
        case _            => throw ToolError(s"$k must be an array of strings")
      }
    case ujson.Str(s) => s.split(",").toSeq.map(_.trim).filter(_.nonEmpty)
    case _            => throw ToolError(s"$k must be an array of strings")
  }

  /** A flat object of scalar values converted to strings (plugin control values, extra params). */
  def stringMap(k: String): Option[Map[String, String]] = objOpt(k).map(_.value.toMap.map {
    case (kk, ujson.Str(s))  => kk -> s
    case (kk, ujson.Num(n))  => kk -> (if n.isWhole && math.abs(n) < 1e15 then n.toLong.toString else n.toString)
    case (kk, ujson.Bool(b)) => kk -> b.toString
    case (kk, _)             => throw ToolError(s"$k.$kk must be a string, number or boolean")
  })

/** JSON Schema building blocks (draft 2020-12 compatible subset). */
object Schema:
  def obj(props: Seq[(String, ujson.Value)], required: Seq[String] = Nil, additional: Boolean = false): ujson.Obj =
    val o = ujson.Obj("type" -> "object", "properties" -> ujson.Obj.from(props))
    if required.nonEmpty then o("required") = ujson.Arr.from(required)
    o("additionalProperties") = additional
    o

  private def withDesc(o: ujson.Obj, d: String): ujson.Obj =
    if d.nonEmpty then o("description") = d
    o

  def str(d: String = "", enumValues: Seq[String] = Nil, default: Option[String] = None): ujson.Obj =
    val o = withDesc(ujson.Obj("type" -> "string"), d)
    if enumValues.nonEmpty then o("enum") = ujson.Arr.from(enumValues)
    default.foreach(v => o("default") = v)
    o

  def int(d: String = "", min: Option[Long] = None, max: Option[Long] = None, default: Option[Long] = None): ujson.Obj =
    val o = withDesc(ujson.Obj("type" -> "integer"), d)
    min.foreach(v => o("minimum") = v.toDouble)
    max.foreach(v => o("maximum") = v.toDouble)
    default.foreach(v => o("default") = v.toDouble)
    o

  def num(d: String = "", min: Option[Double] = None, max: Option[Double] = None, default: Option[Double] = None): ujson.Obj =
    val o = withDesc(ujson.Obj("type" -> "number"), d)
    min.foreach(v => o("minimum") = v)
    max.foreach(v => o("maximum") = v)
    default.foreach(v => o("default") = v)
    o

  def bool(d: String = "", default: Option[Boolean] = None): ujson.Obj =
    val o = withDesc(ujson.Obj("type" -> "boolean"), d)
    default.foreach(v => o("default") = v)
    o

  def arr(items: ujson.Value, d: String = "", maxItems: Option[Int] = None): ujson.Obj =
    val o = withDesc(ujson.Obj("type" -> "array", "items" -> items), d)
    maxItems.foreach(v => o("maxItems") = v)
    o

  def scalarMap(d: String): ujson.Obj =
    withDesc(ujson.Obj("type" -> "object", "additionalProperties" -> ujson.Obj("type" -> ujson.Arr("string", "number", "boolean"))), d)

  val empty: ujson.Obj = obj(Nil)
