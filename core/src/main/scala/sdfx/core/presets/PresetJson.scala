package sdfx.core.presets

import java.nio.ByteBuffer
import java.nio.charset.{CodingErrorAction, StandardCharsets}
import java.util.UUID
import scala.collection.mutable
import scala.util.control.NonFatal

private[presets] final case class RawMember(name: String, keyToken: RawJson, value: RawJson)
private[presets] enum SourceValue:
  case Raw(value: RawJson)
  case Atom(value: PresetAtom)

/** The native envelope and source JSON use one strict lexical reader. RawJson is never evaluated. */
object PresetJson:
  private final case class Problem(code: String, path: String, detail: String) extends RuntimeException(detail)
  private def reject(code: String, path: String, detail: String): Nothing = throw Problem(code, path, detail)
  private def attempt[A](body: => A): Either[PresetError, A] =
    try Right(body)
    catch
      case p: Problem => Left(PresetError(Vector(PresetIssue(p.code, p.path, p.detail))))
      case NonFatal(e) => Left(PresetError(Vector(PresetIssue("invalid-json", "$", Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))))

  private sealed trait Node:
    def start: Int
    def end: Int
  private final case class Obj(entries: Vector[(String, Node)], start: Int, end: Int,
    keySlices: Vector[(Int, Int)]) extends Node
  private final case class Arr(items: Vector[Node], start: Int, end: Int) extends Node
  private final case class Str(value: String, start: Int, end: Int) extends Node
  private final case class Num(value: String, start: Int, end: Int) extends Node
  private final case class Bool(value: Boolean, start: Int, end: Int) extends Node
  private final case class Nil(start: Int, end: Int) extends Node

  private def utf8Length(s: String, maximum: Int, path: String): Int =
    if s == null then reject("invalid-text", path, "null text")
    var i = 0
    var size = 0L
    while i < s.length do
      val cp = s.codePointAt(i)
      if Character.isSurrogate(s.charAt(i)) && (cp >= 0xd800 && cp <= 0xdfff) then
        reject("invalid-unicode", path, "unpaired surrogate")
      size += (if cp < 0x80 then 1 else if cp < 0x800 then 2 else if cp < 0x10000 then 3 else 4)
      if size > maximum then reject("limit", path, s"UTF-8 byte limit $maximum exceeded")
      i += Character.charCount(cp)
    size.toInt

  private def decode(bytes: Array[Byte], limits: PresetLimits): String =
    if bytes == null then reject("invalid-json", "$", "null input")
    if bytes.length > limits.maxBytes then reject("limit", "$", s"file exceeds ${limits.maxBytes} UTF-8 bytes")
    try
      StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes)).toString
    catch case _: java.nio.charset.CharacterCodingException => reject("invalid-utf8", "$", "malformed UTF-8")

  private final class Reader(text: String, limits: PresetLimits):
    private var at = 0
    private var nodes = 0
    private def bump(): Unit =
      nodes += 1
      if nodes > limits.maxNodes then reject("limit", "$", s"JSON node limit ${limits.maxNodes} exceeded")
    private def space(): Unit =
      while at < text.length && (text.charAt(at) == ' ' || text.charAt(at) == '\r' || text.charAt(at) == '\n' || text.charAt(at) == '\t') do at += 1
    private def take(c: Char): Unit =
      if at >= text.length || text.charAt(at) != c then reject("invalid-json", "$", s"expected '$c' at character $at")
      at += 1
    private def hex(pos: Int): Int =
      if pos >= text.length then reject("invalid-json", "$", "incomplete Unicode escape")
      text.charAt(pos) match
        case c if c >= '0' && c <= '9' => c - '0'
        case c if c >= 'A' && c <= 'F' => c - 'A' + 10
        case c if c >= 'a' && c <= 'f' => c - 'a' + 10
        case _ => reject("invalid-json", "$", "Unicode escape requires ASCII hexadecimal digits")
    private def escapedUnit(): Char =
      if at + 4 > text.length then reject("invalid-json", "$", "incomplete Unicode escape")
      val code = (hex(at) << 12) | (hex(at + 1) << 8) | (hex(at + 2) << 4) | hex(at + 3)
      at += 4
      code.toChar
    private def string(cap: Int): Str =
      val start = at
      take('"')
      val value = new StringBuilder
      var size = 0
      var closed = false
      while at < text.length && !closed do
        val c = text.charAt(at)
        if c == '"' then
          at += 1
          closed = true
        else
          var cp = 0
          if c == '\\' then
            at += 1
            if at >= text.length then reject("invalid-json", "$", "unfinished escape")
            text.charAt(at) match
              case '"' => cp = '"'; at += 1
              case '\\' => cp = '\\'; at += 1
              case '/' => cp = '/'; at += 1
              case 'b' => cp = '\b'; at += 1
              case 'f' => cp = '\f'; at += 1
              case 'n' => cp = '\n'; at += 1
              case 'r' => cp = '\r'; at += 1
              case 't' => cp = '\t'; at += 1
              case 'u' =>
                at += 1
                val first = escapedUnit()
                if Character.isHighSurrogate(first) then
                  if at + 2 > text.length || text.charAt(at) != '\\' || text.charAt(at + 1) != 'u' then
                    reject("invalid-unicode", "$", "unpaired high surrogate")
                  at += 2
                  val second = escapedUnit()
                  if !Character.isLowSurrogate(second) then reject("invalid-unicode", "$", "unpaired high surrogate")
                  cp = Character.toCodePoint(first, second)
                else if Character.isLowSurrogate(first) then reject("invalid-unicode", "$", "unpaired low surrogate")
                else cp = first.toInt
              case _ => reject("invalid-json", "$", "invalid string escape")
          else
            if c < ' ' then reject("invalid-json", "$", "unescaped control character")
            if Character.isHighSurrogate(c) then
              if at + 1 >= text.length || !Character.isLowSurrogate(text.charAt(at + 1)) then
                reject("invalid-unicode", "$", "unpaired high surrogate")
              cp = Character.toCodePoint(c, text.charAt(at + 1))
              at += 2
            else if Character.isLowSurrogate(c) then reject("invalid-unicode", "$", "unpaired low surrogate")
            else
              cp = c.toInt
              at += 1
          size += (if cp < 0x80 then 1 else if cp < 0x800 then 2 else if cp < 0x10000 then 3 else 4)
          if size > cap then reject("limit", "$", s"scalar/key exceeds $cap UTF-8 bytes")
          if cp <= 0xffff then value.append(cp.toChar)
          else
            value.append(Character.highSurrogate(cp))
            value.append(Character.lowSurrogate(cp))
      if !closed then reject("invalid-json", "$", "unterminated string")
      Str(value.toString, start, at)
    private def number(): Num =
      val start = at
      if text.charAt(at) == '-' then at += 1
      if at >= text.length then reject("invalid-json", "$", "incomplete number")
      if text.charAt(at) == '0' then at += 1
      else
        if text.charAt(at) < '1' || text.charAt(at) > '9' then reject("invalid-json", "$", "invalid number")
        while at < text.length && text.charAt(at) >= '0' && text.charAt(at) <= '9' do at += 1
      if at < text.length && text.charAt(at) == '.' then
        at += 1
        if at >= text.length || text.charAt(at) < '0' || text.charAt(at) > '9' then reject("invalid-json", "$", "invalid fraction")
        while at < text.length && text.charAt(at) >= '0' && text.charAt(at) <= '9' do at += 1
      if at < text.length && (text.charAt(at) == 'e' || text.charAt(at) == 'E') then
        at += 1
        if at < text.length && (text.charAt(at) == '+' || text.charAt(at) == '-') then at += 1
        if at >= text.length || text.charAt(at) < '0' || text.charAt(at) > '9' then reject("invalid-json", "$", "invalid exponent")
        while at < text.length && text.charAt(at) >= '0' && text.charAt(at) <= '9' do at += 1
      if at - start > limits.maxScalarBytes then reject("limit", "$", "numeric scalar too long")
      Num(text.substring(start, at), start, at)
    private def literal(s: String): Unit =
      if !text.startsWith(s, at) then reject("invalid-json", "$", s"invalid literal at character $at")
      at += s.length
    private def value(depth: Int): Node =
      space()
      if at >= text.length then reject("invalid-json", "$", "missing value")
      bump()
      val start = at
      text.charAt(at) match
        case '{' =>
          if depth >= limits.maxDepth then reject("limit", "$", "JSON nesting too deep")
          at += 1
          space()
          val entries = Vector.newBuilder[(String, Node)]
          val keySlices = Vector.newBuilder[(Int, Int)]
          val seen = mutable.HashSet.empty[String]
          if at < text.length && text.charAt(at) == '}' then at += 1
          else
            var done = false
            while !done do
              space()
              if at >= text.length || text.charAt(at) != '"' then reject("invalid-json", "$", "object key expected")
              bump()
              val keyNode = string(limits.maxKeyBytes)
              val key = keyNode.value
              if !seen.add(key) then reject("duplicate-key", "$", s"duplicate decoded key: $key")
              keySlices += keyNode.start -> keyNode.end
              space(); take(':')
              val item = value(depth + 1)
              entries += key -> item
              space()
              if at < text.length && text.charAt(at) == ',' then at += 1
              else
                take('}')
                done = true
          Obj(entries.result(), start, at, keySlices.result())
        case '[' =>
          if depth >= limits.maxDepth then reject("limit", "$", "JSON nesting too deep")
          at += 1
          space()
          val items = Vector.newBuilder[Node]
          if at < text.length && text.charAt(at) == ']' then at += 1
          else
            var done = false
            while !done do
              items += value(depth + 1)
              space()
              if at < text.length && text.charAt(at) == ',' then at += 1
              else
                take(']')
                done = true
          Arr(items.result(), start, at)
        case '"' => string(limits.maxScalarBytes)
        case 't' => literal("true"); Bool(true, start, at)
        case 'f' => literal("false"); Bool(false, start, at)
        case 'n' => literal("null"); Nil(start, at)
        case c if c == '-' || (c >= '0' && c <= '9') => number()
        case _ => reject("invalid-json", "$", s"invalid value at character $at")
    def root(): Node =
      val out = value(0)
      space()
      if at != text.length then reject("invalid-json", "$", s"trailing characters at $at")
      out

  private def parse(text: String, limits: PresetLimits): Node =
    utf8Length(text, limits.maxBytes, "$")
    new Reader(text, limits).root()
  private def raw(node: Node, text: String): RawJson = RawJson.validated(text.substring(node.start, node.end))
  private def obj(node: Node, path: String): Obj = node match
    case o: Obj => o
    case _ => reject("schema", path, "object expected")
  private def arr(node: Node, path: String): Vector[Node] = node match
    case a: Arr => a.items
    case _ => reject("schema", path, "array expected")
  private def str(node: Node, path: String): String = node match
    case s: Str => s.value
    case _ => reject("schema", path, "string expected")
  private def required(o: Obj, key: String, path: String): Node =
    o.entries.find(_._1 == key).map(_._2).getOrElse(reject("schema", path, s"missing $key"))
  private def optional(o: Obj, key: String): Option[Node] = o.entries.find(_._1 == key).map(_._2)
  private def extras(o: Obj, known: Set[String], text: String): Vector[(String, RawJson)] =
    o.entries.collect { case (key, value) if !known(key) => key -> raw(value, text) }
  private def key(id: String, path: String, limits: PresetLimits): FieldKey =
    if utf8Length(id, limits.maxKeyBytes, path) > limits.maxKeyBytes then reject("limit", path, "key too long")
    FieldKey.parse(id).fold(e => reject(e.issues.head.code, path, e.issues.head.message), identity)
  private def idString(node: Node, path: String, limits: PresetLimits): String =
    val value = str(node, path)
    utf8Length(value, limits.maxKeyBytes, path)
    if value.isEmpty then reject("id", path, "empty ID")
    value
  private def integer(value: String, path: String): Long =
    try
      val n = new java.math.BigDecimal(value)
      if n.compareTo(java.math.BigDecimal.valueOf(Long.MinValue)) < 0 ||
        n.compareTo(java.math.BigDecimal.valueOf(Long.MaxValue)) > 0 then reject("integer", path, "out of Long range")
      n.longValueExact()
    catch
      case p: Problem => throw p
      case NonFatal(_) => reject("integer", path, "fractional or invalid Long")
  private def atom(node: Node, path: String): PresetAtom =
    val o = obj(node, path)
    val t = str(required(o, "type", path), path + ".type")
    val v = str(required(o, "value", path), path + ".value")
    if o.entries.size != 2 then reject("schema", path, "typed atom has unexpected properties")
    t match
      case "text" => PresetAtom.Text(v)
      case "integer" => PresetAtom.Integer(integer(v, path))
      case "decimal" =>
        try PresetAtom.Decimal(BigDecimal(new java.math.BigDecimal(v)))
        catch case NonFatal(_) => reject("decimal", path, "invalid decimal")
      case "flag" =>
        if v == "true" then PresetAtom.Flag(true)
        else if v == "false" then PresetAtom.Flag(false)
        else reject("flag", path, "invalid boolean")
      case _ => reject("schema", path, s"unknown atom type $t")
  private def mode(s: String, path: String): PresetMode = s match
    case "txt2img" => PresetMode.Txt2Img
    case "img2img" => PresetMode.Img2Img
    case "inpaint" => PresetMode.Inpaint
    case _ => reject("schema", path, "unknown mode")
  private def source(node: Node, path: String, text: String, limits: PresetLimits): Option[PresetSource] = node match
    case _: Nil => None
    case _ =>
      val o = obj(node, path)
      val original = required(o, "originalFields", path)
      val originalObject = obj(original, path + ".originalFields")
      if originalObject.entries.size > limits.maxFieldsPerRecord then reject("limit", path, "too many original fields")
      val sourceMode = required(o, "mode", path) match
        case _: Nil => None
        case x => Some(mode(str(x, path + ".mode"), path + ".mode"))
      if o.entries.size != 4 then reject("schema", path, "source has unexpected properties")
      Some(PresetSource(
        idString(required(o, "format", path), path + ".format", limits),
        name(str(required(o, "name", path), path + ".name"), path + ".name", limits),
        sourceMode,
        raw(original, text)
      ))
  private def name(s: String, path: String, limits: PresetLimits): String =
    val cps = s.codePointCount(0, s.length)
    if cps > limits.maxNameCodePoints || !s.codePoints().toArray.exists(cp => !Character.isWhitespace(cp)) ||
      s.codePoints().toArray.exists(cp => Character.isISOControl(cp)) then
      reject("name", path, "invalid preset name")
    s
  private def record(node: Node, path: String, text: String, limits: PresetLimits): PresetRecord =
    val o = obj(node, path)
    val idText = str(required(o, "id", path), path + ".id")
    val id =
      try UUID.fromString(idText)
      catch case NonFatal(_) => reject("id", path, "invalid UUID")
    if id.toString != idText then reject("id", path, "noncanonical UUID")
    val fieldsObj = obj(required(o, "fields", path), path + ".fields")
    if fieldsObj.entries.size > limits.maxFieldsPerRecord then reject("limit", path, "too many fields")
    val fields = fieldsObj.entries.map { (id, value) =>
      key(id, path + ".fields", limits) -> atom(value, path + ".fields." + id)
    }.toMap
    if fields.size != fieldsObj.entries.size then reject("duplicate-key", path, "duplicate canonical field")
    val dirty = arr(required(o, "dirty", path), path + ".dirty").map(x => key(str(x, path + ".dirty"), path + ".dirty", limits))
    if dirty.size > limits.maxFieldsPerRecord then reject("limit", path, "too many dirty fields")
    if dirty.distinct.size != dirty.size then reject("duplicate-key", path, "duplicate dirty field")
    val opaqueNodes = obj(required(o, "opaque", path), path + ".opaque").entries
    if fieldsObj.entries.size + opaqueNodes.size > limits.maxFieldsPerRecord then reject("limit", path, "too many known/opaque fields")
    val opaque = opaqueNodes.map { (k, v) => k -> raw(v, text) }
    PresetRecord(
      id,
      name(str(required(o, "name", path), path + ".name"), path + ".name", limits),
      fields,
      source(required(o, "source", path), path + ".source", text, limits),
      dirty.toSet,
      opaque,
      extras(o, Set("id", "name", "fields", "source", "dirty", "opaque"), text)
    )
  private def collection(node: Node, path: String, text: String, limits: PresetLimits): PresetCollection =
    val o = obj(node, path)
    val id = idString(required(o, "id", path), path + ".id", limits)
    if !Set("config-txt2img", "config-img2img", "utilities")(id) then reject("id", path, "unknown collection")
    val tracked = arr(required(o, "tracked", path), path + ".tracked").map(x => key(str(x, path + ".tracked"), path + ".tracked", limits))
    if tracked.size > limits.maxRegisteredFieldsPerMode then reject("limit", path, "too many tracked fields")
    if tracked.distinct.size != tracked.size then reject("duplicate-key", path, "duplicate tracked field")
    val bindingNodes = arr(required(o, "bindings", path), path + ".bindings")
    if bindingNodes.size > limits.maxRegisteredFieldsPerMode then reject("limit", path, "too many bindings")
    val bindings = bindingNodes.map { x =>
      val b = obj(x, path + ".bindings")
      if b.entries.size != 3 then reject("schema", path, "binding has unexpected properties")
      SourceBinding(
        idString(required(b, "format", path), path + ".bindings.format", limits),
        idString(required(b, "sourceKey", path), path + ".bindings.sourceKey", limits),
        key(str(required(b, "target", path), path + ".bindings.target"), path + ".bindings.target", limits)
      )
    }
    if bindings.map(b => b.format -> b.sourceKey).distinct.size != bindings.size then
      reject("duplicate-key", path, "duplicate source binding")
    val recordNodes = arr(required(o, "presets", path), path + ".presets")
    if recordNodes.size > limits.maxRecords then reject("limit", path, "too many records")
    val records = recordNodes.zipWithIndex.map { (x, i) =>
      record(x, s"$path.presets[$i]", text, limits)
    }
    if records.map(_.id).distinct.size != records.size then reject("duplicate-id", path, "duplicate preset UUID")
    if records.map(_.name).distinct.size != records.size then reject("duplicate-name", path, "duplicate preset name")
    PresetCollection(id, tracked.toSet, bindings, records, extras(o, Set("id", "tracked", "bindings", "presets"), text))
  private def native(text: String, limits: PresetLimits): PresetLibrary =
    val o = obj(parse(text, limits), "$")
    if str(required(o, "format", "$"), "$.format") != "sdfx-native-presets" then reject("format", "$", "unsupported format")
    required(o, "version", "$" ) match
      case Num("1", _, _) => ()
      case _ => reject("version", "$", "unsupported version")
    val collections = arr(required(o, "collections", "$"), "$.collections")
    if collections.size > limits.maxCollections then reject("limit", "$.collections", "too many collections")
    val totalRecords = collections.map(x => arr(required(obj(x, "$.collections"), "presets", "$.collections"), "$.collections.presets").size).sum
    if totalRecords > limits.maxRecords then reject("limit", "$.collections", "too many records")
    val parsed = collections.zipWithIndex.map { (x, i) => collection(x, s"$$.collections[$i]", text, limits) }
    if parsed.map(_.id).distinct.size != parsed.size then reject("duplicate-id", "$.collections", "duplicate collection")
    PresetLibrary(1, parsed, extras(o, Set("format", "version", "collections"), text))

  def readSource(bytes: Array[Byte], limits: PresetLimits): Either[PresetError, RawJson] = attempt {
    val text = decode(bytes, limits)
    val node = parse(text, limits)
    raw(node, text)
  }

  /** Validated members with exact original key/value tokens. No public parser nodes escape. */
  private[presets] def objectMembers(value: RawJson, limits: PresetLimits): Either[PresetError, Vector[RawMember]] = attempt {
    if value == null then reject("schema", "$", "null raw JSON")
    val source = value.text
    val root = obj(parse(source, limits), "$")
    root.entries.zip(root.keySlices).map { case ((name, item), (start, end)) =>
      RawMember(name, RawJson.validated(source.substring(start, end)), raw(item, source))
    }
  }

  /** Scalar conversion keeps Long and BigDecimal exact; null and containers remain inert. */
  private[presets] def scalarAtom(value: RawJson, limits: PresetLimits): Either[PresetError, Option[PresetAtom]] = attempt {
    if value == null then reject("schema", "$", "null raw JSON")
    parse(value.text, limits) match
      case Str(text, _, _) => Some(PresetAtom.Text(text))
      case Bool(flag, _, _) => Some(PresetAtom.Flag(flag))
      case Num(token, _, _) =>
        if !token.exists(c => c == '.' || c == 'e' || c == 'E') then
          try Some(PresetAtom.Integer(java.lang.Long.parseLong(token)))
          catch case _: NumberFormatException => Some(PresetAtom.Decimal(BigDecimal(new java.math.BigDecimal(token))))
        else Some(PresetAtom.Decimal(BigDecimal(new java.math.BigDecimal(token))))
      case _ => None
  }

  def readNative(bytes: Array[Byte], limits: PresetLimits): Either[PresetError, PresetLibrary] = attempt {
    native(decode(bytes, limits), limits)
  }

  private def boundedDecimalText(value: BigDecimal, budget: Int): String =
    if value == null || budget < 1 then reject("limit", "$write", "decimal exceeds output budget")
    val decimal = value.bigDecimal
    val coefficient = decimal.unscaledValue()
    // bitLength is available without rendering the coefficient. When its cheap bound is
    // inconclusive, compare against a power with at most budget decimal digits.
    val surelyWithin = coefficient.bitLength().toLong <= 3L * budget - 1L
    if !surelyWithin then
      val threshold = java.math.BigInteger.TEN.pow(budget)
      val tooLarge =
        if coefficient.signum() >= 0 then coefficient.compareTo(threshold) >= 0
        else coefficient.compareTo(threshold.negate()) <= 0
      if tooLarge then reject("limit", "$write", "decimal coefficient exceeds output budget")
    // precision() may render internally on some JDK paths; it is reached only after the
    // coefficient has been proven to fit the bounded output budget.
    val precision = decimal.precision().toLong
    val scale = decimal.scale().toLong
    val adjusted = precision - scale - 1L
    val sign = if coefficient.signum() < 0 then 1L else 0L
    val encodedLength =
      if scale >= 0 && adjusted >= -6 then
        sign + (if scale == 0 then precision else if scale < precision then precision + 1L else 2L + scale)
      else
        val coefficientLength = if precision == 1 then 1L else precision + 1L
        coefficientLength + sign + 2L + math.abs(adjusted).toString.length
    if encodedLength > budget then reject("limit", "$write", "decimal exceeds output budget")
    decimal.toString

  private final class Writer(limits: PresetLimits, countOnly: Boolean = false):
    private val out = if countOnly then None else Some(new StringBuilder)
    private var byteCount = 0
    def remainingBytes: Int = limits.maxBytes - byteCount
    def result: Array[Byte] = out.get.toString.getBytes(StandardCharsets.UTF_8)
    def append(s: String): Unit =
      byteCount += utf8Length(s, limits.maxBytes - byteCount, "$write")
      out.foreach(_.append(s))
    def quoted(s: String, cap: Int = -1): Unit =
      utf8Length(s, if cap < 0 then limits.maxScalarBytes else cap, "$write")
      append("\"")
      var i = 0
      while i < s.length do
        val cp = s.codePointAt(i)
        cp match
          case 34 => append("\\\"")
          case 92 => append("\\\\")
          case x if x < 32 => append(f"\\u$x%04X")
          case x => append(new String(Character.toChars(x)))
        i += Character.charCount(cp)
      append("\"")
    def rawValue(value: RawJson): Unit =
      if value == null then reject("schema", "$write", "null raw JSON")
      parse(value.text, limits)
      append(value.text)
    def sourceAtom(value: PresetAtom): Unit =
      if value == null then reject("schema", "$write", "null source atom")
      value match
        case PresetAtom.Text(s) => quoted(s)
        case PresetAtom.Integer(n) =>
          val token = n.toString
          // Ordinary output receives this check in its final lexical read. Include
          // the sign in count-only validation while preserving the writer's order.
          if countOnly && token.length > limits.maxScalarBytes then
            reject("limit", "$write", "numeric scalar too long")
          append(token)
        case PresetAtom.Decimal(n) =>
          append(boundedDecimalText(n, math.min(limits.maxScalarBytes, remainingBytes)))
        case PresetAtom.Flag(b) => append(b.toString)

  private def sourceNodeCount(node: Node): Int = node match
    case o: Obj => 1 + o.entries.map((_, value) => 1 + sourceNodeCount(value)).sum
    case a: Arr => 1 + a.items.map(sourceNodeCount).sum
    case _ => 1

  /** Source dictionary writer: one materialized record at a time, with the same strict
    * scalar, UTF-8, depth and node budgets as the native writer/reader.
    */
  private[presets] def writeSourceRecords(
      records: Iterator[Either[PresetError, (String, Vector[(String, SourceValue)])]],
      limits: PresetLimits): Either[PresetError, Array[Byte]] = attempt {
    val w = new Writer(limits)
    traverseSourceRecords(records, limits, w, countOnly = false)
    val bytes = w.result
    readSource(bytes, limits) match
      case Left(e) =>
        val issue = e.issues.head
        reject(issue.code, issue.path, issue.message)
      case Right(_) => bytes
  }

  /** Same source traversal and byte/node budgets, without an output buffer or encoded bytes. */
  private[presets] def validateSourceRecords(
      records: Iterator[Either[PresetError, (String, Vector[(String, SourceValue)])]],
      limits: PresetLimits): Either[PresetError, Unit] = attempt {
    traverseSourceRecords(records, limits, new Writer(limits, countOnly = true), countOnly = true)
  }

  /** Validate even an untouched/unsupported native scalar without charging it to source output twice. */
  private[presets] def validateSourceAtom(value: PresetAtom, limits: PresetLimits): Either[PresetError, Unit] =
    attempt { new Writer(limits, countOnly = true).sourceAtom(value) }

  private def traverseSourceRecords(
      records: Iterator[Either[PresetError, (String, Vector[(String, SourceValue)])]],
      limits: PresetLimits, w: Writer, countOnly: Boolean): Unit =
    if records == null then reject("schema", "$write", "null source records")
    if countOnly && (limits.maxDepth < 1 || limits.maxNodes < 1) then
      reject("limit", "$write", "source root exceeds JSON limits")
    val seenNames = mutable.HashSet.empty[String]
    var recordCount = 0
    var nodes = 1L // root object
    w.append("{")
    while records.hasNext do
      recordCount += 1
      if recordCount > limits.maxRecords then reject("limit", "$write", "too many source records")
      val (recordName, entries) = records.next() match
        case Right(value) => value
        case Left(e) =>
          val issue = e.issues.headOption.getOrElse(PresetIssue("source", "$write", "source record failed"))
          reject(issue.code, issue.path, issue.message)
      if recordName == null then reject("name", "$write", "null preset name")
      name(recordName, "$write", limits)
      // The ordinary writer's final readSource enforces name-key bytes and the two
      // enclosing object depths. Count-only traversal must enforce those without encoding.
      if countOnly then
        utf8Length(recordName, limits.maxKeyBytes, "$write")
        if limits.maxDepth < 2 then reject("limit", recordName, "JSON nesting too deep")
      if !seenNames.add(recordName) then reject("duplicate-name", recordName, "duplicate source preset name")
      if entries == null || entries.size > limits.maxFieldsPerRecord then
        reject("limit", recordName, "too many source fields")
      val seenKeys = mutable.HashSet.empty[String]
      nodes += 2 // record name key and field object
      if nodes > limits.maxNodes then reject("limit", recordName, "too many JSON nodes")
      if recordCount > 1 then w.append(",")
      w.quoted(recordName)
      w.append(":{")
      entries.zipWithIndex.foreach { case ((sourceKey, value), index) =>
        if sourceKey == null || sourceKey.isEmpty then reject("source-key", recordName, "empty source key")
        utf8Length(sourceKey, limits.maxKeyBytes, recordName)
        if !seenKeys.add(sourceKey) then reject("duplicate-key", recordName, "duplicate source key")
        val nested = value match
          case SourceValue.Raw(raw) =>
            if raw == null then reject("schema", sourceKey, "null raw source value")
            sourceNodeCount(parse(raw.text,
              if countOnly then limits.copy(maxDepth = limits.maxDepth - 2) else limits))
          case SourceValue.Atom(atom) =>
            if atom == null then reject("schema", sourceKey, "null source atom")
            1
        nodes += 1L + nested
        if nodes > limits.maxNodes then reject("limit", sourceKey, "too many JSON nodes")
        if index > 0 then w.append(",")
        w.quoted(sourceKey, limits.maxKeyBytes)
        w.append(":")
        value match
          case SourceValue.Raw(raw) => w.append(raw.text)
          case SourceValue.Atom(atom) => w.sourceAtom(atom)
      }
      w.append("}")
    w.append("}")

  private def emit(library: PresetLibrary, limits: PresetLimits): Array[Byte] =
    if library == null then reject("schema", "$write", "null library")
    if library.version != 1 then reject("version", "$write", "unsupported version")
    if library.collections.size > limits.maxCollections || library.collections.map(_.presets.size).sum > limits.maxRecords then
      reject("limit", "$write", "too many collections or records")
    val w = new Writer(limits)
    def fieldId(k: FieldKey): String =
      if k == null || k.segments.isEmpty || k.segments.exists(s => s == null || s.isEmpty) then
        reject("field-key", "$write", "invalid field key")
      var sourceBytes = k.segments.size - 1
      k.segments.foreach(s => sourceBytes += utf8Length(s, limits.maxKeyBytes, "$write"))
      if sourceBytes > limits.maxKeyBytes then reject("limit", "$write", "field key too long")
      val id = k.render
      utf8Length(id, limits.maxKeyBytes, "$write")
      if FieldKey.parse(id).isLeft then reject("field-key", "$write", "invalid field key")
      id
    def field(k: FieldKey): Unit =
      val id = fieldId(k)
      w.quoted(id, limits.maxKeyBytes)
    def rawProperties(values: Vector[(String, RawJson)]): Unit =
      values.foreach { (k, v) =>
        w.append(",")
        w.quoted(k, limits.maxKeyBytes)
        w.append(":")
        w.rawValue(v)
      }
    def atom(value: PresetAtom): Unit =
      if value == null then reject("schema", "$write", "null atom")
      val kind = value match
        case _: PresetAtom.Text => "text"
        case _: PresetAtom.Integer => "integer"
        case _: PresetAtom.Decimal => "decimal"
        case _: PresetAtom.Flag => "flag"
      w.append("{\"type\":")
      w.quoted(kind)
      w.append(",\"value\":")
      value match
        case PresetAtom.Text(s) => w.quoted(s)
        case PresetAtom.Integer(n) => w.quoted(n.toString)
        case PresetAtom.Decimal(n) =>
          val budget = math.min(limits.maxScalarBytes, w.remainingBytes - 3)
          w.quoted(boundedDecimalText(n, budget))
        case PresetAtom.Flag(b) => w.quoted(b.toString)
      w.append("}")
    def source(s: Option[PresetSource]): Unit = s match
      case None => w.append("null")
      case Some(p) =>
        w.append("{\"format\":")
        w.quoted(p.format, limits.maxKeyBytes)
        w.append(",\"name\":")
        w.quoted(p.name)
        w.append(",\"mode\":")
        p.mode match
          case None => w.append("null")
          case Some(PresetMode.Txt2Img) => w.quoted("txt2img")
          case Some(PresetMode.Img2Img) => w.quoted("img2img")
          case Some(PresetMode.Inpaint) => w.quoted("inpaint")
        w.append(",\"originalFields\":")
        w.rawValue(p.originalFields)
        w.append("}")
    w.append("{\"format\":\"sdfx-native-presets\",\"version\":1,\"collections\":[")
    library.collections.zipWithIndex.foreach { (c, ci) =>
      if ci > 0 then w.append(",")
      if c.tracked.size > limits.maxRegisteredFieldsPerMode || c.bindings.size > limits.maxRegisteredFieldsPerMode then
        reject("limit", "$write", "too many registered fields or bindings")
      w.append("{\"id\":")
      w.quoted(c.id, limits.maxKeyBytes)
      w.append(",\"tracked\":[")
      c.tracked.toVector.sortBy(fieldId).zipWithIndex.foreach { (k, i) =>
        if i > 0 then w.append(",")
        field(k)
      }
      w.append("],\"bindings\":[")
      c.bindings.zipWithIndex.foreach { (b, i) =>
        if i > 0 then w.append(",")
        w.append("{\"format\":")
        w.quoted(b.format, limits.maxKeyBytes)
        w.append(",\"sourceKey\":")
        w.quoted(b.sourceKey, limits.maxKeyBytes)
        w.append(",\"target\":")
        field(b.target)
        w.append("}")
      }
      w.append("],\"presets\":[")
      c.presets.zipWithIndex.foreach { (r, ri) =>
        if ri > 0 then w.append(",")
        if r.fields.size + r.opaque.size > limits.maxFieldsPerRecord || r.dirty.size > limits.maxFieldsPerRecord then
          reject("limit", "$write", "too many fields")
        w.append("{\"id\":")
        w.quoted(r.id.toString)
        w.append(",\"name\":")
        w.quoted(r.name)
        w.append(",\"fields\":{")
        r.fields.toVector.sortBy(entry => fieldId(entry._1)).zipWithIndex.foreach { (entry, i) =>
          if i > 0 then w.append(",")
          field(entry._1)
          w.append(":")
          atom(entry._2)
        }
        w.append("},\"source\":")
        source(r.source)
        w.append(",\"dirty\":[")
        r.dirty.toVector.sortBy(fieldId).zipWithIndex.foreach { (k, i) =>
          if i > 0 then w.append(",")
          field(k)
        }
        w.append("],\"opaque\":{")
        r.opaque.zipWithIndex.foreach { (entry, i) =>
          if i > 0 then w.append(",")
          w.quoted(entry._1, limits.maxKeyBytes)
          w.append(":")
          w.rawValue(entry._2)
        }
        w.append("}")
        rawProperties(r.extra)
        w.append("}")
      }
      w.append("]")
      rawProperties(c.extra)
      w.append("}")
    }
    w.append("]")
    rawProperties(library.extra)
    w.append("}")
    w.result

  def writeNative(library: PresetLibrary, limits: PresetLimits): Either[PresetError, Array[Byte]] = attempt {
    val bytes = emit(library, limits)
    // The reader is the single validation authority for duplicate/colliding keys and every native invariant.
    readNative(bytes, limits) match
      case Left(error) => throw Problem(error.issues.head.code, error.issues.head.path, error.issues.head.message)
      case Right(_) => bytes
  }
