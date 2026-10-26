package sdfx.core.presets

import java.nio.ByteBuffer
import java.nio.charset.{CodingErrorAction, StandardCharsets}
import java.util.UUID

enum PresetProfile:
  case Config, Utilities

enum PresetMode:
  case Txt2Img, Img2Img, Inpaint

enum ApplyBase:
  case Current, Defaults

final case class PresetIssue(code: String, path: String, message: String)
final case class PresetError(issues: Vector[PresetIssue])

final case class PresetLimits(
  maxBytes: Int = 4194304,
  maxDepth: Int = 32,
  maxNodes: Int = 100000,
  maxCollections: Int = 3,
  maxRecords: Int = 1000,
  maxFieldsPerRecord: Int = 2048,
  maxRegisteredFieldsPerMode: Int = 4096,
  maxKeyBytes: Int = 512,
  maxScalarBytes: Int = 262144,
  maxCustomFileBytes: Int = 262144,
  maxBindings: Int = 2048,
  maxNameCodePoints: Int = 256
)

final case class FieldKey(segments: Vector[String]):
  def render: String = segments.map(FieldKey.encodeSegment).mkString(".")

object FieldKey:
  private def error(message: String): Left[PresetError, FieldKey] =
    Left(PresetError(Vector(PresetIssue("field-key", "fieldKey", message))))

  private def utf8(s: String): Option[Array[Byte]] =
    try
      val encoder = StandardCharsets.UTF_8.newEncoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
      val b = encoder.encode(java.nio.CharBuffer.wrap(s))
      val out = new Array[Byte](b.remaining())
      b.get(out)
      Some(out)
    catch case _: java.nio.charset.CharacterCodingException => None

  private def boundedUtf8Length(s: String, maximum: Int): Option[Int] =
    if s == null || s.length > maximum then return None
    var i = 0
    var bytes = 0
    while i < s.length do
      val first = s.charAt(i)
      val cp =
        if Character.isHighSurrogate(first) then
          if i + 1 >= s.length || !Character.isLowSurrogate(s.charAt(i + 1)) then return None
          Character.toCodePoint(first, s.charAt(i + 1))
        else if Character.isLowSurrogate(first) then return None
        else first.toInt
      bytes += (if cp < 0x80 then 1 else if cp < 0x800 then 2 else if cp < 0x10000 then 3 else 4)
      if bytes > maximum then return None
      i += Character.charCount(cp)
    Some(bytes)

  private[presets] def encodeSegment(s: String): String =
    val b = new StringBuilder
    var i = 0
    while i < s.length do
      val cp = s.codePointAt(i)
      if cp == '.'.toInt || cp == '%'.toInt || Character.isISOControl(cp) then
        val part = new String(Character.toChars(cp)).getBytes(StandardCharsets.UTF_8)
        part.foreach { byte =>
          val n = byte & 0xff
          b.append('%')
          b.append("0123456789ABCDEF".charAt(n >>> 4))
          b.append("0123456789ABCDEF".charAt(n & 15))
        }
      else b.append(new String(Character.toChars(cp)))
      i += Character.charCount(cp)
    b.toString

  def parse(id: String): Either[PresetError, FieldKey] =
    if id == null then return error("null ID")
    if id.isEmpty || boundedUtf8Length(id, 512).isEmpty then return error("invalid or oversized ID")
    val parts = id.split("\\.", -1).toVector
    if parts.exists(_.isEmpty) then return error("empty ID segment")
    val decoded = parts.map { part =>
      val b = new java.io.ByteArrayOutputStream()
      var i = 0
      var bad = false
      while i < part.length && !bad do
        if part.charAt(i) == '%' then
          if i + 2 >= part.length then bad = true
          else
            val hi = Character.digit(part.charAt(i + 1), 16)
            val lo = Character.digit(part.charAt(i + 2), 16)
            if hi < 0 || lo < 0 then bad = true
            else
              b.write((hi << 4) | lo)
              i += 3
        else
          val cp = part.codePointAt(i)
          val chars = new String(Character.toChars(cp))
          utf8(chars) match
            case Some(bs) => b.write(bs)
            case None => bad = true
          i += Character.charCount(cp)
      if bad then None
      else
        try
          val decoder = StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
          Some(decoder.decode(ByteBuffer.wrap(b.toByteArray)).toString)
        catch case _: java.nio.charset.CharacterCodingException => None
    }
    if decoded.exists(_.isEmpty) then return error("invalid percent encoding")
    val key = FieldKey(decoded.flatten)
    if key.render != id then error("noncanonical ID") else Right(key)

enum PresetAtom:
  case Text(value: String)
  case Integer(value: Long)
  case Decimal(value: BigDecimal)
  case Flag(value: Boolean)

final class RawJson private (val text: String):
  override def equals(other: Any): Boolean = other match
    case that: RawJson => text == that.text
    case _ => false
  override def hashCode(): Int = text.hashCode
  override def toString: String = s"RawJson(${text.length} chars)"

object RawJson:
  private[presets] def validated(text: String): RawJson = new RawJson(text)

final case class PresetSource(format: String, name: String, mode: Option[PresetMode], originalFields: RawJson)

final case class PresetRecord(
  id: UUID,
  name: String,
  fields: Map[FieldKey, PresetAtom],
  source: Option[PresetSource],
  dirty: Set[FieldKey],
  opaque: Vector[(String, RawJson)],
  extra: Vector[(String, RawJson)]
)

object PresetRecord:
  def initialId(collectionId: String, name: String): UUID =
    UUID.nameUUIDFromBytes((collectionId + "\u0000" + name).getBytes(StandardCharsets.UTF_8))

final case class SourceBinding(format: String, sourceKey: String, target: FieldKey)

final case class PresetCollection(
  id: String,
  tracked: Set[FieldKey],
  bindings: Vector[SourceBinding],
  presets: Vector[PresetRecord],
  extra: Vector[(String, RawJson)]
)

final case class PresetLibrary(version: Int, collections: Vector[PresetCollection], extra: Vector[(String, RawJson)])
