package sdfx.core.payload

import scala.collection.mutable.ArrayDeque
import scala.util.control.NonFatal

/** Limits diagnostic allocation before JSON parsing/rendering. This lexical guard is not a syntax validator.
  * Tokens conservatively include object keys as well as containers and scalar values.
  * View expansion's separate depth/node limits do not change these whole-document admission limits.
  */
object PayloadJsonBudget:
  final case class Limits(maxBytes: Int = 24 * 1024 * 1024, maxNodes: Int = 100000, maxDepth: Int = 512):
    require(maxBytes >= 0 && maxBytes < Int.MaxValue && maxNodes > 0 && maxDepth > 0)

  def guard(frozen: CharSequence, limits: Limits = Limits()): Either[String, Unit] =
    var index = 0
    var bytes = 0L
    var nodes = 0
    var depth = 0
    var quoted = false
    var escaped = false
    var atom = false
    val length = frozen.length()
    while index < length do
      val char = frozen.charAt(index)
      val pair = Character.isHighSurrogate(char) && index + 1 < length && Character.isLowSurrogate(frozen.charAt(index + 1))
      bytes += (if pair then 4 else if char <= 0x7f then 1 else if char <= 0x7ff then 2 else if Character.isSurrogate(char) then 1 else 3)
      if bytes > limits.maxBytes then return Left("bytes_limit")
      if quoted then
        if escaped then escaped = false
        else if char == '\\' then escaped = true
        else if char == '"' then quoted = false
      else
        char match
          case '"' => nodes += 1; quoted = true; atom = false
          case '{' | '[' => nodes += 1; depth += 1; atom = false
          case '}' | ']' => depth -= 1; atom = false
          case ',' | ':' | ' ' | '\n' | '\r' | '\t' => atom = false
          case _ => if !atom then { nodes += 1; atom = true }
      if nodes > limits.maxNodes then return Left("nodes_limit")
      if depth > limits.maxDepth then return Left("depth_limit")
      index += (if pair then 2 else 1)
    Right(())

  def admit[A](frozen: CharSequence, limits: Limits = Limits())(parse: => A): Either[String, A] =
    guard(frozen, limits).flatMap { _ =>
      try Right(parse)
      catch case NonFatal(_) => Left("invalid_json")
    }

  /** Cursor frames retain only the current container iterator, never an object-wide child copy/frontier. */
  def write(value: ujson.Value, limits: Limits = Limits()): Either[String, String] =
    sealed trait Work
    final case class Value(value: ujson.Value, depth: Int) extends Work
    final case class ObjectCursor(entries: Iterator[(String, ujson.Value)], depth: Int, first: Boolean) extends Work
    final case class ArrayCursor(entries: Iterator[ujson.Value], depth: Int, first: Boolean) extends Work
    val pending = ArrayDeque[Work](Value(value, 0))
    val output = new Writer(limits.maxBytes)
    var nodes = 0
    try
      while pending.nonEmpty do
        pending.removeLast() match
          case Value(item, depth) =>
            nodes += 1
            if nodes > limits.maxNodes then return Left("nodes_limit")
            item match
              case obj: ujson.Obj =>
                if depth + 1 > limits.maxDepth then return Left("depth_limit")
                output.appendRaw("{")
                pending.append(ObjectCursor(obj.value.iterator, depth + 1, true))
              case arr: ujson.Arr =>
                if depth + 1 > limits.maxDepth then return Left("depth_limit")
                output.appendRaw("[")
                pending.append(ArrayCursor(arr.value.iterator, depth + 1, true))
              case ujson.Str(text) => output.appendQuoted(text)
              case ujson.Bool(boolean) => output.appendRaw(if boolean then "true" else "false")
              case ujson.Null => output.appendRaw("null")
              // Numeric scalar rendering has constant bounded size; documents and strings never use ujson.write.
              case ujson.Num(number) =>
                if !number.isFinite then return Left("nonfinite_number")
                output.appendRaw(ujson.write(ujson.Num(number)))
          case ObjectCursor(entries, depth, first) =>
            if !entries.hasNext then output.appendRaw("}")
            else
              nodes += 1 // Object key.
              if nodes > limits.maxNodes then return Left("nodes_limit")
              val (key, child) = entries.next()
              if !first then output.appendRaw(",")
              output.appendQuoted(key)
              output.appendRaw(":")
              pending.append(ObjectCursor(entries, depth, false))
              pending.append(Value(child, depth))
          case ArrayCursor(entries, depth, first) =>
            if !entries.hasNext then output.appendRaw("]")
            else
              if !first then output.appendRaw(",")
              pending.append(ArrayCursor(entries, depth, false))
              pending.append(Value(entries.next(), depth))
        if output.bytesSeen > limits.maxBytes then return Left("bytes_limit")
      output.result.toRight("bytes_limit")
    catch case NonFatal(_) => Left("invalid_json")

  /** Incremental UTF-8/JSON string writer shared with native image diagnostics. */
  class Writer(byteLimit: Int):
    private val limit = math.max(0, byteLimit)
    private val output = new java.lang.StringBuilder(math.min(1024, limit))
    private var seen = 0
    private var overflow = false
    def bytesSeen: Int = seen
    def result: Option[String] = if overflow then None else Some(output.toString)

    private def reserve(bytes: Int): Boolean =
      if overflow then false
      else if seen.toLong + bytes > limit then
        seen = limit + 1
        overflow = true
        false
      else
        seen += bytes
        true

    private def ascii(c: Char): Unit = if reserve(1) then output.append(c)
    private def escape(c: Char): Unit =
      if reserve(6) then
        output.append("\\u")
        var shift = 12
        while shift >= 0 do
          output.append("0123456789abcdef".charAt((c.toInt >> shift) & 15))
          shift -= 4

    private def ordinary(text: CharSequence, i: Int, c: Char): Int =
      if Character.isHighSurrogate(c) && i + 1 < text.length && Character.isLowSurrogate(text.charAt(i + 1)) then
        if reserve(4) then
          output.append(c)
          output.append(text.charAt(i + 1))
        i + 2
      else if Character.isSurrogate(c) then
        escape(c)
        i + 1
      else
        if reserve(if c <= 0x7f then 1 else if c <= 0x7ff then 2 else 3) then output.append(c)
        i + 1

    def appendRaw(text: CharSequence): Unit =
      var i = 0
      while !overflow && i < text.length do
        i = ordinary(text, i, text.charAt(i))

    def appendQuoted(text: CharSequence): Unit =
      ascii('"')
      var i = 0
      while !overflow && i < text.length do
        val c = text.charAt(i)
        if c == '"' || c == '\\' then
          if reserve(2) then output.append('\\').append(c)
          i += 1
        else if c < ' ' then
          escape(c)
          i += 1
        else i = ordinary(text, i, c)
      ascii('"')
