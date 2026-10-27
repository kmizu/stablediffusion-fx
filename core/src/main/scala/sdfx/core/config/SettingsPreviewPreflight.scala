package sdfx.core.config

import scala.collection.mutable
import java.nio.charset.StandardCharsets

/** Strict streaming preflight: preview values are admitted incrementally and removed from
  * the document before the general JSON parser can materialize them. */
object SettingsPreviewPreflight:
  final case class Result(json: String, overrides: Map[String, String], folders: Map[String, Seq[String]],
      problem: Option[String], retainedUnits: Long)

  def apply(input: String): Result = new Reader(input).run()

  private final class Reader(input: String):
    private var i = 0
    private var units = 0L
    private var bytes = 64L
    private var roots = 0
    private var overridesCount = 0
    private var kindsCount = 0
    private var problem: Option[String] = None
    private var fieldProblem = false
    private var overrideUnits = 0L
    private var overrideBytes = 0L
    private var folderUnits = 0L
    private var folderBytes = 0L
    private val overrides = mutable.LinkedHashMap.empty[String, String]
    private val folders = mutable.LinkedHashMap.empty[String, Seq[String]]
    private val cleaned = new java.lang.StringBuilder
    private var copied = 0
    private var seenOverrides = false
    private var seenFolders = false

    private def fail(message: String): Unit =
      fieldProblem = true
      if problem.isEmpty then problem = Some(message)
    private def syntax(): Nothing = throw new IllegalArgumentException(s"invalid JSON at offset $i")
    private def ws(): Unit = while i < input.length && " \t\r\n".contains(input.charAt(i)) do i += 1
    private def take(c: Char): Unit = { ws(); if i >= input.length || input.charAt(i) != c then syntax(); i += 1 }
    private def accept(c: Char): Boolean =
      ws()
      if i < input.length && input.charAt(i) == c then { i += 1; true } else false

    // Captures at most max UTF-16 units. Ignored strings are validated without allocation.
    private def string(max: Int): Option[String] =
      take('"')
      val b = if max >= 0 then new StringBuilder else null
      var size = 0
      def add(c: Char): Unit =
        size += 1
        if b != null && size <= max then b.append(c)
      var closed = false
      while i < input.length && !closed do
        val c = input.charAt(i); i += 1
        if c == '"' then closed = true
        else if c < ' ' then syntax()
        else if c != '\\' then add(c)
        else
          if i >= input.length then syntax()
          val e = input.charAt(i); i += 1
          e match
            case '"' => add('"')
            case '\\' => add('\\')
            case '/' => add('/')
            case 'b' => add('\b')
            case 'f' => add('\f')
            case 'n' => add('\n')
            case 'r' => add('\r')
            case 't' => add('\t')
            case 'u' =>
              if i + 4 > input.length then syntax()
              var v = 0
              for _ <- 0 until 4 do
                val h = input.charAt(i)
                val digit = if h >= '0' && h <= '9' then h - '0'
                  else if h >= 'a' && h <= 'f' then h - 'a' + 10
                  else if h >= 'A' && h <= 'F' then h - 'A' + 10
                  else -1
                if digit < 0 then syntax()
                v = v * 16 + digit; i += 1
              add(v.toChar)
            case _ => syntax()
      if !closed then syntax()
      if b == null || size > max then None else Some(b.toString)

    private def budget(value: Option[String], max: Int): Option[String] =
      if fieldProblem then None
      else value match
        case None => fail(s"preview string exceeds $max UTF-16 units"); None
        case Some(v) =>
          val nextUnits = units + v.length
          val nextBytes = bytes + ujson.write(v).getBytes(StandardCharsets.UTF_8).length + 16L
          if nextUnits > 1000000L || nextBytes > 4L * 1024 * 1024 then
            fail("preview configuration aggregate limit exceeded"); None
          else
            units = nextUnits
            bytes = nextBytes
            Some(v)

    private def generic(depth: Int): Unit =
      if depth > 128 then throw new IllegalArgumentException("settings JSON nesting exceeds 128")
      ws()
      if i >= input.length then syntax()
      input.charAt(i) match
        case '"' => string(-1); ()
        case '{' =>
          take('{')
          if !accept('}') then
            var more = true
            while more do
              string(-1); take(':'); generic(depth + 1)
              if accept('}') then more = false else take(',')
        case '[' =>
          take('[')
          if !accept(']') then
            var more = true
            while more do
              generic(depth + 1)
              if accept(']') then more = false else take(',')
        case _ =>
          val start = i
          while i < input.length && !",]} \t\r\n".contains(input.charAt(i)) do i += 1
          if start == i then syntax()
          val token = input.substring(start, i)
          if token != "true" && token != "false" && token != "null" &&
              !token.matches("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?") then syntax()

    private def valueString(max: Int): Option[String] =
      ws()
      if i < input.length && input.charAt(i) == '"' then budget(string(if !fieldProblem then max else -1), max)
      else { fail("preview value must be a string"); generic(1); None }

    private def overrideObject(): Unit =
      ws()
      if i >= input.length || input.charAt(i) != '{' then
        fail("preview overrides must be an object"); generic(1)
      else
        take('{')
        if !accept('}') then
          var more = true
          while more do
            overridesCount += 1
            if overridesCount > 10000 then fail("too many preview overrides")
            val key = budget(string(if !fieldProblem then 8192 else -1), 8192)
            take(':')
            val value = valueString(8192)
            for k <- key; v <- value do overrides.update(k, v)
            if accept('}') then more = false else take(',')

    private def folderObject(): Unit =
      ws()
      if i >= input.length || input.charAt(i) != '{' then
        fail("preview folders must be an object"); generic(1)
      else
        take('{')
        if !accept('}') then
          var more = true
          while more do
            kindsCount += 1
            if kindsCount > 64 then fail("too many preview kinds")
            val key = budget(string(if !fieldProblem then 128 else -1), 128)
            take(':'); ws()
            val paths = mutable.ArrayBuffer.empty[String]
            if i < input.length && input.charAt(i) == '[' then
              take('[')
              if !accept(']') then
                var morePaths = true
                while morePaths do
                  roots += 1
                  if roots > 256 then fail("too many preview roots")
                  valueString(8192).foreach(paths += _)
                  if accept(']') then morePaths = false else take(',')
            else { fail("preview roots must be an array"); generic(1) }
            key.foreach(k => if !fieldProblem then folders.update(k, paths.toSeq))
            if accept('}') then more = false else take(',')

    def run(): Result =
      take('{')
      if !accept('}') then
        var more = true
        while more do
          val key = string(128)
          take(':'); ws()
          val start = i
          key match
            case Some("model_preview_overrides") =>
              fieldProblem = false
              if seenOverrides then
                units -= overrideUnits; bytes -= overrideBytes
                overrideUnits = 0; overrideBytes = 0; overrides.clear()
                fail("duplicate preview overrides")
              val baseUnits = units; val baseBytes = bytes
              seenOverrides = true; overrideObject()
              if fieldProblem then
                units = baseUnits; bytes = baseBytes; overrides.clear()
              else
                overrideUnits = units - baseUnits; overrideBytes = bytes - baseBytes
              cleaned.append(input, copied, start); cleaned.append("{}"); copied = i
            case Some("model_preview_folders") =>
              fieldProblem = false
              if seenFolders then
                units -= folderUnits; bytes -= folderBytes
                folderUnits = 0; folderBytes = 0; folders.clear()
                fail("duplicate preview folders")
              val baseUnits = units; val baseBytes = bytes
              seenFolders = true; folderObject()
              if fieldProblem then
                units = baseUnits; bytes = baseBytes; folders.clear()
              else
                folderUnits = units - baseUnits; folderBytes = bytes - baseBytes
              cleaned.append(input, copied, start); cleaned.append("{}"); copied = i
            case _ => generic(1)
          if accept('}') then more = false else take(',')
      ws()
      if i != input.length then syntax()
      cleaned.append(input, copied, input.length)
      Result(cleaned.toString, overrides.toMap, folders.toMap, problem, units)
