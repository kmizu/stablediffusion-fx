package sdfx.core.model

import java.util.regex.Pattern

/** Pure pre-style prompt composition; source behavior documented in the Hires Fix Tweaks plan. */
object HiresPromptModes:
  val modes = Seq("Default", "Append", "Prepend", "Prompt S/R")
  private val networks = "(?U)<(\\w+):([^>]+)>".r
  // Deliberately preserve the upstream greedy group's trailing newline behavior.
  private val edgeNewline = "(?s)^\\r?\\n?(.*)\\r?\\n?$".r

  def expandSeparator(source: String): String =
    val output = new StringBuilder
    var index = 0
    while index < source.length do
      if source.startsWith("{{", index) then { output.append('{'); index += 2 }
      else if source.startsWith("}}", index) then { output.append('}'); index += 2 }
      else if source.startsWith("{newline}", index) then { output.append('\n'); index += 9 }
      else
        require(source(index) != '{' && source(index) != '}', "Separator accepts only {newline} and escaped braces")
        output.append(source(index)); index += 1
    output.toString

  def transform(base: String, hires: String, mode: String, removeNetworks: Boolean = false,
                separator: String = "{newline}", marker: String = "@"): (String, String) =
    require(modes.contains(mode), "Unknown Hires prompt mode")
    val stripped = if removeNetworks then networks.replaceAllIn(base, "") else base
    mode match
      case "Default" => (base, if hires.isEmpty then stripped else hires)
      case "Append" | "Prepend" =>
        if removeNetworks || hires.strip.nonEmpty then
          val sep = expandSeparator(separator)
          (base, if mode == "Append" then stripped + sep + hires else hires + sep + stripped)
        else (base, hires)
      case _ =>
        require(marker.codePointCount(0, marker.length) == 1 && !marker.matches("(?U)[\\s\\w]"),
          "Marker must be one non-word, non-whitespace character")
        val quoted = Pattern.quote(marker)
        val instructions = Pattern.compile(s"^$quoted((?:[^$quoted]|$quoted$quoted)+)$quoted", Pattern.MULTILINE)
        val matcher = instructions.matcher(hires)
        val entries = Vector.newBuilder[(String, Int, Int)]
        while matcher.find() do entries += ((matcher.group(1), matcher.start(), matcher.end()))
        val found = entries.result()
        var first = base
        var second = stripped
        found.zipWithIndex.foreach { case ((rawKey, _, end), index) =>
          val key = rawKey.replace(marker + marker, marker)
          val raw = hires.substring(end, found.lift(index + 1).map(_._2).getOrElse(hires.length))
            .replace(marker + marker, marker)
          val replacement = edgeNewline.findFirstMatchIn(raw).get.group(1)
          val insertion = marker + key + marker
          if first.contains(insertion) then
            first = first.replace(insertion, "")
            second = second.replace(insertion, replacement)
          else second = second.replace(key, replacement)
        }
        (first, second)
