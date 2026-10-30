package sdfx.core.infotext

import scala.collection.immutable.ListMap
import scala.util.Try

/** Read-only A1111 provenance; native NUI saves raw expressions directly. */
object EmbeddingMergeMetadata:
  val key = "EmbeddingMerge"
  private val limit = 65536
  private val keyPattern = """(?:<'EM[_/-][0-9]+'>|\{'EM[_/-][0-9]+'\})=""".r

  private def span(text: String, start: Int): Int =
    require(start + 1 < text.length && (text(start) == '<' || text(start) == '{') && text(start + 1) == '\'', "Expected merge expression")
    val closing = if text(start) == '<' then '>' else '}'
    var quoted = false
    var p = start + 1
    while p < text.length do
      if text(p) == '\'' then
        if quoted && p + 1 < text.length && text(p + 1) == '\'' then p += 1
        else quoted = !quoted
      else if text(p) == closing && !quoted then return p + 1
      p += 1
    throw new IllegalArgumentException("Unclosed merge expression")

  def mapping(raw: String): Option[ListMap[String, String]] = Try {
    require(raw.length <= limit)
    val text = if raw.startsWith("\"") then ujson.read(raw).str else raw
    var p = 0
    var result = ListMap.empty[String, String]
    while p < text.length do
      while p < text.length && text(p).isWhitespace do p += 1
      val found = keyPattern.findPrefixOf(text.substring(p)).getOrElse(throw new IllegalArgumentException("Invalid merge provenance"))
      val name = found.dropRight(1)
      require(!result.contains(name), "Duplicate merge provenance")
      p += found.length
      val end = span(text, p)
      result = result.updated(name, text.substring(p, end))
      p = end
      while p < text.length && text(p).isWhitespace do p += 1
      if p < text.length then
        require(text(p) == ',' && p + 1 < text.length)
        p += 1
    require(result.nonEmpty)
    result
  }.toOption

  private def substitute(text: String, replacements: ListMap[String, String]): String =
    require(text.length <= limit)
    val output = new StringBuilder
    var p = 0
    while p < text.length do
      if p + 1 < text.length && (text(p) == '<' || text(p) == '{') && text(p + 1) == '\'' then
        val end = span(text, p)
        val original = text.substring(p, end)
        output.append(replacements.getOrElse(original, original))
        p = end
      else
        output.append(text(p))
        p += 1
    output.toString

  def restore(parsed: ParsedInfotext): Option[ParsedInfotext] =
    parsed.get(key).flatMap(mapping).flatMap { replacements => Try {
      val fields = (parsed.fields - key).map { (name, value) =>
        name -> (if name == "Hires prompt" || name == "Hires negative prompt" then substitute(value, replacements) else value)
      }
      ParsedInfotext(substitute(parsed.prompt, replacements), substitute(parsed.negativePrompt, replacements), ListMap.from(fields))
    }.toOption }
