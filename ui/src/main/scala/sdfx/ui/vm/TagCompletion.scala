package sdfx.ui.vm

import java.nio.file.{Files, Path}
import java.util.Locale

case class CompletionTag(name: String, count: Long, aliases: Vector[String]):
  def insertion: String = name.replace('_', ' ').replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

case class CompletionToken(start: Int, end: Int, query: String)

object TagCompletion:
  private def normalized(s: String): String = s.toLowerCase(Locale.ROOT).replace('_', ' ')
  def suggestions(tags: Vector[CompletionTag], query: String, limit: Int = 12): Vector[CompletionTag] =
    val q = normalized(query.trim)
    if q.isEmpty then Vector.empty
    else tags.iterator.filter(t => normalized(t.name).startsWith(q) || t.aliases.exists(a => normalized(a).startsWith(q))).take(limit).toVector

  /** Prompt syntax boundaries are retained verbatim, including weight suffixes. */
  def token(text: String, caret: Int, selectionStart: Int, selectionEnd: Int): Option[CompletionToken] =
    if caret < 0 || caret > text.length then return None
    def boundary(i: Int): Boolean =
      val c = text.charAt(i)
      ",\n\r:()[]<>|".contains(c) && (i == 0 || text.charAt(i - 1) != '\\')
    var left = caret
    var right = caret
    while left > 0 && !boundary(left - 1) do left -= 1
    while right < text.length && !boundary(right) do right += 1
    while left < right && text.charAt(left).isWhitespace do left += 1
    while right > left && text.charAt(right - 1).isWhitespace do right -= 1
    if left >= right || caret < left || caret > right || selectionStart < left || selectionEnd > right then None
    else
      val q = text.substring(left, caret).trim
      // Avoid completing weight numbers or names inside extra-network syntax.
      val inNetwork = text.lastIndexOf('<', left) > text.lastIndexOf('>', left)
      if q.isEmpty || inNetwork || q.forall(c => c.isDigit || c == '.' || c == '-' || c == '+') then None
      else Some(CompletionToken(left, right, q))

  def csvRow(line: String): Vector[String] =
    val result = Vector.newBuilder[String]
    val field = new StringBuilder
    var quoted = false
    var i = 0
    while i < line.length do
      line.charAt(i) match
        case '"' if quoted && i + 1 < line.length && line.charAt(i + 1) == '"' => field.append('"'); i += 1
        case '"' => quoted = !quoted
        case ',' if !quoted => result += field.toString; field.clear()
        case c => field.append(c)
      i += 1
    require(!quoted, "Unclosed CSV quote")
    result += field.toString
    result.result()

  def parse(lines: Iterator[String]): Vector[CompletionTag] =
    var columns = Map("name" -> 0, "count" -> 2, "aliases" -> 3)
    var first = true
    val tags = lines.zipWithIndex.filter(_._1.trim.nonEmpty).flatMap { (line, index) =>
      require(index < 300000, "Tag CSV exceeds 300,000 rows")
      val fields = csvRow(line.stripPrefix("\ufeff"))
      require(fields.size >= 2, s"Tag CSV row ${index + 1}: expected name,type[,count,aliases]")
      val header = fields.map(_.trim.toLowerCase(Locale.ROOT))
      if first && header.contains("name") && (header.contains("type") || header.contains("category")) then
        first = false
        columns = header.zipWithIndex.toMap
        None
      else
        first = false
        def field(key: String) = columns.get(key).flatMap(fields.lift)
        val name = field("name").getOrElse("").trim
        require(name.nonEmpty && !name.exists(c => c == '\r' || c == '\n'), s"Invalid tag at row ${index + 1}")
        Some(CompletionTag(name, field("count").flatMap(_.toLongOption).getOrElse(0L), field("aliases").toVector.flatMap(_.split(',')).map(_.trim).filter(_.nonEmpty)))
    }.toVector
    require(tags.nonEmpty, "Tag CSV is empty")
    tags.sortBy(t => (-t.count.max(0L), t.name)).distinctBy(_.name)

  def load(path: Path): Vector[CompletionTag] =
    require(Files.size(path) <= 32L * 1024 * 1024, "Tag CSV exceeds 32 MiB")
    val source = scala.io.Source.fromFile(path.toFile, "UTF-8")
    try parse(source.getLines()) finally source.close()
