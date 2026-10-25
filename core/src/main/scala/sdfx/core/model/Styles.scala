package sdfx.core.model

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** A1111 styles.csv compatible prompt styles. `{prompt}` in a style is replaced by the user prompt. */
case class PromptStyle(name: String, prompt: String, negativePrompt: String):
  def applyTo(p: String, n: String): (String, String) =
    (merge(prompt, p), merge(negativePrompt, n))

  // modules/styles.py merge_prompts
  private def merge(style: String, text: String): String =
    if style.contains("{prompt}") then style.replace("{prompt}", text)
    else Seq(text.trim, style.trim).filter(_.nonEmpty).mkString(", ")

object Styles:
  def parseCsv(content: String): Vector[PromptStyle] =
    // A1111 reads styles.csv as utf-8-sig: drop a BOM so the header still matches.
    val rows = Csv.parse(content.stripPrefix("\uFEFF"))
    if rows.isEmpty then Vector.empty
    else
      val header = rows.head.map(_.trim.toLowerCase)
      val ni = header.indexOf("name")
      val pi = if header.contains("prompt") then header.indexOf("prompt") else header.indexOf("text") // legacy column
      val qi = header.indexOf("negative_prompt")
      rows.tail.filter(r => ni >= 0 && r.length > ni && r(ni).nonEmpty).map { r =>
        def at(i: Int) = if i >= 0 && i < r.length then r(i) else ""
        PromptStyle(at(ni), at(pi), at(qi))
      }.toVector

  def load(file: Path): Vector[PromptStyle] =
    if Files.exists(file) then parseCsv(Files.readString(file)) else Vector.empty

  def toCsv(styles: Seq[PromptStyle]): String =
    (Seq(Seq("name", "prompt", "negative_prompt")) ++ styles.map(s => Seq(s.name, s.prompt, s.negativePrompt)))
      .map(_.map(Csv.escape).mkString(","))
      .mkString("", "\r\n", "\r\n")

  def apply(styles: Seq[PromptStyle], names: Seq[String], prompt: String, negative: String): (String, String) =
    names.flatMap(n => styles.find(_.name == n)).foldLeft((prompt, negative)) { case ((p, n), s) => s.applyTo(p, n) }

/** Minimal RFC 4180 CSV reader/writer (quoted fields, embedded newlines and quotes). */
object Csv:
  def escape(s: String): String =
    if s.exists(c => c == ',' || c == '"' || c == '\n' || c == '\r') then "\"" + s.replace("\"", "\"\"") + "\"" else s

  def parse(text: String): Vector[Vector[String]] =
    val rows = Vector.newBuilder[Vector[String]]
    var row = Vector.newBuilder[String]
    val field = new StringBuilder
    var inQuotes = false
    var i = 0
    var rowHasData = false
    def endField(): Unit =
      row += field.toString
      field.clear()
      rowHasData = true
    def endRow(): Unit =
      if rowHasData || field.nonEmpty then
        endField()
        rows += row.result()
      row = Vector.newBuilder[String]
      rowHasData = false
    while i < text.length do
      val c = text.charAt(i)
      if inQuotes then
        if c == '"' then
          if i + 1 < text.length && text.charAt(i + 1) == '"' then
            field += '"'
            i += 1
          else inQuotes = false
        else field += c
      else
        c match
          case '"' if field.isEmpty => inQuotes = true // a quote inside an unquoted field is literal (Python csv)
          case ','  => endField()
          case '\r' => ()
          case '\n' => endRow()
          case _    => field += c
      i += 1
    endRow()
    rows.result()
