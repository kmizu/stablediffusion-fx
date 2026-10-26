package sdfx.core.model

import java.nio.ByteBuffer
import java.nio.charset.{CodingErrorAction, StandardCharsets}
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import java.util.UUID
import scala.util.Try

/** One physical CSV record. Runtime identities distinguish rows without rewriting a user's CSV schema. */
case class StyleEntry(id: String, cells: Vector[String], raw: String = "")

/** Lossless CSV editing: untouched records/header keep their original bytes; edits retain foreign columns.
  * `version` is the actual disk-byte version, not a hash of a normalized parse.
  */
case class StyleDocument(path: Path, header: Vector[String], entries: Vector[StyleEntry], version: String,
    bom: Boolean = false, eol: String = "\r\n", rawHeader: String = "",
    masterVersion: Option[String] = None, mirrorVersions: Map[Path, String] = Map.empty):
  private def key(s: String) = s.trim.toLowerCase(java.util.Locale.ROOT)
  private def column(field: String): Int =
    val direct = header.indexWhere(h => key(h) == field)
    if direct < 0 && field == "prompt" then header.indexWhere(h => key(h) == "text") else direct
  def value(row: StyleEntry, field: String): String =
    if field == "sort" && (column(field) < 0 || row.cells.lift(column(field)).isEmpty) then (entries.indexWhere(_.id == row.id) + 1).toString
    else row.cells.lift(column(field)).getOrElse("")
  def promptStyles: Vector[PromptStyle] = entries.filter(e => value(e, "name").nonEmpty)
    .map(e => PromptStyle(value(e, "name"), value(e, "prompt"), value(e, "negative_prompt")))
  def fields(row: StyleEntry): Map[String, String] = header.zipWithIndex.map((h, i) => key(h) -> row.cells.lift(i).getOrElse("")).toMap
  private def unique(name: String, except: String): String =
    val used = entries.filterNot(_.id == except).map(value(_, "name")).toSet
    var n = name
    while used(n) do n += "x"
    n
  def edit(id: String, changes: Map[String, String], uniqueNames: Boolean = true): StyleDocument =
    require(entries.exists(_.id == id), "The selected style no longer exists; reload the list")
    val missing = changes.keys.filter(column(_) < 0).toVector
    val doc = if missing.isEmpty then this else copy(header = header ++ missing, rawHeader = "")
    val rows = doc.entries.map { row =>
      if row.id != id then row
      else
        var cells = row.cells.padTo(doc.header.size, "")
        changes.foreach { (field, raw) =>
          val v = if field == "name" && uniqueNames then doc.unique(raw, id) else raw
          cells = cells.updated(doc.column(field), v)
        }
        row.copy(cells = cells, raw = "")
    }
    doc.copy(entries = rows)
  def append(changes: Map[String, String], uniqueNames: Boolean = true): StyleDocument =
    val row = StyleEntry(UUID.randomUUID().toString, Vector.fill(header.size)(""))
    val fields = if column("sort") >= 0 && !changes.contains("sort") then changes.updated("sort", (entries.size + 1).toString) else changes
    copy(entries = entries :+ row).edit(row.id, fields, uniqueNames)
  def duplicate(ids: Seq[String]): StyleDocument = ids.foldLeft(this) { (doc, id) =>
    entries.find(_.id == id).map(e => doc.append(fields(e))).getOrElse(doc)
  }
  def delete(ids: Seq[String]): StyleDocument = copy(entries = entries.filterNot(e => ids.contains(e.id)))
  def reorder(ids: Seq[String]): StyleDocument =
    require(ids.distinct.size == entries.size && ids.toSet == entries.map(_.id).toSet, "Row order must contain each style exactly once")
    copy(entries = ids.map(id => entries.find(_.id == id).get).toVector)
  def numericSort: StyleDocument =
    val values = entries.map(e => Try(value(e, "sort").toDouble).toOption.filter(d => !d.isNaN && !d.isInfinite))
    if values.exists(_.isEmpty) then this else copy(entries = entries.zip(values.flatten).sortBy(_._2).map(_._1))
  def replace(search: String, replacement: String, uniqueNames: Boolean = true, protectedFields: Set[String] = Set.empty): StyleDocument =
    if search.isEmpty then this
    else entries.foldLeft(this) { (doc, entry) =>
      val changes = fields(entry).collect { case (k, v) if !protectedFields(k) && v.contains(search) => k -> v.replace(search, replacement) }
      if changes.isEmpty then doc else doc.edit(entry.id, changes, uniqueNames)
    }
  def filtered(query: String, mode: String): Either[String, Vector[StyleEntry]] =
    try
      val regex = if mode == "regex" then Some(java.util.regex.Pattern.compile(query)) else None
      Right(entries.filter { e =>
        (e.cells :+ value(e, "sort")).exists { s =>
          regex.fold(if mode == "Case insensitive" then s.toLowerCase(java.util.Locale.ROOT).contains(query.toLowerCase(java.util.Locale.ROOT)) else s.contains(query))
            (_.matcher(s).find())
        }
      })
    catch case e: java.util.regex.PatternSyntaxException => Left(e.getDescription)
  def bytes: Array[Byte] =
    val out = new StringBuilder(if bom then "\uFEFF" else "")
    val records = (if rawHeader.nonEmpty then rawHeader else header.map(Csv.escape).mkString(",") + eol) +:
      entries.map(e => if e.raw.nonEmpty then e.raw else e.cells.map(Csv.escape).mkString(",") + eol)
    records.foreach { r =>
      if out.nonEmpty && out.last != '\n' && out.last != '\uFEFF' then out.append(eol)
      out.append(r)
    }
    out.toString.getBytes(StandardCharsets.UTF_8)

object StyleDocument:
  val standardHeader = Vector("name", "prompt", "negative_prompt")
  def fingerprint(bytes: Array[Byte], exists: Boolean = true): String =
    (if exists then "file:" else "missing:") + MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
  def read(path: Path): StyleDocument =
    if !Files.exists(path) then StyleDocument(path, standardHeader, Vector.empty, fingerprint(Array.emptyByteArray, false))
    else parse(path, Files.readAllBytes(path))
  def parse(path: Path, bytes: Array[Byte]): StyleDocument =
    val decoder = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
    val text = decoder.decode(ByteBuffer.wrap(bytes)).toString
    val bom = text.startsWith("\uFEFF")
    val body = text.stripPrefix("\uFEFF")
    val records = scala.collection.mutable.ArrayBuffer[String]()
    var quoted = false
    var fieldStart = true
    var start = 0
    var i = 0
    while i < body.length do
      val c = body(i)
      if quoted then
        if c == '"' then
          if i + 1 < body.length && body(i + 1) == '"' then i += 1 else quoted = false
      else if c == '"' && fieldStart then { quoted = true; fieldStart = false }
      else if c == ',' then fieldStart = true
      else if c == '\n' then
        records += body.substring(start, i + 1)
        start = i + 1
        fieldStart = true
      else if c != '\r' then fieldStart = false
      i += 1
    require(!quoted, "Unterminated quoted field in styles CSV; fix the source or reload a valid file")
    if start < body.length then records += body.substring(start)
    val eol = records.find(_.endsWith("\n")).map(r => if r.endsWith("\r\n") then "\r\n" else "\n").getOrElse("\r\n")
    if records.isEmpty then return StyleDocument(path, standardHeader, Vector.empty, fingerprint(bytes), bom, eol)
    val header = Csv.parse(records.head).headOption.getOrElse(Vector.empty)
    val normalized = header.map(_.trim.toLowerCase(java.util.Locale.ROOT))
    require(normalized.contains("name") && (normalized.contains("prompt") || normalized.contains("text")), "Styles CSV needs name and prompt/text columns")
    require(normalized.distinct.size == normalized.size, "Duplicate CSV column names are ambiguous; fix the source header before editing (no source bytes were changed)")
    val entries = records.tail.map { raw =>
      val cells = Csv.parse(raw).headOption.getOrElse(Vector.empty)
      StyleEntry(UUID.randomUUID().toString, cells, raw)
    }.toVector
    StyleDocument(path, header, entries, fingerprint(bytes), bom, eol, records.head)
