package sdfx.core.presets

import java.nio.ByteBuffer
import java.nio.charset.{CodingErrorAction, StandardCharsets}
import java.nio.file.{Files, Path}
import java.util.Arrays

final case class SourceDocument(format: String, mode: Option[PresetMode], raw: RawJson, sha256: String)

/** Explicit source interchange files. These never become authoritative native storage. */
object PresetSourceFiles:
  private val textPath = Path.of("tracked-ids")
  private val additionalPath = Path.of("additional-components")

  private def strictText(bytes: Array[Byte], maximum: Int, operation: String, path: Path): String =
    if bytes == null then PresetDisk.fail("input", operation, path, "null input")
    if bytes.length > maximum then PresetDisk.fail("limit", operation, path, s"file exceeds $maximum bytes")
    try
      StandardCharsets.UTF_8.newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes)).toString
    catch case _: java.nio.charset.CharacterCodingException =>
      PresetDisk.fail("invalid-utf8", operation, path, "malformed UTF-8")

  private def utf8Bytes(value: String, maximum: Int, operation: String, path: Path): Array[Byte] =
    if value == null then PresetDisk.fail("input", operation, path, "null text")
    // Each valid UTF-16 code unit costs at least one UTF-8 byte. This rejects
    // obviously oversized caller strings before an encoder can allocate from them.
    if value.length > maximum then
      PresetDisk.fail("limit", operation, path, s"text exceeds $maximum UTF-8 bytes")
    var index = 0
    var byteCount = 0
    while index < value.length do
      val unit = value.charAt(index)
      val width =
        if Character.isHighSurrogate(unit) then
          if index + 1 >= value.length || !Character.isLowSurrogate(value.charAt(index + 1)) then
            PresetDisk.fail("invalid-unicode", operation, path, "unpaired surrogate")
          index += 2
          4
        else if Character.isLowSurrogate(unit) then
          PresetDisk.fail("invalid-unicode", operation, path, "unpaired surrogate")
        else
          index += 1
          if unit < 0x80 then 1 else if unit < 0x800 then 2 else 3
      if byteCount > maximum - width then
        PresetDisk.fail("limit", operation, path, s"text exceeds $maximum UTF-8 bytes")
      byteCount += width
    // One exact-size byte array is allocated only after the bounded preflight.
    val encoded = new Array[Byte](byteCount)
    try
      val encoder = StandardCharsets.UTF_8.newEncoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
      val output = ByteBuffer.wrap(encoded)
      val step = encoder.encode(java.nio.CharBuffer.wrap(value), output, true)
      if step.isError then step.throwException()
      if step.isOverflow then PresetDisk.fail("limit", operation, path, s"text exceeds $maximum UTF-8 bytes")
      val flush = encoder.flush(output)
      if flush.isError then flush.throwException()
      if flush.isOverflow then PresetDisk.fail("limit", operation, path, s"text exceeds $maximum UTF-8 bytes")
      if output.position() != byteCount then
        PresetDisk.fail("invalid-unicode", operation, path, "UTF-8 length changed during encoding")
    catch case _: java.nio.charset.CharacterCodingException =>
      PresetDisk.fail("invalid-unicode", operation, path, "unpaired surrogate")
    encoded

  private def sourceLimits(limits: PresetLimits): PresetLimits =
    limits.copy(maxBytes = math.min(limits.maxBytes, limits.maxCustomFileBytes))

  private def validateId(id: String, limits: PresetLimits, operation: String, path: Path): Unit =
    utf8Bytes(id, limits.maxKeyBytes, operation, path)
    if id.isEmpty || id != id.trim || id.exists(ch => ch == '\r' || ch == '\n' || ch == 0.toChar) then
      PresetDisk.fail("id", operation, path, "invalid ID line or label")

  def read(path: Path, format: String, mode: Option[PresetMode], limits: PresetLimits): Either[PresetError, SourceDocument] =
    PresetDisk.checked("read source", path) {
      validateId(format, limits, "read source", path)
      val bytes = PresetDisk.boundedRead(path, limits.maxBytes, "read source", noFollow = false)
      val raw = PresetDisk.requireRight(PresetJson.readSource(bytes, limits), "read source", path)
      SourceDocument(format, mode, raw, PresetDisk.sha256(bytes))
    }

  def readTrackedIds(bytes: Array[Byte], limits: PresetLimits): Either[PresetError, Vector[String]] =
    PresetDisk.checked("read tracked IDs", textPath) {
      val source = strictText(bytes, limits.maxCustomFileBytes, "read tracked IDs", textPath)
      val ids = source.linesIterator.map(_.trim).filter(s => s.nonEmpty && !s.startsWith("#")).toVector
      if ids.size > limits.maxBindings then
        PresetDisk.fail("limit", "read tracked IDs", textPath, s"more than ${limits.maxBindings} IDs")
      ids.foreach(id => validateId(id, limits, "read tracked IDs", textPath))
      ids
    }

  def writeTrackedIds(ids: Vector[String], limits: PresetLimits): Either[PresetError, Array[Byte]] =
    PresetDisk.checked("write tracked IDs", textPath) {
      if ids == null || ids.size > limits.maxBindings then
        PresetDisk.fail("limit", "write tracked IDs", textPath, s"more than ${limits.maxBindings} IDs")
      val out = new java.io.ByteArrayOutputStream()
      ids.foreach { id =>
        validateId(id, limits, "write tracked IDs", textPath)
        if id.startsWith("#") then PresetDisk.fail("id", "write tracked IDs", textPath, "ID would be read as a comment")
        val line = utf8Bytes(id + "\n", limits.maxCustomFileBytes, "write tracked IDs", textPath)
        if out.size().toLong + line.length > limits.maxCustomFileBytes then
          PresetDisk.fail("limit", "write tracked IDs", textPath, "text file exceeds byte cap")
        out.write(line)
      }
      out.toByteArray
    }

  def readAdditionalComponents(bytes: Array[Byte], limits: PresetLimits): Either[PresetError, Vector[(String, Option[String])]] =
    PresetDisk.checked("read additional components", additionalPath) {
      strictText(bytes, limits.maxCustomFileBytes, "read additional components", additionalPath)
      val raw = PresetDisk.requireRight(PresetJson.readSource(bytes, sourceLimits(limits)),
        "read additional components", additionalPath)
      val root = ujson.read(raw.text) match
        case o: ujson.Obj => o
        case _ => PresetDisk.fail("schema", "read additional components", additionalPath, "object expected")
      if root.value.size != 1 then
        PresetDisk.fail("schema", "read additional components", additionalPath, "additionalComponents is the only supported root key")
      val entries = root.value.get("additionalComponents") match
        case Some(o: ujson.Obj) => o.value.toVector
        case _ => PresetDisk.fail("schema", "read additional components", additionalPath, "additionalComponents object expected")
      if entries.size > limits.maxBindings then
        PresetDisk.fail("limit", "read additional components", additionalPath, "too many mappings")
      entries.map { (label, value) =>
        utf8Bytes(label, limits.maxKeyBytes, "read additional components", additionalPath)
        val id = value match
          case ujson.Null => None
          case s: ujson.Str =>
            utf8Bytes(s.value, limits.maxKeyBytes, "read additional components", additionalPath)
            Some(s.value)
          case _ => PresetDisk.fail("schema", "read additional components", additionalPath, "ID must be string or null")
        label -> id
      }
    }

  def writeAdditionalComponents(entries: Vector[(String, Option[String])], limits: PresetLimits): Either[PresetError, Array[Byte]] =
    PresetDisk.checked("write additional components", additionalPath) {
      if entries == null || entries.size > limits.maxBindings then
        PresetDisk.fail("limit", "write additional components", additionalPath, "too many mappings")
      if entries.map(_._1).distinct.size != entries.size then
        PresetDisk.fail("duplicate-key", "write additional components", additionalPath, "duplicate label")
      val out = new java.io.ByteArrayOutputStream()
      def append(text: String): Unit =
        val bytes = utf8Bytes(text, limits.maxCustomFileBytes, "write additional components", additionalPath)
        if out.size().toLong + bytes.length > limits.maxCustomFileBytes then
          PresetDisk.fail("limit", "write additional components", additionalPath, "JSON file exceeds byte cap")
        out.write(bytes)
      append("{\"additionalComponents\":{")
      entries.zipWithIndex.foreach { case ((label, maybeId), index) =>
        utf8Bytes(label, limits.maxKeyBytes, "write additional components", additionalPath)
        if index > 0 then append(",")
        append(ujson.write(ujson.Str(label)))
        append(":")
        maybeId match
          case None => append("null")
          case Some(id) =>
            utf8Bytes(id, limits.maxKeyBytes, "write additional components", additionalPath)
            append(ujson.write(ujson.Str(id)))
      }
      append("}}")
      val bytes = out.toByteArray
      val parsed = PresetDisk.requireRight(readAdditionalComponents(bytes, limits),
        "write additional components", additionalPath)
      if parsed != entries then
        PresetDisk.fail("schema", "write additional components", additionalPath, "roundtrip changed a mapping")
      bytes
    }

  def exportFile(path: Path, bytes: Array[Byte], expected: PresetRevision): Either[PresetError, PresetRevision] =
    PresetDisk.checked("export source", path) {
      val cap = PresetLimits().maxBytes
      strictText(bytes, cap, "export source", path)
      val target = path.toAbsolutePath.normalize()
      val directory = target.getParent
      if directory == null || !Files.isDirectory(directory) then
        PresetDisk.fail("path", "export source", target, "destination directory is absent")
      def current(): Option[Array[Byte]] =
        if PresetDisk.exists(target) then
          if PresetDisk.isLinkOrReparse(target) then
            PresetDisk.fail("path", "export source", target, "symbolic-link or reparse-backed target refused")
          Some(PresetDisk.boundedRead(target, cap, "export source", noFollow = true))
        else None
      val original = current()
      if PresetDisk.revision(original) != expected then
        PresetDisk.fail("conflict", "export source", target, "destination revision differs from expected revision")
      val temp = PresetDisk.tempPath(directory, "preset-export")
      try
        PresetFileOps.nio.writeForced(temp, bytes)
        val latest = current()
        val changed = (original, latest) match
          case (None, None) => false
          case (Some(a), Some(b)) => !Arrays.equals(a, b)
          case _ => true
        if changed then PresetDisk.fail("conflict", "export source", target, "destination changed before atomic replace")
        PresetFileOps.nio.atomicReplace(temp, target)
      finally PresetDisk.deleteTemp(temp)
      PresetRevision(Some(PresetDisk.sha256(bytes)))
    }
