package sdfx.core.modelpresets

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, LinkOption, Path, StandardCopyOption, StandardOpenOption}
import java.security.MessageDigest

final case class LoadedModelPresets(document: ModelPresetDocument, revision: Option[String])

/** Per-full-checkpoint identity; exchange JSON remains compatible with upstream short-hash files. */
final class ModelPresetStore(val directory: Path):
  private def key(hash: String): String =
    require(hash.matches("[0-9a-fA-F]{64}"), "A full checkpoint SHA256 is required")
    hash.toLowerCase(java.util.Locale.ROOT)
  def path(hash: String): Path = directory.resolve(key(hash) + ".json")
  private def digest(bytes: Array[Byte]): String = MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
  private def bytes(file: Path): Array[Byte] =
    val in = Files.newInputStream(file, LinkOption.NOFOLLOW_LINKS)
    try
      val result = in.readNBytes(ModelPresetDocument.MaxBytes + 1)
      require(result.length <= ModelPresetDocument.MaxBytes, "Preset file is too large")
      result
    finally in.close()
  def load(hash: String): LoadedModelPresets =
    val file = path(hash)
    if !Files.exists(file, LinkOption.NOFOLLOW_LINKS) then LoadedModelPresets(ModelPresetDocument.empty, None)
    else
      val data = bytes(file)
      LoadedModelPresets(ModelPresetDocument.parseBytes(data), Some(digest(data)))
  def save(hash: String, document: ModelPresetDocument, expected: Option[String]): LoadedModelPresets = synchronized {
    val file = path(hash)
    val data = document.validate().json.getBytes(UTF_8)
    Files.createDirectories(directory)
    val lock = FileChannel.open(directory.resolve(key(hash) + ".lock"), StandardOpenOption.CREATE,
      StandardOpenOption.WRITE, LinkOption.NOFOLLOW_LINKS)
    try
      val held = lock.lock()
      try
        requireRevision(load(hash).revision, expected)
        val temp = Files.createTempFile(directory, ".preset-", ".tmp")
        try
          val out = FileChannel.open(temp, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)
          try
            val buffer = ByteBuffer.wrap(data)
            while buffer.hasRemaining do out.write(buffer)
            out.force(true)
          finally out.close()
          Files.move(temp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
        finally Files.deleteIfExists(temp)
        LoadedModelPresets(document, Some(digest(data)))
      finally held.release()
    finally lock.close()
  }
  private def requireRevision(actual: Option[String], expected: Option[String]): Unit =
    if actual != expected then throw new IllegalStateException("Presets changed on disk. Reload before saving.")
  def importDocument(source: Path): ModelPresetDocument =
    val decoder = UTF_8.newDecoder().onMalformedInput(java.nio.charset.CodingErrorAction.REPORT)
      .onUnmappableCharacter(java.nio.charset.CodingErrorAction.REPORT)
    val text = try decoder.decode(ByteBuffer.wrap(bytes(source))).toString
      catch case e: java.nio.charset.CharacterCodingException => throw new IllegalArgumentException("Preset file is not valid UTF-8", e)
    ModelPresetDocument.parseShared(text)
  def exportDocument(document: ModelPresetDocument, target: Path): Unit =
    val data = document.validate().json.getBytes(UTF_8)
    val parent = target.toAbsolutePath.getParent
    val temp = Files.createTempFile(parent, ".preset-export-", ".tmp")
    try
      Files.write(temp, data)
      // Hard-link publication is atomic and never replaces an existing destination.
      Files.createLink(target, temp)
    finally Files.deleteIfExists(temp)
