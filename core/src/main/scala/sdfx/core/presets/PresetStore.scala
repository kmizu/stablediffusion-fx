package sdfx.core.presets

import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.{Files, LinkOption, Path, StandardCopyOption, StandardOpenOption}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.time.Instant
import java.util.{Arrays, UUID}
import scala.util.control.NonFatal

final case class PresetRevision(sha256: Option[String])
final case class LoadedPresets(library: PresetLibrary, revision: PresetRevision)

/** Narrow seam for deterministic failures at the durability boundary. */
private[presets] trait PresetFileOps:
  def writeForced(path: Path, bytes: Array[Byte]): Unit
  def atomicReplace(from: Path, to: Path): Unit

private[presets] object PresetFileOps:
  val nio: PresetFileOps = new PresetFileOps:
    def writeForced(path: Path, bytes: Array[Byte]): Unit =
      val channel = FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE,
        LinkOption.NOFOLLOW_LINKS)
      try
        val buffer = ByteBuffer.wrap(bytes)
        while buffer.hasRemaining do channel.write(buffer)
        channel.force(true)
      finally channel.close()

    def atomicReplace(from: Path, to: Path): Unit =
      Files.move(from, to, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)

/** Shared bounded disk primitives; all source JSON validation remains in PresetJson. */
private[presets] object PresetDisk:
  private final case class Abort(error: PresetError) extends RuntimeException
  private def pathText(path: Path): String = Option(path).fold("<null path>")(_.toString)

  def fail(code: String, operation: String, path: Path, detail: String): Nothing =
    val safe = Option(detail).getOrElse("unknown error").replace('\r', ' ').replace('\n', ' ').take(240)
    throw Abort(PresetError(Vector(PresetIssue(code, pathText(path), s"$operation: $safe"))))

  def checked[A](operation: String, path: Path)(body: => A): Either[PresetError, A] =
    try Right(body)
    catch
      case Abort(error) => Left(error)
      case NonFatal(e) =>
        val detail = Option(e.getMessage).getOrElse(e.getClass.getSimpleName)
        Left(PresetError(Vector(PresetIssue("io", pathText(path),
          s"$operation: ${detail.replace('\r', ' ').replace('\n', ' ').take(240)}"))))

  def requireRight[A](value: Either[PresetError, A], operation: String, path: Path): A =
    value match
      case Right(good) => good
      case Left(error) =>
        val first = error.issues.headOption.getOrElse(PresetIssue("invalid", "$", "invalid document"))
        fail(first.code, operation, path, s"${first.path}: ${first.message}")

  def exists(path: Path): Boolean = Files.exists(path, LinkOption.NOFOLLOW_LINKS)

  def isLinkOrReparse(path: Path): Boolean =
    val attr = Files.readAttributes(path, classOf[BasicFileAttributes], LinkOption.NOFOLLOW_LINKS)
    val dosReparse = try
      Files.readAttributes(path, "dos:reparsePoint", LinkOption.NOFOLLOW_LINKS)
        .get("reparsePoint") == java.lang.Boolean.TRUE
    catch case NonFatal(_) => false
    attr.isSymbolicLink || attr.isOther || dosReparse

  def boundedRead(path: Path, maximum: Int, operation: String, noFollow: Boolean): Array[Byte] =
    if maximum < 0 || maximum == Int.MaxValue then fail("limit", operation, path, "invalid byte cap")
    val in =
      if noFollow then Files.newInputStream(path, StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)
      else Files.newInputStream(path, StandardOpenOption.READ)
    try
      val bytes = in.readNBytes(maximum + 1)
      if bytes.length > maximum then fail("limit", operation, path, s"file exceeds $maximum bytes")
      bytes
    finally in.close()

  def sha256(bytes: Array[Byte]): String =
    val digits = "0123456789abcdef"
    val out = new StringBuilder(64)
    MessageDigest.getInstance("SHA-256").digest(bytes).foreach { b =>
      val n = b & 0xff
      out.append(digits.charAt(n >>> 4))
      out.append(digits.charAt(n & 15))
    }
    out.toString

  def revision(bytes: Option[Array[Byte]]): PresetRevision = PresetRevision(bytes.map(sha256))

  def tempPath(directory: Path, stem: String): Path =
    directory.resolve(s".$stem-${UUID.randomUUID()}.tmp")

  def deleteTemp(path: Path): Unit = Files.deleteIfExists(path)

/** One authoritative native library with expected-revision writes. */
final class PresetStore private[presets] (
  directory: Path,
  limits: PresetLimits,
  fileOps: PresetFileOps
):
  def this(directory: Path, limits: PresetLimits) = this(directory, limits, PresetFileOps.nio)

  private val root = directory.toAbsolutePath.normalize()
  val libraryPath: Path = root.resolve("library-v1.json")
  private val backupPath = root.resolve("library-v1.json.bak")
  private val lockPath = root.resolve(".library-v1.lock")
  private val empty = PresetLibrary(1, Vector.empty, Vector.empty)
  private var directoryIdentity: Option[(Path, Object)] = None
  private var seenExistingLibrary = false

  private def rejectLink(path: Path, operation: String): Unit =
    if PresetDisk.exists(path) && PresetDisk.isLinkOrReparse(path) then
      PresetDisk.fail("path", operation, path, "symbolic-link or reparse-backed path refused")

  private def checkRoot(operation: String, create: Boolean): Boolean =
    if !PresetDisk.exists(root) then
      if directoryIdentity.nonEmpty then PresetDisk.fail("conflict", operation, root, "configured directory disappeared")
      if !create then return false
      Files.createDirectories(root)
    rejectLink(root, operation)
    if !Files.isDirectory(root, LinkOption.NOFOLLOW_LINKS) then
      PresetDisk.fail("path", operation, root, "configured path is not a directory")
    val attrs = Files.readAttributes(root, classOf[BasicFileAttributes], LinkOption.NOFOLLOW_LINKS)
    val now = (root.toRealPath(LinkOption.NOFOLLOW_LINKS), attrs.fileKey().asInstanceOf[Object])
    directoryIdentity match
      case Some(previous) =>
        val changed = previous._1 != now._1 ||
          (previous._2 != null && now._2 != null && previous._2 != now._2)
        if changed then PresetDisk.fail("conflict", operation, root, "configured directory was replaced")
      case None => directoryIdentity = Some(now)
    true

  private def fixed(path: Path, operation: String): Unit =
    if path.normalize().getParent != root then PresetDisk.fail("path", operation, path, "path escapes configured directory")
    checkRoot(operation, create = false)
    rejectLink(path, operation)

  private def readCurrent(operation: String): Option[Array[Byte]] =
    fixed(libraryPath, operation)
    if PresetDisk.exists(libraryPath) then
      Some(PresetDisk.boundedRead(libraryPath, limits.maxBytes, operation, noFollow = true))
    else None

  private def readBackup(operation: String): Array[Byte] =
    fixed(backupPath, operation)
    if !PresetDisk.exists(backupPath) then PresetDisk.fail("missing-backup", operation, backupPath, "backup is absent")
    PresetDisk.boundedRead(backupPath, limits.maxBytes, operation, noFollow = true)

  private def lock[A](operation: String)(body: => A): A =
    fixed(lockPath, operation)
    val channel = FileChannel.open(lockPath, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
      LinkOption.NOFOLLOW_LINKS)
    try
      val held = channel.lock()
      try
        fixed(lockPath, operation)
        body
      finally held.release()
    finally channel.close()

  private def writeTemp(stem: String, bytes: Array[Byte], operation: String): Path =
    checkRoot(operation, create = false)
    val path = PresetDisk.tempPath(root, stem)
    fixed(path, operation)
    try
      fileOps.writeForced(path, bytes)
      path
    catch
      case NonFatal(e) =>
        PresetDisk.deleteTemp(path)
        throw e

  private def replace(from: Path, to: Path, operation: String): Unit =
    fixed(from, operation)
    fixed(to, operation)
    fileOps.atomicReplace(from, to)

  private def expectCurrent(expected: PresetRevision, operation: String): Option[Array[Byte]] =
    val bytes = readCurrent(operation)
    if PresetDisk.revision(bytes) != expected then
      PresetDisk.fail("conflict", operation, libraryPath, "current file revision differs from expected revision")
    bytes

  private def stillCurrent(previous: Option[Array[Byte]], operation: String): Unit =
    val now = readCurrent(operation)
    val changed = (previous, now) match
      case (None, None) => false
      case (Some(a), Some(b)) => !Arrays.equals(a, b)
      case _ => true
    if changed then PresetDisk.fail("conflict", operation, libraryPath, "current bytes changed before atomic replace")

  def load(): Either[PresetError, LoadedPresets] = synchronized {
    PresetDisk.checked("load", libraryPath) {
      if !checkRoot("load", create = false) then LoadedPresets(empty, PresetRevision(None))
      else
        val current = readCurrent("load")
        current match
          case Some(bytes) =>
            val doc = PresetDisk.requireRight(PresetJson.readNative(bytes, limits), "load", libraryPath)
            seenExistingLibrary = true
            LoadedPresets(doc, PresetDisk.revision(current))
          case None =>
            fixed(backupPath, "load")
            if seenExistingLibrary || PresetDisk.exists(backupPath) then
              PresetDisk.fail("conflict", "load", libraryPath, "previous library is missing; explicit restore required")
            LoadedPresets(empty, PresetRevision(None))
    }
  }

  def save(expected: PresetRevision, next: PresetLibrary): Either[PresetError, LoadedPresets] = synchronized {
    PresetDisk.checked("save", libraryPath) {
      val candidate = PresetDisk.requireRight(PresetJson.writeNative(next, limits), "save", libraryPath)
      checkRoot("save", create = expected.sha256.isEmpty)
      lock("save") {
        val old = expectCurrent(expected, "save")
        fixed(backupPath, "save")
        if old.isEmpty && PresetDisk.exists(backupPath) then
          PresetDisk.fail("conflict", "save", libraryPath, "library absent but backup exists; explicit restore required")
        old.foreach(bytes => PresetDisk.requireRight(PresetJson.readNative(bytes, limits), "save", libraryPath))
        val candidateTemp = writeTemp("library-v1", candidate, "save")
        try
          old.foreach { bytes =>
            val backupTemp = writeTemp("library-v1-backup", bytes, "save")
            try replace(backupTemp, backupPath, "save backup")
            finally PresetDisk.deleteTemp(backupTemp)
          }
          stillCurrent(old, "save")
          replace(candidateTemp, libraryPath, "save")
        finally PresetDisk.deleteTemp(candidateTemp)
        seenExistingLibrary = true
        LoadedPresets(next, PresetRevision(Some(PresetDisk.sha256(candidate))))
      }
    }
  }

  def restoreBackup(expected: PresetRevision): Either[PresetError, LoadedPresets] = synchronized {
    PresetDisk.checked("restore backup", libraryPath) {
      checkRoot("restore backup", create = false)
      lock("restore backup") {
        val current = expectCurrent(expected, "restore backup")
        val backup = readBackup("restore backup")
        val recovered = PresetDisk.requireRight(PresetJson.readNative(backup, limits), "restore backup", backupPath)
        val targetTemp = writeTemp("library-v1-restore", backup, "restore backup")
        try
          current.foreach { original =>
            val timestamp = Instant.now().toString.replace(':', '-')
            val rejectedPath = root.resolve(s"library-v1.rejected-$timestamp-${UUID.randomUUID()}.json")
            fixed(rejectedPath, "preserve rejected")
            val rejectedTemp = writeTemp("library-v1-rejected", original, "preserve rejected")
            try replace(rejectedTemp, rejectedPath, "preserve rejected")
            finally PresetDisk.deleteTemp(rejectedTemp)
          }
          stillCurrent(current, "restore backup")
          replace(targetTemp, libraryPath, "restore backup")
        finally PresetDisk.deleteTemp(targetTemp)
        seenExistingLibrary = true
        LoadedPresets(recovered, PresetRevision(Some(PresetDisk.sha256(backup))))
      }
    }
  }
