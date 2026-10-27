package sdfx.core.modelcards

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.nio.ByteBuffer
import java.nio.channels.SeekableByteChannel
import java.nio.charset.StandardCharsets.{UTF_8, UTF_16LE}
import java.nio.file.{Files, LinkOption, Path, StandardOpenOption}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.{Base64, Locale}
import java.util.concurrent.{Executors, ThreadFactory, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** Metadata validation and bounded reads, always called off the FX application thread.
  * Observable chain changes fail closed. Java does not atomically bind every ancestor to an open
  * handle; hostile concurrent junction mutation still has a residual TOCTOU window.
  */
object LocalPreviewFiles:
  private val MaxSourceBytes = 32 * 1024 * 1024
  private val NoFollow = LinkOption.NOFOLLOW_LINKS

  final case class PathStamp(fileKey: Option[String], size: Long, modifiedMillis: Long,
      directory: Boolean, regularFile: Boolean, reparseTarget: Option[Path])

  trait Platform:
    def windows: Boolean
    def driveType(root: Path): Either[String, String]
    def stamp(path: Path): Either[String, PathStamp]
    def realPath(path: Path): Either[String, Path]
    def openNoFollow(path: Path): Either[String, SeekableByteChannel]

  val systemPlatform: Platform = NativePlatform
  def scoped(platform: Platform = systemPlatform): Scope = new Scope(platform)
  def validate(selected: Path, allowedRealParent: Option[Path]): Either[String, LocalPreviewIdentity] =
    scoped().validate(selected, allowedRealParent)
  def readSnapshot(ref: PreviewReference, buffer: Array[Byte], cancelled: () => Boolean): Either[String, PreviewSourceSnapshot] =
    scoped().readSnapshot(ref, buffer, cancelled)

  /** Validate before creating/probing a Windows filesystem path; relative roots alone may be resolved by AppHome. */
  def validateWindowsPath(raw: String, allowRelative: Boolean): Either[String, Unit] =
    if raw == null || raw.isEmpty || raw.length > 8192 || raw.indexOf('\u0000') >= 0 then Left("Invalid local path")
    else
      val path = raw.replace('/', '\\')
      val driveAbsolute = path.length >= 3 && isAsciiLetter(path.charAt(0)) && path.charAt(1) == ':' && path.charAt(2) == '\\'
      val relative = !path.startsWith("\\") && path.indexOf(':') < 0
      val suffix = if driveAbsolute then path.substring(3) else path
      val invalidPart = suffix.split("\\\\", -1).exists { part =>
        part.exists(c => c < ' ' || "<>:\"|?*".contains(c)) ||
          (part != "." && part != ".." && (part.endsWith(" ") || part.endsWith("."))) ||
          Set("CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9").contains(part.takeWhile(_ != '.').toUpperCase(Locale.ROOT))
      }
      if (!driveAbsolute && !(allowRelative && relative)) || invalidPart then Left("Choose an ordinary local drive path")
      else Right(())

  private def isAsciiLetter(c: Char): Boolean = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
  private def attempt[A](reason: String)(f: => A): Either[String, A] =
    try Right(f) catch case NonFatal(_) => Left(reason)

  /** Lifetime is one scan or validation. Only drive classification is cached; file/link stamps never are. */
  final class Scope private[LocalPreviewFiles] (platform: Platform):
    private val drives = mutable.Map.empty[Path, Either[String, String]]
    private var admission: (Long, Long) => Option[() => Unit] = (_, _) => Some(() => ())
    private var probeClock: () => Long = () => System.nanoTime()
    private var probeObserver: (String, Long) => Unit = (_, _) => ()
    private var beforeProbe: String => Boolean = _ => true
    private final class AdmissionDenied extends RuntimeException
    def withProbeObserver[A](clock: () => Long, observe: (String, Long) => Unit,
        before: String => Boolean = _ => true)(body: => A): A =
      val oldClock = probeClock
      val oldObserver = probeObserver
      val oldBefore = beforeProbe
      probeClock = clock
      probeObserver = observe
      beforeProbe = before
      try body finally
        probeClock = oldClock
        probeObserver = oldObserver
        beforeProbe = oldBefore
    private def observed[A](name: String)(body: => A): A =
      if !beforeProbe(name) then throw new AdmissionDenied
      val start = probeClock()
      try body finally probeObserver(name, (probeClock() - start).max(0L))
    /** Scope is operation/thread confined. Charges precede owned identity collections and probe scratch. */
    def withAdmission[A](admit: (Long, Long) => Option[() => Unit])(body: => A): A =
      val previous = admission
      admission = admit
      try body finally admission = previous
    private def charge(chars: Long, bytes: Long): Unit =
      if admission(chars, bytes).isEmpty then throw new AdmissionDenied
    private def scratch[A](chars: Long, bytes: Long)(body: => A): A =
      val release = admission(chars, bytes).getOrElse(throw new AdmissionDenied)
      try body finally release()
    def windows: Boolean = platform.windows

    /** Classifies a child using the same fail-closed reparse metadata as model validation. */
    def childReparse(path: Path): Either[String, Boolean] =
      try
        local(path).flatMap { normalized =>
          scratch(131072L, 524288L) {
            observed("stamp") { platform.stamp(normalized) }.map(_.reparseTarget.nonEmpty)
          }
        }
      catch
        case _: AdmissionDenied => Left("Metadata admission limit reached")
        case error: AssertionError => throw error
        case NonFatal(_) => Left("Child reparse information is unavailable")

    def resolveRoot(raw: String, home: Path): Either[String, Path] =
      try
        if raw == null || raw.length > 8192 then Left("Invalid local model root")
        else
          charge(raw.length.toLong * 8L, 2048L + raw.length.toLong * 16L)
          val syntax = if windows then validateWindowsPath(raw, allowRelative = true)
            else if raw.isEmpty || raw.indexOf('\u0000') >= 0 || raw.contains("://") then Left("Invalid local model root")
            else Right(())
          syntax.map { _ =>
            val path = Path.of(raw)
            (if path.isAbsolute then path else home.resolve(path)).toAbsolutePath.normalize
          }
      catch
        case _: AdmissionDenied => Left("Metadata admission limit reached")
        case error: AssertionError => throw error
        case NonFatal(_) => Left("Invalid local model root")

    def validate(selected: Path, allowedRealParent: Option[Path]): Either[String, LocalPreviewIdentity] =
      inspect(selected, allowedRealParent, directory = false)

    def validateModel(selected: Path, directoryModel: Boolean): Either[String, LocalPreviewIdentity] =
      inspect(selected, None, directoryModel)

    private def local(path: Path): Either[String, Path] =
      if path == null || !path.isAbsolute then Left("A fully qualified local path is required")
      else
        val pathChars = path.toString.length.toLong
        charge(pathChars * 8L, 2048L + pathChars * 16L)
        val syntax = if windows then validateWindowsPath(path.toString, allowRelative = false) else Right(())
        syntax.flatMap { _ =>
          val normalized = path.normalize
          if !windows then Right(normalized)
          else
            drives.getOrElseUpdate(normalized.getRoot, scratch(131072L, 524288L) {
              observed("drive") { platform.driveType(normalized.getRoot) }
            }).flatMap {
              case "Fixed" | "Removable" | "Ram" => Right(normalized)
              case _ => Left("Preview drive is not an available local drive")
            }
        }

    private def inspect(selected: Path, allowedParent: Option[Path], directory: Boolean): Either[String, LocalPreviewIdentity] =
      try
        local(selected).flatMap { normalized =>
          charge(0L, 512L)
          val identities = mutable.LinkedHashMap.empty[Path, AncestorIdentity]
          val stamps = mutable.Map.empty[Path, PathStamp]

          def capture(path: Path, links: Set[Path] = Set.empty): Either[String, Unit] =
            local(path).flatMap { current =>
              def capturePart(part: Path): Either[String, Unit] =
                // The structured metadata result and reparse target are bounded scratch.
                scratch(131072L, 524288L) {
                  observed("stamp") { platform.stamp(part) }.flatMap { stamp =>
                    stamp.fileKey.filter(_.nonEmpty).toRight("Filesystem identity is unavailable").flatMap { key =>
                      val targetChars = stamp.reparseTarget.fold(0L)(_.toString.length.toLong)
                      val chars = part.toString.length.toLong * 3L + key.length.toLong + targetChars * 2L
                      charge(chars, 1024L + chars * 2L)
                      val target = stamp.reparseTarget.map(p => (if p.isAbsolute then p else part.getParent.resolve(p)).normalize)
                      val identity = AncestorIdentity(part, key, target)
                      if identities.get(part).exists(_ != identity) then Left("Preview ancestor changed during validation")
                      else
                        identities.update(part, identity)
                        stamps.update(part, stamp)
                        target match
                          case Some(destination) if links(part) || links.size >= 32 => Left("Preview link chain is cyclic or too deep")
                          case Some(destination) => capture(destination, links + part)
                          case None => Right(())
                    }
                  }
                }
              var prefix = current.getRoot
              var result = capturePart(prefix)
              val parts = current.iterator()
              while result.isRight && parts.hasNext do
                val part = parts.next()
                val chars = prefix.toString.length.toLong + part.toString.length.toLong + 1L
                charge(chars, 384L + chars * 2L)
                prefix = prefix.resolve(part)
                result = capturePart(prefix)
              result
            }

          for
            _ <- capture(normalized)
            resolved <-
              charge(32768L, 131072L)
              observed("realPath") { platform.realPath(normalized) }
            real <- local(resolved)
            _ <- capture(real)
            _ <- allowedParent match
              case None => Right(())
              case Some(parent) =>
                for
                  approved <- local(parent)
                  _ <- capture(approved)
                  currentParent <-
                    charge(32768L, 131072L)
                    observed("realPath") { platform.realPath(approved) }
                  _ <- if currentParent == approved && real.startsWith(approved) && real != approved then Right(())
                    else Left("Automatic preview escaped its approved parent")
                yield ()
            file <- stamps.get(real).toRight("Resolved filesystem identity is unavailable")
            _ <- if (if directory then file.directory else file.regularFile) && file.reparseTarget.isEmpty then Right(())
              else Left(if directory then "Model folder is unavailable" else "Preview is not a regular local file")
            key <- file.fileKey.filter(_.nonEmpty).toRight("Filesystem identity is unavailable")
            _ <- if file.size >= 0 then Right(()) else Left("Invalid local file size")
          yield
            charge(0L, 256L + identities.size.toLong * 128L)
            LocalPreviewIdentity(real, key, file.size, file.modifiedMillis, identities.values.toVector)
        }
      catch
        case _: AdmissionDenied => Left("Metadata admission limit reached")
        case error: AssertionError => throw error
        case NonFatal(_) => Left("Local file or ancestor is unavailable")

    def readSnapshot(ref: PreviewReference, buffer: Array[Byte], cancelled: () => Boolean): Either[String, PreviewSourceSnapshot] =
      // A resolver's earlier drive admission is not authority for a later source read.
      val fresh = new Scope(platform)
      def unchanged(): Either[String, LocalPreviewIdentity] =
        fresh.validate(ref.localPath, ref.allowedRealParent).flatMap { current =>
          if current == ref.identity then Right(current) else Left("Preview changed; refresh its reference")
        }
      try
        if cancelled() then Left("Preview read cancelled")
        else if buffer == null || buffer.length > MaxSourceBytes then Left("Invalid admitted preview buffer")
        else if ref.identity.size > MaxSourceBytes || ref.identity.size > buffer.length then Left("Preview exceeds the admitted source buffer")
        else unchanged().flatMap { _ =>
          if cancelled() then Left("Preview read cancelled")
          else platform.openNoFollow(ref.identity.realPath).flatMap { channel =>
            try
              unchanged().flatMap { _ =>
                if channel.size() != ref.identity.size then Left("Preview changed while opening")
                else
                  val expected = ref.identity.size.toInt
                  val bytes = ByteBuffer.wrap(buffer, 0, expected)
                  var failed: Option[String] = None
                  var zeroReads = 0
                  while bytes.hasRemaining && failed.isEmpty do
                    if cancelled() then failed = Some("Preview read cancelled")
                    else
                      val count = channel.read(bytes)
                      if count < 0 then failed = Some("Preview was truncated while reading")
                      else if count == 0 then
                        zeroReads += 1
                        if zeroReads >= 8 then failed = Some("Preview source made no read progress")
                      else zeroReads = 0
                  if cancelled() then Left("Preview read cancelled")
                  else if failed.nonEmpty then Left(failed.get)
                  else if channel.size() != expected then Left("Preview changed while reading")
                  else unchanged().map { current =>
                    val digest = MessageDigest.getInstance("SHA-256")
                    digest.update(buffer, 0, expected)
                    val hash = digest.digest().map(b => f"${b & 0xff}%02x").mkString
                    PreviewSourceSnapshot(buffer, expected, hash, current)
                  }
              }
            finally channel.close()
          }
        }
      catch case NonFatal(_) => Left("Preview source could not be read safely")

  private object NativePlatform extends Platform:
    val windows: Boolean = System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win")
    def driveType(root: Path): Either[String, String] =
      if windows then WindowsProbe.query("drive", root).flatMap(value =>
        attempt("Drive classification is unavailable")(value("drive").str))
      else Right("Fixed")

    def stamp(path: Path): Either[String, PathStamp] =
      attempt("Local file attributes are unavailable")(Files.readAttributes(path, classOf[BasicFileAttributes], NoFollow)).flatMap { attributes =>
        val metadata: Either[String, (Option[Path], Option[String])] =
          if windows then WindowsProbe.query("stamp", path).flatMap { value =>
            attempt("Windows file identity or reparse information is unavailable") {
              val key = value("fileKey").str
              if !key.matches("win32-fileid:[0-9A-F]{16}:[0-9A-F]{32}") then
                throw new IllegalArgumentException("Unsupported Windows file identity")
              val target = if value("reparse").bool then
                val raw = value("target").str
                if raw.isEmpty then throw new IllegalArgumentException("No reparse target")
                Some(Path.of(raw))
              else None
              (target, Some(key))
            }
          }
          else if attributes.isSymbolicLink then attempt("Link target is unavailable") {
            (Some(Files.readSymbolicLink(path)), Option(attributes.fileKey()).map(_.toString))
          }
          else Right((None, Option(attributes.fileKey()).map(_.toString)))
        metadata.flatMap { (link, key) =>
          if link.isEmpty && (attributes.isSymbolicLink || attributes.isOther) then Left("Unsupported or changed reparse information")
          else Right(PathStamp(key, attributes.size(),
            attributes.lastModifiedTime().toMillis, attributes.isDirectory, attributes.isRegularFile, link))
        }
      }

    def realPath(path: Path): Either[String, Path] = attempt("Local link target is unavailable")(path.toRealPath())
    def openNoFollow(path: Path): Either[String, SeekableByteChannel] =
      attempt("Preview source cannot be opened")(Files.newByteChannel(path, StandardOpenOption.READ, NoFollow))

  /** One process-lifetime probe worker and one hidden child, shared across operation scopes.
    * No drive/file facts are cached here. Input paths are JSON on stdin, never PowerShell source.
    * A timed-out worker retains ownership until it actually exits; it is never replaced while alive.
    */
  private object WindowsProbe:
    private val script = """
      |$ErrorActionPreference = 'Stop'
      |$ProgressPreference = 'SilentlyContinue'
      |[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
      |[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
      |# Fixed native adapter compiled once in this retained process. No user input is C# source.
      |Add-Type -ErrorAction Stop -TypeDefinition @'
      |using System;
      |using System.ComponentModel;
      |using System.Globalization;
      |using System.IO;
      |using System.Runtime.InteropServices;
      |using Microsoft.Win32.SafeHandles;
      |
      |public static class SdfxPreviewFileIdentity {
      |    [StructLayout(LayoutKind.Sequential)]
      |    private struct FILE_ID_INFO {
      |        public UInt64 VolumeSerialNumber;
      |        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
      |        public byte[] FileId;
      |    }
      |
      |    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
      |    private static extern SafeFileHandle CreateFileW(string name, UInt32 access, UInt32 share,
      |        IntPtr security, UInt32 disposition, UInt32 flags, IntPtr template);
      |
      |    [DllImport("kernel32.dll", ExactSpelling = true, SetLastError = true)]
      |    [return: MarshalAs(UnmanagedType.Bool)]
      |    private static extern bool GetFileInformationByHandleEx(SafeFileHandle file, Int32 infoClass,
      |        out FILE_ID_INFO info, UInt32 size);
      |
      |    public static string Read(string path) {
      |        // Validate again immediately before the metadata handle; never admit UNC/device paths.
      |        if (String.IsNullOrEmpty(path) || path.Length < 3 || path.Length > 8192 ||
      |            !((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) ||
      |            path[1] != ':' || (path[2] != '\\' && path[2] != '/') || path.IndexOf(':', 2) >= 0)
      |            throw new IOException("Not an ordinary absolute drive path");
      |        DriveType drive = new DriveInfo(Path.GetPathRoot(path)).DriveType;
      |        if (drive != DriveType.Fixed && drive != DriveType.Removable && drive != DriveType.Ram)
      |            throw new IOException("Not an available local drive");
      |        // Desired access 0 queries metadata only. Share all; open the final reparse object itself.
      |        // FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, OPEN_EXISTING.
      |        using (SafeFileHandle handle = CreateFileW(path, 0, 7, IntPtr.Zero, 3, 0x02200000, IntPtr.Zero)) {
      |            if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
      |            FILE_ID_INFO info;
      |            if (!GetFileInformationByHandleEx(handle, 18, out info, (UInt32)Marshal.SizeOf(typeof(FILE_ID_INFO))))
      |                throw new Win32Exception(Marshal.GetLastWin32Error());
      |            if (info.FileId == null || info.FileId.Length != 16) throw new IOException("No 128-bit file identity");
      |            return "win32-fileid:" + info.VolumeSerialNumber.ToString("X16", CultureInfo.InvariantCulture) +
      |                ":" + BitConverter.ToString(info.FileId).Replace("-", "");
      |        }
      |    }
      |}
      |'@
      |while ($null -ne ($line = [Console]::ReadLine())) {
      |  try {
      |    $request = ConvertFrom-Json -InputObject $line
      |    if ($request.op -eq 'drive') {
      |      $drive = [System.IO.DriveInfo]::new([string]$request.path)
      |      $answer = @{ ok = $true; drive = $drive.DriveType.ToString() }
      |    } elseif ($request.op -eq 'stamp') {
      |      $item = Get-Item -LiteralPath ([string]$request.path) -Force -ErrorAction Stop
      |      $reparse = (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)
      |      $target = $null
      |      if ($reparse) {
      |        if ($null -eq $item.PSObject.Properties['Target']) { throw 'Unsupported reparse target' }
      |        $targets = @($item.Target)
      |        if ($targets.Count -ne 1 -or [string]::IsNullOrEmpty([string]$targets[0])) { throw 'Ambiguous reparse target' }
      |        $target = [string]$targets[0]
      |      }
      |      $fileKey = [SdfxPreviewFileIdentity]::Read([string]$request.path)
      |      $answer = @{ ok = $true; reparse = $reparse; target = $target; fileKey = $fileKey }
      |    } else { throw 'Unknown probe operation' }
      |    [Console]::WriteLine((ConvertTo-Json -InputObject $answer -Compress))
      |  } catch { [Console]::WriteLine('{"ok":false}') }
      |}
      |""".stripMargin
    private val busy = new AtomicBoolean(false)
    private val lock = new ReentrantLock()
    private val executor = Executors.newSingleThreadExecutor(new ThreadFactory:
      def newThread(runnable: Runnable): Thread =
        val thread = new Thread(runnable, "preview-local-probe")
        thread.setDaemon(true)
        thread)
    @volatile private var process: Process = null
    private var writer: BufferedWriter = null
    private var reader: BufferedReader = null

    Runtime.getRuntime.addShutdownHook(new Thread(() => terminate(), "preview-local-probe-shutdown"))

    private def terminate(): Unit =
      val current = process
      if current != null && current.isAlive then current.destroyForcibly()

    private def ensureProcess(): Unit =
      if process == null || !process.isAlive then
        if writer != null then try writer.close() catch case NonFatal(_) => ()
        if reader != null then try reader.close() catch case NonFatal(_) => ()
        val encoded = Base64.getEncoder.encodeToString(script.getBytes(UTF_16LE))
        val executable = Path.of(System.getenv("SystemRoot"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe").toString
        val command = new ProcessBuilder(executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded)
        // PowerShell can emit CLIXML diagnostics even when stdout contains valid JSON.
        // OS discard has no unread pipe or drainer thread; protocol/process stages below retain
        // bounded failure categories without storing arbitrary diagnostic output.
        command.redirectError(ProcessBuilder.Redirect.DISCARD)
        process = command.start()
        writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream, UTF_8))
        reader = new BufferedReader(new InputStreamReader(process.getInputStream, UTF_8))

    private def boundedLine(): String =
      val line = new StringBuilder
      var c = reader.read()
      while c != -1 && c != '\n' && line.length < 32768 do
        if c != '\r' then line.append(c.toChar)
        c = reader.read()
      if c == -1 || line.length >= 32768 then throw new java.io.IOException("Invalid probe response")
      line.toString

    def query(operation: String, path: Path): Either[String, ujson.Value] =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
      var acquired = false
      try
        acquired = lock.tryLock(2, TimeUnit.SECONDS)
        if !acquired || path.toString.length > 8192 || !busy.compareAndSet(false, true) then
          Left("Local drive probe is unavailable")
        else queryOwned(operation, path, deadline)
      catch
        case _: InterruptedException =>
          Thread.currentThread().interrupt()
          Left("Local drive probe interrupted")
      finally if acquired then lock.unlock()

    private def queryOwned(operation: String, path: Path, deadline: Long): Either[String, ujson.Value] =
        val future = executor.submit(new java.util.concurrent.Callable[Either[String, ujson.Value]]:
          def call(): Either[String, ujson.Value] =
            var stage = "process start"
            try
              ensureProcess()
              stage = "request transport"
              writer.write(ujson.write(ujson.Obj("op" -> operation, "path" -> path.toString)))
              writer.newLine()
              writer.flush()
              stage = "response transport"
              val line = boundedLine()
              stage = "JSON protocol"
              val response = ujson.read(line)
              response.obj.get("ok") match
                case Some(ujson.Bool(true)) => Right(response)
                case Some(ujson.Bool(false)) => Left("Local drive or reparse information is unavailable")
                case _ => throw new java.io.IOException("Missing probe outcome")
            catch
              case NonFatal(_) =>
                val exit = if process != null && !process.isAlive then s" (exit ${process.exitValue()})" else ""
                terminate()
                Left(s"Local drive probe $stage failed$exit")
            finally busy.set(false))
        try future.get(math.max(1L, deadline - System.nanoTime()), TimeUnit.NANOSECONDS)
        catch
          case _: InterruptedException =>
            terminate()
            Thread.currentThread().interrupt()
            Left("Local drive probe interrupted")
          case NonFatal(_) =>
            terminate()
            Left("Local drive probe timed out or failed")
