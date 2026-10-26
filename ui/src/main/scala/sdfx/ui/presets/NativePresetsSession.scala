package sdfx.ui.presets

import java.nio.file.{Files, LinkOption, NoSuchFileException, Path, StandardOpenOption}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.{Executors, ThreadFactory}
import scala.util.control.NonFatal

import sdfx.core.presets.*
import sdfx.ui.Fx
import sdfx.ui.vm.Mode

/** Serializes preset file operations and delivers their results to the UI owner. */
final class NativePresetsSession private[ui] (
  directory: Path,
  controllers: Map[Mode, NativePresetsController],
  onResult: Either[PresetError, LoadedPresets] => Unit,
  dispatch: (() => Unit) => Unit,
  createDefaultDirectory: Boolean
):
  def this(directory: Path, controllers: Map[Mode, NativePresetsController],
           onResult: Either[PresetError, LoadedPresets] => Unit) =
    this(directory, controllers, onResult, thunk => Fx.run(thunk()), false)

  private[ui] def this(directory: Path, controllers: Map[Mode, NativePresetsController],
                       onResult: Either[PresetError, LoadedPresets] => Unit,
                       dispatch: (() => Unit) => Unit) =
    this(directory, controllers, onResult, dispatch, false)

  private val root = directory.toAbsolutePath.normalize()
  private val limits = PresetLimits()
  private lazy val store = new PresetStore(root, limits)
  private val gate = new Object
  private val io = Executors.newSingleThreadExecutor((task => {
    val thread = new Thread(task, "sdfx-native-presets-io")
    thread.setDaemon(true)
    thread
  }): ThreadFactory)

  private var closed = false
  private var pending = false
  private var generation = 0L
  private var hasLoadedRevision = false
  private var loadedRevision = PresetRevision(None)
  private var readOnlyAfterReloadError = false
  // Accessed only by the serialized I/O worker. A later disappearance is an error.
  private var defaultDirectorySeen = false

  private def error(code: String, path: Path, operation: String, detail: String): PresetError =
    val safe = Option(detail).getOrElse("unknown error").replace('\r', ' ').replace('\n', ' ').take(240)
    PresetError(Vector(PresetIssue(code, Option(path).fold("<null path>")(_.toString), s"$operation: $safe")))

  private def ensureDirectory(operation: String): Either[PresetError, Unit] =
    try
      if createDefaultDirectory && !defaultDirectorySeen then Files.createDirectories(root)
      if !Files.exists(root, LinkOption.NOFOLLOW_LINKS) then
        Left(error("path", root, operation, "configured presets directory does not exist"))
      else
        val attrs = Files.readAttributes(root, classOf[BasicFileAttributes], LinkOption.NOFOLLOW_LINKS)
        val reparse =
          try Files.getAttribute(root, "dos:reparsePoint", LinkOption.NOFOLLOW_LINKS) == java.lang.Boolean.TRUE
          catch case NonFatal(_) => false
        if !attrs.isDirectory || attrs.isSymbolicLink || attrs.isOther || reparse then
          Left(error("path", root, operation, "configured presets path is not an ordinary directory"))
        else
          defaultDirectorySeen = true
          Right(())
    catch case NonFatal(e) =>
      Left(error("io", root, operation, Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))

  private def busy[A](operation: String, token: Long,
                      callback: Either[PresetError, A] => Unit): Unit =
    val result = Left(error("busy", root, operation, "another preset operation is pending"))
    dispatch(() => {
      val current = gate.synchronized { !closed && pending && generation == token }
      if current then callback(result)
    })

  private def deliver[A](token: Long, result: Either[PresetError, A],
                         callback: Either[PresetError, A] => Unit,
                         update: Either[PresetError, A] => Unit): Unit =
    dispatch(() => {
      val current = gate.synchronized {
        if closed || !pending || generation != token then false
        else
          try update(result)
          finally pending = false
          true
      }
      if current then callback(result)
    })

  private def submit[A](operation: String, path: Path,
                        callback: Either[PresetError, A] => Unit)(
                        work: => Either[PresetError, A])(
                        update: Either[PresetError, A] => Unit): Unit =
    val admission = gate.synchronized {
      if closed then None
      else if pending then Some((false, generation))
      else
        pending = true
        generation += 1
        Some((true, generation))
    }
    admission.foreach { case (accepted, token) =>
      if !accepted then busy(operation, token, callback)
      else
        try
          io.execute(() => {
            val result =
              try work
              catch case NonFatal(e) =>
                Left(error("io", path, operation, Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))
            deliver(token, result, callback, update)
          })
        catch case NonFatal(e) =>
          deliver(token, Left(error("io", path, operation,
            Option(e.getMessage).getOrElse(e.getClass.getSimpleName))), callback, update)
    }

  private def installLibrary(result: Either[PresetError, LoadedPresets]): Unit =
    result.foreach { loaded =>
      loadedRevision = loaded.revision
      hasLoadedRevision = true
    }

  def reload(): Unit =
    submit("reload", root, onResult)(
      ensureDirectory("reload").flatMap(_ => store.load())
    ) { result =>
      installLibrary(result)
      readOnlyAfterReloadError = result.isLeft
    }

  def save(expected: PresetRevision, next: PresetLibrary): Unit =
    submit("save", root, onResult)(
      ensureDirectory("save").flatMap { _ =>
        val state = gate.synchronized {
          if readOnlyAfterReloadError then
            Some(error("read-only", root, "save", "reload failed; reload or restore backup first"))
          else if hasLoadedRevision && expected != loadedRevision then
            Some(error("conflict", root, "save", "caller revision differs from the loaded library"))
          else None
        }
        state match
          case Some(problem) => Left(problem)
          case None => store.load().flatMap(_ => store.save(expected, next))
      }
    )(installLibrary)

  def restoreBackup(expected: PresetRevision): Unit =
    submit("restore backup", root, onResult)(
      ensureDirectory("restore backup").flatMap(_ => store.restoreBackup(expected))
    ) { result =>
      installLibrary(result)
      if result.isRight then readOnlyAfterReloadError = false
    }

  def readSource(path: Path, format: String, mode: Option[PresetMode],
                 onRead: Either[PresetError, SourceDocument] => Unit): Unit =
    submit("read source", path, onRead)(
      ensureDirectory("read source").flatMap(_ => PresetSourceFiles.read(path, format, mode, limits))
    )(_ => ())

  private def readCustom(path: Path, operation: String): Either[PresetError, Array[Byte]] =
    try
      val in = Files.newInputStream(path)
      try
        val bytes = in.readNBytes(limits.maxCustomFileBytes + 1)
        if bytes.length > limits.maxCustomFileBytes then
          Left(error("limit", path, operation, s"file exceeds ${limits.maxCustomFileBytes} bytes"))
        else Right(bytes)
      finally in.close()
    catch case NonFatal(e) =>
      Left(error("io", path, operation, Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))

  def readTrackedIds(path: Path, onRead: Either[PresetError, Vector[String]] => Unit): Unit =
    submit("read tracked IDs", path, onRead)(
      ensureDirectory("read tracked IDs").flatMap(_ =>
        readCustom(path, "read tracked IDs").flatMap(PresetSourceFiles.readTrackedIds(_, limits)))
    )(_ => ())

  def readAdditionalComponents(path: Path,
      onRead: Either[PresetError, Vector[(String, Option[String])]] => Unit): Unit =
    submit("read additional components", path, onRead)(
      ensureDirectory("read additional components").flatMap(_ =>
        readCustom(path, "read additional components").flatMap(PresetSourceFiles.readAdditionalComponents(_, limits)))
    )(_ => ())

  private def rejectedName(name: String): Boolean =
    name.toLowerCase(Locale.ROOT).startsWith("library-v1.rejected-")

  /** Source interchange must never replace Store-owned state or recovery evidence. */
  private def exportDestinationAllowed(path: Path): Either[PresetError, Unit] =
    try
      // Match the writer's normalized target, then compare actual filesystem identity.
      val target = path.toAbsolutePath.normalize()
      val parent = target.getParent
      val inStoreDirectory = parent != null && Files.isSameFile(parent, root)
      val name = target.getFileName.toString.toLowerCase(Locale.ROOT)
      val reservedName = name == "library-v1.json" || name == "library-v1.json.bak" ||
        name == ".library-v1.lock" || rejectedName(name)
      val fixed = Vector(root.resolve("library-v1.json"), root.resolve("library-v1.json.bak"),
        root.resolve(".library-v1.lock"))
      val existing = Files.exists(target)
      val sameFixed = existing && fixed.exists { candidate =>
        Files.exists(candidate) && Files.isSameFile(target, candidate)
      }
      val sameRejected =
        if !existing || (inStoreDirectory && reservedName) then false
        else
          val entries = Files.newDirectoryStream(root)
          try
            val iterator = entries.iterator()
            var same = false
            while iterator.hasNext && !same do
              val candidate = iterator.next()
              if rejectedName(candidate.getFileName.toString) && Files.isSameFile(target, candidate) then
                same = true
            same
          finally entries.close()
      if (inStoreDirectory && reservedName) || sameFixed || sameRejected then
        Left(error("reserved-path", target, "export source", "destination belongs to the native preset library"))
      else Right(())
    catch case NonFatal(e) =>
      Left(error("io", path, "export source", Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))

  private def readRawRevision(path: Path, operation: String): Either[PresetError, PresetRevision] =
    try
      val attributes =
        try Some(Files.readAttributes(path, classOf[BasicFileAttributes], LinkOption.NOFOLLOW_LINKS))
        catch case _: NoSuchFileException => None
      attributes match
        case None => Right(PresetRevision(None))
        case Some(attrs) =>
          val reparse =
            try Files.getAttribute(path, "dos:reparsePoint", LinkOption.NOFOLLOW_LINKS) == java.lang.Boolean.TRUE
            catch case NonFatal(_) => false
          if !attrs.isRegularFile || attrs.isSymbolicLink || attrs.isOther || reparse then
            Left(error("path", path, operation, "revision target is not an ordinary file"))
          else
            val in = Files.newInputStream(path, StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS)
            try
              val bytes = in.readNBytes(limits.maxBytes + 1)
              if bytes.length > limits.maxBytes then
                Left(error("limit", path, operation, s"file exceeds ${limits.maxBytes} bytes"))
              else
                val digits = "0123456789abcdef"
                val sha = MessageDigest.getInstance("SHA-256").digest(bytes)
                val hex = new StringBuilder(64)
                sha.foreach { byte =>
                  val n = byte & 0xff
                  hex.append(digits.charAt(n >>> 4))
                  hex.append(digits.charAt(n & 15))
                }
                Right(PresetRevision(Some(hex.toString)))
            finally in.close()
    catch case NonFatal(e) =>
      Left(error("io", path, operation, Option(e.getMessage).getOrElse(e.getClass.getSimpleName)))

  def readLibraryRevision(onRead: Either[PresetError, PresetRevision] => Unit): Unit =
    val path = root.resolve("library-v1.json")
    submit("read library revision", path, onRead)(
      ensureDirectory("read library revision").flatMap(_ => readRawRevision(path, "read library revision"))
    )(_ => ())

  def readExportRevision(path: Path, onRead: Either[PresetError, PresetRevision] => Unit): Unit =
    submit("read export revision", path, onRead)(
      ensureDirectory("read export revision").flatMap(_ =>
        exportDestinationAllowed(path).flatMap(_ => readRawRevision(path, "read export revision")))
    )(_ => ())

  def exportFile(path: Path, bytes: Array[Byte], expected: PresetRevision,
                 onExport: Either[PresetError, PresetRevision] => Unit): Unit =
    // The caller may reuse its array after returning. Capture at admission, before I/O.
    val captured =
      if bytes == null then Left(error("input", path, "export source", "null bytes"))
      else if bytes.length > limits.maxBytes then
        Left(error("limit", path, "export source", s"file exceeds ${limits.maxBytes} bytes"))
      else Right(bytes.clone())
    submit("export source", path, onExport)(
      ensureDirectory("export source").flatMap(_ =>
        exportDestinationAllowed(path).flatMap(_ =>
          captured.flatMap(PresetSourceFiles.exportFile(path, _, expected))))
    )(_ => ())

  def directoryPath: Path = root

  def isMutationPending: Boolean = gate.synchronized { pending }

  def close(): Unit =
    gate.synchronized {
      if !closed then
        closed = true
        generation += 1
        pending = false
        io.shutdown()
    }
