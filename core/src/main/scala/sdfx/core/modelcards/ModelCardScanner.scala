package sdfx.core.modelcards

import java.io.InputStream
import java.nio.file.{DirectoryStream, Files, LinkOption, Path}
import java.util.Locale
import scala.collection.mutable
import scala.util.control.NonFatal

final case class CardScanLimits(directoryEntries: Int = 4096, directoryChars: Long = 1024L * 1024L,
    metadataChars: Long = 2000000L, metadataBytes: Long = 96L * 1024L * 1024L,
    records: Int = 50000, depth: Int = 32, pendingDirectories: Int = 10000,
    problemDetails: Int = 100, problemChars: Int = 512, sliceEntries: Int = 5000,
    sliceNanos: Long = 250000000L, cachedGroupEntries: Int = 50000)
object CardScanLimits:
  val defaults: CardScanLimits = CardScanLimits()

trait CardScanObserver:
  def reserve(ticket: Long, category: String, chars: Long, bytes: Long): Unit
  def materialize(ticket: Long): Unit
  def release(ticket: Long): Unit
  def entryAdmitted(directory: Path, chars: Long): Unit
  def directorySorted(directory: Path): Unit
  def weightOpened(path: Path): Unit
  def visited(path: Path): Unit
  def pendingDirectories(count: Int): Unit
  def cancellationCheck(): Unit
  def blockingCall(name: String, elapsedNanos: Long): Unit = ()
  def beforeBlocking(name: String): Unit = ()
object CardScanObserver:
  val none: CardScanObserver = new CardScanObserver:
    def reserve(ticket: Long, category: String, chars: Long, bytes: Long): Unit = ()
    def materialize(ticket: Long): Unit = ()
    def release(ticket: Long): Unit = ()
    def entryAdmitted(directory: Path, chars: Long): Unit = ()
    def directorySorted(directory: Path): Unit = ()
    def weightOpened(path: Path): Unit = ()
    def visited(path: Path): Unit = ()
    def pendingDirectories(count: Int): Unit = ()
    def cancellationCheck(): Unit = ()

/** All metadata paths are validated by the separate Task1 scope. */
trait CardScanDirectoryOps:
  def openDirectory(path: Path): DirectoryStream[Path]
  def openFile(path: Path): InputStream
  def isChildReparse(path: Path): Boolean
  def isDirectory(path: Path): Boolean
  def isRegularFile(path: Path): Boolean
object CardScanDirectoryOps:
  val system: CardScanDirectoryOps = new CardScanDirectoryOps:
    def openDirectory(path: Path) = Files.newDirectoryStream(path)
    def openFile(path: Path) = Files.newInputStream(path)
    def isChildReparse(path: Path) =
      val a = Files.readAttributes(path, classOf[java.nio.file.attribute.BasicFileAttributes], LinkOption.NOFOLLOW_LINKS)
      a.isSymbolicLink || a.isOther
    def isDirectory(path: Path) = Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)
    def isRegularFile(path: Path) = Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)
  class Delegating(delegate: CardScanDirectoryOps) extends CardScanDirectoryOps:
    def openDirectory(path: Path) = delegate.openDirectory(path)
    def openFile(path: Path) = delegate.openFile(path)
    def isChildReparse(path: Path) = delegate.isChildReparse(path)
    def isDirectory(path: Path) = delegate.isDirectory(path)
    def isRegularFile(path: Path) = delegate.isRegularFile(path)

/** Shared accounting across the published snapshot and the in-flight replacement.
  * A ticket is transferred to exactly one result lease or released as temporary workspace.
  */
final class CardScanReservations(val maxChars: Long, val maxBytes: Long, val observer: CardScanObserver):
  require(maxChars >= 24L && maxBytes >= 512L, "Metadata budget must include an emergency result slot")
  private var next = 0L
  private var chars = 0L
  private var bytes = 0L
  private val live = mutable.LongMap.empty[(Long, Long)]
  private val results = new java.util.IdentityHashMap[CardScanResult, Vector[Long]]()
  def tryReserve(category: String, count: Long, size: Long): Option[Long] = synchronized {
    // Keep one control slot free for a replacement scan even when the old snapshot is near its cap.
    val guardChars = if category == "scan-control" then 0L else 24L
    val guardBytes = if category == "scan-control" then 0L else 512L
    if count < 0 || size < 0 || count > maxChars - chars - guardChars || size > maxBytes - bytes - guardBytes then None
    else
      next += 1
      val id = next
      chars += count
      bytes += size
      live(id) = (count, size)
      observer.reserve(id, category, count, size)
      Some(id)
  }
  def release(ticket: Long): Unit = synchronized {
    live.remove(ticket).foreach { (c, b) => chars -= c; bytes -= b; observer.release(ticket) }
  }
  def retain(result: CardScanResult, tickets: Vector[Long]): Unit = synchronized { results.put(result, tickets); () }
  def releaseResult(result: CardScanResult): Unit = synchronized {
    Option(results.remove(result)).foreach(_.foreach(release))
  }

final case class CardScanHooks(limits: CardScanLimits = CardScanLimits.defaults,
    observer: CardScanObserver = CardScanObserver.none, clockNanos: () => Long = () => System.nanoTime(),
    directoryOps: CardScanDirectoryOps = CardScanDirectoryOps.system,
    scope: Option[LocalPreviewFiles.Scope] = None, reservations: Option[CardScanReservations] = None,
    retainResult: Boolean = false, cachedEntries: () => Iterator[CardScanCachedEntry] = () => Iterator.empty)


/** A borrowed cached catalog entry; it is never authority to load a model. */
final case class CardScanCachedEntry(kind: String, path: String, family: Option[String], inspectOnly: Boolean,
    directoryModel: Boolean = false, displayName: Option[String] = None)

private[sdfx] object ModelCardScanner:
  def scan(request: CardScanRequest, cancelled: () => Boolean): CardScanResult = scan(request, cancelled, CardScanHooks())

  def scan(request: CardScanRequest, cancelled: () => Boolean, hooks: CardScanHooks): CardScanResult =
    new Scan(request, cancelled, hooks).run()

  private final class Scan(request: CardScanRequest, cancelled: () => Boolean, hooks: CardScanHooks):
    private val limits = hooks.limits
    private val observer = hooks.observer
    private val ops = hooks.directoryOps
    private val scope = hooks.scope.getOrElse(LocalPreviewFiles.scoped())
    private val ledger = hooks.reservations.getOrElse(new CardScanReservations(limits.metadataChars, limits.metadataBytes, observer))
    private val retained = mutable.ArrayBuffer.empty[Long]
    private val temporary = mutable.HashSet.empty[Long]
    private val records = mutable.ArrayBuffer.empty[ModelCardRecord]
    private val problems = mutable.ArrayBuffer.empty[String]
    private val seen = mutable.HashSet.empty[(String, String)]
    private val cache = mutable.HashMap.empty[(String, String), CardScanCachedEntry]
    private val ambiguous = mutable.HashSet.empty[(String, String)]
    private final case class CachedGroup(parent: Path, realParent: Path,
        items: mutable.ArrayBuffer[CardScanCachedEntry], ticket: Long)
    private val groups = mutable.LinkedHashMap.empty[String, CachedGroup]
    private var groupedEntries = 0
    private final case class PendingRequest(kind: String, base: Path, baseReal: Path, depth: Int, ticket: Long, directoryModel: Boolean)
    private final case class PendingGroup(dir: Path, requests: mutable.ArrayBuffer[PendingRequest])
    private val pending = mutable.HashMap.empty[String, PendingGroup]
    private val pendingOrder = new java.util.PriorityQueue[PendingGroup](new java.util.Comparator[PendingGroup]:
      def compare(a: PendingGroup, b: PendingGroup): Int =
        val depth = Integer.compare(a.dir.getNameCount, b.dir.getNameCount)
        if depth != 0 then depth else a.dir.toString.compareTo(b.dir.toString))
    private var status = CardScanStatus.Complete
    private var problemCount = 0L
    private var visits = 0L
    private var sliceVisits = 0
    private var sliceAt = hooks.clockNanos()
    private var stopped = false
    private final class ScanStopped extends RuntimeException
    private val narrower = "Choose a narrower folder"
    private var emergencySlot = false

    private def reserve(category: String, chars: Long, bytes: Long, keep: Boolean): Option[Long] =
      val ticket = ledger.tryReserve(category, chars, bytes)
      ticket.foreach { t =>
        if keep then retained += t else temporary += t
        observer.materialize(t)
      }
      ticket

    private def free(ticket: Long): Unit =
      temporary -= ticket
      ledger.release(ticket)

    /** Scope charges before growing its identity structures; scratch releases are immediate. */
    private def admitted[A](category: String)(body: => A): (A, Vector[Long]) =
      val owned = mutable.ArrayBuffer.empty[Long]
      val result = scope.withProbeObserver(hooks.clockNanos, (name, elapsed) => {
        observer.blockingCall(name, elapsed)
        observeCancellation()
      }, name => {
        observer.beforeBlocking(name)
        !observeCancellation()
      }) {
        scope.withAdmission { (chars, bytes) =>
          (if check() then None else reserve(category, chars, bytes, keep = false)).map { ticket =>
            owned += ticket
            () => free(ticket)
          }
        }(body)
      }
      observeCancellation()
      (result, owned.iterator.filter(temporary.contains).toVector)

    private def issue(message: => String): Unit =
      if status == CardScanStatus.Complete then status = CardScanStatus.Partial
      problemCount += 1
      if problems.size < limits.problemDetails then
        val cap = limits.problemChars.max(0)
        reserve("error", cap.toLong, 256L + 2L * cap, keep = true) match
          case Some(_) =>
            val text = Option(message).getOrElse("Unavailable")
            problems += text.take(cap)
          case None if problems.isEmpty && emergencySlot =>
            // A control slot was reserved before work began for an explicit cap diagnostic.
            problems += narrower.take(cap)
          case None => ()

    private def capped(reason: String): Unit = issue(s"$reason; $narrower")

    private def observeCancellation(): Boolean =
      observer.cancellationCheck()
      if cancelled() then
        status = CardScanStatus.Cancelled
        stopped = true
      stopped

    private def check(force: Boolean = false): Boolean =
      val now = hooks.clockNanos()
      if force || sliceVisits >= limits.sliceEntries || now - sliceAt >= limits.sliceNanos then
        sliceVisits = 0
        sliceAt = now
        observeCancellation()
      stopped

    private def visit(path: Path): Unit =
      visits += 1
      sliceVisits += 1
      observer.visited(path)
      check()

    private def windowsFold(text: String): String =
      text.map(ch => Character.toLowerCase(Character.toUpperCase(ch)))

    private def key(kind: String, real: Path): (String, String) =
      val text = real.toString
      (kind, if scope.windows then windowsFold(text) else text)

    private def enqueue(kind: String, base: Path, baseReal: Path, dir: Path, depth: Int, directoryModel: Boolean = false): Unit =
      if depth > limits.depth then capped(s"Maximum scan depth reached at $dir")
      else
        val size = dir.toString.length.toLong * 2L + base.toString.length.toLong + baseReal.toString.length.toLong + kind.length.toLong
        reserve("pending-directory", size, 640L + size * 2L, keep = false) match
          case None => capped("Metadata reservation limit reached")
          case Some(t) =>
            val normalized = groupKey(dir)
            if !pending.contains(normalized) && pending.size >= limits.pendingDirectories then
              free(t)
              capped("Pending directory limit reached")
            else
              val group = pending.getOrElseUpdate(normalized, {
                val created = PendingGroup(dir, mutable.ArrayBuffer.empty)
                pendingOrder.add(created)
                created
              })
              group.requests += PendingRequest(kind, base, baseReal, depth, t, directoryModel)
              observer.pendingDirectories(pending.size)

    private def blocking[A](name: String)(body: => A): A =
      observer.beforeBlocking(name)
      observeCancellation()
      if stopped && name != "close-directory" then throw new ScanStopped
      val start = hooks.clockNanos()
      try body finally
        observer.blockingCall(name, (hooks.clockNanos() - start).max(0L))
        observeCancellation()

    private def directoryEntries(dir: Path): (Vector[(Path, Long)], Boolean) =
      var complete = true
      val entries = mutable.ArrayBuffer.empty[(Path, Long)]
      var chars = 0L
      var stream: DirectoryStream[Path] = null
      try
        stream = blocking("open-directory") { ops.openDirectory(dir) }
        val it = if stopped then null else blocking("directory-iterator") { stream.iterator() }
        var full = false
        var hasNext = it != null && !stopped && blocking("directory-has-next") { it.hasNext }
        while !stopped && !full && hasNext do
          val path = blocking("directory-next") { it.next() }
          visit(path)
          if !stopped then
            val count = path.toString.length.toLong + path.getFileName.toString.length.toLong
            if entries.size >= limits.directoryEntries || count > limits.directoryChars - chars then
              capped(s"Directory admission limit reached at $dir")
              complete = false
              full = true
            else
              reserve("directory-entry", count, 384L + count * 2L, keep = false) match
                case None =>
                  capped("Metadata reservation limit reached")
                  complete = false
                  full = true
                case Some(t) =>
                  entries += ((path, t))
                  chars += count
                  observer.entryAdmitted(dir, count)
          if !stopped && !full then hasNext = blocking("directory-has-next") { it.hasNext }
      catch
        case _: ScanStopped => complete = false
        case error: AssertionError => throw error
        case NonFatal(e) =>
          complete = false
          issue(s"Cannot read $dir: ${Option(e.getMessage).getOrElse(e.toString)}")
      finally if stream != null then
        try blocking("close-directory") { stream.close() }
        catch
          case error: AssertionError => throw error
          case NonFatal(e) =>
            complete = false
            issue(s"Cannot close $dir: ${Option(e.getMessage).getOrElse(e.toString)}")
      if stopped then complete = false
      observer.directorySorted(dir)
      entries.sortInPlaceWith { (a, b) =>
        check()
        val x = a._1.getFileName.toString
        val y = b._1.getFileName.toString
        val compare = String.CASE_INSENSITIVE_ORDER.compare(x, y)
        if compare == 0 then x.compareTo(y) < 0 else compare < 0
      }
      (entries.toVector, complete)

    private def add(kind: String, base: Path, path: Path, directoryModel: Boolean, resolver: ModelPreviewResolver): Unit =
      if !check() then
        val (validation, identityTickets) = admitted("model-identity-validation") {
          scope.validateModel(path, directoryModel)
        }
        try
          validation match
            case Left(reason) =>
              if reason.contains("Metadata admission limit") then capped(reason) else issue(s"$path: $reason")
            case Right(identity) =>
              val k = key(kind, identity.realPath)
              if !seen(k) then
                if records.size >= limits.records then capped("Card limit reached")
                else
                  val pathText = path.toAbsolutePath.normalize.toString
                  val nameChars = pathText.length.toLong - base.toString.length.toLong + 1L
                  val cachedNameChars = cache.get(k).flatMap(_.displayName).fold(0L)(_.length.toLong)
                  val chars = pathText.length.toLong * 2L + nameChars.max(0L) * 2L + kind.length.toLong + cachedNameChars * 2L
                  reserve("record-and-index", chars, 1024L + chars * 2L, keep = true) match
                    case None => capped("Metadata reservation limit reached")
                    case Some(_) =>
                      val id = CardId(kind, pathText)
                      val rawName = try base.relativize(path).toString.replace('\\', '/') catch case NonFatal(_) => path.getFileName.toString
                      val dot = rawName.lastIndexOf('.')
                      val localName = if !directoryModel && dot > rawName.lastIndexOf('/') then rawName.substring(0, dot) else rawName
                      val name = cache.get(k).flatMap(_.displayName).getOrElse(localName)
                      val overridePath = request.settings.modelPreviewOverrides.get(id.storageKey)
                      val candidateChars = (math.max(pathText.length, overridePath.fold(0)(_.length)).toLong + 256L) * 12L
                      val candidateTicket = reserve("preview-reference", candidateChars,
                        4096L + candidateChars * 2L, keep = false)
                      var resolved: Either[String, Option[PreviewReference]] = Left("Metadata admission limit reached")
                      var previewTickets = Vector.empty[Long]
                      candidateTicket.foreach { _ =>
                        val operation = admitted("preview-identity-validation") {
                          resolver.resolve(id, directoryModel, overridePath)
                        }
                        resolved = operation._1
                        previewTickets = operation._2
                      }
                      resolved match
                        case Right(Some(ref)) =>
                          val id = ref.identity
                          val ancestorChars = id.ancestorKeys.iterator.map { a =>
                            a.path.toString.length.toLong + a.fileKey.length.toLong + a.reparseTarget.fold(0L)(_.toString.length.toLong)
                          }.sum
                          val actualChars = (ref.localPath.toString.length.toLong + ref.source.length.toLong +
                            id.realPath.toString.length.toLong + id.fileKey.length.toLong + ancestorChars) * 2L
                          val bytes = 1024L + actualChars * 2L + id.ancestorKeys.size.toLong * 256L
                          if reserve("preview-graph", actualChars, bytes, keep = true).isEmpty then
                            resolved = Left("Metadata admission limit reached")
                        case _ => ()
                      previewTickets.foreach(free)
                      candidateTicket.foreach(free)
                      val family = cache.get(k).flatMap(_.family)
                      val rec = resolved match
                        case Left(problem) =>
                          if problem.contains("Metadata admission limit") then capped(problem)
                          ModelCardRecord(id, name, family, None, Some(problem.take(limits.problemChars)))
                        case Right(preview) => ModelCardRecord(id, name, family, preview, None)
                      records += rec
                      seen += k
        finally identityTickets.foreach(free)

    private def suffixAllowed(kind: String, path: Path): Boolean =
      val s = path.getFileName.toString.toLowerCase(Locale.ROOT)
      val suffixes = if kind == "checkpoint" then Vector(".safetensors", ".ckpt")
        else Vector(".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".patch")
      suffixes.exists(s.endsWith)

    private def marker(kind: String): Option[String] = kind match
      case "controlnet" => Some("config.json")
      case "svd" | "zero123" => Some("model_index.json")
      case "checkpoint" | "vae" | "lora" | "embedding" | "upscaler" | "face_restore" | "photomaker" => None
      case _ => Some("model_index.json")

    private def withDirectory[A](dir: Path)(f: (Vector[(Path, Long)], ModelPreviewResolver) => A): A =
      val (entries, listedCompletely) = directoryEntries(dir)
      val comparator: java.util.Comparator[String] =
        if scope.windows then String.CASE_INSENSITIVE_ORDER else java.util.Comparator.naturalOrder[String]()
      val structureTicket = reserve("preview-index-structure", 0L, 512L, keep = false)
      val index = structureTicket.map(_ => new java.util.TreeMap[String, List[Path]](comparator))
      val indexTickets = mutable.ArrayBuffer.empty[Long]
      var complete = listedCompletely && index.nonEmpty
      if index.isEmpty then capped("Metadata reservation limit reached")
      try
        entries.foreach { (path, _) =>
          val name = path.getFileName.toString
          (if index.nonEmpty && !check() then reserve("preview-name-index", name.length.toLong, 384L + name.length.toLong * 2L, keep = false) else None) match
            case None =>
              if complete then capped("Metadata reservation limit reached")
              complete = false
            case Some(ticket) =>
              indexTickets += ticket
              val map = index.get
              map.put(name, path :: Option(map.get(name)).getOrElse(Nil))
        }
        val lookup = (parent: Path, name: String) =>
          if !complete then Left("Preview index is incomplete; choose a narrower folder")
          else if parent.toAbsolutePath.normalize != dir.toAbsolutePath.normalize then Left("Preview lookup parent changed")
          else Right(index.flatMap(map => Option(map.get(name))).getOrElse(Nil))
        f(entries, new ModelPreviewResolver(scope, Some(lookup)))
      finally
        indexTickets.foreach(free)
        structureTicket.foreach(free)
        entries.foreach((_, ticket) => free(ticket))

    private def groupKey(path: Path): String =
      val text = path.toAbsolutePath.normalize.toString
      if scope.windows then windowsFold(text) else text

    private def validateDirectory(base: Path, baseReal: Path, dir: Path, selectedRoot: Boolean): Boolean =
      val (validation, tickets) = admitted("directory-identity-validation") { scope.validateModel(dir, directoryModel = true) }
      try validation match
        case Left(reason) => issue(s"$dir: $reason"); false
        case Right(id) =>
          val childLink = id.ancestorKeys.exists { a =>
            a.reparseTarget.nonEmpty && (!selectedRoot || (a.path.startsWith(base) && a.path != base))
          }
          if childLink || !id.realPath.startsWith(baseReal) then
            issue(s"Child link or escaped directory skipped: $dir"); false
          else true
      finally tickets.foreach(free)

    private def consumeGroup(dir: Path, resolver: ModelPreviewResolver): Unit =
      groups.remove(groupKey(dir)).foreach { group =>
        try group.items.foreach { item =>
          if !check() then
            val path = Path.of(item.path)
            add(item.kind, path.getParent, path, item.directoryModel, resolver)
        }
        finally free(group.ticket)
      }

    private def scanDirectory(group: PendingGroup): Unit =
      val dir = group.dir
      val valid = group.requests.filter(r => validateDirectory(r.base, r.baseReal, dir, selectedRoot = true))
      if valid.nonEmpty && !stopped then
        withDirectory(dir) { (entries, resolver) =>
          consumeGroup(dir, resolver)
          valid.foreach { request =>
            val kind = request.kind
            val base = request.base
            val baseReal = request.baseReal
            if request.directoryModel then add(kind, base, dir, directoryModel = true, resolver = resolver)
            else entries.foreach { (path, _) =>
              if !check() then
                try
                  val (reparse, reparseTickets) = admitted("child-reparse-probe") { scope.childReparse(path) }
                  reparseTickets.foreach(free)
                  if stopped then ()
                  else if blocking("child-attributes") { ops.isChildReparse(path) } || reparse == Right(true) then issue(s"Child link or reparse point skipped: $path")
                  else if stopped then ()
                  else if reparse.isLeft then
                    if reparse.swap.toOption.exists(_.contains("Metadata admission limit")) then capped("Metadata admission limit reached")
                    else issue(s"Child reparse information unavailable: $path")
                  else if blocking("directory-attributes") { ops.isDirectory(path) } then
                    if !stopped then
                      val isModel = marker(kind).exists(m => blocking("marker-attributes") { ops.isRegularFile(path.resolve(m)) })
                      if !stopped && isModel then
                        enqueue(kind, base, baseReal, path, request.depth + 1, directoryModel = true)
                      else if !stopped && !path.getFileName.toString.startsWith(".") then enqueue(kind, base, baseReal, path, request.depth + 1)
                  else if !stopped && blocking("file-attributes") { ops.isRegularFile(path) } && !stopped && suffixAllowed(kind, path) then
                    add(kind, base, path, directoryModel = false, resolver = resolver)
                catch
                  case _: ScanStopped => ()
                  case error: AssertionError => throw error
                  case NonFatal(e) => issue(s"Cannot inspect $path: ${Option(e.getMessage).getOrElse(e.toString)}")
            }
          }
        }

    private def registerGroup(item: CardScanCachedEntry, path: Path): Unit =
      val parent = if item.directoryModel then path else path.getParent
      val (validation, tickets) = admitted("cached-parent-identity") { scope.validateModel(parent, directoryModel = true) }
      try validation match
        case Right(id) if !id.ancestorKeys.exists(_.reparseTarget.nonEmpty) =>
          if groupedEntries >= limits.cachedGroupEntries then capped("Cached grouping limit reached")
          else
            val keyChars = parent.toString.length.toLong
            val keyTicket = reserve("cached-group-key", keyChars, 256L + keyChars * 2L, keep = false)
            val key = if keyTicket.nonEmpty then groupKey(parent) else ""
            if keyTicket.isEmpty then capped("Metadata reservation limit reached")
            val group = if keyTicket.isEmpty then None else groups.get(key).orElse {
              val chars = parent.toString.length.toLong + id.realPath.toString.length.toLong
              reserve("cached-parent-group", chars, 512L + chars * 2L, keep = false).map { t =>
                val created = CachedGroup(parent, id.realPath, mutable.ArrayBuffer.empty, t)
                groups.put(key, created)
                created
              }
            }
            group match
              case None => capped("Metadata reservation limit reached")
              case Some(g) =>
                val chars = item.path.length.toLong + item.kind.length.toLong
                reserve("cached-group-entry", chars, 384L + chars * 2L, keep = false) match
                  case None => capped("Metadata reservation limit reached")
                  case Some(_) => g.items += item; groupedEntries += 1
        case Left(reason) if reason.contains("Metadata admission limit") => capped(reason)
        case _ => issue(s"Cached parent is unavailable or reparse-backed: $parent")
      finally tickets.foreach(free)

    private def collectCache(): Unit =
      val it = hooks.cachedEntries()
      while !stopped && it.hasNext do
        val item = it.next()
        if item.kind != null && item.path != null && item.kind.nonEmpty && item.path.nonEmpty then
          val (resolved, pathTickets) = admitted("cached-path") { scope.resolveRoot(item.path, request.home) }
          try
            if resolved.swap.toOption.exists(_.contains("Metadata admission limit")) then capped("Metadata admission limit reached")
            resolved.foreach { path =>
              visit(path)
              if !stopped then
                val n = item.kind.length.toLong + item.path.length.toLong + item.family.fold(0)(_.length).toLong + item.displayName.fold(0)(_.length).toLong
                reserve("cached-index", n * 2L, 384L + n * 4L, keep = false) match
                  case None => capped("Metadata reservation limit reached"); stopped = true
                  case Some(_) =>
                    val (reparse, reparseTickets) = admitted("cached-reparse-probe") { scope.childReparse(path) }
                    reparseTickets.foreach(free)
                    if reparse == Right(false) && !stopped then
                      val isDirectory = blocking("cached-directory-attributes") { ops.isDirectory(path) }
                      val markerValid = !isDirectory || (item.inspectOnly && !stopped &&
                        blocking("cached-marker-attributes") { ops.isRegularFile(path.resolve("model_index.json")) })
                      val (validation, identityTickets) = admitted("cached-identity-validation") {
                        scope.validateModel(path, directoryModel = isDirectory)
                      }
                      try validation match
                        case Right(identity) if markerValid && (!item.inspectOnly || !identity.ancestorKeys.exists(_.reparseTarget.nonEmpty)) =>
                          val idKey = key(item.kind, identity.realPath)
                          val normalized = item.copy(path = path.toString, directoryModel = isDirectory)
                          cache.get(idKey) match
                            case Some(previous) if previous.displayName != normalized.displayName =>
                              cache.remove(idKey)
                              ambiguous += idKey
                            case _ if !ambiguous(idKey) => cache.update(idKey, normalized)
                            case _ => ()
                          if normalized.inspectOnly then registerGroup(normalized, path)
                        case Left(reason) if reason.contains("Metadata admission limit") => capped(reason)
                        case _ => () // stale or reparse-backed cache never authorizes a card
                      finally identityTickets.foreach(free)
                    else if reparse.swap.toOption.exists(_.contains("Metadata admission limit")) then
                      capped("Metadata admission limit reached")
            }
          finally pathTickets.foreach(free)
        check()

    def run(): CardScanResult =
      try
        check(force = true)
        emergencySlot = reserve("scan-control", narrower.length.toLong, 512L, keep = true).nonEmpty
        if !emergencySlot then
          status = CardScanStatus.Partial
          stopped = true
          problemCount = 1L
          problems += narrower
        if !stopped && scope.windows && reserve("drive-cache", 128L, 4096L, keep = false).isEmpty then
          capped("Metadata reservation limit reached")
          stopped = true
        if !stopped then collectCache()
        val d = request.settings.modelDirs
        val builtins = Vector("checkpoint" -> d.checkpoints, "vae" -> d.vae, "lora" -> d.lora,
          "controlnet" -> d.controlnet, "embedding" -> d.embeddings, "upscaler" -> d.upscalers,
          "face_restore" -> d.faceRestore, "svd" -> d.svd, "zero123" -> d.zero123, "photomaker" -> d.photomaker)
        val roots = builtins.iterator ++ request.settings.modelPreviewFolders.iterator
        while !stopped && roots.hasNext do
          val (kind, paths) = roots.next()
          paths.foreach { raw =>
            if !stopped then
              val (resolved, rootTickets) = admitted("root-path") { scope.resolveRoot(raw, request.home) }
              try resolved match
                case Left(reason) =>
                  if reason.contains("Metadata admission limit") then capped(reason) else issue(s"$raw: $reason")
                case Right(path) =>
                  val (validation, identityTickets) = admitted("root-identity-validation") {
                    scope.validateModel(path, directoryModel = true)
                  }
                  try validation match
                    case Left(reason) =>
                      if reason.contains("Metadata admission limit") then capped(reason) else issue(s"$path: $reason")
                    case Right(identity) => enqueue(kind, path, identity.realPath, path, 0)
                  finally identityTickets.foreach(free)
              finally rootTickets.foreach(free)
          }
        while !stopped && pendingOrder.size() > 0 do
          val group = pendingOrder.remove()
          pending.remove(groupKey(group.dir))
          observer.pendingDirectories(pending.size)
          try scanDirectory(group)
          finally group.requests.foreach(r => free(r.ticket))
        while !stopped && groups.nonEmpty do
          val group = groups.head._2
          if validateDirectory(group.parent, group.realParent, group.parent, selectedRoot = false) && !stopped then
            withDirectory(group.parent)((ignored, resolver) => consumeGroup(group.parent, resolver))
          else
            groups.remove(groupKey(group.parent))
            free(group.ticket)
        records.sortInPlaceWith { (a, b) =>
          check()
          val names = String.CASE_INSENSITIVE_ORDER.compare(a.name, b.name)
          if names != 0 then names < 0
          else if a.id.kind != b.id.kind then a.id.kind < b.id.kind
          else a.id.modelPath < b.id.modelPath
        }
        val result = CardScanResult(request.revision, records.toVector, status, problems.toVector, problemCount, visits)
        temporary.toVector.foreach(free)
        if hooks.retainResult then ledger.retain(result, retained.toVector)
        else retained.foreach(ledger.release)
        result
      catch
        case _: ScanStopped =>
          temporary.toVector.foreach(free)
          retained.foreach(ledger.release)
          CardScanResult(request.revision, Vector.empty, CardScanStatus.Cancelled, Vector.empty, problemCount, visits)
        case error: AssertionError => throw error
        case NonFatal(e) =>
          temporary.toVector.foreach(free)
          retained.foreach(ledger.release)
          CardScanResult(request.revision, Vector.empty, CardScanStatus.Failed,
            Vector(s"Scan failed: ${Option(e.getMessage).getOrElse(e.toString)}".take(limits.problemChars)), 1L, visits)
