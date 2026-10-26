package sdfx.ui

import java.nio.file.Files
import java.nio.charset.StandardCharsets.UTF_8
import java.security.MessageDigest
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import scala.util.control.NonFatal
import sdfx.core.presets.*
import sdfx.ui.presets.NativePresetsSession

/** Pure session and CLI contracts. Actual JavaFX controls are exercised by
  * RealNativePresetsPaneCheck; this suite never starts Platform or touches the clipboard.
  */
class NativePresetsPaneSuite extends munit.FunSuite:
  private type Result = Either[PresetError, LoadedPresets]

  /** The private session seam queues the same delivery thunks that production posts to FX.
    * Draining on this test thread makes the pending/revision transition deterministic.
    */
  private final class Harness(directory: java.nio.file.Path,
                              createDefaultDirectory: Boolean = false,
                              seedExistingLibrary: Boolean = true):
    // These cases isolate the session's concurrency contract from first-use seeding,
    // which the actual MainWindow helper exercises with a missing library.
    if seedExistingLibrary && Files.isDirectory(directory) then
      assert(new PresetStore(directory, PresetLimits()).save(PresetRevision(None), emptyLibrary).isRight)
    val deliveries = new LinkedBlockingQueue[() => Unit]()
    val dispatchThreads = new LinkedBlockingQueue[Thread]()
    val results = new LinkedBlockingQueue[Result]()
    val callbackThreads = new LinkedBlockingQueue[Thread]()
    private val onResult: Result => Unit =
      r => { callbackThreads.add(Thread.currentThread()); results.add(r); () }
    private val dispatch: (() => Unit) => Unit =
      thunk => { dispatchThreads.add(Thread.currentThread()); deliveries.add(thunk); () }
    val session =
      if createDefaultDirectory then
        new NativePresetsSession(directory, Map.empty, onResult, dispatch, true)
      else new NativePresetsSession(directory, Map.empty, onResult, dispatch)

    def nextDelivery(): () => Unit =
      val delivery = deliveries.poll(10, TimeUnit.SECONDS)
      assert(delivery != null, "preset I/O callback was never scheduled")
      delivery

    def receive(): Result =
      val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
      var result: Result = null
      while result == null && System.nanoTime() < end do
        val delivery = deliveries.poll(100, TimeUnit.MILLISECONDS)
        if delivery != null then delivery()
        result = results.poll()
      assert(result != null, "preset I/O completion did not arrive")
      result

    def waitForDelivery(): Unit =
      deliveries.add(nextDelivery())

    def drain(): Unit =
      var delivery = deliveries.poll()
      while delivery != null do
        delivery()
        delivery = deliveries.poll()

  private def collection(id: String, names: String*): PresetCollection =
    PresetCollection(id, Set.empty, Vector.empty, names.toVector.map { name =>
      PresetRecord(PresetRecord.initialId(id, name), name, Map.empty, None, Set.empty, Vector.empty, Vector.empty)
    }, Vector.empty)

  private def emptyLibrary: PresetLibrary = PresetLibrary(1, Vector(
    collection("config-txt2img"), collection("config-img2img"), collection("utilities")), Vector.empty)

  private def revisionOf(bytes: Array[Byte]): PresetRevision =
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
    PresetRevision(Some(digest.map(b => f"${b & 0xff}%02x").mkString))

  private def receiveRevision(h: Harness)(
      start: (Either[PresetError, PresetRevision] => Unit) => Unit): Either[PresetError, PresetRevision] =
    val owner = Thread.currentThread()
    var result: Either[PresetError, PresetRevision] = null
    start(value => {
      assertEquals(Thread.currentThread(), owner)
      result = value
    })
    val delivery = h.nextDelivery()
    assert(result == null, "revision callback bypassed the owner dispatcher")
    delivery()
    assert(result != null, "revision callback was lost")
    assert(!h.session.isMutationPending)
    result

  test("all results including Busy reach onResult only through the owner dispatcher") {
    val directory = Files.createTempDirectory("presets-dispatch-")
    val h = new Harness(directory)
    try
      val owner = Thread.currentThread()
      h.session.reload()
      val active = h.nextDelivery()
      h.session.reload()
      val busy = h.nextDelivery()
      val worker = h.dispatchThreads.poll()
      assert(worker != null && worker.isDaemon && worker != owner,
        "preset load was not dispatched from the daemon I/O worker")
      assertEquals(h.dispatchThreads.poll(), owner)
      assert(h.results.isEmpty, "onResult ran before either delivery was drained")
      assert(h.callbackThreads.isEmpty)
      busy()
      val rejected = h.results.poll()
      assert(rejected != null && rejected.isLeft, "overlapping reload did not report Busy")
      assert(rejected.swap.toOption.get.issues.exists(_.code.equalsIgnoreCase("busy")))
      assertEquals(h.callbackThreads.poll(), owner)
      assert(h.session.isMutationPending, "Busy delivery cleared the active reload")
      active()
      val completed = h.results.poll()
      assert(completed != null && completed.isRight, "active reload was lost after Busy")
      assertEquals(h.callbackThreads.poll(), owner)
      assert(!h.session.isMutationPending)
    finally h.session.close()
  }

  test("Busy delivery leaves a saved revision pending until the active result is delivered") {
    val directory = Files.createTempDirectory("presets-busy-pending-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val accepted = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Accepted")))
      val rejected = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Rejected")))
      h.session.save(initial.revision, accepted)
      val active = h.nextDelivery()
      h.session.save(initial.revision, rejected)
      val busy = h.nextDelivery()
      assert(h.results.isEmpty, "disk completion bypassed the delivery dispatcher")
      busy()
      val busyResult = h.results.poll()
      assert(busyResult != null && busyResult.isLeft, "overlapping save did not report Busy")
      assert(h.session.isMutationPending, "Busy response reopened writes before revision install")
      active()
      val completion = h.results.poll()
      assert(completion != null && completion.isRight, "active save was lost after Busy")
      val saved = completion.toOption.get
      assert(saved.revision != initial.revision)
      assert(!h.session.isMutationPending)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get, saved)
      assertEquals(saved.library.collections(2).presets.map(_.name), Vector("Accepted"))
    finally h.session.close()
  }

  test("a delivered missing-library result can start a validated first save in its callback") {
    val directory = Files.createTempDirectory("presets-callback-seed-")
    val deliveries = new LinkedBlockingQueue[() => Unit]()
    val results = new LinkedBlockingQueue[Result]()
    var session: NativePresetsSession = null
    session = new NativePresetsSession(directory, Map.empty,
      result => {
        results.add(result)
        result match
          case Right(loaded) if loaded.revision == PresetRevision(None) =>
            session.save(loaded.revision, emptyLibrary)
          case _ => ()
      },
      thunk => { deliveries.add(thunk); () })
    try
      session.reload()
      val first = deliveries.poll(10, TimeUnit.SECONDS)
      assert(first != null)
      assert(session.isMutationPending)
      first()
      val loaded = results.poll()
      assert(loaded != null && loaded.isRight)
      assertEquals(loaded.toOption.get.revision, PresetRevision(None))
      assert(session.isMutationPending, "callback save was rejected as Busy")
      val second = deliveries.poll(10, TimeUnit.SECONDS)
      assert(second != null)
      second()
      val seeded = results.poll()
      assert(seeded != null && seeded.isRight)
      assert(seeded.toOption.get.revision.sha256.nonEmpty)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.library, emptyLibrary)
    finally session.close()
  }

  test("a delayed Busy callback from an older operation cannot reach a newer generation") {
    val directory = Files.createTempDirectory("presets-stale-generation-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val first = h.nextDelivery()
      h.session.reload()
      val oldBusy = h.nextDelivery()
      first()
      val firstResult = h.results.poll()
      assert(firstResult != null && firstResult.isRight)
      h.session.reload()
      val next = h.nextDelivery()
      oldBusy()
      assert(h.results.isEmpty, "stale Busy callback reached the newer operation")
      assert(h.session.isMutationPending, "stale Busy callback cleared the newer operation")
      next()
      val nextResult = h.results.poll()
      assert(nextResult != null && nextResult.isRight)
      assert(!h.session.isMutationPending)
    finally h.session.close()
  }

  test("an absent explicit preset directory is rejected without creating it on reload or save") {
    val parent = Files.createTempDirectory("presets-explicit-absent-")
    val directory = parent.resolve("not-created-yet")
    val h = new Harness(directory)
    try
      assert(!Files.exists(directory))
      h.session.save(PresetRevision(None), emptyLibrary)
      assert(h.receive().isLeft, "explicit missing path was created by PresetStore.save")
      assert(!Files.exists(directory))
      h.session.reload()
      assert(h.receive().isLeft)
      assert(!Files.exists(directory))
    finally h.session.close()
  }

  test("default-only admission creates its directory on the daemon before missing-library reload") {
    val parent = Files.createTempDirectory("presets-default-first-use-")
    val directory = parent.resolve("config").resolve("native-presets")
    val h = new Harness(directory, createDefaultDirectory = true)
    try
      assert(!Files.exists(directory), "constructor performed filesystem I/O on the caller thread")
      h.session.reload()
      val delivery = h.nextDelivery()
      val worker = h.dispatchThreads.poll()
      assert(worker != null && worker.isDaemon && worker != Thread.currentThread())
      assert(Files.isDirectory(directory), "default directory was not created by the I/O worker")
      assert(!Files.exists(directory.resolve("library-v1.json")), "session seeded a library without pane validation")
      assert(h.results.isEmpty)
      delivery()
      val loaded = h.results.poll()
      assert(loaded != null && loaded.isRight)
      assertEquals(loaded.toOption.get.revision, PresetRevision(None))
      assert(loaded.toOption.get.library.collections.isEmpty)
      assert(!Files.exists(directory.resolve("library-v1.json")))
    finally h.session.close()
  }

  test("default-only admission reports an existing non-directory without replacing it") {
    val path = Files.createTempFile("presets-default-wrong-type-", ".txt")
    val original = "keep this file".getBytes(UTF_8)
    Files.write(path, original)
    val h = new Harness(path, createDefaultDirectory = true)
    try
      h.session.reload()
      assert(h.receive().isLeft)
      assertEquals(Files.readAllBytes(path).toVector, original.toVector)
    finally
      h.session.close()
      Files.deleteIfExists(path)
  }

  test("a default directory disappearing after first load is reported rather than recreated") {
    val parent = Files.createTempDirectory("presets-default-disappeared-")
    val directory = parent.resolve("native-presets")
    val h = new Harness(directory, createDefaultDirectory = true)
    try
      h.session.reload()
      assert(h.receive().isRight)
      Files.delete(directory)
      h.session.reload()
      assert(h.receive().isLeft)
      assert(!Files.exists(directory))
    finally h.session.close()
  }

  test("typed source reads and export use owner-thread delivery without changing library revision") {
    val directory = Files.createTempDirectory("presets-typed-io-")
    val h = new Harness(directory)
    val owner = Thread.currentThread()
    val source = directory.resolve("source.json")
    val tracked = directory.resolve("tracked.txt")
    val additional = directory.resolve("additional.json")
    val exported = directory.resolve("export.json")
    val sourceBytes = """{"One":{"txt2img_steps":7}}""".getBytes(UTF_8)
    val exportBytes = """{"Exported":{}}""".getBytes(UTF_8)
    val expectedExport = exportBytes.toVector
    Files.write(source, sourceBytes)
    Files.writeString(tracked, "# note\ntxt2img_steps\ncustom.script-value\n")
    Files.writeString(additional, """{"additionalComponents":{"Optional":null,"X type":"script_xyz_x_type"}}""")
    try
      h.session.reload()
      val loaded = h.receive().toOption.get

      var document: Either[PresetError, SourceDocument] = null
      h.session.readSource(source, "config-presets", Some(PresetMode.Txt2Img), r => {
        assertEquals(Thread.currentThread(), owner)
        document = r
      })
      val deliverSource = h.nextDelivery()
      assert(document == null)
      deliverSource()
      assertEquals(document.toOption.get.format, "config-presets")
      assertEquals(document.toOption.get.mode, Some(PresetMode.Txt2Img))
      assert(document.toOption.get.raw.text.contains("txt2img_steps"))

      var ids: Either[PresetError, Vector[String]] = null
      h.session.readTrackedIds(tracked, r => {
        assertEquals(Thread.currentThread(), owner)
        ids = r
      })
      val deliverIds = h.nextDelivery()
      assert(ids == null)
      deliverIds()
      assertEquals(ids.toOption.get, Vector("txt2img_steps", "custom.script-value"))

      var mappings: Either[PresetError, Vector[(String, Option[String])]] = null
      h.session.readAdditionalComponents(additional, r => {
        assertEquals(Thread.currentThread(), owner)
        mappings = r
      })
      val deliverMappings = h.nextDelivery()
      assert(mappings == null)
      deliverMappings()
      assertEquals(mappings.toOption.get, Vector("Optional" -> None, "X type" -> Some("script_xyz_x_type")))

      var exportedRevision: Either[PresetError, PresetRevision] = null
      h.session.exportFile(exported, exportBytes, PresetRevision(None), r => {
        assertEquals(Thread.currentThread(), owner)
        exportedRevision = r
      })
      java.util.Arrays.fill(exportBytes, '!'.toByte)
      val deliverExport = h.nextDelivery()
      assert(exportedRevision == null)
      deliverExport()
      assert(exportedRevision != null && exportedRevision.isRight)
      assertEquals(Files.readAllBytes(exported).toVector, expectedExport)
      assert(h.results.isEmpty, "typed operations emitted a library result")
      assert(!h.session.isMutationPending)

      h.session.save(loaded.revision, emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "After export"))))
      assert(h.receive().isRight, "export revision replaced the library revision")
    finally h.session.close()
  }

  test("raw library revision distinguishes missing, empty, malformed JSON and invalid UTF-8") {
    val directory = Files.createTempDirectory("presets-raw-library-revision-")
    val h = new Harness(directory, seedExistingLibrary = false)
    val library = directory.resolve("library-v1.json")
    try
      assertEquals(receiveRevision(h)(cb => h.session.readLibraryRevision(cb)),
        Right(PresetRevision(None)))
      assert(!Files.exists(library), "metadata read created the authoritative file")
      val rawCases = Vector(
        Array.empty[Byte],
        "{broken".getBytes(UTF_8),
        Array[Byte](0xff.toByte, 0xfe.toByte, 0x80.toByte))
      rawCases.foreach { bytes =>
        Files.write(library, bytes)
        assertEquals(receiveRevision(h)(cb => h.session.readLibraryRevision(cb)),
          Right(revisionOf(bytes)))
        assertEquals(Files.readAllBytes(library).toVector, bytes.toVector)
      }
      assert(h.results.isEmpty, "metadata reads changed the library result stream")
    finally h.session.close()
  }

  test("raw revision reads refuse oversized library and export files without changing bytes") {
    val directory = Files.createTempDirectory("presets-raw-revision-limit-")
    val outside = Files.createTempDirectory("presets-raw-revision-outside-")
    val h = new Harness(directory, seedExistingLibrary = false)
    val library = directory.resolve("library-v1.json")
    val source = outside.resolve("large-source.json")
    val oversized = Array.fill[Byte](PresetLimits().maxBytes + 1)('x'.toByte)
    Files.write(library, oversized)
    Files.write(source, oversized)
    try
      val native = receiveRevision(h)(cb => h.session.readLibraryRevision(cb))
      assertEquals(native.swap.toOption.get.issues.head.code, "limit")
      val interchange = receiveRevision(h)(cb => h.session.readExportRevision(source, cb))
      assertEquals(interchange.swap.toOption.get.issues.head.code, "limit")
      assertEquals(Files.size(library), oversized.length.toLong)
      assertEquals(Files.size(source), oversized.length.toLong)
      assert(h.results.isEmpty)
    finally h.session.close()
  }

  test("raw library revision inspection never installs an external edit as the loaded revision") {
    val directory = Files.createTempDirectory("presets-raw-no-install-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val loaded = h.receive().toOption.get
      val external = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "External edit")))
      val bytes = PresetJson.writeNative(external, PresetLimits()).toOption.get
      val library = directory.resolve("library-v1.json")
      Files.write(library, bytes)
      val observed = receiveRevision(h)(cb => h.session.readLibraryRevision(cb)).toOption.get
      assertEquals(observed, revisionOf(bytes))
      assert(observed != loaded.revision)
      h.session.save(observed, external)
      assert(h.receive().isLeft, "raw metadata read silently installed an external library revision")
      assertEquals(Files.readAllBytes(library).toVector, bytes.toVector)
    finally h.session.close()
  }

  test("revision metadata uses the same Busy gate and suppresses queued delivery after close") {
    val directory = Files.createTempDirectory("presets-revision-delivery-")
    val outside = Files.createTempFile("presets-revision-busy-", ".json")
    val h = new Harness(directory)
    var native: Either[PresetError, PresetRevision] = null
    var interchange: Either[PresetError, PresetRevision] = null
    try
      h.session.readLibraryRevision(r => native = r)
      val active = h.nextDelivery()
      h.session.readExportRevision(outside, r => interchange = r)
      val busy = h.nextDelivery()
      assert(native == null && interchange == null)
      busy()
      assert(interchange != null && interchange.isLeft)
      assert(interchange.swap.toOption.get.issues.exists(_.code.equalsIgnoreCase("busy")))
      assert(h.session.isMutationPending)
      h.session.close()
      active()
      assert(native == null, "closed session delivered a queued raw revision")
      assert(h.results.isEmpty)
    finally
      h.session.close()
      Files.deleteIfExists(outside)
  }

  test("export revision read rejects protected native paths and accepts raw outside bytes") {
    val directory = Files.createTempDirectory("presets-export-revision-guard-")
    val outside = Files.createTempDirectory("presets-export-revision-outside-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      h.session.save(initial.revision, emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Version A"))))
      val savedA = h.receive().toOption.get
      val versionB = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Version B")))
      h.session.save(savedA.revision, versionB)
      val savedB = h.receive().toOption.get
      val library = directory.resolve("library-v1.json")
      val backup = directory.resolve("library-v1.json.bak")
      val lock = directory.resolve(".library-v1.lock")
      val rejected = directory.resolve("library-v1.rejected-existing.json")
      Files.writeString(rejected, "preserved rejection")
      val reserved = Vector(library, backup, lock, rejected)
      val before = reserved.map(p => p -> Files.readAllBytes(p).toVector).toMap
      reserved.foreach { path =>
        val result = receiveRevision(h)(cb => h.session.readExportRevision(path, cb))
        assertEquals(result.swap.toOption.get.issues.head.code, "reserved-path")
        reserved.foreach(p => assertEquals(Files.readAllBytes(p).toVector, before(p)))
      }
      val futureRejected = directory.resolve("library-v1.rejected-future.json")
      val futureResult = receiveRevision(h)(cb => h.session.readExportRevision(futureRejected, cb))
      assertEquals(futureResult.swap.toOption.get.issues.head.code, "reserved-path")
      assert(!Files.exists(futureRejected))

      val insideSource = directory.resolve("ordinary-source.json")
      assertEquals(receiveRevision(h)(cb => h.session.readExportRevision(insideSource, cb)),
        Right(PresetRevision(None)))

      val destination = outside.resolve("existing-source.json")
      assertEquals(receiveRevision(h)(cb => h.session.readExportRevision(destination, cb)),
        Right(PresetRevision(None)))
      val raw = Array[Byte](0xff.toByte, 0x00.toByte, 0x7f.toByte)
      Files.write(destination, raw)
      val observed = receiveRevision(h)(cb => h.session.readExportRevision(destination, cb)).toOption.get
      assertEquals(observed, revisionOf(raw))
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.revision, savedB.revision)

      val replacement = """{"Reviewed":{}}""".getBytes(UTF_8)
      var replaced: Either[PresetError, PresetRevision] = null
      h.session.exportFile(destination, replacement, observed, r => replaced = r)
      h.nextDelivery()()
      assert(replaced != null && replaced.isRight)
      assertEquals(Files.readAllBytes(destination).toVector, replacement.toVector)
      val reviewedAgain = receiveRevision(h)(cb => h.session.readExportRevision(destination, cb)).toOption.get
      assertEquals(reviewedAgain, replaced.toOption.get)

      val external = "external edit after review".getBytes(UTF_8)
      Files.write(destination, external)
      var refused: Either[PresetError, PresetRevision] = null
      h.session.exportFile(destination, "{}".getBytes(UTF_8), reviewedAgain, r => refused = r)
      h.nextDelivery()()
      assert(refused != null && refused.isLeft, "stale reviewed export overwrote external bytes")
      assertEquals(Files.readAllBytes(destination).toVector, external.toVector)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.revision, savedB.revision)
      h.session.save(savedB.revision, versionB)
      assert(h.receive().isRight, "metadata read replaced the loaded library revision")
    finally h.session.close()
  }

  test("corrupt library can be explicitly restored using its observed raw revision") {
    val directory = Files.createTempDirectory("presets-observed-restore-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val versionA = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Recover A")))
      val versionB = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Before corruption")))
      h.session.save(initial.revision, versionA)
      val savedA = h.receive().toOption.get
      h.session.save(savedA.revision, versionB)
      assert(h.receive().isRight)
      val library = directory.resolve("library-v1.json")
      val corrupt = "{unparseable".getBytes(UTF_8)
      Files.write(library, corrupt)
      h.session.reload()
      assert(h.receive().isLeft)
      val observed = receiveRevision(h)(cb => h.session.readLibraryRevision(cb)).toOption.get
      assertEquals(observed, revisionOf(corrupt))
      h.session.restoreBackup(observed)
      val restored = h.receive()
      assert(restored.isRight)
      assertEquals(restored.toOption.get.library, versionA)
      val rejected = Files.list(directory)
      try
        val preserved = rejected.filter(p => p.getFileName.toString.contains(".rejected-"))
          .toArray.map(_.asInstanceOf[java.nio.file.Path]).toVector
        assertEquals(preserved.size, 1)
        assertEquals(Files.readAllBytes(preserved.head).toVector, corrupt.toVector)
      finally rejected.close()
    finally h.session.close()
  }

  test("an edit after raw revision review prevents restore and preserves both files") {
    val directory = Files.createTempDirectory("presets-stale-observed-restore-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val versionA = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Backup A")))
      val versionB = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Current B")))
      h.session.save(initial.revision, versionA)
      val savedA = h.receive().toOption.get
      h.session.save(savedA.revision, versionB)
      assert(h.receive().isRight)
      val library = directory.resolve("library-v1.json")
      val backup = directory.resolve("library-v1.json.bak")
      val backupBytes = Files.readAllBytes(backup).toVector
      Files.writeString(library, "{first corrupt")
      h.session.reload()
      assert(h.receive().isLeft)
      val observed = receiveRevision(h)(cb => h.session.readLibraryRevision(cb)).toOption.get
      val later = "{edited after review".getBytes(UTF_8)
      Files.write(library, later)
      h.session.restoreBackup(observed)
      val refused = h.receive()
      assert(refused.isLeft, "stale observed revision restored over an external edit")
      assertEquals(Files.readAllBytes(library).toVector, later.toVector)
      assertEquals(Files.readAllBytes(backup).toVector, backupBytes)
    finally h.session.close()
  }

  test("source export refuses every reserved native file and preserves both library versions") {
    val directory = Files.createTempDirectory("presets-export-reserved-")
    val outside = Files.createTempDirectory("presets-export-outside-")
    val h = new Harness(directory)
    val sourceBytes = """{"Source only":{}}""".getBytes(UTF_8)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val versionA = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Version A")))
      val versionB = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Version B")))
      h.session.save(initial.revision, versionA)
      val savedA = h.receive().toOption.get
      h.session.save(savedA.revision, versionB)
      val savedB = h.receive().toOption.get
      val library = directory.resolve("library-v1.json")
      val backup = directory.resolve("library-v1.json.bak")
      val lock = directory.resolve(".library-v1.lock")
      val rejected = directory.resolve("library-v1.rejected-2000-01-01T00-00-00Z-evidence.json")
      Files.writeString(rejected, "corrupt bytes held for explicit recovery")
      val reserved = Vector(library, backup, lock, rejected)
      val before = reserved.map(path => path -> Files.readAllBytes(path).toVector).toMap
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.revision, savedB.revision)
      assertEquals(PresetJson.readNative(before(backup).toArray, PresetLimits()).toOption.get, versionA)

      reserved.foreach { path =>
        var answer: Either[PresetError, PresetRevision] = null
        h.session.exportFile(path, sourceBytes, revisionOf(before(path).toArray), r => answer = r)
        val delivery = h.nextDelivery()
        assert(answer == null, s"export to $path bypassed the owner dispatcher")
        delivery()
        assert(answer != null && answer.isLeft, s"reserved export target $path was accepted")
        assert(!h.session.isMutationPending, s"reserved export to $path left the gate pending")
        reserved.foreach(p => assertEquals(Files.readAllBytes(p).toVector, before(p)))
      }

      val futureRejected = directory.resolve("library-v1.rejected-future.json")
      var futureAnswer: Either[PresetError, PresetRevision] = null
      h.session.exportFile(futureRejected, sourceBytes, PresetRevision(None), r => futureAnswer = r)
      h.nextDelivery()()
      assert(futureAnswer != null && futureAnswer.isLeft, "new recovery namespace path was accepted")
      assert(!Files.exists(futureRejected))
      assert(!h.session.isMutationPending)

      val outsideFile = outside.resolve("source-export.json")
      var ordinary: Either[PresetError, PresetRevision] = null
      h.session.exportFile(outsideFile, sourceBytes, PresetRevision(None), r => ordinary = r)
      val outsideDelivery = h.nextDelivery()
      assert(ordinary == null)
      outsideDelivery()
      assert(ordinary != null && ordinary.isRight, "ordinary outside source export was blocked")
      assertEquals(Files.readAllBytes(outsideFile).toVector, sourceBytes.toVector)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.revision, savedB.revision)
      reserved.foreach(p => assertEquals(Files.readAllBytes(p).toVector, before(p)))
    finally h.session.close()
  }

  test("source export after a corrupt reload preserves corrupt bytes and the good backup") {
    val directory = Files.createTempDirectory("presets-export-corrupt-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val versionA = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Good A")))
      val versionB = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Good B")))
      h.session.save(initial.revision, versionA)
      val savedA = h.receive().toOption.get
      h.session.save(savedA.revision, versionB)
      assert(h.receive().isRight)
      val library = directory.resolve("library-v1.json")
      val backup = directory.resolve("library-v1.json.bak")
      val goodBackup = Files.readAllBytes(backup).toVector
      val corrupt = "{broken".getBytes(UTF_8)
      Files.write(library, corrupt)
      h.session.reload()
      assert(h.receive().isLeft)

      var answer: Either[PresetError, PresetRevision] = null
      h.session.exportFile(library, """{"Looks valid":{}}""".getBytes(UTF_8),
        revisionOf(corrupt), r => answer = r)
      val delivery = h.nextDelivery()
      assert(answer == null)
      delivery()
      assert(answer != null && answer.isLeft)
      assert(!h.session.isMutationPending)
      assertEquals(Files.readAllBytes(library).toVector, corrupt.toVector)
      assertEquals(Files.readAllBytes(backup).toVector, goodBackup)
    finally h.session.close()
  }

  test("case aliases of the authoritative library are refused when the host resolves them") {
    val directory = Files.createTempDirectory("presets-export-case-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val loaded = h.receive().toOption.get
      val library = directory.resolve("library-v1.json")
      val alias = directory.resolve("LiBrArY-V1.JsOn")
      assume(Files.exists(alias) && Files.isSameFile(alias, library),
        "host uses case-sensitive file names; Windows case-alias path needs a case-insensitive host")
      val original = Files.readAllBytes(library).toVector
      val metadata = receiveRevision(h)(cb => h.session.readExportRevision(alias, cb))
      assertEquals(metadata.swap.toOption.get.issues.head.code, "reserved-path")
      var answer: Either[PresetError, PresetRevision] = null
      h.session.exportFile(alias, "{}".getBytes(UTF_8), loaded.revision, r => answer = r)
      h.nextDelivery()()
      assert(answer != null && answer.isLeft)
      assertEquals(Files.readAllBytes(library).toVector, original)
      assert(!h.session.isMutationPending)
    finally h.session.close()
  }

  test("resolved parent aliases of the authoritative library are refused when supported") {
    val directory = Files.createTempDirectory("presets-export-parent-")
    val aliasParent = Files.createTempDirectory("presets-export-link-")
    val alias = aliasParent.resolve("presets-link")
    val h = new Harness(directory)
    try
      h.session.reload()
      val loaded = h.receive().toOption.get
      val library = directory.resolve("library-v1.json")
      val linked =
        try
          Files.createSymbolicLink(alias, directory)
          true
        catch case NonFatal(_) => false
      assume(linked, "host cannot create a directory symlink; parent-alias coverage needs that facility")
      assert(Files.isSameFile(alias.resolve("library-v1.json"), library))
      val original = Files.readAllBytes(library).toVector
      val metadata = receiveRevision(h)(cb => h.session.readExportRevision(alias.resolve("library-v1.json"), cb))
      assertEquals(metadata.swap.toOption.get.issues.head.code, "reserved-path")
      var answer: Either[PresetError, PresetRevision] = null
      h.session.exportFile(alias.resolve("library-v1.json"), "{}".getBytes(UTF_8),
        loaded.revision, r => answer = r)
      h.nextDelivery()()
      assert(answer != null && answer.isLeft)
      assertEquals(Files.readAllBytes(library).toVector, original)
      assert(!h.session.isMutationPending)
    finally
      h.session.close()
      Files.deleteIfExists(alias)
  }

  test("typed I/O and restore are rejected through their own callbacks while a save is pending") {
    val directory = Files.createTempDirectory("presets-typed-busy-")
    val h = new Harness(directory)
    val source = directory.resolve("source.json")
    val tracked = directory.resolve("tracked.txt")
    val additional = directory.resolve("additional.json")
    val exported = directory.resolve("must-not-export.json")
    Files.writeString(source, "{}")
    Files.writeString(tracked, "txt2img_steps\n")
    Files.writeString(additional, """{"additionalComponents":{}}""")
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val accepted = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Accepted")))
      h.session.save(initial.revision, accepted)
      val saveDelivery = h.nextDelivery()

      var sourceResult: Either[PresetError, SourceDocument] = null
      var idsResult: Either[PresetError, Vector[String]] = null
      var mappingsResult: Either[PresetError, Vector[(String, Option[String])]] = null
      var exportResult: Either[PresetError, PresetRevision] = null
      h.session.readSource(source, "config-presets", Some(PresetMode.Txt2Img), r => sourceResult = r)
      h.session.readTrackedIds(tracked, r => idsResult = r)
      h.session.readAdditionalComponents(additional, r => mappingsResult = r)
      h.session.exportFile(exported, "{}".getBytes(UTF_8), PresetRevision(None), r => exportResult = r)
      h.session.restoreBackup(initial.revision)
      val rejectedDeliveries = Vector.fill(5)(h.nextDelivery())
      assert(sourceResult == null && idsResult == null && mappingsResult == null && exportResult == null)
      assert(h.results.isEmpty)
      rejectedDeliveries.foreach(deliver => deliver())
      assert(sourceResult != null && sourceResult.isLeft)
      assert(idsResult != null && idsResult.isLeft)
      assert(mappingsResult != null && mappingsResult.isLeft)
      assert(exportResult != null && exportResult.isLeft)
      val rejectedRestore = h.results.poll()
      assert(rejectedRestore != null && rejectedRestore.isLeft)
      assert(h.session.isMutationPending, "rejected typed I/O cleared the active save")
      assert(!Files.exists(exported))

      saveDelivery()
      val saved = h.results.poll()
      assert(saved != null && saved.isRight)
      assertEquals(saved.toOption.get.library.collections(2).presets.map(_.name), Vector("Accepted"))
      assert(!h.session.isMutationPending)
    finally h.session.close()
  }

  test("oversized custom files report a bounded read error on the owner thread") {
    val directory = Files.createTempDirectory("presets-custom-limit-")
    val h = new Harness(directory)
    val tracked = directory.resolve("oversized.txt")
    val additional = directory.resolve("oversized.json")
    Files.write(tracked, Array.fill[Byte](PresetLimits().maxCustomFileBytes + 1)('#'.toByte))
    Files.write(additional, Array.fill[Byte](PresetLimits().maxCustomFileBytes + 1)(' '.toByte))
    val owner = Thread.currentThread()
    try
      var ids: Either[PresetError, Vector[String]] = null
      h.session.readTrackedIds(tracked, r => {
        assertEquals(Thread.currentThread(), owner)
        ids = r
      })
      val deliverIds = h.nextDelivery()
      assert(ids == null)
      deliverIds()
      assertEquals(ids.swap.toOption.get.issues.head.code, "limit")

      var mappings: Either[PresetError, Vector[(String, Option[String])]] = null
      h.session.readAdditionalComponents(additional, r => {
        assertEquals(Thread.currentThread(), owner)
        mappings = r
      })
      val deliverMappings = h.nextDelivery()
      assert(mappings == null)
      deliverMappings()
      assertEquals(mappings.swap.toOption.get.issues.head.code, "limit")
      assertEquals(Files.size(tracked), (PresetLimits().maxCustomFileBytes + 1).toLong)
      assertEquals(Files.size(additional), (PresetLimits().maxCustomFileBytes + 1).toLong)
    finally h.session.close()
  }

  test("restore backup is explicit and reports the recovered library after owner-thread delivery") {
    val directory = Files.createTempDirectory("presets-session-restore-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val changed = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Temporary")))
      h.session.save(initial.revision, changed)
      val saved = h.receive().toOption.get
      assertEquals(saved.library.collections(2).presets.map(_.name), Vector("Temporary"))
      h.session.restoreBackup(saved.revision)
      val delivery = h.nextDelivery()
      assert(h.results.isEmpty)
      assert(h.session.isMutationPending)
      delivery()
      val restored = h.results.poll()
      assert(restored != null && restored.isRight)
      assertEquals(restored.toOption.get.library, emptyLibrary)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.revision,
        restored.toOption.get.revision)
      assert(!h.session.isMutationPending)
    finally h.session.close()
  }

  test("explicit restore remains available after a corrupt reload and preserves rejected bytes") {
    val directory = Files.createTempDirectory("presets-corrupt-restore-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val initial = h.receive().toOption.get
      val changed = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Before corruption")))
      h.session.save(initial.revision, changed)
      assert(h.receive().isRight)
      val backing = directory.resolve("library-v1.json")
      val corrupt = "{broken".getBytes(UTF_8)
      Files.write(backing, corrupt)
      h.session.reload()
      assert(h.receive().isLeft)

      h.session.restoreBackup(revisionOf(corrupt))
      val delivery = h.nextDelivery()
      assert(h.results.isEmpty)
      delivery()
      val recovered = h.results.poll()
      assert(recovered != null && recovered.isRight)
      assertEquals(recovered.toOption.get.library, emptyLibrary)
      val rejected = Files.list(directory)
      try
        val preserved = rejected.filter(p => p.getFileName.toString.contains(".rejected-"))
          .toArray.map(_.asInstanceOf[java.nio.file.Path]).toVector
        assertEquals(preserved.size, 1)
        assertEquals(Files.readAllBytes(preserved.head).toVector, corrupt.toVector)
      finally rejected.close()
    finally h.session.close()
  }

  test("close after a typed I/O completion was queued suppresses its callback at delivery") {
    val directory = Files.createTempDirectory("presets-typed-close-")
    val h = new Harness(directory)
    val tracked = directory.resolve("tracked.txt")
    Files.writeString(tracked, "txt2img_steps\n")
    var callbackCount = 0
    h.session.readTrackedIds(tracked, _ => callbackCount += 1)
    val delivery = h.nextDelivery()
    h.session.close()
    delivery()
    assertEquals(callbackCount, 0)
    assert(h.results.isEmpty)
  }

  test("a typed Busy callback is suppressed if its active generation has been replaced") {
    val directory = Files.createTempDirectory("presets-typed-stale-")
    val h = new Harness(directory)
    val tracked = directory.resolve("tracked.txt")
    Files.writeString(tracked, "txt2img_steps\n")
    try
      h.session.reload()
      val active = h.nextDelivery()
      var staleCallbackCount = 0
      h.session.readTrackedIds(tracked, _ => staleCallbackCount += 1)
      val oldBusy = h.nextDelivery()
      active()
      val firstResult = h.results.poll()
      assert(firstResult != null && firstResult.isRight)
      h.session.reload()
      val next = h.nextDelivery()
      oldBusy()
      assertEquals(staleCallbackCount, 0)
      assert(h.session.isMutationPending)
      next()
      val nextResult = h.results.poll()
      assert(nextResult != null && nextResult.isRight)
    finally h.session.close()
  }

  test("explicit invalid CLI directory remains explicit rather than silently falling back") {
    val invalid = Files.createTempFile("presets-cli-invalid-", ".txt")
    try
      val parsed = CliOptions.parse(Seq("--configpresets-dir", invalid.toString))
      assertEquals(parsed.configPresetsDir, Some(invalid.toString))
      val chosen = java.nio.file.Path.of(parsed.configPresetsDir.get)
      val h = new Harness(chosen)
      try
        assertEquals(h.session.directoryPath, chosen.toAbsolutePath.normalize())
        h.session.reload()
        assert(h.receive().isLeft)
        assert(Files.isRegularFile(invalid))
      finally h.session.close()
    finally Files.deleteIfExists(invalid)
  }

  test("one collection may delete its final preset without resetting another collection") {
    val directory = Files.createTempDirectory("presets-last-delete-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val first = h.receive().toOption.get
      val initial = emptyLibrary.copy(collections = Vector(
        collection("config-txt2img", "same"), collection("config-img2img", "same"), collection("utilities", "same")))
      h.session.save(first.revision, initial)
      val saved = h.receive().toOption.get
      assertEquals(saved.library.collections.map(_.presets.size), Vector(1, 1, 1))
      val afterDelete = saved.library.copy(collections = saved.library.collections.updated(0,
        saved.library.collections(0).copy(presets = Vector.empty)))
      h.session.save(saved.revision, afterDelete)
      val deleted = h.receive().toOption.get
      assertEquals(deleted.library.collections.map(_.presets.size), Vector(0, 1, 1))
      assertEquals(deleted.library.collections(1).presets.head.name, "same")
      assertEquals(deleted.library.collections(2).presets.head.name, "same")
    finally h.session.close()
  }

  test("overlapping whole-library save cannot erase the first accepted save") {
    val directory = Files.createTempDirectory("presets-overlap-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val first = h.receive().toOption.get
      val alice = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2, collection("utilities", "Alice")))
      val staleBob = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2, collection("utilities", "Bob")))
      h.session.save(first.revision, alice)
      assert(h.session.isMutationPending, "pending gate must close before scheduling disk I/O")
      val active = h.nextDelivery()
      // This candidate was captured from the same old revision. It must be rejected as Busy
      // or Conflict, never queued as an overwrite behind the first operation.
      h.session.save(first.revision, staleBob)
      assert(h.session.isMutationPending, "Busy rejection must not reopen mutation controls")
      val busy = h.nextDelivery()
      busy()
      val rejected = h.results.poll()
      assert(rejected != null && rejected.isLeft)
      assert(rejected.swap.toOption.get.issues.exists(_.code.equalsIgnoreCase("busy")))
      assert(h.session.isMutationPending)
      active()
      val saved = h.results.poll()
      assert(saved != null && saved.isRight)
      assert(!h.session.isMutationPending, "success delivery must release pending controls")
      val disk = new PresetStore(directory, PresetLimits()).load().toOption.get
      assertEquals(disk.library.collections.find(_.id == "utilities").get.presets.map(_.name), Vector("Alice"))
    finally h.session.close()
  }

  test("caller revision is checked even after a prior completion installed a newer revision") {
    val directory = Files.createTempDirectory("presets-caller-revision-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val first = h.receive().toOption.get
      h.session.save(first.revision, emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "Fresh"))))
      val saved = h.receive().toOption.get
      assert(saved.revision != first.revision)
      h.session.save(first.revision, emptyLibrary)
      assert(h.receive().isLeft)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.revision, saved.revision)
    finally h.session.close()
  }

  test("close discards a queued completion callback before delivery") {
    val directory = Files.createTempDirectory("presets-close-")
    val h = new Harness(directory)
    h.session.reload()
    val first = h.receive().toOption.get
    h.session.save(first.revision, emptyLibrary)
    h.waitForDelivery()
    h.session.close()
    h.drain()
    assertEquals(h.results.size(), 0, "late callback reached the disposed owner")
  }

  test("external edit rejects a stale caller revision and a corrupt refresh blocks saves") {
    val directory = Files.createTempDirectory("presets-external-edit-")
    val h = new Harness(directory)
    try
      h.session.reload()
      val first = h.receive().toOption.get
      h.session.save(first.revision, emptyLibrary)
      val saved = h.receive().toOption.get
      val backing = directory.resolve("library-v1.json")
      val external = emptyLibrary.copy(collections = emptyLibrary.collections.updated(2,
        collection("utilities", "External")))
      val bytes = PresetJson.writeNative(external, PresetLimits()).toOption.get
      Files.write(backing, bytes)
      h.session.save(saved.revision, emptyLibrary)
      assert(h.receive().isLeft)
      assertEquals(new PresetStore(directory, PresetLimits()).load().toOption.get.library, external)
      Files.writeString(backing, "{broken")
      h.session.reload()
      assert(h.receive().isLeft)
      h.session.save(saved.revision, emptyLibrary)
      assert(h.receive().isLeft)
      assertEquals(Files.readString(backing), "{broken")
    finally h.session.close()
  }
