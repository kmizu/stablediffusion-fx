package sdfx.engine

import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.modelcards.*
import sdfx.core.protocol.{FileEntry, ModelEntry}
import java.nio.file.{Files, Path}
import java.util.concurrent.Executor
import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration.*

/** Executor injection drives ownership deterministically without sleeps or an engine. */
class ModelCardCatalogSuite extends munit.FunSuite:
  private final class ManualExecutor extends Executor:
    val tasks = mutable.Queue.empty[Runnable]
    def execute(task: Runnable): Unit = tasks.enqueue(task)
    def runNext(): Unit = tasks.dequeue().run()
    def drain(): Unit = while tasks.nonEmpty do runNext()

  private def read[A](lease: CardScanLease)(f: CardScanView => A): A =
    lease.withView(f).fold(fail(_), identity)

  private def empty(revision: Long, status: CardScanStatus = CardScanStatus.Complete) =
    CardScanResult(revision, Vector.empty, status, Vector.empty, 0L, 0L)

  private def awaitBlocked(release: java.util.concurrent.CountDownLatch): Unit =
    val deadline = System.nanoTime() + 5000000000L
    var done = false
    while !done && System.nanoTime() < deadline do
      try done = release.await(deadline - System.nanoTime(), java.util.concurrent.TimeUnit.NANOSECONDS)
      catch case _: InterruptedException => () // deliberately simulate an uninterruptible filesystem call
    if !done then throw new AssertionError("test did not release blocked scan")

  test("latest revision wins with one active scan and one replacement") {
    val executor = new ManualExecutor
    val calls = mutable.ArrayBuffer.empty[Long]
    val published = mutable.ArrayBuffer.empty[Long]
    var active = 0
    var peak = 0
    var catalog: ModelCardCatalog = null
    val scan: (CardScanRequest, () => Boolean) => CardScanResult = (request, cancelled) =>
      active += 1
      peak = peak.max(active)
      calls += request.revision
      if calls.size == 1 then
        catalog.refresh()
        catalog.refresh() // replaces the intermediate request
        assert(cancelled())
      active -= 1
      empty(request.revision)
    catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = scan, worker = executor)
    try
      catalog.addListener(r => { published += r.revision; r.close() }).fold(fail(_), identity)
      catalog.refresh()
      executor.drain()
      assertEquals(peak, 1)
      assertEquals(calls.size, 2)
      assert(calls.last > calls.head)
      assertEquals(published.toVector, Vector(calls.last))
      assertEquals(catalog.snapshotInfo.map(_.revision), Some(calls.last))
    finally catalog.close()
  }

  test("close cancels active and queued work without waiting or publishing stale results") {
    val executor = new ManualExecutor
    var closedInside = false
    var published = 0
    var catalog: ModelCardCatalog = null
    val scan: (CardScanRequest, () => Boolean) => CardScanResult = (request, cancelled) =>
      catalog.refresh()
      catalog.close() // returns even while this scan is still executing
      closedInside = true
      assert(cancelled())
      empty(request.revision)
    catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = scan, worker = executor)
    catalog.addListener(lease => { published += 1; lease.close() }).fold(fail(_), identity)
    catalog.refresh()
    executor.drain()
    assert(closedInside)
    assertEquals(published, 0)
  }

  test("cached engine metadata enriches a locally present model without starting or refreshing the engine") {
    val root = Files.createTempDirectory("sdfx-model-card-catalog-")
    try
      val model = Files.write(root.resolve("example.safetensors"), Array[Byte](1, 2, 3))
      val missing = root.resolve("gone.safetensors")
      val future = Files.write(root.resolve("future.patch"), Array[Byte](4, 5))
      val futureDir = Files.createDirectory(root.resolve("future-directory"))
      Files.write(futureDir.resolve("model_index.json"), Array[Byte](1))
      val loraDir = Files.createDirectory(root.resolve("lora"))
      val lora = Files.write(loraDir.resolve("style.safetensors"), Array[Byte](8))
      val embeddingDir = Files.createDirectory(root.resolve("embeddings"))
      val upscalerDir = Files.createDirectory(root.resolve("upscalers"))
      val embedding = Files.write(embeddingDir.resolve("token.pt"), Array[Byte](6))
      val upscaler = Files.write(upscalerDir.resolve("scale.pth"), Array[Byte](7))
      val settings = Settings(modelDirs = ModelDirs(checkpoints = Seq(root.toString), vae = Nil, lora = Seq(loraDir.toString),
        controlnet = Nil, embeddings = Seq(embeddingDir.toString), upscalers = Seq(upscalerDir.toString),
        faceRestore = Nil, svd = Nil, zero123 = Nil, photomaker = Nil))
      val cached = CatalogSnapshot(checkpoints = Seq(ModelEntry("Engine/Example [literal]", model.toString, "sdxl"),
        ModelEntry("gone", missing.toString, "sd15")),
        loras = Seq(FileEntry("LoRA/Literal 名", lora.toString)),
        embeddings = Seq(FileEntry("Embedding Literal", embedding.toString)),
        upscalers = Seq(FileEntry("scale", upscaler.toString)),
        extraFiles = Map("future_kind" -> Seq(FileEntry("future", future.toString),
          FileEntry("future-directory", futureDir.toString), FileEntry("gone", missing.toString))))
      val executor = new ManualExecutor
      var reads = 0
      val catalog = new ModelCardCatalog(AppHome(root), () => settings, () => { reads += 1; cached }, worker = executor)
      try
        val result = catalog.refresh()
        executor.drain()
        val lease = Await.result(result, 1.second)
        try read(lease) { snapshot =>
          val record = snapshot.records.find(_.id.modelPath == model.toAbsolutePath.normalize.toString).get
          assertEquals(record.id.kind, "checkpoint")
          assertEquals(record.name, "Engine/Example [literal]")
          assertEquals(record.family, Some("sdxl"))
          val kindsByPath = snapshot.records.map(r => r.id.modelPath -> r.id.kind).toMap
          assertEquals(kindsByPath.get(embedding.toAbsolutePath.normalize.toString), Some("embedding"))
          assertEquals(snapshot.records.find(_.id.modelPath == embedding.toAbsolutePath.normalize.toString).map(_.name), Some("Embedding Literal"))
          assertEquals(snapshot.records.find(r => r.id.kind == "lora" && r.id.modelPath == lora.toAbsolutePath.normalize.toString).map(_.name), Some("LoRA/Literal 名"))
          assertEquals(kindsByPath.get(upscaler.toAbsolutePath.normalize.toString), Some("upscaler"))
          assert(!snapshot.records.exists(_.id.modelPath == missing.toAbsolutePath.normalize.toString))
          assert(snapshot.records.exists(r => r.id.kind == "future_kind" && r.id.modelPath == future.toAbsolutePath.normalize.toString))
          assert(snapshot.records.exists(r => r.id.kind == "future_kind" && r.id.modelPath == futureDir.toAbsolutePath.normalize.toString))
          assert(reads > 0)
        }
        finally lease.close()
      finally catalog.close()
    finally
      val stream = Files.walk(root)
      try stream.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally stream.close()
  }

  test("a blocked old worker is cancelled and only the latest replacement is published") {
    val root = Files.createTempDirectory("sdfx-model-card-blocked-")
    val worker = java.util.concurrent.Executors.newSingleThreadExecutor()
    val entered = new java.util.concurrent.CountDownLatch(1)
    val release = new java.util.concurrent.CountDownLatch(1)
    val published = new java.util.concurrent.CountDownLatch(1)
    val calls = new java.util.concurrent.ConcurrentLinkedQueue[Long]()
    val notices = new java.util.concurrent.ConcurrentLinkedQueue[Long]()
    val oldSawCancellation = new java.util.concurrent.atomic.AtomicBoolean(false)
    val scan: (CardScanRequest, () => Boolean) => CardScanResult = (request, cancelled) =>
      calls.add(request.revision)
      if calls.size() == 1 then
        entered.countDown()
        awaitBlocked(release)
        oldSawCancellation.set(cancelled())
      empty(request.revision)
    val catalog = new ModelCardCatalog(AppHome(root), () => Settings(), () => CatalogSnapshot(),
      scan = scan, worker = worker)
    try
      catalog.addListener(r => { notices.add(r.revision); r.close(); published.countDown() }).fold(fail(_), identity)
      catalog.refresh()
      val enteredInTime = entered.await(5, java.util.concurrent.TimeUnit.SECONDS)
      assert(enteredInTime, "worker did not reach the controlled boundary")
      catalog.refresh()
      val latest = catalog.refresh()
      assertEquals(calls.size(), 1) // the replacement cannot start while the only worker is blocked
      release.countDown()
      val publishedInTime = published.await(5, java.util.concurrent.TimeUnit.SECONDS)
      if !publishedInTime && latest.isCompleted then Await.result(latest, 1.second).close()
      assert(publishedInTime, "latest scan did not notify its listener")
      val current = Await.result(latest, 5.seconds)
      import scala.jdk.CollectionConverters.*
      assert(oldSawCancellation.get())
      assertEquals(calls.asScala.toVector.size, 2)
      assertEquals(notices.asScala.toVector, Vector(current.revision))
      assertEquals(catalog.snapshotInfo.map(_.revision), Some(current.revision))
      current.close()
    finally
      release.countDown()
      catalog.close()
      worker.shutdownNow()
      worker.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
      Files.deleteIfExists(root)
  }

  test("close returns while the worker is blocked and prevents stale publication") {
    val root = Files.createTempDirectory("sdfx-model-card-close-")
    val worker = java.util.concurrent.Executors.newSingleThreadExecutor()
    val closer = java.util.concurrent.Executors.newSingleThreadExecutor()
    val entered = new java.util.concurrent.CountDownLatch(1)
    val release = new java.util.concurrent.CountDownLatch(1)
    val exited = new java.util.concurrent.CountDownLatch(1)
    val sawCancellation = new java.util.concurrent.atomic.AtomicBoolean(false)
    val notices = new java.util.concurrent.atomic.AtomicInteger(0)
    val scan: (CardScanRequest, () => Boolean) => CardScanResult = (request, cancelled) =>
      entered.countDown()
      try
        awaitBlocked(release)
        sawCancellation.set(cancelled())
        empty(request.revision)
      finally exited.countDown()
    val catalog = new ModelCardCatalog(AppHome(root), () => Settings(), () => CatalogSnapshot(),
      scan = scan, worker = worker)
    try
      catalog.addListener(lease => { notices.incrementAndGet(); lease.close() }).fold(fail(_), identity)
      catalog.refresh()
      val enteredInTime = entered.await(5, java.util.concurrent.TimeUnit.SECONDS)
      assert(enteredInTime, "worker did not reach the controlled boundary")
      catalog.refresh()
      val close = closer.submit(new Runnable { def run(): Unit = catalog.close() })
      close.get(2, java.util.concurrent.TimeUnit.SECONDS) // old scan remains behind release
      assertEquals(exited.getCount, 1L)
      release.countDown()
      val exitedInTime = exited.await(5, java.util.concurrent.TimeUnit.SECONDS)
      assert(exitedInTime, "worker did not exit after release")
      worker.shutdown()
      val workerTerminated = worker.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
      assert(workerTerminated, "worker did not terminate")
      assert(sawCancellation.get())
      assertEquals(notices.get(), 0)
    finally
      release.countDown()
      catalog.close()
      worker.shutdownNow()
      closer.shutdownNow()
      worker.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
      closer.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
      Files.deleteIfExists(root)
  }


  test("snapshot graph remains charged while old Future, retained lease, or derived view owns it") {
    val root = Files.createTempDirectory("sdfx-model-card-leases-")
    val path = Files.write(root.resolve("model.safetensors"), Array[Byte](1))
    val worker = new ManualExecutor
    val settings = Settings(modelDirs = ModelDirs(checkpoints = Seq(root.toString), vae = Nil, lora = Nil,
      controlnet = Nil, embeddings = Nil, upscalers = Nil, faceRestore = Nil, svd = Nil, zero123 = Nil, photomaker = Nil))
    val live = mutable.Map.empty[Long, (String, Long, Long)]
    val released = mutable.Set.empty[Long]
    var invalid = false
    val observer = new CardScanObserver:
      def reserve(id: Long, category: String, chars: Long, bytes: Long): Unit =
        if live.contains(id) || released.contains(id) then invalid = true
        live(id) = (category, chars, bytes)
      def materialize(id: Long): Unit = if !live.contains(id) then invalid = true
      def release(id: Long): Unit =
        if live.remove(id).isEmpty then invalid = true
        released += id
      def entryAdmitted(dir: Path, chars: Long): Unit = ()
      def directorySorted(dir: Path): Unit = ()
      def weightOpened(path: Path): Unit = invalid = true
      def visited(path: Path): Unit = ()
      def pendingDirectories(count: Int): Unit = ()
      def cancellationCheck(): Unit = ()
    val ledger = new CardScanReservations(2000000L, 96L * 1024L * 1024L, observer)
    val hooks = CardScanHooks(reservations = Some(ledger), observer = observer)
    val catalog = new ModelCardCatalog(AppHome(root), () => settings, () => CatalogSnapshot(), worker = worker, hooks = hooks)
    try
      val firstFuture = catalog.refresh()
      worker.drain()
      val first = Await.result(firstFuture, 1.second)
      assertEquals(read(first)(_.records.size), 1)
      val oldGraph = live.collect { case (id, ("record-and-index", _, _)) => id }.toSet
      assert(oldGraph.nonEmpty)
      val retained = first.retain().fold(fail(_), identity)
      val derived = first.reserveDerived(256L, 1024L).fold(fail(_), identity)
      val secondFuture = catalog.refresh()
      worker.drain()
      val second = Await.result(secondFuture, 1.second)
      assert(oldGraph.forall(live.contains), "replacement dropped a consumer-owned graph")
      catalog.close()
      assertEquals(catalog.snapshotInfo, None)
      assert(oldGraph.forall(live.contains), "catalog close dropped a consumer-owned graph")
      assertEquals(read(retained)(_.records.size), 1)
      first.close()
      assert(first.withView(_.records.size).isLeft)
      assert(!first.hasGraphForTest, "a closed lease retained a graph through its completed Future")
      assert(oldGraph.forall(live.contains))
      retained.close()
      assert(oldGraph.forall(live.contains), "derived view must retain its source graph")
      derived.close()
      assert(oldGraph.forall(released.contains))
      second.close()
      assert(!invalid, "reservation must be admitted and released exactly once")
      assert(live.isEmpty, live.toString)
      assert(firstFuture.value.nonEmpty && secondFuture.value.nonEmpty) // still retain the completed Futures
    finally
      catalog.close()
      Files.deleteIfExists(path)
      Files.deleteIfExists(root)
  }

  test("lease, derived-view, and listener quotas refuse visibly and recover after release") {
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    val held = mutable.ArrayBuffer.empty[CardScanLease]
    val removers = mutable.ArrayBuffer.empty[() => Unit]
    try
      val firstFuture = catalog.refresh()
      worker.drain()
      val first = Await.result(firstFuture, 1.second)
      held += first
      (0 until 127).foreach(_ => held += first.retain().fold(fail(_), identity))
      assert(first.retain().isLeft)
      assert(catalog.acquireSnapshot().isLeft)
      held.remove(1).close()
      held += catalog.acquireSnapshot().fold(fail(_), identity)
      val tooLarge = first.reserveDerived(2000001L, 96L * 1024L * 1024L)
      assert(tooLarge.isLeft)
      (0 until 64).foreach(_ => removers += catalog.addListener(lease => lease.close()).fold(fail(_), identity))
      assert(catalog.addListener(lease => lease.close()).isLeft)
      removers.remove(0)()
      removers += catalog.addListener(lease => lease.close()).fold(fail(_), identity)
    finally
      removers.foreach(_())
      held.foreach(_.close())
      catalog.close()
  }

  test("listener A closes catalog and prevents B from receiving the captured delivery") {
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    var callsA = 0
    var callsB = 0
    catalog.addListener(lease => { callsA += 1; catalog.close(); lease.close() }).fold(fail(_), identity)
    catalog.addListener(lease => { callsB += 1; lease.close() }).fold(fail(_), identity)
    val future = catalog.refresh()
    worker.drain()
    Await.result(future, 1.second).close()
    assertEquals(callsA, 1)
    assertEquals(callsB, 0)
    assertEquals(catalog.snapshotInfo, None)
  }

  test("refresh during listener delivery suppresses the remaining obsolete deliveries") {
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    val received = mutable.ArrayBuffer.empty[Long]
    var next: Option[scala.concurrent.Future[CardScanLease]] = None
    catalog.addListener(lease => {
      if next.isEmpty then next = Some(catalog.refresh())
      lease.close()
    }).fold(fail(_), identity)
    catalog.addListener(lease => { received += lease.revision; lease.close() }).fold(fail(_), identity)
    try
      val old = catalog.refresh()
      worker.drain()
      val oldLease = Await.result(old, 1.second)
      val newLease = Await.result(next.get, 1.second)
      assert(newLease.revision > oldLease.revision)
      assertEquals(received.toVector, Vector(newLease.revision))
      oldLease.close(); newLease.close()
    finally catalog.close()
  }

  test("closing catalog does not wait on an admitted blocked listener") {
    val worker = java.util.concurrent.Executors.newSingleThreadExecutor()
    val closer = java.util.concurrent.Executors.newSingleThreadExecutor()
    val entered = new java.util.concurrent.CountDownLatch(1)
    val release = new java.util.concurrent.CountDownLatch(1)
    val exited = new java.util.concurrent.CountDownLatch(1)
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    catalog.addListener(lease => {
      entered.countDown()
      try awaitBlocked(release) finally { lease.close(); exited.countDown() }
    }).fold(fail(_), identity)
    try
      val result = catalog.refresh()
      val enteredInTime = entered.await(5, java.util.concurrent.TimeUnit.SECONDS)
      assert(enteredInTime, "worker did not reach the controlled boundary")
      closer.submit(new Runnable { def run(): Unit = catalog.close() }).get(2, java.util.concurrent.TimeUnit.SECONDS)
      assertEquals(exited.getCount, 1L)
      release.countDown()
      val exitedInTime = exited.await(5, java.util.concurrent.TimeUnit.SECONDS)
      assert(exitedInTime, "worker did not exit after release")
      Await.result(result, 1.second).close()
    finally
      release.countDown(); catalog.close(); worker.shutdownNow(); closer.shutdownNow()
      worker.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
      closer.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
  }

  test("queued UI handoff rechecks revision and closes stale leases before replacing rows") {
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    val ui = mutable.Queue.empty[() => Unit]
    val deliveries = mutable.ArrayBuffer.empty[CardScanLease]
    val painted = mutable.ArrayBuffer.empty[Long]
    var current: Option[CardScanLease] = None
    catalog.addListener(lease => {
      deliveries += lease
      ui.enqueue(() => {
        if catalog.snapshotInfo.exists(_.revision == lease.revision) then
          // Real UI clears rows/indexes first, then closes the old owner.
          current.foreach(_.close())
          current = Some(lease)
          painted += lease.revision
        else lease.close()
      })
    }).fold(fail(_), identity)
    try
      val firstFuture = catalog.refresh(); worker.drain()
      val secondFuture = catalog.refresh(); worker.drain()
      val first = Await.result(firstFuture, 1.second)
      val second = Await.result(secondFuture, 1.second)
      assertEquals(ui.size, 2)
      while ui.nonEmpty do ui.dequeue()()
      assertEquals(painted.toVector, Vector(second.revision))
      assert(deliveries.head.withView(_.records.size).isLeft)
      val thirdFuture = catalog.refresh(); worker.drain()
      val third = Await.result(thirdFuture, 1.second)
      current.foreach(_.close()); current = None // UI clears rows before releasing its current view
      catalog.close()
      while ui.nonEmpty do ui.dequeue()()
      assertEquals(painted.toVector, Vector(second.revision))
      assert(deliveries.last.withView(_.records.size).isLeft)
      first.close(); second.close(); third.close()
      assert(deliveries.forall(_.withView(_.records.size).isLeft))
    finally
      current.foreach(_.close())
      deliveries.foreach(_.close())
      catalog.close()
  }

  test("listener removed during an earlier callback is not delivered from a captured list") {
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    var removeB: () => Unit = () => ()
    var callsB = 0
    catalog.addListener(lease => { removeB(); lease.close() }).fold(fail(_), identity)
    removeB = catalog.addListener(lease => { callsB += 1; lease.close() }).fold(fail(_), identity)
    try
      val future = catalog.refresh(); worker.drain()
      Await.result(future, 1.second).close()
      assertEquals(callsB, 0)
    finally catalog.close()
  }

  test("conflicting cached names for one model do not invent a unique actionable name") {
    val root = Files.createTempDirectory("sdfx-ambiguous-model-name-")
    val model = Files.write(root.resolve("local-stem.safetensors"), Array[Byte](1))
    val settings = Settings(modelDirs = ModelDirs(checkpoints = Seq(root.toString), vae = Nil, lora = Nil,
      controlnet = Nil, embeddings = Nil, upscalers = Nil, faceRestore = Nil, svd = Nil, zero123 = Nil, photomaker = Nil))
    val cached = CatalogSnapshot(checkpoints = Seq(
      ModelEntry("First Alias", model.toString, "sd15"),
      ModelEntry("Second Alias", model.toString, "sd15")))
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(root), () => settings, () => cached, worker = worker)
    try
      val future = catalog.refresh(); worker.drain()
      val lease = Await.result(future, 1.second)
      try read(lease) { view =>
        val record = view.records.find(_.id.modelPath == model.toAbsolutePath.normalize.toString).get
        assertEquals(record.name, "local-stem")
      }
      finally lease.close()
    finally
      catalog.close()
      Files.deleteIfExists(model)
      Files.deleteIfExists(root)
  }

  test("blocked withView retains graph accounting when parent and catalog close concurrently") {
    val worker = new ManualExecutor
    val borrower = java.util.concurrent.Executors.newSingleThreadExecutor()
    val closer = java.util.concurrent.Executors.newSingleThreadExecutor()
    val entered = new java.util.concurrent.CountDownLatch(1)
    val release = new java.util.concurrent.CountDownLatch(1)
    val live = new java.util.concurrent.ConcurrentHashMap[Long, String]()
    val observer = new CardScanObserver:
      def reserve(id: Long, category: String, chars: Long, bytes: Long): Unit = { live.put(id, category); () }
      def materialize(id: Long): Unit =
        if !live.containsKey(id) then throw new AssertionError("unreserved materialization")
      def release(id: Long): Unit =
        if live.remove(id) == null then throw new AssertionError("duplicate or unknown release")
      def entryAdmitted(dir: Path, chars: Long): Unit = ()
      def directorySorted(dir: Path): Unit = ()
      def weightOpened(path: Path): Unit = fail(s"unexpected weight read: $path")
      def visited(path: Path): Unit = ()
      def pendingDirectories(count: Int): Unit = ()
      def cancellationCheck(): Unit = ()
    val ledger = new CardScanReservations(2000000L, 96L * 1024L * 1024L, observer)
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker,
      hooks = CardScanHooks(reservations = Some(ledger), observer = observer))
    try
      val future = catalog.refresh(); worker.drain()
      val lease = Await.result(future, 1.second)
      val graphIds = live.entrySet().toArray.toVector.collect {
        case entry: java.util.Map.Entry[?, ?] if entry.getValue == "catalog-graph" => entry.getKey.asInstanceOf[Long]
      }
      assert(graphIds.nonEmpty)
      val callback = borrower.submit(new java.util.concurrent.Callable[Either[String, Int]]:
        def call() = lease.withView { view =>
          entered.countDown()
          awaitBlocked(release)
          view.records.size
        })
      val enteredInTime = entered.await(5, java.util.concurrent.TimeUnit.SECONDS)
      if !enteredInTime && callback.isDone then callback.get(1, java.util.concurrent.TimeUnit.SECONDS)
      assert(enteredInTime, "borrow callback failed or did not reach the controlled boundary")
      closer.submit(new Runnable { def run(): Unit = { catalog.close(); lease.close() } }).get(2, java.util.concurrent.TimeUnit.SECONDS)
      assert(!lease.hasGraphForTest)
      assert(graphIds.forall(live.containsKey), "active callback lost its retained graph reservation")
      release.countDown()
      assertEquals(callback.get(5, java.util.concurrent.TimeUnit.SECONDS), Right(0))
      assert(graphIds.forall(id => !live.containsKey(id)))
      assert(live.isEmpty)
    finally
      release.countDown(); catalog.close(); borrower.shutdownNow(); closer.shutdownNow()
      borrower.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
      closer.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)
  }

  test("reentrant close in withView keeps borrowed data available until callback returns") {
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    try
      val future = catalog.refresh(); worker.drain()
      val lease = Await.result(future, 1.second)
      val result = lease.withView { view =>
        catalog.close()
        lease.close()
        assert(!lease.hasGraphForTest)
        view.records.size
      }
      assertEquals(result, Right(0))
      assert(lease.withView(_.records.size).isLeft)
    finally catalog.close()
  }

  test("public success callback runs without catalog monitor and cannot block close") {
    import java.util.concurrent.{Callable, CountDownLatch, Executors, TimeUnit, TimeoutException}
    import java.util.concurrent.atomic.AtomicBoolean
    import scala.concurrent.ExecutionContext
    val worker = new ManualExecutor
    val runner = Executors.newSingleThreadExecutor()
    val closer = Executors.newSingleThreadExecutor()
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val callbackHeldMonitor = new AtomicBoolean(true)
    val callbackExited = new CountDownLatch(1)
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    val result = catalog.refresh()
    result.onComplete { outcome =>
      callbackHeldMonitor.set(Thread.holdsLock(catalog))
      entered.countDown()
      try
        release.await(5, TimeUnit.SECONDS)
        outcome.toOption.foreach(_.close())
      finally callbackExited.countDown()
    }(using ExecutionContext.parasitic)
    try
      val running = runner.submit(new Runnable { def run(): Unit = worker.runNext() })
      val reached = entered.await(5, TimeUnit.SECONDS)
      val closeFuture = closer.submit(new Callable[Unit] { def call(): Unit = catalog.close() })
      val closeReturned = try { closeFuture.get(2, TimeUnit.SECONDS); true }
        catch case _: TimeoutException => false
      // Never assert while either controlled callback remains blocked.
      release.countDown()
      running.get(5, TimeUnit.SECONDS)
      closeFuture.get(5, TimeUnit.SECONDS)
      val exited = callbackExited.await(5, TimeUnit.SECONDS)
      assert(reached, "success callback did not begin")
      assert(closeReturned, "catalog.close waited for an unrelated Future callback")
      assert(exited, "success callback did not exit")
      assert(!callbackHeldMonitor.get(), "success callback ran under catalog monitor")
    finally
      release.countDown(); catalog.close(); runner.shutdownNow(); closer.shutdownNow()
      runner.awaitTermination(5, TimeUnit.SECONDS); closer.awaitTermination(5, TimeUnit.SECONDS)
  }

  test("public admission failure callback runs without catalog monitor") {
    import java.util.concurrent.atomic.AtomicBoolean
    import scala.concurrent.ExecutionContext
    val worker = new ManualExecutor
    val limited = new CardScanReservations(24L, 512L, CardScanObserver.none)
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker,
      hooks = CardScanHooks(reservations = Some(limited)))
    try
      val failureHeld = new AtomicBoolean(true)
      val failureSeen = new AtomicBoolean(false)
      val future = catalog.refresh()
      future.onComplete { result =>
        failureHeld.set(Thread.holdsLock(catalog))
        failureSeen.set(result.isFailure)
      }(using ExecutionContext.parasitic)
      worker.runNext()
      assert(failureSeen.get(), "graph-admission failure callback not observed")
      assert(!failureHeld.get(), "failure callback ran under catalog monitor")
    finally catalog.close()
  }

  test("public supersession callback runs without catalog monitor") {
    import java.util.concurrent.atomic.AtomicBoolean
    import scala.concurrent.ExecutionContext
    val queuedWorker = new ManualExecutor
    val queued = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = queuedWorker)
    try
      val supersededHeld = new AtomicBoolean(true)
      val supersededSeen = new AtomicBoolean(false)
      val first = queued.refresh()
      first.onComplete { result =>
        supersededHeld.set(Thread.holdsLock(queued))
        supersededSeen.set(result.isFailure)
      }(using ExecutionContext.parasitic)
      val second = queued.refresh()
      assert(supersededSeen.get(), "superseded callback not observed")
      assert(!supersededHeld.get(), "supersession callback ran under catalog monitor")
      queuedWorker.drain()
      Await.result(second, 1.second).close()
    finally queued.close()
  }

  test("public pending-close failure callback runs without catalog monitor") {
    import java.util.concurrent.atomic.AtomicBoolean
    import scala.concurrent.ExecutionContext
    val worker = new ManualExecutor
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = worker)
    try
      val held = new AtomicBoolean(true)
      val seen = new AtomicBoolean(false)
      val future = catalog.refresh()
      future.onComplete { result =>
        held.set(Thread.holdsLock(catalog))
        seen.set(result.isFailure)
      }(using ExecutionContext.parasitic)
      catalog.close()
      assert(seen.get(), "close failure callback not observed")
      assert(!held.get(), "close callback ran under catalog monitor")
      worker.drain()
    finally catalog.close()
  }

  test("active superseded scan failure callback runs without catalog monitor") {
    import java.util.concurrent.atomic.AtomicBoolean
    import scala.concurrent.ExecutionContext
    val worker = new ManualExecutor
    var catalog: ModelCardCatalog = null
    var replacement: scala.concurrent.Future[CardScanLease] = null
    val held = new AtomicBoolean(true)
    val seen = new AtomicBoolean(false)
    catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => {
        if replacement == null then replacement = catalog.refresh()
        empty(request.revision)
      }, worker = worker)
    try
      val first = catalog.refresh()
      first.onComplete { result =>
        held.set(Thread.holdsLock(catalog)); seen.set(result.isFailure)
      }(using ExecutionContext.parasitic)
      worker.drain()
      assert(seen.get(), "active supersession callback not observed")
      assert(!held.get(), "active supersession callback ran under catalog monitor")
      Await.result(replacement, 1.second).close()
    finally catalog.close()
  }

  test("inline scan executor invokes catalog listeners outside refresh monitor") {
    import java.util.concurrent.atomic.AtomicBoolean
    val held = new AtomicBoolean(true)
    val seen = new AtomicBoolean(false)
    val direct = new Executor { def execute(r: Runnable): Unit = r.run() }
    val catalog = new ModelCardCatalog(AppHome(Path.of(".")), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => empty(request.revision), worker = direct)
    catalog.addListener(lease => {
      held.set(Thread.holdsLock(catalog)); seen.set(true); lease.close()
    }).fold(fail(_), identity)
    try
      val result = catalog.refresh()
      Await.result(result, 1.second).close()
      assert(seen.get(), "inline listener did not run")
      assert(!held.get(), "inline executor left refresh catalog monitor held")
    finally catalog.close()
  }
