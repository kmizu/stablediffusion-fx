package sdfx.ui

import munit.FunSuite
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.modelcards.{CardId, CardPreviewMode, CardScanStatus, ModelCardRecord}
import sdfx.engine.AppServices
import sdfx.ui.view.{BrowserOwnedView, BrowserViewState, ModelBrowserSettings, ModelBrowserState}
import sdfx.ui.vm.SettingsMerge

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.collection.mutable

/** Headless presenter contracts. Native list virtualization, focus and layout are Task7 gates. */
class ModelBrowserStateSuite extends FunSuite:
  private def record(kind: String, name: String, path: String): ModelCardRecord =
    ModelCardRecord(CardId(kind, Path.of(path).toAbsolutePath.normalize.toString), name, None, None, None)

  private def fixture(body: (AppHome, AppServices) => Unit): Unit =
    val root = Files.createTempDirectory("browser-state")
    val home = AppHome(root)
    val svc = new AppServices(home, Settings(), dryRun = true, builtins = Nil,
      launchOverride = Some(() => throw new IllegalStateException("unexpected engine start")),
      installShutdownHook = false)
    try body(home, svc)
    finally
      svc.shutdown()
      val walk = Files.walk(root)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally walk.close()

  test("filter uses all whitespace terms, kind and stable identity without mutating source") {
    val rows = Vector(record("checkpoint", "Blue Sky", "models/blue sky.safetensors"),
      record("lora", "Blue Line", "models/line.safetensors"),
      record("checkpoint", "Red Sky", "models/red.safetensors"))
    val original = rows.map(_.id)
    assertEquals(ModelBrowserState.filter(rows, "checkpoint", " SKY   blue ").map(_.id), Vector(rows.head.id))
    assertEquals(ModelBrowserState.filter(rows, "lora", "line").map(_.id), Vector(rows(1).id))
    assertEquals(ModelBrowserState.filter(rows, "all", "models sky").map(_.id), Vector(rows.head.id, rows(2).id))
    assertEquals(rows.map(_.id), original)
  }

  test("menu omits current mode and saved default is explicitly applied") {
    fixture { (_, svc) =>
      val a = new ModelBrowserSettings(svc)
      val b = new ModelBrowserSettings(svc)
      assertEquals(a.menuModes.toSet, Set(CardPreviewMode.Hide, CardPreviewMode.Show))
      a.setTemporary(CardPreviewMode.Show)
      assertEquals(b.currentMode, CardPreviewMode.Show)
      assertEquals(b.menuModes.toSet, Set(CardPreviewMode.Blur, CardPreviewMode.Hide))
      assert(a.saveDefault("Hide").isRight)
      assertEquals(b.currentMode, CardPreviewMode.Show)
      val reopened = new ModelBrowserSettings(svc)
      assertEquals(reopened.currentMode, CardPreviewMode.Show)
      reopened.useSavedDefaultNow()
      assertEquals(a.currentMode, CardPreviewMode.Hide)
    }
  }

  test("dialog merge copies edited preview defaults and folders while retaining concurrent state") {
    val current = Settings(lastModel = Some("new"), pluginValues = Map("p" -> Map("k" -> "v")),
      modelPreviewOverrides = Map("new-key" -> Path.of("new.png").toAbsolutePath.normalize.toString))
    val edited = Settings(nsfwCardBlurDefault = "Show", modelPreviewFolders = Map("hypernetwork" -> Seq("inspect-only")),
      modelPreviewOverrides = Map("stale" -> Path.of("stale.png").toAbsolutePath.normalize.toString))
    val merged = SettingsMerge(current, edited)
    assertEquals(merged.nsfwCardBlurDefault, "Show")
    assertEquals(merged.modelPreviewFolders, edited.modelPreviewFolders)
    assertEquals(merged.modelPreviewOverrides, current.modelPreviewOverrides)
    assertEquals(merged.lastModel, current.lastModel)
    assertEquals(merged.pluginValues, current.pluginValues)
    assertEquals(merged.modelDirs, edited.modelDirs)
  }

  test("association updates one key against current settings and failed write is visible") {
    fixture { (_, svc) =>
      val ui = new ModelBrowserSettings(svc)
      val id = record("lora", "one", "models/one.safetensors").id
      svc.updateSettings(_.copy(lastModel = Some("later"), pluginValues = Map("p" -> Map("x" -> "y")),
        modelPreviewOverrides = Map("other" -> Path.of("other.png").toAbsolutePath.normalize.toString)))
      assert(ui.saveAssociation(id, Path.of("selected.png").toAbsolutePath.normalize.toString).isRight)
      assertEquals(svc.settings.modelPreviewOverrides, Map("other" -> Path.of("other.png").toAbsolutePath.normalize.toString, id.storageKey -> Path.of("selected.png").toAbsolutePath.normalize.toString))
      assert(ui.removeAssociation(id).isRight)
      assertEquals(svc.settings.modelPreviewOverrides, Map("other" -> Path.of("other.png").toAbsolutePath.normalize.toString))
      assertEquals(svc.settings.lastModel, Some("later"))
      assertEquals(svc.settings.pluginValues, Map("p" -> Map("x" -> "y")))
    }
  }

  test("recovery hold rejects ordinary saves and only successful explicit recovery applies effects") {
    val root = Files.createTempDirectory("browser-recovery")
    val home = AppHome(root)
    Files.createDirectories(home.settingsFile.getParent)
    val original = "{bad".getBytes(StandardCharsets.UTF_8)
    Files.write(home.settingsFile, original)
    val loaded = Settings.loadResult(home.settingsFile)
    var failWrite = true
    var writes = 0
    var effects = 0
    val svc = new AppServices(home, loaded.settings, true, Nil,
      launchOverride = Some(() => throw new IllegalStateException("unexpected engine start")),
      installShutdownHook = false, initialRecovery = loaded.recovery,
      settingsWriter = (path, value) => {
        writes += 1
        if failWrite then throw new java.io.IOException("injected")
        Settings.save(path, value)
      })
    try
      val ui = new ModelBrowserSettings(svc, _ => effects += 1)
      val id = record("lora", "one", "models/one.safetensors").id
      assert(ui.saveDefault("Show").isLeft)
      assert(ui.saveAssociation(id, Path.of("one.png").toAbsolutePath.normalize.toString).isLeft)
      assertEquals(writes, 0)
      assert(ui.saveRecoveredSettings().isLeft)
      assertEquals(writes, 1)
      assertEquals(effects, 0)
      assertEquals(svc.settingsRecovery, loaded.recovery)
      assertEquals(Files.readAllBytes(home.settingsFile).toSeq, original.toSeq)
      failWrite = false
      assert(ui.saveRecoveredSettings().isRight)
      assertEquals(writes, 2)
      assertEquals(effects, 1)
      assertEquals(svc.settingsRecovery, None)
      assertEquals(Settings.load(home.settingsFile), svc.settings)
    finally
      svc.shutdown()
      val walk = Files.walk(root)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally walk.close()
  }

  test("partial and failure states remain distinct and capacity gives a narrowing instruction") {
    val partial = ModelBrowserState.statusText(CardScanStatus.Partial, Vector("metadata limit"), 1)
    assert(partial.toLowerCase.contains("partial"))
    assert(partial.toLowerCase.contains("narrower"))
    assert(partial.contains("metadata limit"))
    val failure = ModelBrowserState.statusText(CardScanStatus.Failed, Vector("permission denied"), 1)
    assert(failure.toLowerCase.contains("fail"))
    assert(failure.contains("permission denied"))
    assertNotEquals(partial, failure)
  }

  test("latest handoff coalesces, releases stale and clears rows before ownership") {
    val tasks = mutable.Queue.empty[() => Unit]
    val events = mutable.ArrayBuffer.empty[String]
    final class Owned(val revision: Long) extends BrowserOwnedView:
      def close(): Unit = events += s"close-$revision"
    val state = new BrowserViewState(task => tasks.enqueue(task),
      view => events += s"install-${view.revision}", () => events += "clear")
    val old = new Owned(1); val next = new Owned(2); val latest = new Owned(3)
    state.offer(old)
    assertEquals(tasks.size, 1)
    tasks.dequeue()()
    state.offer(next); state.offer(latest)
    assertEquals(tasks.size, 1)
    assert(events.contains("close-2"))
    tasks.dequeue()()
    assert(events.indexOf("clear") < events.indexOf("close-1"))
    assert(events.contains("install-3"))
    state.dispose()
    assert(events.lastIndexOf("clear") < events.indexOf("close-3"))
    val stale = new Owned(4)
    state.offer(stale)
    assert(events.contains("close-4"))
    assertEquals(tasks.size, 0)
  }

  test("capacity callback retries only current visible refused cards up to available slots") {
    var available = 2
    val attempted = mutable.ArrayBuffer.empty[Int]
    ModelBrowserState.retryVisibleCapacity(Seq(1, 2, 3, 4), () => available, i => {
      attempted += i
      available -= 1
    })
    assertEquals(attempted.toSeq, Seq(1, 2))
    assertEquals(available, 0)
  }

  test("derived rows reserve before materialization and release on refusal or replacement") {
    val events = mutable.ArrayBuffer.empty[String]
    val source = Vector(record("checkpoint", "Blue", "models/blue.safetensors"))
    val denied = ModelBrowserState.derive(source, "all", "blue", (chars, bytes) => {
      events += s"reserve-$chars-$bytes"
      Left("metadata capacity")
    }, () => events += "materialize")
    assert(denied.isLeft)
    assertEquals(events.size, 1)
    assert(events.head.startsWith("reserve-"))
    val admitted = ModelBrowserState.derive(source, "all", "blue", (chars, bytes) => {
      events += s"reserve-$chars-$bytes"
      Right(() => events += "release")
    }, () => events += "materialize").fold(fail(_), identity)
    assertEquals(admitted.rows.map(_.id), Vector(source.head.id))
    assert(events.indexOf("materialize") > events.lastIndexWhere(_.startsWith("reserve-")))
    admitted.close()
    admitted.close()
    assertEquals(events.count(_ == "release"), 1)
  }

  test("association write failure does not publish settings or signal success") {
    val root = Files.createTempDirectory("browser-write-failure")
    val home = AppHome(root)
    var effects = 0
    val svc = new AppServices(home, Settings(), true, Nil,
      launchOverride = Some(() => throw new IllegalStateException("unexpected engine start")),
      installShutdownHook = false, settingsWriter = (_, _) => throw new java.io.IOException("disk full"))
    try
      val ui = new ModelBrowserSettings(svc, _ => effects += 1)
      val id = record("lora", "one", "models/one.safetensors").id
      val result = ui.saveAssociation(id, Path.of("one.png").toAbsolutePath.normalize.toString)
      assert(result.isLeft)
      assert(result.swap.toOption.exists(_.contains("disk full")))
      assertEquals(svc.settings.modelPreviewOverrides, Map.empty)
      assertEquals(effects, 0)
    finally
      svc.shutdown()
      val walk = Files.walk(root)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally walk.close()
  }

  test("late revision and pending delivery after disposal are closed without installation") {
    val tasks = mutable.Queue.empty[() => Unit]
    val installed = mutable.ArrayBuffer.empty[Long]
    val closed = mutable.ArrayBuffer.empty[Long]
    final class Owned(val revision: Long) extends BrowserOwnedView:
      def close(): Unit = closed += revision
    val state = new BrowserViewState(task => tasks.enqueue(task),
      view => installed += view.revision, () => ())
    state.offer(new Owned(9))
    state.offer(new Owned(8))
    tasks.dequeue()()
    assertEquals(installed.toSeq, Seq(9L))
    assert(closed.contains(8L))
    state.offer(new Owned(10))
    state.dispose()
    while tasks.nonEmpty do tasks.dequeue()()
    assertEquals(installed.toSeq, Seq(9L))
    assert(closed.contains(9L))
    assert(closed.contains(10L))
  }

  test("real catalog and derived leases remain charged until browser clears and releases") {
    import sdfx.core.modelcards.*
    import sdfx.engine.{CatalogSnapshot, ModelCardCatalog}
    import sdfx.ui.view.CatalogBrowserView
    import scala.concurrent.Await
    import scala.concurrent.duration.*
    import java.util.concurrent.Executor
    val root = Files.createTempDirectory("browser-ledger")
    val worker = mutable.Queue.empty[Runnable]
    val fx = mutable.Queue.empty[() => Unit]
    val live = mutable.Map.empty[Long, String]
    val events = mutable.ArrayBuffer.empty[String]
    val observer = new CardScanObserver:
      def reserve(id: Long, category: String, chars: Long, bytes: Long): Unit = live(id) = category
      def materialize(id: Long): Unit = ()
      def release(id: Long): Unit = { events += s"release-${live.getOrElse(id, "unknown")}"; live.remove(id); () }
      def entryAdmitted(path: Path, chars: Long): Unit = ()
      def directorySorted(path: Path): Unit = ()
      def weightOpened(path: Path): Unit = ()
      def visited(path: Path): Unit = ()
      def pendingDirectories(count: Int): Unit = ()
      def cancellationCheck(): Unit = ()
    val ledger = new CardScanReservations(2000000L, 96L * 1024L * 1024L, observer)
    val executor = new Executor { def execute(r: Runnable): Unit = worker.enqueue(r) }
    val row = record("checkpoint", "blue", root.resolve("blue.safetensors").toString)
    val catalog = new ModelCardCatalog(AppHome(root), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => CardScanResult(request.revision, Vector(row), CardScanStatus.Complete, Vector.empty, 0, 1),
      worker = executor, hooks = CardScanHooks(observer = observer, reservations = Some(ledger)))
    var rows = Vector.empty[ModelCardRecord]
    val state = new BrowserViewState(f => fx.enqueue(f), view => {
      val owner = view.asInstanceOf[CatalogBrowserView]
      val derived = owner.lease.withView(v => ModelBrowserState.derive(v.records, "all", "",
        (c, b) => owner.lease.reserveDerived(c, b).map(d => () => d.close()))).fold(fail(_), identity).fold(fail(_), identity)
      owner.replaceDerived(derived)
      rows = derived.rows
    }, () => { rows = Vector.empty; events += "clear" })
    def next(): CatalogBrowserView =
      val future = catalog.refresh()
      worker.dequeue().run()
      new CatalogBrowserView(Await.result(future, 2.seconds))
    try
      val first = next(); state.offer(first); fx.dequeue()()
      assertEquals(rows.map(_.id), Vector(row.id))
      assert(live.values.count(_ == "card-derived-view") == 1)
      val middle = next(); state.offer(middle)
      val latest = next(); state.offer(latest)
      assertEquals(fx.size, 1)
      assertEquals(live.values.count(_ == "catalog-graph"), 2)
      fx.dequeue()()
      assert(events.lastIndexOf("clear") < events.lastIndexOf("release-card-derived-view"))
      assertEquals(live.values.count(_ == "catalog-graph"), 1)
      var copied = false
      val refusal = latest.lease.withView(v => ModelBrowserState.derive(v.records, "all", "",
        (_, _) => latest.lease.reserveDerived(2000001L, 96L * 1024L * 1024L).map(d => () => d.close()),
        () => copied = true)).fold(fail(_), identity)
      assert(refusal.isLeft); assert(!copied)
      catalog.close()
      assertEquals(live.values.count(_ == "catalog-graph"), 1)
      state.dispose()
      assertEquals(rows, Vector.empty)
      assertEquals(live.values.count(_ == "catalog-graph"), 0)
      assertEquals(live.values.count(_ == "card-derived-view"), 0)
    finally
      state.dispose(); catalog.close()
      val walk = Files.walk(root)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally walk.close()
  }

  test("chosen local path round trips through checked settings and resolver as encoded selected URI") {
    import sdfx.core.modelcards.{CardPreviewPolicy, ModelPreviewResolver}
    fixture { (home, svc) =>
      val chosen = Files.write(home.root.resolve("NSFW space % preview.png"), Array[Byte](1))
      val id = record("lora", "one", home.root.resolve("one.safetensors").toString).id
      val ref = ModelPreviewResolver.validateExplicit(chosen).fold(fail(_), identity)
      val ui = new ModelBrowserSettings(svc)
      assert(ui.saveAssociation(id, ref.localPath.toString).isRight)
      val stored = Settings.load(home.settingsFile).modelPreviewOverrides(id.storageKey)
      assertEquals(stored, chosen.toAbsolutePath.normalize.toString)
      val resolved = ModelPreviewResolver.resolve(id, false, Some(stored)).fold(fail(_), identity).get
      assertEquals(resolved.source, chosen.toAbsolutePath.normalize.toUri.toASCIIString)
      assert(CardPreviewPolicy.matches(Some(resolved.source)))
      assert(resolved.source.contains("%20")); assert(resolved.source.contains("%25"))
    }
  }

  test("shutdown cleanup attempts later resources once after an earlier failure") {
    val cleanup = new sdfx.ui.vm.UiCleanup
    val seen = mutable.ArrayBuffer.empty[String]
    val steps: Seq[(String, () => Unit)] = Seq(
      "listener" -> (() => { seen += "listener"; throw new IllegalStateException("injected") }),
      "views" -> (() => { seen += "views"; () }),
      "loader" -> (() => { seen += "loader"; () })
    )
    cleanup.run(steps)
    cleanup.run(steps)
    assertEquals(seen.toSeq, Seq("listener", "views", "loader"))
  }

  test("association presenter rejects URI strings without writing or publishing") {
    fixture { (_, svc) =>
      val ui = new ModelBrowserSettings(svc)
      val id = record("lora", "one", "models/one.safetensors").id
      val before = svc.settings
      assert(ui.saveAssociation(id, "file:///tmp/preview.png").isLeft)
      assert(ui.saveAssociation(id, "https://example.invalid/preview.png").isLeft)
      assertEquals(svc.settings, before)
    }
  }

  test("shared catalog listener leases update both consumers and survive one consumer close") {
    import sdfx.core.modelcards.*
    import sdfx.engine.{CatalogSnapshot, ModelCardCatalog}
    import sdfx.ui.view.CatalogBrowserView
    import scala.concurrent.Await
    import scala.concurrent.duration.*
    import java.util.concurrent.Executor
    val root = Files.createTempDirectory("browser-shared-catalog")
    val worker = mutable.Queue.empty[Runnable]
    val fx1 = mutable.Queue.empty[() => Unit]
    val fx2 = mutable.Queue.empty[() => Unit]
    val executor = new Executor { def execute(r: Runnable): Unit = worker.enqueue(r) }
    val row = record("checkpoint", "fixture", root.resolve("fixture.safetensors").toString)
    val catalog = new ModelCardCatalog(AppHome(root), () => Settings(), () => CatalogSnapshot(),
      scan = (request, _) => CardScanResult(request.revision, Vector(row), CardScanStatus.Complete, Vector.empty, 0, 1),
      worker = executor)
    var revision1 = 0L
    var revision2 = 0L
    val state1 = new BrowserViewState(f => fx1.enqueue(f), v => revision1 = v.revision, () => ())
    val state2 = new BrowserViewState(f => fx2.enqueue(f), v => revision2 = v.revision, () => ())
    def offer(state: BrowserViewState)(lease: sdfx.engine.CardScanLease): Unit =
      val revision = lease.revision
      state.offer(new CatalogBrowserView(lease, () => catalog.snapshotInfo.exists(_.revision == revision)))
    val remove1 = catalog.addListener(offer(state1)).fold(fail(_), identity)
    val remove2 = catalog.addListener(offer(state2)).fold(fail(_), identity)
    def scan(): Long =
      val result = catalog.refresh()
      worker.dequeue().run()
      val lease = Await.result(result, 2.seconds)
      val revision = lease.revision
      lease.close()
      revision
    try
      val first = scan()
      fx1.dequeue()(); fx2.dequeue()()
      assertEquals(revision1, first); assertEquals(revision2, first)
      val second = scan()
      fx1.dequeue()(); fx2.dequeue()()
      assertEquals(revision1, second); assertEquals(revision2, second)
      remove1(); state1.dispose()
      val third = scan()
      assertEquals(fx1.size, 0)
      fx2.dequeue()()
      assertEquals(revision2, third)
      assertEquals(catalog.snapshotInfo.map(_.revision), Some(third))
    finally
      remove1(); remove2(); state1.dispose(); state2.dispose(); catalog.close()
      val walk = Files.walk(root)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally walk.close()
  }

  test("actual final cleanup coordinator continues after loader failure without joining") {
    val events = mutable.ArrayBuffer.empty[String]
    val scheduled = mutable.Queue.empty[() => Unit]
    val cleanup = new sdfx.ui.vm.FinalAppCleanup(
      () => { events += "loader"; throw new IllegalStateException("injected loader failure") },
      task => { events += "scheduled"; scheduled.enqueue(task); () },
      () => { events += "services"; throw new IllegalStateException("injected service failure") },
      () => { events += "exit"; () })
    cleanup.run()
    cleanup.run()
    assertEquals(events.toSeq, Seq("loader", "scheduled"))
    assertEquals(scheduled.size, 1)
    scheduled.dequeue()()
    assertEquals(events.toSeq, Seq("loader", "scheduled", "services", "exit"))
  }
