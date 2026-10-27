package sdfx.core

import sdfx.core.config.{ModelDirs, Settings}
import sdfx.core.modelcards.*
import java.nio.file.{Files, Path}

/** Offline fixtures: parent reForge and Python are never imported or launched. */
class ModelCardScannerSuite extends munit.FunSuite with TempDir:
  private val kinds = Vector("checkpoints", "vae", "lora", "controlnet", "embeddings", "upscalers",
    "face_restore", "svd", "zero123", "photomaker", "hypernetwork", "future_kind")

  private def fixture(root: Path): Settings =
    kinds.foreach(k => Files.createDirectories(root.resolve(k)))
    def dirs(k: String) = Seq(root.resolve(k).toString)
    Settings(modelDirs = ModelDirs(checkpoints = dirs("checkpoints"), vae = dirs("vae"),
      lora = dirs("lora"), controlnet = dirs("controlnet"), embeddings = dirs("embeddings"),
      upscalers = dirs("upscalers"), faceRestore = dirs("face_restore"), svd = dirs("svd"),
      zero123 = dirs("zero123"), photomaker = dirs("photomaker")),
      modelPreviewFolders = Map("hypernetwork" -> dirs("hypernetwork"), "future_kind" -> dirs("future_kind")))

  private def file(root: Path, relative: String): Path =
    val p = root.resolve(relative)
    Files.createDirectories(p.getParent)
    Files.write(p, Array[Byte](0, 1, 2, 3)) // deliberately invalid weights; no header can be read
    p

  private def scan(root: Path, settings: Settings) =
    ModelCardScanner.scan(CardScanRequest(42L, root, settings), () => false)

  /** Real temporary files with deterministic fixture identities, not Windows file-ID evidence. */
  private class FixturePlatform extends LocalPreviewFiles.Platform:
    private val native = LocalPreviewFiles.systemPlatform
    def windows: Boolean = native.windows
    def driveType(root: Path): Either[String, String] = Right("Fixed")
    def stamp(path: Path): Either[String, LocalPreviewFiles.PathStamp] =
      try
        val a = Files.readAttributes(path, classOf[java.nio.file.attribute.BasicFileAttributes],
          java.nio.file.LinkOption.NOFOLLOW_LINKS)
        val target = if a.isSymbolicLink then Some(Files.readSymbolicLink(path)) else None
        Right(LocalPreviewFiles.PathStamp(Some("fixture:" + path.toAbsolutePath.normalize.toString),
          a.size(), a.lastModifiedTime().toMillis, a.isDirectory, a.isRegularFile, target))
      catch case scala.util.control.NonFatal(e) => Left(e.toString)
    def realPath(path: Path): Either[String, Path] =
      try Right(path.toRealPath()) catch case scala.util.control.NonFatal(e) => Left(e.toString)
    def openNoFollow(path: Path): Either[String, java.nio.channels.SeekableByteChannel] =
      Left("fixture does not open model weights or preview bytes")
  private def fixtureScope() = LocalPreviewFiles.scoped(new FixturePlatform)

  tmp.test("all configured kinds, generic hypernetwork, suffixes and no weight-derived family") { root =>
    val settings = fixture(root)
    kinds.foreach(k => file(root, s"$k/$k.SAFETENSORS"))
    val exts = Vector("pt", "pth", "ckpt", "bin", "patch")
    exts.foreach(e => file(root, s"future_kind/item.$e"))
    file(root, "checkpoints/rejected.pt")
    val result = scan(root, settings)
    assertEquals(result.revision, 42L)
    assertEquals(result.status, CardScanStatus.Complete)
    val paths = result.records.map(_.id.modelPath).toSet
    val canonicalKind = Map("checkpoints" -> "checkpoint", "embeddings" -> "embedding",
      "upscalers" -> "upscaler")
    kinds.foreach { folder =>
      val path = root.resolve(s"$folder/$folder.SAFETENSORS").toAbsolutePath.normalize.toString
      val record = result.records.find(_.id.modelPath == path).getOrElse(fail(s"missing $path"))
      assertEquals(record.id.kind, canonicalKind.getOrElse(folder, folder), folder)
    }
    exts.foreach(e => assert(paths.contains(root.resolve(s"future_kind/item.$e").toAbsolutePath.normalize.toString)))
    assert(!paths.contains(root.resolve("checkpoints/rejected.pt").toAbsolutePath.normalize.toString))
    assert(result.records.forall(_.family.isEmpty), result.records.toString)
    assert(result.records.forall(_.preview.isEmpty), result.records.toString)
  }

  tmp.test("documented Python marker boundaries and hidden traversal without executing Python") { root =>
    val settings = fixture(root)
    Vector("controlnet" -> "config.json", "svd" -> "model_index.json",
      "zero123" -> "model_index.json", "future_kind" -> "model_index.json").foreach { (kind, marker) =>
      file(root, s"$kind/bundle/$marker")
      file(root, s"$kind/bundle/weights.safetensors")
      file(root, s"$kind/normal/nested.pt")
    }
    file(root, "lora/.hidden/secret.pt")
    file(root, "controlnet/.hidden-marker/config.json")
    file(root, "controlnet/.hidden-marker/weights.safetensors")
    val result = scan(root, settings)
    val paths = result.records.map(_.id.modelPath).toSet
    Vector("controlnet", "svd", "zero123", "future_kind").foreach { kind =>
      assert(paths.contains(root.resolve(s"$kind/bundle").toAbsolutePath.normalize.toString), kind)
      assert(!paths.contains(root.resolve(s"$kind/bundle/weights.safetensors").toAbsolutePath.normalize.toString), kind)
      assert(paths.contains(root.resolve(s"$kind/normal/nested.pt").toAbsolutePath.normalize.toString), kind)
    }
    assert(!paths.contains(root.resolve("lora/.hidden/secret.pt").toAbsolutePath.normalize.toString))
    assert(paths.contains(root.resolve("controlnet/.hidden-marker").toAbsolutePath.normalize.toString))
    assert(!paths.contains(root.resolve("controlnet/.hidden-marker/weights.safetensors").toAbsolutePath.normalize.toString))
  }

  tmp.test("missing roots are visible and inaccessible overrides never silently select a sidecar") { root =>
    val settings = fixture(root)
    val model = file(root, "lora/example.pt")
    file(root, "lora/example.png")
    val id = CardId("lora", model.toAbsolutePath.normalize.toString)
    val s = settings.copy(modelDirs = settings.modelDirs.copy(vae = Seq(root.resolve("missing").toString)),
      modelPreviewOverrides = Map(id.storageKey -> root.resolve("missing-preview.png").toString))
    val result = scan(root, s)
    assert(result.status != CardScanStatus.Complete)
    assert(result.problemCount > 0L)
    assert(result.problems.exists(_.toLowerCase.contains("missing")), result.problems.toString)
    val record = result.records.find(_.id == id).get
    assertEquals(record.preview, None)
    assert(record.problem.nonEmpty)
  }

  tmp.test("depth limit and bounded problem detail count remain explicit") { root =>
    val settings = fixture(root)
    var d = root.resolve("future_kind")
    (0 until 34).foreach { _ => d = Files.createDirectory(d.resolve("d")) }
    file(root, root.relativize(d.resolve("too-deep.pt")).toString)
    val missing = (0 until 110).map(i => root.resolve(s"missing-$i").toString)
    val result = scan(root, settings.copy(modelDirs = settings.modelDirs.copy(vae = missing)))
    assertEquals(result.status, CardScanStatus.Partial)
    assert(result.problemCount >= 110L)
    assert(result.problems.size <= 100)
    assert(result.problems.forall(_.length <= 512))
    assert(!result.records.exists(_.id.modelPath.endsWith("too-deep.pt")))
  }

  tmp.test("cancellation is surfaced, not presented as a complete catalog") { root =>
    val settings = fixture(root)
    file(root, "lora/a.pt")
    val result = ModelCardScanner.scan(CardScanRequest(7L, root, settings), () => true)
    assertEquals(result.status, CardScanStatus.Cancelled)
    assertEquals(result.revision, 7L)
  }

  // Proposed injection seam: hooks observe allocations at their actual ownership boundary.
  // A ticket identifies one reservation, not a record after it was already materialized.
  private class Observer extends CardScanObserver:
    val live = scala.collection.mutable.Map.empty[Long, (Long, Long)]
    val categories = scala.collection.mutable.Map.empty[Long, String]
    val reserveCategories = scala.collection.mutable.Set.empty[String]
    var groupAdmissions = 0
    var peakChars = 0L
    var peakBytes = 0L
    private var currentChars = 0L
    private var currentBytes = 0L
    def totalsMatch: Boolean =
      currentChars == live.valuesIterator.map(_._1).sum && currentBytes == live.valuesIterator.map(_._2).sum
    var weightOpens = 0
    var invalidReservation = false
    var invalidMaterialization = false
    var invalidRelease = false
    var directoryAdmitted = 0
    val events = scala.collection.mutable.ArrayBuffer.empty[(String, Path, Long)]
    var checks = Vector.empty[(Long, Long)]
    var blocks = Vector.empty[(String, Long)]
    override def blockingCall(name: String, elapsedNanos: Long): Unit = blocks :+= (name -> elapsedNanos)
    var visits = 0L
    var peakPendingDirectories = 0
    def pendingDirectories(count: Int): Unit = peakPendingDirectories = peakPendingDirectories.max(count)
    var now = 0L
    def reserve(ticket: Long, category: String, chars: Long, bytes: Long): Unit =
      if live.contains(ticket) then invalidReservation = true
      else
        live(ticket) = (chars, bytes)
        currentChars += chars
        currentBytes += bytes
      categories(ticket) = category
      reserveCategories += category
      if category == "cached-group-entry" then groupAdmissions += 1
      peakChars = peakChars.max(currentChars)
      peakBytes = peakBytes.max(currentBytes)
    def materialize(ticket: Long): Unit = if !live.contains(ticket) then invalidMaterialization = true
    def release(ticket: Long): Unit =
      live.remove(ticket) match
        case Some((chars, bytes)) => currentChars -= chars; currentBytes -= bytes
        case None => invalidRelease = true
      categories.remove(ticket)
    def entryAdmitted(directory: Path, chars: Long): Unit =
      directoryAdmitted += 1
      events += (("admit", directory, chars))
    def directorySorted(directory: Path): Unit = events += (("sort", directory, 0L))
    def weightOpened(path: Path): Unit = weightOpens += 1
    def visited(path: Path): Unit = visits += 1
    def cancellationCheck(): Unit = checks :+= (visits -> now)

  tmp.test("reservations precede materialization and a small shared budget is never exceeded") { root =>
    val settings = fixture(root)
    (0 until 30).foreach(i => file(root, s"future_kind/long-name-$i.pt"))
    val observer = new Observer
    val limits = CardScanLimits.defaults.copy(metadataChars = 1200L, metadataBytes = 8192L)
    val hooks = CardScanHooks(limits = limits, observer = observer, clockNanos = () => observer.now)
    val result = ModelCardScanner.scan(CardScanRequest(2L, root, settings), () => false, hooks)
    assertEquals(result.status, CardScanStatus.Partial)
    assert(observer.peakChars <= limits.metadataChars)
    assert(observer.peakBytes <= limits.metadataBytes)
    assert(!observer.invalidReservation && !observer.invalidMaterialization && !observer.invalidRelease)
    assert(observer.totalsMatch)
    assertEquals(observer.weightOpens, 0)
    assert(result.problems.exists(_.toLowerCase.contains("narrower")))
  }

  tmp.test("entry admission is bounded before sorting and no weight file is opened") { root =>
    val settings = fixture(root)
    (0 until 12).foreach(i => file(root, f"future_kind/$i%02d.pt"))
    val observer = new Observer
    val hooks = CardScanHooks(limits = CardScanLimits.defaults.copy(directoryEntries = 5), observer = observer)
    val result = ModelCardScanner.scan(CardScanRequest(3L, root, settings), () => false, hooks)
    assertEquals(result.status, CardScanStatus.Partial)
    assert(observer.directoryAdmitted <= 5)
    assertEquals(observer.weightOpens, 0)
  }

  tmp.test("slices check cancellation at the entry and elapsed-time boundaries") { root =>
    val settings = fixture(root)
    (0 until 12).foreach(i => file(root, s"future_kind/$i.pt"))
    val observer = new Observer:
      override def visited(path: Path): Unit =
        super.visited(path)
        now += 50000000L
    val hooks = CardScanHooks(limits = CardScanLimits.defaults.copy(sliceEntries = 3, sliceNanos = 250000000L),
      observer = observer, clockNanos = () => observer.now, scope = Some(fixtureScope()))
    ModelCardScanner.scan(CardScanRequest(4L, root, settings), () => false, hooks)
    assert(observer.checks.nonEmpty)
    observer.checks.sliding(2).foreach { pair =>
      if pair.size == 2 then
        assert(pair(1)._1 - pair(0)._1 <= 3L)
        assert(pair(1)._2 - pair(0)._2 <= 250000000L)
    }
  }

  /** The seam wraps directory enumeration, without replacing Task1's path validation. */
  private class DirectoryProbe(observer: Observer) extends CardScanDirectoryOps.Delegating(CardScanDirectoryOps.system):
    override def openDirectory(dir: Path): java.nio.file.DirectoryStream[Path] =
      observer.events += (("open", dir, 0L))
      val underlying = super.openDirectory(dir)
      new java.nio.file.DirectoryStream[Path]:
        def iterator(): java.util.Iterator[Path] =
          val it = underlying.iterator()
          new java.util.Iterator[Path]:
            def hasNext: Boolean = it.hasNext
            def next(): Path =
              val path = it.next()
              observer.events += (("pull", dir, 0L))
              path
        def close(): Unit =
          observer.events += (("close", dir, 0L))
          underlying.close()
    override def openFile(path: Path): java.io.InputStream =
      if Set(".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".patch")
          .exists(path.toString.toLowerCase(java.util.Locale.ROOT).endsWith) then
        observer.weightOpens += 1
        fail(s"scanner opened a model weight: $path")
      super.openFile(path)

  tmp.test("UTF-16 per-directory character admission stops before materialization") { root =>
    val settings = fixture(root)
    val directory = root.resolve("future_kind")
    val observer = new Observer
    // Synthetic names exercise the real 1MiB bound without creating invalid Windows long-path files.
    val ops = new DirectoryProbe(observer):
      override def openDirectory(dir: Path): java.nio.file.DirectoryStream[Path] =
        if dir != directory then super.openDirectory(dir)
        else new java.nio.file.DirectoryStream[Path]:
          def iterator(): java.util.Iterator[Path] =
            val entries = (0 until 4096).iterator.map(i => dir.resolve(f"$i%04d" + "x" * 330 + ".pt"))
            new java.util.Iterator[Path]:
              def hasNext: Boolean = entries.hasNext
              def next(): Path =
                val path = entries.next()
                observer.events += (("pull", dir, 0L))
                path
          def close(): Unit = observer.events += (("close", dir, 0L))
    val result = ModelCardScanner.scan(CardScanRequest(6L, root, settings), () => false,
      CardScanHooks(observer = observer, directoryOps = ops))
    val events = observer.events.filter(_._2 == directory).toVector
    val admitted = events.filter(_._1 == "admit")
    assertEquals(result.status, CardScanStatus.Partial)
    assert(admitted.nonEmpty && admitted.size < 4096)
    assert(admitted.map(_._3).sum <= 1024L * 1024L)
    assert(events.count(_._1 == "pull") <= admitted.size + 1)
    assertEquals(events.count(_._1 == "close"), 1)
    val sort = events.indexWhere(_._1 == "sort")
    if sort >= 0 then assert(!events.drop(sort + 1).exists(_._1 == "admit"))
  }

  tmp.test("small card and pending-directory budgets report partial without overshooting") { root =>
    val settings = fixture(root)
    (0 until 12).foreach(i => file(root, s"future_kind/$i.pt"))
    val cardResult = ModelCardScanner.scan(CardScanRequest(8L, root, settings), () => false,
      CardScanHooks(limits = CardScanLimits.defaults.copy(records = 5)))
    assertEquals(cardResult.status, CardScanStatus.Partial)
    assertEquals(cardResult.records.size, 5)
    assert(cardResult.problems.exists(_.toLowerCase.contains("narrower")))
    (0 until 12).foreach(i => Files.createDirectory(root.resolve(s"hypernetwork/dir$i")))
    val observer = new Observer
    val pendingResult = ModelCardScanner.scan(CardScanRequest(9L, root, settings), () => false,
      CardScanHooks(limits = CardScanLimits.defaults.copy(pendingDirectories = 5), observer = observer))
    assertEquals(pendingResult.status, CardScanStatus.Partial)
    assert(observer.peakPendingDirectories <= 5)
    assert(pendingResult.problems.exists(_.toLowerCase.contains("narrower")))
  }

  test("production scan limits are the specified real bounds") {
    val l = CardScanLimits.defaults
    assertEquals(l.directoryEntries, 4096)
    assertEquals(l.directoryChars, 1024L * 1024L)
    assertEquals(l.metadataChars, 2000000L)
    assertEquals(l.metadataBytes, 96L * 1024L * 1024L)
    assertEquals(l.records, 50000)
    assertEquals(l.depth, 32)
    assertEquals(l.pendingDirectories, 10000)
    assertEquals(l.problemDetails, 100)
    assertEquals(l.problemChars, 512)
    assertEquals(l.sliceEntries, 5000)
    assertEquals(l.sliceNanos, 250000000L)
  }

  tmp.test("child reparse escape and inaccessible directory become visible problems") { root =>
    val settings = fixture(root)
    val outside = Files.createTempDirectory("sdfx-outside-")
    try
      val outsideWeight = file(outside, "secret.pt")
      val child = root.resolve("future_kind/escape")
      // Even when creating a Windows symlink is forbidden, the injected directory seam supplies
      // the same child-reparse metadata; Task1 still owns all real-path/drive validation.
      val realLink = try
        Files.createSymbolicLink(child, outside)
        true
      catch
        case _: java.io.IOException | _: UnsupportedOperationException | _: SecurityException =>
          Files.createDirectory(child)
          false
      if realLink then assertEquals(LocalPreviewFiles.scoped().childReparse(child), Right(true))
      val denied = Files.createDirectory(root.resolve("future_kind/denied"))
      val observer = new Observer
      val ops = new DirectoryProbe(observer):
        override def isChildReparse(path: Path): Boolean = path == child || super.isChildReparse(path)
        override def openDirectory(path: Path): java.nio.file.DirectoryStream[Path] =
          if path == denied then throw new java.nio.file.AccessDeniedException(path.toString)
          else super.openDirectory(path)
      val result = ModelCardScanner.scan(CardScanRequest(10L, root, settings), () => false,
        CardScanHooks(observer = observer, directoryOps = ops))
      assert(result.status != CardScanStatus.Complete)
      assert(result.problemCount >= 2L, s"real link fixture=$realLink: ${result.problems}")
      assert(!result.records.exists(_.id.modelPath == outsideWeight.toAbsolutePath.normalize.toString))
      assert(result.problems.exists(_.toLowerCase.contains("escape")))
      assert(result.problems.exists(_.toLowerCase.contains("denied")))
    finally TestFiles.deleteRecursively(outside)
  }

  tmp.test("scanner and Task1 scoped validator never open model weights") { root =>
    val settings = fixture(root)
    kinds.foreach(k => file(root, s"$k/no-header.safetensors"))
    val native = LocalPreviewFiles.systemPlatform
    var probes = 0
    var weightOpens = 0
    val platform = new LocalPreviewFiles.Platform:
      def windows = native.windows
      def driveType(path: Path) = { probes += 1; Right("Fixed") }
      def stamp(path: Path) = native.stamp(path)
      def realPath(path: Path) = native.realPath(path)
      def openNoFollow(path: Path) =
        if path.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors") then
          weightOpens += 1
          fail(s"model bytes opened through local validator: $path")
        native.openNoFollow(path)
    val scope = LocalPreviewFiles.scoped(platform)
    val observer = new Observer
    val result = ModelCardScanner.scan(CardScanRequest(11L, root, settings), () => false,
      CardScanHooks(scope = Some(scope), directoryOps = new DirectoryProbe(observer), observer = observer))
    assertEquals(result.records.size, kinds.size)
    assert(result.records.forall(_.family.isEmpty))
    assertEquals(weightOpens, 0)
    assertEquals(observer.weightOpens, 0)
    if native.windows then assert(probes <= 1, s"drive classification must be shared in one scan: $probes")
  }

  tmp.test("detail count and length limits preserve total problem count") { root =>
    val settings = fixture(root)
    val denied = (0 until 110).map(i => Files.createDirectory(root.resolve(s"future_kind/denied-$i"))).toSet
    val observer = new Observer
    val ops = new DirectoryProbe(observer):
      override def openDirectory(path: Path): java.nio.file.DirectoryStream[Path] =
        if denied.contains(path) then throw new java.nio.file.AccessDeniedException(path.toString, null, "X" * 800)
        super.openDirectory(path)
    val result = ModelCardScanner.scan(CardScanRequest(12L, root, settings), () => false,
      CardScanHooks(observer = observer, directoryOps = ops))
    assertEquals(result.status, CardScanStatus.Partial)
    assertEquals(result.problemCount, 110L)
    assertEquals(result.problems.size, 100)
    assert(result.problems.forall(_.length <= 512))
  }

  tmp.test("small depth and problem budgets are separately enforced") { root =>
    val settings = fixture(root)
    file(root, "future_kind/a/b/c/too-deep.pt")
    val denied = (0 until 6).map(i => Files.createDirectory(root.resolve(s"hypernetwork/denied-$i"))).toSet
    val observer = new Observer
    val ops = new DirectoryProbe(observer):
      override def openDirectory(path: Path): java.nio.file.DirectoryStream[Path] =
        if denied.contains(path) then throw new java.nio.file.AccessDeniedException(path.toString, null, "Y" * 80)
        super.openDirectory(path)
    val limits = CardScanLimits.defaults.copy(depth = 2, problemDetails = 3, problemChars = 20)
    val result = ModelCardScanner.scan(CardScanRequest(13L, root, settings), () => false,
      CardScanHooks(limits = limits, observer = observer, directoryOps = ops))
    assertEquals(result.status, CardScanStatus.Partial)
    assert(result.problemCount >= 7L)
    assert(result.problems.size <= 3)
    assert(result.problems.forall(_.length <= 20))
    assert(!result.records.exists(_.id.modelPath.endsWith("too-deep.pt")))
  }

  tmp.test("real aggregate character and byte budgets include existing owned reservations") { root =>
    val settings = fixture(root)
    (0 until 80).foreach(i => file(root, s"future_kind/long-model-name-$i.safetensors"))
    // Seed old snapshot ownership into the same ledger; the scan must reserve before each copy.
    for (oldChars, oldBytes) <- Vector((1996000L, 1000000L), (1000L, 96L * 1024L * 1024L - 4096L)) do
      val observer = new Observer
      val ledger = new CardScanReservations(2000000L, 96L * 1024L * 1024L, observer)
      val old = ledger.tryReserve("old-snapshot", oldChars, oldBytes).get
      observer.materialize(old)
      val result = ModelCardScanner.scan(CardScanRequest(14L, root, settings), () => false,
        CardScanHooks(observer = observer, reservations = Some(ledger)))
      assertEquals(result.status, CardScanStatus.Partial)
      assert(observer.peakChars <= 2000000L)
      assert(observer.peakBytes <= 96L * 1024L * 1024L)
      assert(observer.live.contains(old), "new scan must not release an old owner's reservation")
      assert(result.problems.exists(_.toLowerCase.contains("narrower")))
      ledger.release(old)
  }

  tmp.test("default 5000-visit cancellation boundary applies across directories") { root =>
    val settings = fixture(root)
    Files.createDirectories(root.resolve("future_kind/first"))
    Files.createDirectories(root.resolve("future_kind/second"))
    (0 until 2501).foreach { i =>
      file(root, f"future_kind/first/$i%05d.pt")
      file(root, f"future_kind/second/$i%05d.pt")
    }
    val observer = new Observer
    val hooks = CardScanHooks(observer = observer, clockNanos = () => 0L, scope = Some(fixtureScope()))
    val result = ModelCardScanner.scan(CardScanRequest(15L, root, settings), () => observer.visits >= 5000L, hooks)
    assertEquals(result.status, CardScanStatus.Cancelled)
    assert(observer.visits <= 5000L, s"late cancellation at ${observer.visits}")
    assert(observer.totalsMatch)
    assert(!observer.invalidReservation && !observer.invalidMaterialization && !observer.invalidRelease)
    assert(observer.checks.nonEmpty)
    assert(observer.checks.head._1 <= 5000L)
    observer.checks.sliding(2).foreach { pair =>
      if pair.size == 2 then assert(pair(1)._1 - pair(0)._1 <= 5000L)
    }
  }

  tmp.test("default 250ms clock boundary cancels before a 5000-entry slice") { root =>
    val settings = fixture(root)
    (0 until 30).foreach(i => file(root, s"future_kind/$i.pt"))
    val observer = new Observer:
      override def visited(path: Path): Unit =
        super.visited(path)
        now += 50000000L
    val result = ModelCardScanner.scan(CardScanRequest(16L, root, settings),
      () => observer.now >= 250000000L,
      CardScanHooks(observer = observer, clockNanos = () => observer.now, scope = Some(fixtureScope())))
    assertEquals(result.status, CardScanStatus.Cancelled)
    assert(observer.visits < 5000L)
    assert(observer.now <= 250000000L)
    assert(observer.checks.exists(_._2 == 250000000L))
  }

  tmp.test("one directory enumeration serves every card and sidecar lookup") { root =>
    val settings = fixture(root)
    (0 until 8).foreach { i =>
      file(root, s"lora/model$i.pt")
      file(root, s"lora/model$i.preview.png")
    }
    val observer = new Observer
    val platform = new LocalPreviewFiles.Platform:
      private val native = LocalPreviewFiles.systemPlatform
      def windows = native.windows
      def driveType(path: Path) = Right("Fixed")
      def stamp(path: Path) = native.stamp(path)
      def realPath(path: Path) = native.realPath(path)
      def openNoFollow(path: Path) = native.openNoFollow(path)
    val result = ModelCardScanner.scan(CardScanRequest(17L, root, settings), () => false,
      CardScanHooks(observer = observer, directoryOps = new DirectoryProbe(observer),
        scope = Some(LocalPreviewFiles.scoped(platform))))
    val lora = result.records.filter(_.id.kind == "lora")
    assertEquals(lora.size, 8)
    assert(lora.forall(_.preview.nonEmpty), lora.toString)
    assertEquals(observer.events.count(e => e._1 == "open" && e._2 == root.resolve("lora")), 1)
  }

  tmp.test("identity admission rejects before any ancestor stamp and restores the default scope") { root =>
    val weight = file(root, "lora/model.pt")
    val native = LocalPreviewFiles.systemPlatform
    var stamps = 0
    val platform = new LocalPreviewFiles.Platform:
      def windows = native.windows
      def driveType(path: Path) = Right("Fixed")
      def stamp(path: Path) = { stamps += 1; native.stamp(path) }
      def realPath(path: Path) = native.realPath(path)
      def openNoFollow(path: Path) = native.openNoFollow(path)
    val scope = LocalPreviewFiles.scoped(platform)
    val denied = scope.withAdmission((_, _) => None) { scope.validateModel(weight, directoryModel = false) }
    assert(denied.isLeft)
    assertEquals(stamps, 0)
    assert(scope.validateModel(weight, directoryModel = false).isRight)
    assert(stamps > 0)
  }

  tmp.test("child reparse uses scoped platform metadata and fails closed when unavailable") { root =>
    val child = Files.createDirectory(root.resolve("child"))
    val native = LocalPreviewFiles.systemPlatform
    val platform = new LocalPreviewFiles.Platform:
      def windows = native.windows
      def driveType(path: Path) = Right("Fixed")
      def stamp(path: Path) =
        if path == child then native.stamp(path).map(_.copy(reparseTarget = Some(root.resolve("outside"))))
        else native.stamp(path)
      def realPath(path: Path) = native.realPath(path)
      def openNoFollow(path: Path) = native.openNoFollow(path)
    assertEquals(LocalPreviewFiles.scoped(platform).childReparse(child), Right(true))
    val unavailable = new LocalPreviewFiles.Platform:
      def windows = native.windows
      def driveType(path: Path) = Right("Fixed")
      def stamp(path: Path) = Left("unavailable")
      def realPath(path: Path) = native.realPath(path)
      def openNoFollow(path: Path) = native.openNoFollow(path)
    assert(LocalPreviewFiles.scoped(unavailable).childReparse(child).isLeft)
  }

  tmp.test("a blocking directory call records its gap and cancellation is checked as it returns") { root =>
    val settings = fixture(root)
    val observer = new Observer
    val ops = new DirectoryProbe(observer):
      override def openDirectory(path: Path): java.nio.file.DirectoryStream[Path] =
        val stream = super.openDirectory(path)
        observer.now += 300000000L
        stream
    val result = ModelCardScanner.scan(CardScanRequest(18L, root, settings), () => observer.now >= 300000000L,
      CardScanHooks(observer = observer, directoryOps = ops, clockNanos = () => observer.now, scope = Some(fixtureScope())))
    assertEquals(result.status, CardScanStatus.Cancelled)
    assert(observer.blocks.contains("open-directory" -> 300000000L))
    assert(observer.checks.exists(_._2 == 300000000L))
  }

  tmp.test("injected preview lookup preserves priority and rejects a candidate outside its parent") { root =>
    val model = file(root, "model.pt")
    val chosen = file(root, "model.preview.png")
    file(root, "model.png")
    val otherDir = Files.createDirectory(root.resolve("outside"))
    val escaped = file(root, "outside/model.preview.png")
    val native = LocalPreviewFiles.systemPlatform
    val platform = new LocalPreviewFiles.Platform:
      def windows = native.windows
      def driveType(path: Path) = Right("Fixed")
      def stamp(path: Path) = native.stamp(path)
      def realPath(path: Path) = native.realPath(path)
      def openNoFollow(path: Path) = native.openNoFollow(path)
    val scope = LocalPreviewFiles.scoped(platform)
    val id = CardId("lora", model.toAbsolutePath.normalize.toString)
    var names = Vector.empty[String]
    val resolver = new ModelPreviewResolver(scope, Some((parent, name) => {
      assertEquals(parent, root)
      names :+= name
      Right(if name == "model.preview.png" then Vector(chosen) else Vector.empty)
    }))
    val result = resolver.resolve(id, false, None).fold(fail(_), identity)
    assertEquals(result.map(_.localPath), Some(chosen.toAbsolutePath.normalize))
    assertEquals(names, Vector("model.preview.png"))
    val bad = new ModelPreviewResolver(scope, Some((_, _) => Right(Vector(escaped))))
    assert(bad.resolve(id, false, None).isLeft)
  }

  tmp.test("sub-250ms cancellation during directory open prevents hasNext and next") { root =>
    val settings = fixture(root)
    var cancelled = false
    var hasNextCalls = 0
    var nextCalls = 0
    var closes = 0
    val observer = new Observer
    val ops = new DirectoryProbe(observer):
      override def openDirectory(path: Path): java.nio.file.DirectoryStream[Path] =
        val underlying = super.openDirectory(path)
        cancelled = true // fake clock remains at zero
        new java.nio.file.DirectoryStream[Path]:
          def iterator(): java.util.Iterator[Path] =
            val it = underlying.iterator()
            new java.util.Iterator[Path]:
              def hasNext: Boolean = { hasNextCalls += 1; it.hasNext }
              def next(): Path = { nextCalls += 1; it.next() }
          def close(): Unit = { closes += 1; underlying.close() }
    val result = ModelCardScanner.scan(CardScanRequest(30L, root, settings), () => cancelled,
      CardScanHooks(observer = observer, directoryOps = ops, scope = Some(fixtureScope()), clockNanos = () => 0L))
    assertEquals(result.status, CardScanStatus.Cancelled)
    assertEquals(hasNextCalls, 0)
    assertEquals(nextCalls, 0)
    assertEquals(closes, 1)
  }

  tmp.test("cancellation observed after a short metadata stamp prevents the next stamp or realPath") { root =>
    val settings = fixture(root)
    var cancelled = false
    var stampCalls = 0
    var postCancelCalls = 0
    val platform = new FixturePlatform:
      override def stamp(path: Path) =
        if cancelled then postCancelCalls += 1
        stampCalls += 1
        val answer = super.stamp(path)
        cancelled = true
        answer
      override def realPath(path: Path) =
        if cancelled then postCancelCalls += 1
        super.realPath(path)
    val result = ModelCardScanner.scan(CardScanRequest(31L, root, settings), () => cancelled,
      CardScanHooks(scope = Some(LocalPreviewFiles.scoped(platform)), clockNanos = () => 0L))
    assertEquals(result.status, CardScanStatus.Cancelled)
    assertEquals(stampCalls, 1)
    assertEquals(postCancelCalls, 0)
  }

  tmp.test("cancellation observed after a short realPath probe prevents subsequent ancestor stamps") { root =>
    val settings = fixture(root)
    var cancelled = false
    var realCalls = 0
    var postCancelCalls = 0
    val platform = new FixturePlatform:
      override def stamp(path: Path) =
        if cancelled then postCancelCalls += 1
        super.stamp(path)
      override def realPath(path: Path) =
        if cancelled then postCancelCalls += 1
        realCalls += 1
        val answer = super.realPath(path)
        cancelled = true
        answer
    val result = ModelCardScanner.scan(CardScanRequest(32L, root, settings), () => cancelled,
      CardScanHooks(scope = Some(LocalPreviewFiles.scoped(platform)), clockNanos = () => 0L))
    assertEquals(result.status, CardScanStatus.Cancelled)
    assertEquals(realCalls, 1)
    assertEquals(postCancelCalls, 0)
  }

  tmp.test("cached generic siblings and a configured-kind overlap reuse one admitted parent index") { root =>
    val settings = fixture(root)
    val external = Files.createDirectory(root.resolve("external"))
    val a = file(root, "external/a.pt")
    val b = file(root, "external/b.patch")
    file(root, "external/a.preview.png")
    val overlap = file(root, "lora/shared.pt")
    val cached = Vector(CardScanCachedEntry("future_a", a.toString, None, true),
      CardScanCachedEntry("future_b", b.toString, None, true),
      CardScanCachedEntry("future_b", overlap.toString, None, true))
    val observer = new Observer
    val result = ModelCardScanner.scan(CardScanRequest(33L, root, settings), () => false,
      CardScanHooks(scope = Some(fixtureScope()), observer = observer, directoryOps = new DirectoryProbe(observer),
        cachedEntries = () => cached.iterator))
    val ids = result.records.map(_.id).toSet
    assert(ids.contains(CardId("future_a", a.toString)))
    assert(ids.contains(CardId("future_b", b.toString)))
    assert(ids.contains(CardId("future_b", overlap.toString)))
    assert(ids.contains(CardId("lora", overlap.toString)))
    assertEquals(observer.events.count(e => e._1 == "open" && e._2 == external), 1)
    assertEquals(observer.events.count(e => e._1 == "open" && e._2 == root.resolve("lora")), 1)
    assertEquals(observer.events.count(e => e._1 == "sort" && e._2 == external), 1)
    assert(result.records.find(_.id == CardId("future_a", a.toString)).exists(_.preview.nonEmpty))
  }

  tmp.test("cached grouping is reserved before materialization and over-budget grouping is Partial") { root =>
    val settings = fixture(root)
    val external = Files.createDirectory(root.resolve("external"))
    val files = (0 until 80).map(i => file(root, f"external/name-$i%03d.pt"))
    val cached = files.map(p => CardScanCachedEntry("future", p.toString, None, true)).toVector
    val observer = new Observer
    val hooks = CardScanHooks(scope = Some(fixtureScope()), observer = observer,
      directoryOps = new DirectoryProbe(observer), cachedEntries = () => cached.iterator,
      limits = CardScanLimits.defaults.copy(cachedGroupEntries = 5))
    val result = ModelCardScanner.scan(CardScanRequest(34L, root, settings), () => false, hooks)
    assertEquals(result.status, CardScanStatus.Partial)
    assert(!observer.invalidReservation && !observer.invalidMaterialization && !observer.invalidRelease)
    assert(observer.peakChars <= 2000000L && observer.peakBytes <= 96L * 1024L * 1024L)
    assert(result.records.count(_.id.kind == "future") <= 5)
    assert(result.problems.exists(_.toLowerCase.contains("narrower")))
    assert(observer.reserveCategories.contains("cached-parent-group"))
    assertEquals(observer.groupAdmissions, 5)
    assert(observer.events.count(e => e._1 == "open" && e._2 == external) <= 1)
  }

  tmp.test("marker probe retarget is rejected before opening a directory model") { root =>
    val settings = fixture(root)
    val child = Files.createDirectory(root.resolve("controlnet/bundle"))
    val marker = file(root, "controlnet/bundle/config.json")
    val outside = Files.createDirectory(root.resolve("outside"))
    file(root, "outside/config.json")
    var changed = false
    val platform = new FixturePlatform:
      override def stamp(path: Path) = super.stamp(path).map { stamp =>
        if changed && path == child then stamp.copy(reparseTarget = Some(outside)) else stamp
      }
      override def realPath(path: Path) =
        if changed && path.startsWith(child) then Right(outside.resolve(child.relativize(path))) else super.realPath(path)
    val observer = new Observer
    val ops = new DirectoryProbe(observer):
      override def isRegularFile(path: Path): Boolean =
        val answer = super.isRegularFile(path)
        if path == marker then changed = true
        answer
    val result = ModelCardScanner.scan(CardScanRequest(35L, root, settings), () => false,
      CardScanHooks(scope = Some(LocalPreviewFiles.scoped(platform)), observer = observer, directoryOps = ops))
    assert(changed)
    assertEquals(result.status, CardScanStatus.Partial)
    assert(result.problemCount > 0)
    assert(!observer.events.exists(e => e._1 == "open" && (e._2 == child || e._2 == outside)))
    assert(!result.records.exists(_.id.modelPath == child.toAbsolutePath.normalize.toString))
  }

  tmp.test("cached parent retarget after admission is rejected before preview listing") { root =>
    val settings = fixture(root)
    val parent = Files.createDirectory(root.resolve("external"))
    val weight = file(root, "external/a.pt")
    val outside = Files.createDirectory(root.resolve("outside"))
    file(root, "outside/a.pt")
    var changed = false
    val platform = new FixturePlatform:
      override def stamp(path: Path) = super.stamp(path).map { stamp =>
        if changed && path == parent then stamp.copy(reparseTarget = Some(outside)) else stamp
      }
      override def realPath(path: Path) =
        if changed && path.startsWith(parent) then Right(outside.resolve(parent.relativize(path))) else super.realPath(path)
    val observer = new Observer:
      override def reserve(ticket: Long, category: String, chars: Long, bytes: Long): Unit =
        super.reserve(ticket, category, chars, bytes)
        if category == "cached-parent-group" then changed = true
    val cached = Vector(CardScanCachedEntry("future", weight.toString, None, true))
    val result = ModelCardScanner.scan(CardScanRequest(36L, root, settings), () => false,
      CardScanHooks(scope = Some(LocalPreviewFiles.scoped(platform)), observer = observer,
        directoryOps = new DirectoryProbe(observer), cachedEntries = () => cached.iterator))
    assert(changed, "must exercise the post-validation grouping boundary")
    assertEquals(result.status, CardScanStatus.Partial)
    assert(result.problemCount > 0)
    assert(!observer.events.exists(e => e._1 == "open" && (e._2 == parent || e._2 == outside)))
    assert(!result.records.exists(_.id.modelPath == weight.toAbsolutePath.normalize.toString))
  }

  tmp.test("many sidecars retain the actual graph, not fixed realPath scratch allowances") { root =>
    val settings = fixture(root)
    (0 until 48).foreach { i =>
      file(root, s"lora/model$i.pt")
      file(root, s"lora/model$i.preview.png")
    }
    val observer = new Observer
    val ledger = new CardScanReservations(2000000L, 96L * 1024L * 1024L, observer)
    val result = ModelCardScanner.scan(CardScanRequest(37L, root, settings), () => false,
      CardScanHooks(scope = Some(fixtureScope()), observer = observer, reservations = Some(ledger), retainResult = true))
    assertEquals(result.status, CardScanStatus.Complete)
    assertEquals(result.records.count(r => r.id.kind == "lora" && r.preview.nonEmpty), 48)
    val retainedChars = observer.live.valuesIterator.map(_._1).sum
    val retainedBytes = observer.live.valuesIterator.map(_._2).sum
    assert(retainedChars < 500000L, s"retained=$retainedChars; categories=${observer.categories.values.toVector.groupMapReduce(identity)(_ => 1)(_ + _)}")
    assert(retainedBytes < 8L * 1024L * 1024L)
    assert(!observer.categories.values.exists(_.contains("scratch")))
    assert(!observer.invalidReservation && !observer.invalidMaterialization && !observer.invalidRelease)
    ledger.releaseResult(result)
    assert(observer.live.isEmpty)
  }

  tmp.test("sub-250ms cancellation after drive probe prevents ancestor stamps") { root =>
    assume(LocalPreviewFiles.systemPlatform.windows)
    val settings = fixture(root)
    var cancelled = false
    var drives = 0
    var stamps = 0
    val platform = new FixturePlatform:
      override def driveType(path: Path) = { drives += 1; cancelled = true; Right("Fixed") }
      override def stamp(path: Path) = { stamps += 1; super.stamp(path) }
    val result = ModelCardScanner.scan(CardScanRequest(38L, root, settings), () => cancelled,
      CardScanHooks(scope = Some(LocalPreviewFiles.scoped(platform)), clockNanos = () => 0L))
    assertEquals(result.status, CardScanStatus.Cancelled)
    assertEquals(drives, 1)
    assertEquals(stamps, 0)
  }

  tmp.test("an explicitly selected root link remains allowed while its children are checked") { root =>
    val settings = fixture(root)
    val selected = Files.createDirectory(root.resolve("selected-root"))
    val target = Files.createDirectory(root.resolve("target-root"))
    val model = file(root, "selected-root/item.pt")
    file(root, "target-root/item.pt")
    val platform = new FixturePlatform:
      override def stamp(path: Path) = super.stamp(path).map { stamp =>
        if path == selected then stamp.copy(reparseTarget = Some(target)) else stamp
      }
      override def realPath(path: Path) =
        if path.startsWith(selected) then Right(target.resolve(selected.relativize(path)))
        else super.realPath(path)
    val configured = settings.copy(modelPreviewFolders = settings.modelPreviewFolders.updated("selected_kind", Seq(selected.toString)))
    val result = ModelCardScanner.scan(CardScanRequest(39L, root, configured), () => false,
      CardScanHooks(scope = Some(LocalPreviewFiles.scoped(platform))))
    assertEquals(result.status, CardScanStatus.Complete)
    assert(result.records.exists(_.id == CardId("selected_kind", model.toAbsolutePath.normalize.toString)))
  }

  tmp.test("cancellation immediately before a probe prevents the probe itself") { root =>
    val settings = fixture(root)
    var cancelled = false
    var calls = 0
    val platform = new FixturePlatform:
      override def driveType(path: Path) = { calls += 1; super.driveType(path) }
      override def stamp(path: Path) = { calls += 1; super.stamp(path) }
      override def realPath(path: Path) = { calls += 1; super.realPath(path) }
    val observer = new Observer:
      override def beforeBlocking(name: String): Unit =
        if name == "drive" || name == "stamp" then cancelled = true
    val result = ModelCardScanner.scan(CardScanRequest(40L, root, settings), () => cancelled,
      CardScanHooks(observer = observer, scope = Some(LocalPreviewFiles.scoped(platform)), clockNanos = () => 0L))
    assert(cancelled)
    assertEquals(result.status, CardScanStatus.Cancelled)
    assertEquals(calls, 0)
  }

  tmp.test("configured kinds sharing a directory enumerate it once and keep separate identities") { root =>
    val settings = fixture(root)
    val shared = Files.createDirectory(root.resolve("shared"))
    val filePath = file(root, "shared/same.pt")
    val configured = settings.copy(modelDirs = settings.modelDirs.copy(lora = Seq(shared.toString), vae = Seq(shared.toString)))
    val observer = new Observer
    val result = ModelCardScanner.scan(CardScanRequest(41L, root, configured), () => false,
      CardScanHooks(scope = Some(fixtureScope()), observer = observer, directoryOps = new DirectoryProbe(observer)))
    assertEquals(result.status, CardScanStatus.Complete)
    assert(result.records.exists(_.id == CardId("lora", filePath.toAbsolutePath.normalize.toString)))
    assert(result.records.exists(_.id == CardId("vae", filePath.toAbsolutePath.normalize.toString)))
    assertEquals(observer.events.count(e => e._1 == "open" && e._2 == shared), 1)
  }
