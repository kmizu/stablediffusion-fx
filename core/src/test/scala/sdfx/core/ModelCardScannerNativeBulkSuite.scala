package sdfx.core

import sdfx.core.config.{ModelDirs, Settings}
import sdfx.core.modelcards.*
import java.nio.file.{Files, Path}
import scala.concurrent.duration.*

/** Actual filesystem/native enumeration gate: separate suite gives only these two tests five minutes each. */
class ModelCardScannerNativeBulkSuite extends munit.FunSuite with TempDir:
  override def munitTimeout: Duration = 5.minutes
  private val kinds = Vector("checkpoints", "vae", "lora", "controlnet", "embeddings", "upscalers",
    "face_restore", "svd", "zero123", "photomaker", "hypernetwork", "future_kind")
  private def fixture(root: Path): Settings =
    kinds.foreach(k => Files.createDirectories(root.resolve(k)))
    def dirs(k: String) = Seq(root.resolve(k).toString)
    Settings(modelDirs = ModelDirs(checkpoints = dirs("checkpoints"), vae = dirs("vae"), lora = dirs("lora"),
      controlnet = dirs("controlnet"), embeddings = dirs("embeddings"), upscalers = dirs("upscalers"),
      faceRestore = dirs("face_restore"), svd = dirs("svd"), zero123 = dirs("zero123"), photomaker = dirs("photomaker")),
      modelPreviewFolders = Map("hypernetwork" -> dirs("hypernetwork"), "future_kind" -> dirs("future_kind")))
  private def file(root: Path, relative: String): Path =
    val path = root.resolve(relative)
    Files.createDirectories(path.getParent)
    Files.write(path, Array[Byte](0, 1, 2, 3))
  private def scan(root: Path, settings: Settings) =
    ModelCardScanner.scan(CardScanRequest(42L, root, settings), () => false)
  private class Observer extends CardScanObserver:
    val events = scala.collection.mutable.ArrayBuffer.empty[(String, Path, Long)]
    val live = scala.collection.mutable.Map.empty[Long, (Long, Long)]
    var peakChars = 0L
    var peakBytes = 0L
    private var currentChars = 0L
    private var currentBytes = 0L
    def totalsMatch: Boolean =
      currentChars == live.valuesIterator.map(_._1).sum && currentBytes == live.valuesIterator.map(_._2).sum
    var invalid = false
    def reserve(id: Long, category: String, c: Long, b: Long): Unit =
      if live.contains(id) then invalid = true
      else
        live(id) = (c, b)
        currentChars += c
        currentBytes += b
      peakChars = peakChars.max(currentChars)
      peakBytes = peakBytes.max(currentBytes)
    def materialize(id: Long): Unit = if !live.contains(id) then invalid = true
    def release(id: Long): Unit = live.remove(id) match
      case Some((c, b)) =>
        currentChars -= c
        currentBytes -= b
      case None => invalid = true
    def entryAdmitted(dir: Path, c: Long): Unit = events += (("admit", dir, c))
    def directorySorted(dir: Path): Unit = events += (("sort", dir, 0L))
    def weightOpened(path: Path): Unit = invalid = true
    def visited(path: Path): Unit = ()
    def pendingDirectories(count: Int): Unit = ()
    def cancellationCheck(): Unit = ()
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
        observer.invalid = true
        fail(s"scanner opened a model weight: $path")
      super.openFile(path)

  tmp.test("directory admission stops at the real entry boundary and reports a partial scan") { root =>
    val settings = fixture(root)
    (0 until 4097).foreach(i => file(root, f"future_kind/$i%05d.pt"))
    val result = scan(root, settings)
    assertEquals(result.status, CardScanStatus.Partial)
    assert(result.records.size <= 4096)
    assert(result.problems.exists(_.toLowerCase.contains("narrower")), result.problems.toString)
  }


  tmp.test("streaming directory admission ends before sorting and closes the iterator at the real 4096 cap") { root =>
    val settings = fixture(root)
    val directory = root.resolve("future_kind")
    (0 until 4097).foreach(i => file(root, f"future_kind/$i%05d.pt"))
    val observer = new Observer
    val ops = new DirectoryProbe(observer)
    val result = ModelCardScanner.scan(CardScanRequest(5L, root, settings), () => false,
      CardScanHooks(observer = observer, directoryOps = ops))
    val events = observer.events.filter(_._2 == directory).toVector
    val admits = events.count(_._1 == "admit")
    val pulls = events.count(_._1 == "pull")
    assertEquals(result.status, CardScanStatus.Partial)
    assertEquals(admits, 4096)
    assert(pulls <= 4097, s"iterator kept being consumed: $pulls")
    assertEquals(events.count(_._1 == "close"), 1)
    val sort = events.indexWhere(_._1 == "sort")
    if sort >= 0 then
      assert(!events.drop(sort + 1).exists(_._1 == "admit"), events.toString)
      assert(events.take(sort).exists(_._1 == "close"), events.toString)
    assert(result.problemCount > 0)
    assert(!observer.invalid)
    assert(observer.totalsMatch)
    assert(observer.peakChars <= 2000000L && observer.peakBytes <= 96L * 1024L * 1024L)
    assert(result.problems.exists(_.toLowerCase.contains("narrower")))
  }
