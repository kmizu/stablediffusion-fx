package sdfx.core

import sdfx.core.gallery.*
import sdfx.core.png.PngMetadata

import java.nio.file.attribute.FileTime
import java.nio.file.{Files, Path}
import java.time.LocalDate
import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, Executors, TimeUnit}
import scala.jdk.CollectionConverters.*

class OutputNamingSuite extends munit.FunSuite with TempDir:

  tmp.test("dayDir creates outputs/<kind>/<date>") { dir =>
    val d = OutputNaming.dayDir(dir, "txt2img-images", LocalDate.of(2026, 2, 13))
    assertEquals(d, dir.resolve("txt2img-images").resolve("2026-02-13"))
    assert(Files.isDirectory(d))
    assertEquals(OutputNaming.dayDir(dir, "txt2img-images", LocalDate.of(2026, 2, 13)), d) // idempotent
  }

  tmp.test("nextIndex: empty / missing dir -> 0") { dir =>
    assertEquals(OutputNaming.nextIndex(dir), 0)
    assertEquals(OutputNaming.nextIndex(dir.resolve("missing")), 0)
  }

  tmp.test("nextIndex: max + 1, ignoring non-matching names") { dir =>
    Seq("00000-1.png", "00007-123.png", "00003-9-grid.png", "grid-0042.png", "notes.txt", "12345.png", ".sdfx-1.png.part")
      .foreach(n => Files.createFile(dir.resolve(n)))
    assertEquals(OutputNaming.nextIndex(dir), 8)
  }

  tmp.test("nextIndex: keeps counting past 99999 like A1111") { dir =>
    Files.createFile(dir.resolve("99999-1.png"))
    Files.createFile(dir.resolve("100000-1.png"))
    assertEquals(OutputNaming.nextIndex(dir), 100001)
  }

  tmp.test("nextFile: name format, reservation and increment") { dir =>
    val a = OutputNaming.nextFile(dir, "txt2img-images", 12345L)
    val b = OutputNaming.nextFile(dir, "txt2img-images", 12345L)
    val g = OutputNaming.nextFile(dir, "txt2img-grids", 7L, "-grid")
    assertEquals(a.getFileName.toString, "00000-12345.png")
    assertEquals(b.getFileName.toString, "00001-12345.png")
    assertEquals(g.getFileName.toString, "00000-7-grid.png")
    assert(Files.exists(a) && Files.size(a) == 0L)
    assertEquals(a.getParent.getFileName.toString, LocalDate.now().toString)
    assertEquals(a.getParent.getParent, dir.resolve("txt2img-images"))
  }

  tmp.test("nextFile: negative seed and continuing after existing files") { dir =>
    val day = OutputNaming.dayDir(dir, "img2img-images")
    Files.createFile(day.resolve("00041-5.png"))
    assertEquals(OutputNaming.nextFile(dir, "img2img-images", -1L).getFileName.toString, "00042--1.png")
  }

  tmp.test("nextFile: concurrent reservations are unique") { dir =>
    val threads = 16
    val perThread = 25
    val pool = Executors.newFixedThreadPool(threads)
    val start = new CountDownLatch(1)
    val out = new ConcurrentLinkedQueue[Path]()
    try
      (0 until threads).foreach { t =>
        pool.submit(new Runnable:
          def run(): Unit =
            start.await()
            (0 until perThread).foreach(_ => out.add(OutputNaming.nextFile(dir, "txt2img-images", t.toLong)))
        )
      }
      start.countDown()
      pool.shutdown()
      assert(pool.awaitTermination(60, TimeUnit.SECONDS))
    finally pool.shutdownNow()
    val all = out.asScala.toSeq
    assertEquals(all.size, threads * perThread)
    assertEquals(all.distinct.size, all.size)
    val idx = all.map(_.getFileName.toString.takeWhile(_ != '-').toInt).sorted
    assertEquals(idx, (0 until threads * perThread).toSeq)
  }

class GalleryIndexSuite extends munit.FunSuite with TempDir:
  import TestFiles.*

  private def png(p: Path, params: Option[String], mtime: Long): Path =
    PngMetadata.write(image(4, 4), p, params.map("parameters" -> _).toSeq)
    Files.setLastModifiedTime(p, FileTime.fromMillis(mtime))
    p

  private val T0 = 1_700_000_000_000L

  tmp.test("missing root -> empty") { dir =>
    assertEquals(new GalleryIndex(dir.resolve("nope")).scan(), Vector.empty)
    assertEquals(new GalleryIndex(dir).rootDir, dir)
  }

  tmp.test("scan: newest first, recursive, PNG only, skips dot dirs (.tmp) and empty reservations") { dir =>
    val a = png(dir.resolve("txt2img-images/2026-01-01/00000-1.png"), Some("cat, cute"), T0)
    val b = png(dir.resolve("txt2img-images/2026-01-02/00000-2.png"), Some("dog"), T0 + 2000)
    val c = png(dir.resolve("img2img-images/2026-01-02/00000-3.PNG"), None, T0 + 1000)
    png(dir.resolve(".tmp/engine-out.png"), Some("tmp"), T0 + 9000)
    png(dir.resolve("x/.hidden/h.png"), Some("hidden"), T0 + 9000)
    Files.createDirectories(dir.resolve("txt2img-images/2026-01-03"))
    Files.createFile(dir.resolve("txt2img-images/2026-01-03/00001-4.png")) // reserved, still empty
    Files.writeString(dir.resolve("notes.txt"), "cat")
    ImageIOWrite.jpg(dir.resolve("photo.jpg"))
    val idx = new GalleryIndex(dir)
    val items = idx.scan()
    assertEquals(items.map(_.path), Vector(b, c, a))
    assertEquals(items.map(_.parameters), Vector(Some("dog"), None, Some("cat, cute")))
    assertEquals(items.head.fileName, "00000-2.png")
    assertEquals(items.head.modified, T0 + 2000)
    assertEquals(items.head.size, Files.size(b))
    assertEquals(idx.scan(limit = 2).map(_.path), Vector(b, c))
    assertEquals(idx.scan(limit = 0), Vector.empty)
  }

  tmp.test("scan: equal mtimes are ordered deterministically by path") { dir =>
    val a = png(dir.resolve("a.png"), None, T0)
    val b = png(dir.resolve("b.png"), None, T0)
    assertEquals(new GalleryIndex(dir).scan().map(_.path), Vector(a, b))
  }

  tmp.test("scan and search exclude root intermediates while retaining unrelated nested directories") { dir =>
    val finalImage = png(dir.resolve("txt2img/2026-10-26/final.png"), Some("cat"), T0)
    png(dir.resolve("intermediates/generation-one/frame.png"), Some("cat intermediate"), T0 + 3)
    val nested = png(dir.resolve("archive/intermediates/keep.png"), Some("cat archived"), T0 + 2)
    val similarlyNamed = png(dir.resolve("intermediates-old/keep.png"), Some("cat old"), T0 + 1)
    val idx = new GalleryIndex(dir)
    assertEquals(idx.scan().map(_.path), Vector(nested, similarlyNamed, finalImage))
    assertEquals(idx.search("cat").map(_.path), Vector(nested, similarlyNamed, finalImage))
    assertEquals(idx.search("cat", limit = 1).map(_.path), Vector(nested))
  }

  tmp.test("cache: same mtime/size -> cached item; changed mtime -> re-read") { dir =>
    val f = png(dir.resolve("a.png"), Some("first"), T0)
    val idx = new GalleryIndex(dir)
    val first = idx.item(f).get
    assertEquals(first.parameters, Some("first"))
    assert(idx.item(f).get eq first)
    // rewrite with new params but restore the old mtime: size differs -> re-read
    png(f, Some("second, longer parameters text"), T0)
    assertEquals(idx.item(f).get.parameters, Some("second, longer parameters text"))
    // same size, new content, new mtime -> re-read
    png(f, Some("third, longer parameters text!"), T0 + 5000)
    val third = idx.item(f).get
    assertEquals(third.parameters, Some("third, longer parameters text!"))
    assertEquals(third.modified, T0 + 5000)
  }

  tmp.test("cache: content changed with same mtime and size stays cached (documented key)") { dir =>
    val f = png(dir.resolve("a.png"), Some("aaaa"), T0)
    val idx = new GalleryIndex(dir)
    assertEquals(idx.item(f).get.parameters, Some("aaaa"))
    png(f, Some("bbbb"), T0)
    assertEquals(idx.item(f).get.parameters, Some("aaaa"))
    Files.setLastModifiedTime(f, FileTime.fromMillis(T0 + 1))
    assertEquals(idx.item(f).get.parameters, Some("bbbb"))
  }

  tmp.test("item: missing or empty file -> None") { dir =>
    val idx = new GalleryIndex(dir)
    assertEquals(idx.item(dir.resolve("none.png")), None)
    assertEquals(idx.item(Files.createFile(dir.resolve("e.png"))), None)
  }

  tmp.test("deleted file disappears from scan") { dir =>
    val a = png(dir.resolve("a.png"), Some("x"), T0)
    val b = png(dir.resolve("b.png"), Some("y"), T0 + 1)
    val idx = new GalleryIndex(dir)
    assertEquals(idx.scan().size, 2)
    Files.delete(b)
    assertEquals(idx.scan().map(_.path), Vector(a))
  }

  tmp.test("search: prompt text, case-insensitive, file name, Japanese, limit") { dir =>
    val a = png(dir.resolve("00001-111.png"), Some("Masterpiece, 1girl, smile\nNegative prompt: lowres\nSteps: 20, Seed: 111"), T0)
    val b = png(dir.resolve("00002-222.png"), Some("landscape, mountains\nSteps: 30, Seed: 222"), T0 + 1)
    val c = png(dir.resolve("00003-333.png"), Some("桜の木, 夜景\nSteps: 20, Seed: 333"), T0 + 2)
    val d = png(dir.resolve("00004-444.png"), None, T0 + 3)
    val idx = new GalleryIndex(dir)
    assertEquals(idx.search("masterpiece").map(_.path), Vector(a))
    assertEquals(idx.search("  MOUNTAINS ").map(_.path), Vector(b))
    assertEquals(idx.search("lowres").map(_.path), Vector(a))
    assertEquals(idx.search("桜").map(_.path), Vector(c))
    assertEquals(idx.search("444").map(_.path), Vector(d))
    assertEquals(idx.search("steps: 20").map(_.path), Vector(c, a))
    assertEquals(idx.search("").map(_.path), Vector(d, c, b, a))
    assertEquals(idx.search("", limit = 1).map(_.path), Vector(d))
    assertEquals(idx.search("nothing-matches"), Vector.empty)
  }

object ImageIOWrite:
  def jpg(p: Path): Unit = javax.imageio.ImageIO.write(TestFiles.image(4, 4), "jpg", p.toFile)
