package sdfx.ui

import java.nio.file.{Files, Path}
import java.util.zip.ZipFile
import scala.jdk.CollectionConverters.*
import sdfx.core.gallery.GalleryItem
import sdfx.ui.vm.GalleryVM

class GalleryBatchSuite extends munit.FunSuite:
  private def item(path: String) = GalleryItem(Path.of(path), 1L, 4L, None)

  test("generation keeps twenty results, excludes grids and history, and exports despite search") {
    val vm = new GalleryVM
    vm.addNewest(item("old.png"))
    vm.beginGeneration("first")
    (1 to 20).foreach(n => vm.addResult("first", item(s"first-$n.png")))
    vm.addResult("first", item("grid.png"), grid = true)
    assertEquals(vm.filtered.size, 20)
    assertEquals(vm.selectedGenerationPaths.size, 20)
    vm.query.set("first-20")
    assertEquals(vm.filtered.size, 1)
    assertEquals(vm.selectedGenerationPaths.size, 20)
    vm.beginGeneration("second")
    vm.addResult("second", item("second.png"))
    assertEquals(vm.filtered.size, 1)
    vm.selectedGeneration.set("first")
    assertEquals(vm.filtered.size, 20)
    vm.remove(Path.of("first-1.png"))
    assertEquals(vm.selectedGenerationPaths.size, 19)
    vm.selectedGeneration.set("")
    assertEquals(vm.filtered.size, 22)
    assertEquals(vm.selectedGenerationPaths.size, 0)
  }

  test("scan refresh preserves the selected job results even when absent from a limited scan") {
    val vm = new GalleryVM
    vm.beginGeneration("job")
    vm.addResult("job", item("result.png"))
    vm.merge(Seq(item("historical.png")), Long.MaxValue)
    assertEquals(vm.filtered.size, 1)
    assertEquals(vm.selectedGenerationPaths, Seq(Path.of("result.png")))
  }

  test("ZIP includes exact bytes with unique safe basenames") {
    val dir = Files.createTempDirectory("sdfx-gallery-zip-")
    val a = Files.createDirectories(dir.resolve("a")).resolve("image.png")
    val b = Files.createDirectories(dir.resolve("b")).resolve("IMAGE.png")
    Files.write(a, Array[Byte](1, 2, 3))
    Files.write(b, Array[Byte](4, 5))
    val target = dir.resolve("images.zip")
    assertEquals(GalleryZipExporter.write(Seq(a, b), target), 2)
    val zip = new ZipFile(target.toFile)
    try
      val entries = zip.entries().asScala.toVector
      assertEquals(entries.size, 2)
      assertEquals(entries.map(_.getName.toLowerCase).distinct.size, 2)
      assert(entries.forall(e => !e.getName.contains("/") && !e.getName.contains("\\")))
      assertEquals(entries.map(e => zip.getInputStream(e).readAllBytes().toSeq), Vector(Seq[Byte](1, 2, 3), Seq[Byte](4, 5)))
    finally zip.close()
  }

  test("failed or empty export preserves existing archive and leaves no partial file") {
    val dir = Files.createTempDirectory("sdfx-gallery-fail-")
    val good = Files.write(dir.resolve("good.png"), Array[Byte](1))
    val target = Files.write(dir.resolve("existing.zip"), Array[Byte](9, 8))
    intercept[Exception](GalleryZipExporter.write(Seq(good, dir.resolve("missing.png")), target))
    assertEquals(Files.readAllBytes(target).toSeq, Seq[Byte](9, 8))
    intercept[IllegalArgumentException](GalleryZipExporter.write(Nil, target))
    val files = Files.list(dir)
    try assertEquals(files.iterator().asScala.map(_.getFileName.toString).toSet, Set("good.png", "existing.zip"))
    finally files.close()
  }

  test("ZIP refuses to replace a source image") {
    val source = Files.write(Files.createTempDirectory("sdfx-gallery-self-").resolve("a.png"), Array[Byte](3))
    intercept[IllegalArgumentException](GalleryZipExporter.write(Seq(source), source))
    assertEquals(Files.readAllBytes(source).toSeq, Seq[Byte](3))
  }

  test("failure after writing the first entry removes the unpublished archive") {
    val dir = Files.createTempDirectory("sdfx-gallery-partial-")
    val good = Files.write(dir.resolve("good.png"), Array[Byte](1))
    val target = dir.resolve("new.zip")
    intercept[IllegalArgumentException](GalleryZipExporter.write(Seq(good, dir.resolve("missing.png")), target))
    assert(!Files.exists(target))
    val files = Files.list(dir)
    try assertEquals(files.iterator().asScala.map(_.getFileName.toString).toSet, Set("good.png"))
    finally files.close()
  }
