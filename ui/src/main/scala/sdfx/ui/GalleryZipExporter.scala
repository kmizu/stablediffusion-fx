package sdfx.ui

import java.nio.file.{Files, Path, StandardCopyOption}
import java.util.Locale
import java.util.zip.{ZipEntry, ZipOutputStream}
import scala.collection.mutable
import scala.util.Using

/** Blocking I/O: callers run this off the FX thread. A failed export never publishes a partial archive. */
object GalleryZipExporter:
  def write(paths: Seq[Path], destination: Path): Int =
    require(paths.nonEmpty, "This generation has no images to save")
    val target = destination.toAbsolutePath.normalize()
    val sources = paths.map(_.toAbsolutePath.normalize()).toVector
    require(!sources.exists(p => p == target || (Files.exists(target) && Files.isSameFile(p, target))),
      "The ZIP destination must not replace a source image")
    val temp = Files.createTempFile(target.getParent, ".sdfx-gallery-", ".zip.tmp")
    try
      val used = mutable.Set.empty[String]
      Using.resource(new ZipOutputStream(Files.newOutputStream(temp))) { zip =>
        sources.foreach { source =>
          require(Files.isRegularFile(source), s"Image is missing or is not a file: $source")
          val raw = source.getFileName.toString.replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_")
          val base = if raw.isBlank || raw == "." || raw == ".." then "image.png" else raw
          val dot = base.lastIndexOf('.')
          val (stem, suffix) = if dot > 0 then (base.take(dot), base.drop(dot)) else (base, "")
          var name = base
          var n = 2
          while used(name.toLowerCase(Locale.ROOT)) do
            name = s"$stem-$n$suffix"
            n += 1
          used += name.toLowerCase(Locale.ROOT)
          zip.putNextEntry(new ZipEntry(name))
          Files.copy(source, zip)
          zip.closeEntry()
        }
      }
      // Refuse filesystems without atomic replacement rather than risk a partial destination.
      Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      sources.size
    finally Files.deleteIfExists(temp)
