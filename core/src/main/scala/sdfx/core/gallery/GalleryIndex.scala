package sdfx.core.gallery

import sdfx.core.png.PngMetadata

import java.nio.file.{Files, Path}
import java.util.concurrent.ConcurrentHashMap
import scala.jdk.CollectionConverters.*
import scala.util.Using

case class GalleryItem(path: Path, modified: Long, size: Long, parameters: Option[String]):
  def fileName: String = path.getFileName.toString

/** Scans the outputs directory for PNG files and caches their embedded parameters.
  * Metadata is read lazily and cached by (path, mtime, size).
  */
final class GalleryIndex(root: Path, additionalRoots: () => Seq[Path] = () => Nil):
  private case class Key(path: Path, modified: Long, size: Long)
  private val cache = new ConcurrentHashMap[Path, GalleryItem]()

  def rootDir: Path = root

  /** Newest first. Skips engine temp files and the reserved root intermediate-artifact directory. */
  def scan(limit: Int = Int.MaxValue): Vector[GalleryItem] =
    val roots = (Seq(root) ++ additionalRoots()).map(_.toAbsolutePath.normalize()).distinct
    val files = roots.flatMap { scanRoot =>
      if !Files.isDirectory(scanRoot) then Vector.empty
      else Using.resource(Files.walk(scanRoot)) { s =>
        s.iterator().asScala
          .filter(p => Files.isRegularFile(p) && p.getFileName.toString.toLowerCase.endsWith(".png"))
          .filterNot(p => scanRoot.relativize(p).iterator().asScala.exists(_.toString.startsWith(".")))
          .filterNot(p => Set("intermediates", "vae-tools", "vae-recovery")(scanRoot.relativize(p).getName(0).toString))
          .toVector
      }
    }
    val items = files.distinct.flatMap(item)
    items.sortBy(i => (-i.modified, i.path.toString)).take(limit).toVector

  def item(p: Path): Option[GalleryItem] =
    try
      val mod = Files.getLastModifiedTime(p).toMillis
      val size = Files.size(p)
      val cached = cache.get(p)
      if cached != null && cached.modified == mod && cached.size == size then Some(cached)
      else if size == 0 then None
      else
        val it = GalleryItem(p, mod, size, PngMetadata.readParameters(p))
        cache.put(p, it)
        Some(it)
    catch case _: Exception => None

  def search(query: String, limit: Int = 200): Vector[GalleryItem] =
    val q = query.trim.toLowerCase
    val all = scan()
    (if q.isEmpty then all else all.filter(i => i.fileName.toLowerCase.contains(q) || i.parameters.exists(_.toLowerCase.contains(q)))).take(limit)
