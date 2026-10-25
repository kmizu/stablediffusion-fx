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
final class GalleryIndex(root: Path):
  private case class Key(path: Path, modified: Long, size: Long)
  private val cache = new ConcurrentHashMap[Path, GalleryItem]()

  def rootDir: Path = root

  /** Newest first. Skips the engine temp directory. */
  def scan(limit: Int = Int.MaxValue): Vector[GalleryItem] =
    if !Files.isDirectory(root) then Vector.empty
    else
      val files = Using.resource(Files.walk(root)) { s =>
        s.iterator().asScala
          .filter(p => Files.isRegularFile(p) && p.getFileName.toString.toLowerCase.endsWith(".png"))
          .filterNot(p => root.relativize(p).iterator().asScala.exists(_.toString.startsWith(".")))
          .toVector
      }
      val items = files.flatMap(item)
      items.sortBy(i => (-i.modified, i.path.toString)).take(limit)

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
