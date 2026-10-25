package sdfx.core.gallery

import java.nio.file.{Files, Path}
import java.time.LocalDate
import scala.jdk.CollectionConverters.*
import scala.util.Using

/** A1111-like file naming: `<outputs>/<kind>/<yyyy-mm-dd>/<00012>-<seed>.png`. Thread-safe per JVM. */
object OutputNaming:
  // Any number of digits: A1111 keeps counting past 99999 ("100000-...").
  private val Counter = """^(\d+)-.*""".r

  def dayDir(outputs: Path, kind: String, date: LocalDate = LocalDate.now()): Path =
    val d = outputs.resolve(kind).resolve(date.toString)
    Files.createDirectories(d)
    d

  def nextIndex(dir: Path): Int =
    if !Files.isDirectory(dir) then 0
    else
      Using.resource(Files.list(dir)) { s =>
        s.iterator().asScala
          .map(_.getFileName.toString)
          .collect { case Counter(n) if n.length < 10 => n.toInt }
          .maxOption
          .map(_ + 1)
          .getOrElse(0)
      }

  def nextFile(outputs: Path, kind: String, seed: Long, suffix: String = ""): Path = synchronized {
    val dir = dayDir(outputs, kind)
    val idx = nextIndex(dir)
    val f = dir.resolve(String.format(java.util.Locale.ROOT, "%05d", Int.box(idx)) + s"-$seed$suffix.png")
    Files.createFile(f) // reserve the name
    f
  }
