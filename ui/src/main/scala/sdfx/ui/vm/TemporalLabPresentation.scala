package sdfx.ui.vm

import java.nio.file.{Files, LinkOption, Path}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import sdfx.core.protocol.TemporalLabResult

final case class TemporalInputInfo(count: Int, width: Int, height: Int)

/** Advisory header inspection only; the worker validates full input safety and decoding on submission. */
object TemporalInputInspection:
  private val extensions = Set("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff")
  private def imageSize(p: Path): Either[String, (Int, Int)] =
    val input = ImageIO.createImageInputStream(p.toFile)
    require(input != null, "Cannot open frame")
    Using.resource(input) { stream =>
      val readers = ImageIO.getImageReaders(stream)
      if !readers.hasNext then Left("Image header inspection unavailable: no Java reader; worker will validate")
      else
        val reader = readers.next()
        try
          reader.setInput(stream, true, true)
          Right((reader.getWidth(0), reader.getHeight(0)))
        finally reader.dispose()
    }
  def inspect(path: String): Either[String, TemporalInputInfo] = inspect(path, () => false, imageSize)
  def inspect(path: String, cancelled: () => Boolean): Either[String, TemporalInputInfo] = inspect(path, cancelled, imageSize)
  def inspect(path: String, cancelled: () => Boolean, readSize: Path => Either[String, (Int, Int)]): Either[String, TemporalInputInfo] =
    try
      def check(): Unit = if cancelled() then throw new InterruptedException("Inspection superseded or closed")
      check()
      require(path != null && path.trim.nonEmpty, "Select a folder")
      val dir = Path.of(path)
      require(Files.isDirectory(dir, LinkOption.NOFOLLOW_LINKS), "Not a regular directory")
      val files = Using.resource(Files.list(dir)) { stream =>
        val iter = stream.iterator.asScala
        val builder = Vector.newBuilder[Path]
        var count = 0
        while iter.hasNext do
          check()
          val p = iter.next(); count += 1
          require(count <= 20000, "Too many directory entries")
          if extensions.contains(p.getFileName.toString.split("\\.").last.toLowerCase(java.util.Locale.ROOT)) then builder += p
        builder.result()
      }
      require(files.nonEmpty && files.size <= 10000, "No frames or too many frames")
      var dimensions = Option.empty[(Int, Int)]
      files.foreach { p =>
        check()
        require(Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS), "Frame is not a regular file")
        val size = readSize(p) match
          case Right(value) => value
          case Left(message) => throw new IllegalArgumentException(message)
        check()
        require(size._1 > 0 && size._2 > 0 && size._1 <= 16384 && size._2 <= 16384 && size._1.toLong * size._2 <= 64000000L, "Invalid frame dimensions")
        require(dimensions.forall(_ == size), "Frame dimensions differ")
        dimensions = Some(size)
      }
      val (w, h) = dimensions.get
      require(w.toLong * h * files.size <= 2000000000L, "Total frame pixels exceed limit")
      Right(TemporalInputInfo(files.size, w, h))
    catch
      case e: InterruptedException => Left(e.getMessage)
      case NonFatal(e) => Left(Option(e.getMessage).getOrElse(e.toString))

/** One running and one latest pending scan; methods and completions run on the UI thread. */
final class TemporalInspectionQueue(
    background: (() => Unit) => Unit,
    onUi: (() => Unit) => Unit,
    inspect: (String, () => Boolean) => Either[String, TemporalInputInfo]):
  @volatile private var revision = 0L
  private var running = false
  @volatile private var closed = false
  private var pending = Option.empty[(Long, String, Either[String, TemporalInputInfo] => Unit)]
  def request(path: String, deliver: Either[String, TemporalInputInfo] => Unit): Unit =
    if !closed then
      revision += 1
      pending = Some((revision, path, deliver))
      start()
  private def start(): Unit =
    if !closed && !running then pending.foreach { (id, path, deliver) =>
      pending = None; running = true
      background(() => {
        val result = try inspect(path, () => closed || id != revision) catch case NonFatal(e) => Left(Option(e.getMessage).getOrElse(e.toString))
        onUi(() => {
          running = false
          if !closed && id == revision then deliver(result)
          start()
        })
      })
    }
  def dispose(): Unit = { closed = true; pending = None; revision += 1 }

object TemporalLabPresentation:
  val dfiHelp = "Reference frames are center-cropped to the generated aspect ratio and resized when dimensions differ. Inter blur is rounded to the nearest integer (ties to even), then even values use the next odd kernel. This independent implementation applies blur; the upstream control did not affect output."
  val deflickerHelp = "Stages run in order: blend, overlay, normalize. Zero strengths and unchecked normalize copy the frames. Overlay composites the first pair and uses that result twice for the first two output frames; later outputs use adjacent pairs."
  def dfiInputs(a: TemporalInputInfo, b: TemporalInputInfo): String =
    val mismatch = if a.count != b.count then " Counts differ; processing requires equal counts." else ""
    val alignment = if a.width != b.width || a.height != b.height then " Reference will be center-cropped as needed and resized to generated dimensions." else ""
    s"Reference: ${a.count} frames, ${a.width} × ${a.height}; generated: ${b.count} frames, ${b.width} × ${b.height}.$mismatch$alignment Header inspection only; worker validates all inputs."
  def completion(result: TemporalLabResult): String =
    if result.dryRun then "Dry run: request validated; no output created."
    else result.outputDir.map { p =>
      val details = if result.operation == "extract" then
        s"; source FPS: ${result.sourceFps.map(_.toString).getOrElse("unknown")}; output FPS: ${result.fps.map(_.toString).getOrElse("unknown")}; duration: ${result.duration.map(_.toString).getOrElse("unknown")} s"
      else if result.referenceAligned.contains(true) then "; reference was center-cropped as needed and resized to generated dimensions"
      else ""
      s"Complete: $p (${result.frameCount.getOrElse(0)} frames)$details"
    }.getOrElse("Preview ready")
