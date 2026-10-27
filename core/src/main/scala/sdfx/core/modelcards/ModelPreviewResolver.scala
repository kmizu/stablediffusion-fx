package sdfx.core.modelcards

import java.nio.file.{Files, Path}
import java.util.Locale
import scala.collection.mutable
import scala.util.control.NonFatal

object ModelPreviewResolver:
  def resolve(id: CardId, directoryModel: Boolean, overridePath: Option[String]): Either[String, Option[PreviewReference]] =
    new ModelPreviewResolver(LocalPreviewFiles.scoped()).resolve(id, directoryModel, overridePath)

  def validateExplicit(path: Path): Either[String, PreviewReference] =
    new ModelPreviewResolver(LocalPreviewFiles.scoped()).validateExplicit(path)

/** One instance can share drive admission within one scanner operation. It never reads model weights. */
final class ModelPreviewResolver(files: LocalPreviewFiles.Scope,
    lookup: Option[(Path, String) => Either[String, Seq[Path]]] = None):
  private val fileSuffixes = Vector(".preview.png", ".preview.jpg", ".preview.jpeg", ".png", ".jpg", ".jpeg",
    ".preview.gif", ".gif", ".preview.bmp", ".bmp")
  private val directoryNames = Vector("preview.png", "preview.jpg", "preview.jpeg", "preview.gif", "preview.bmp")
  private val formats = Set("png", "jpg", "jpeg", "gif", "bmp")

  def validateExplicit(path: Path): Either[String, PreviewReference] = validated(path, None)

  def resolve(id: CardId, directoryModel: Boolean, overridePath: Option[String]): Either[String, Option[PreviewReference]] =
    try
      overridePath match
        case Some(raw) =>
          if raw == null || raw.isEmpty then Left("Preview override is empty")
          else validateExplicit(Path.of(raw)).map(Some(_))
        case None =>
          val model = Path.of(id.modelPath)
          val parent = if directoryModel then model else model.getParent
          files.validateModel(parent, directoryModel = true).flatMap { parentIdentity =>
            val names = if directoryModel then directoryNames else
              val name = model.getFileName.toString
              val dot = name.lastIndexOf('.')
              val stem = if dot > 0 then name.substring(0, dot) else name
              fileSuffixes.map(stem + _)
            val existing = if lookup.isEmpty then candidates(parent).map(Some(_)) else Right(None)
            existing.flatMap { entries =>
              var result: Either[String, Option[PreviewReference]] = Right(None)
              var done = false
              names.iterator.takeWhile(_ => !done).foreach { name =>
                val foundResult = lookup match
                  case Some(find) => find(parent, name)
                  case None => Right(entries.get.filter { entry =>
                    if files.windows then entry.getFileName.toString.equalsIgnoreCase(name)
                    else entry.getFileName.toString == name
                  })
                foundResult match
                  case Left(reason) => result = Left(reason); done = true
                  case Right(found) =>
                    val invalid = found.exists { entry =>
                      val sameParent = entry.toAbsolutePath.normalize.getParent == parent.toAbsolutePath.normalize
                      val sameName = if files.windows then entry.getFileName.toString.equalsIgnoreCase(name)
                        else entry.getFileName.toString == name
                      !sameParent || !sameName
                    }
                    if invalid then
                      result = Left("Preview lookup returned an invalid candidate")
                      done = true
                    else if found.size > 1 then
                      result = Left("Ambiguous preview filename case; choose an explicit preview")
                      done = true
                    else found.headOption.foreach { chosen =>
                      result = validated(chosen, Some(parentIdentity.realPath)).map(Some(_))
                      done = true
                    }
              }
              result
            }
          }
    catch
      case error: AssertionError => throw error
      case NonFatal(_) => Left("Preview path is invalid or unavailable")

  private def validated(path: Path, parent: Option[Path]): Either[String, PreviewReference] =
    try
      val name = Option(path.getFileName).map(_.toString).getOrElse("")
      val suffix = name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT)
      if !formats(suffix) then Left("Unsupported preview format; choose PNG, JPEG, GIF or BMP")
      else files.validate(path, parent).map { identity =>
        val selected = path.toAbsolutePath.normalize
        new PreviewReference(selected, selected.toUri.toASCIIString, identity, parent)
      }
    catch
      case error: AssertionError => throw error
      case NonFatal(_) => Left("Preview path is invalid or unavailable")

  private def candidates(parent: Path): Either[String, Vector[Path]] =
    val stream = Files.newDirectoryStream(parent)
    try
      val result = mutable.ArrayBuffer.empty[Path]
      val iterator = stream.iterator()
      var chars = 0L
      var full = false
      while iterator.hasNext && !full do
        val entry = iterator.next()
        val nextChars = entry.toString.length.toLong + entry.getFileName.toString.length.toLong
        if result.size >= 4096 || chars + nextChars > 1024 * 1024 then full = true
        else
          result += entry
          chars += nextChars
      if full then Left("Preview directory is too large; choose a narrower folder or explicit preview")
      else Right(result.toVector)
    finally stream.close()
