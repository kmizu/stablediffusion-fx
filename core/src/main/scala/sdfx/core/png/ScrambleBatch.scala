package sdfx.core.png

import java.nio.file.{FileAlreadyExistsException, Files, LinkOption, Path}
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

/** Non-destructive folder conversion for the upstream pixel-scramble PNG formats. */
object ScrambleBatch:
  enum Mode:
    case Encode, Decode
  enum Status:
    case Written, Exists, Skipped, Unverified, Failed
  final case class Result(source: Path, destination: Path, status: Status, message: String = "")

  def run(sourceDir: Path, destinationDir: Path, password: String, mode: Mode, allowUnverified: Boolean = false,
      shouldStop: () => Boolean = () => false, onResult: Result => Unit = _ => ()): Seq[Result] =
    require(password != null && password.nonEmpty, "Password must not be empty")
    require(Files.isDirectory(sourceDir), "Source folder does not exist")
    Files.createDirectories(destinationDir)
    require(!Files.isSameFile(sourceDir, destinationDir), "Source and destination must be different folders")
    val sources = Using.resource(Files.list(sourceDir))(_.iterator().asScala.filter { p =>
      Files.isRegularFile(p) && Seq(".png", ".jpg", ".jpeg", ".webp").exists(p.getFileName.toString.toLowerCase(java.util.Locale.ROOT).endsWith)
    }.toVector.sortBy(_.getFileName.toString))
    // Reserve names independent of disk state so reruns consistently report existing outputs.
    val reserved = scala.collection.mutable.HashSet.empty[String]
    val destinations = sources.map { source =>
      val name = source.getFileName.toString
      val base = name.substring(0, name.lastIndexOf('.')) + ".png"
      def key(value: String) = value.toLowerCase(java.util.Locale.ROOT)
      var candidate = base
      if reserved.contains(key(candidate)) then
        candidate = name + ".png"
        var suffix = 2
        while reserved.contains(key(candidate)) do
          candidate = name + "." + suffix + ".png"
          suffix += 1
      reserved += key(candidate)
      source -> destinationDir.resolve(candidate)
    }
    destinations.iterator.takeWhile(_ => !shouldStop()).map { (source, dst) =>
      def result(status: Status, message: String = "") = Result(source, dst, status, message)
      if Files.exists(dst, LinkOption.NOFOLLOW_LINKS) then result(Status.Exists)
      else
        try
          val text = PngMetadata.readTexts(source)
          val marker = text.get("Encrypt")
          if mode == Mode.Encode && marker.nonEmpty then result(Status.Skipped, "Already scrambled")
          else if mode == Mode.Decode && marker.isEmpty then result(Status.Skipped, "Not scrambled")
          else if mode == Mode.Decode && !text.contains("EncryptPwdSha") && !allowUnverified then
            result(Status.Unverified, "Password cannot be verified for this image")
          else
            val sourceInfo = if mode == Mode.Encode then Some(ExternalImageMetadataProbe.inspect(source)) else None
            val metadataWarning = sourceInfo.filter(info => info.format() == "jpeg" || info.format() == "webp")
              .map(_ => "JPEG/WebP metadata may be lost; the restored file is PNG, not the original bytes")
            val pixels = mode match
              case Mode.Encode => PngMetadata.readImage(source)
              case Mode.Decode => ScrambledPng.read(source, Some(password)).image
            val extra = text.toSeq.filterNot((key, _) => key == "Encrypt" || key == "EncryptPwdSha")
            val temp = Files.createTempFile(destinationDir, ".sdfx-batch-", ".part")
            try
              ScrambledPng.write(pixels, temp, extra, if mode == Mode.Encode then Some(password) else None, Some(source))
              try
                Files.move(temp, dst)
                val note = if mode == Mode.Decode && !text.contains("EncryptPwdSha") then "Password not verified"
                  else metadataWarning.getOrElse("")
                result(Status.Written, note)
              catch case _: FileAlreadyExistsException => result(Status.Exists)
            finally Files.deleteIfExists(temp)
        catch case NonFatal(e) => result(Status.Failed, Option(e.getMessage).getOrElse(e.getClass.getSimpleName).stripPrefix("requirement failed: "))
    }.map { r => onResult(r); r }.toVector
