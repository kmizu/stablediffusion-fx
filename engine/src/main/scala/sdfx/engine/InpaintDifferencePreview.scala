package sdfx.engine

import java.nio.file.{Files, LinkOption, Path}
import scala.concurrent.{ExecutionContext, Future}

object InpaintDifferencePreview:
  final case class Artifact private[engine](directory: Path, mask: Path, preview: Path, width: Int, height: Int):
    /** Never recurse: a replaced directory or unexpected file is not ours to remove. */
    def cleanup(): Unit =
      if !Files.isSymbolicLink(directory) && Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS) then
        Files.deleteIfExists(mask)
        Files.deleteIfExists(preview)
        try Files.deleteIfExists(directory)
        catch case _: java.nio.file.DirectoryNotEmptyException => ()

  def accept(root: Path, value: ujson.Value): Artifact =
    val parent = root.toAbsolutePath.normalize()
    val dir = Path.of(value("artifact_dir").str).toAbsolutePath.normalize()
    val mask = Path.of(value("mask_path").str).toAbsolutePath.normalize()
    val preview = Path.of(value("preview_path").str).toAbsolutePath.normalize()
    val width = value("width").num
    val height = value("height").num
    require(dir.getParent == parent && dir.getFileName.toString.startsWith("inpaint-difference-") &&
      !Files.isSymbolicLink(dir) && Files.isDirectory(dir, LinkOption.NOFOLLOW_LINKS), "Invalid preview directory")
    require(mask == dir.resolve("mask.png") && preview == dir.resolve("preview.png") &&
      !Files.isSymbolicLink(mask) && !Files.isSymbolicLink(preview) &&
      Files.isRegularFile(mask, LinkOption.NOFOLLOW_LINKS) && Files.isRegularFile(preview, LinkOption.NOFOLLOW_LINKS),
      "Invalid preview images")
    require(width.isFinite && height.isFinite && width > 0 && height > 0 && width <= 16777216 &&
      height <= 16777216 && width * height <= 16777216 && width == math.floor(width) && height == math.floor(height),
      "Invalid preview dimensions")
    Artifact(dir, mask, preview, width.toInt, height.toInt)

  def run(services: AppServices, base: String, altered: String, output: Path, options: ujson.Obj)
      (using ec: ExecutionContext): Future[Artifact] =
    Files.createDirectories(output)
    val params = ujson.Obj.from(options.value.toSeq)
    params("base_path") = base
    params("altered_path") = altered
    params("output_dir") = output.toAbsolutePath.normalize().toString
    services.client.raw("inpaint_difference_preview", params).map(accept(output, _))
