package sdfx.core.modelpresets

import java.awt.image.BufferedImage
import java.nio.file.{Files, LinkOption, Path, StandardCopyOption}
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import javax.imageio.ImageIO

final case class PresetModelIdentity(path: Path, sha256: String)

/** No model weights are loaded or changed. Operations bind to an explicit content identity. */
final class ModelPresetService(val directory: Path, val civitai: CivitaiPresetClient = new CivitaiPresetClient()):
  val store = new ModelPresetStore(directory)
  def identify(file: Path, check: () => Unit): PresetModelIdentity =
    check()
    val path = file.toRealPath()
    val before = Files.readAttributes(path, classOf[BasicFileAttributes])
    require(before.isRegularFile, "Select a checkpoint file")
    val digest = MessageDigest.getInstance("SHA-256")
    val input = Files.newInputStream(path)
    try
      val buffer = new Array[Byte](1024 * 1024)
      var n = input.read(buffer)
      while n >= 0 do
        check(); digest.update(buffer, 0, n); n = input.read(buffer)
    finally input.close()
    check()
    val after = Files.readAttributes(path, classOf[BasicFileAttributes])
    require(before.size == after.size && before.lastModifiedTime == after.lastModifiedTime && before.fileKey == after.fileKey,
      "Checkpoint changed while hashing; reload it")
    PresetModelIdentity(path, digest.digest().map(b => f"${b & 255}%02x").mkString)
  def load(model: PresetModelIdentity): LoadedModelPresets = store.load(model.sha256)
  def save(model: PresetModelIdentity, document: ModelPresetDocument, expected: Option[String]): LoadedModelPresets =
    store.save(model.sha256, document, expected)
  def thumbnailPath(model: PresetModelIdentity): Path =
    val name = model.path.getFileName.toString
    val dot = name.lastIndexOf('.')
    model.path.resolveSibling((if dot > 0 then name.take(dot) else name) + ".png")
  def saveThumbnail(model: PresetModelIdentity, data: Array[Byte], replace: Boolean, check: () => Unit): Path =
    check()
    val target = thumbnailPath(model)
    if !replace && Files.exists(target, LinkOption.NOFOLLOW_LINKS) then return target
    require(data.length <= 16 * 1024 * 1024, "Thumbnail input is too large")
    val stream = ImageIO.createImageInputStream(new ByteArrayInputStream(data))
    require(stream != null, "Invalid thumbnail image")
    val readers = ImageIO.getImageReaders(stream)
    require(readers.hasNext, "Unsupported thumbnail image")
    val reader = readers.next()
    val image = try
      reader.setInput(stream, true, true)
      val width = reader.getWidth(0); val height = reader.getHeight(0)
      require(width > 0 && height > 0 && width.toLong * height <= 64L * 1024 * 1024, "Thumbnail dimensions are too large")
      check(); reader.read(0)
    finally
      reader.dispose(); stream.close()
    check()
    val ratio = math.min(1.0, 300.0 / math.max(image.getWidth, image.getHeight))
    val resized = new BufferedImage(math.max(1, (image.getWidth * ratio).round.toInt),
      math.max(1, (image.getHeight * ratio).round.toInt), BufferedImage.TYPE_INT_ARGB)
    val graphics = resized.createGraphics()
    try
      graphics.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_BICUBIC)
      graphics.drawImage(image, 0, 0, resized.getWidth, resized.getHeight, null)
    finally graphics.dispose()
    val encoded = new ByteArrayOutputStream()
    require(ImageIO.write(resized, "png", encoded), "PNG encoding unavailable")
    val temp = Files.createTempFile(target.getParent, ".model-thumbnail-", ".tmp")
    try
      Files.write(temp, encoded.toByteArray)
      check()
      if replace then Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
      else
        try Files.createLink(target, temp)
        catch case _: java.nio.file.FileAlreadyExistsException => ()
      target
    finally Files.deleteIfExists(temp)
