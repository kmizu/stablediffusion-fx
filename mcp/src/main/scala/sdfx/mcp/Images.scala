package sdfx.mcp

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File}
import java.nio.file.{Files, Path, Paths}
import java.util.Base64
import javax.imageio.ImageIO
import javax.imageio.stream.ImageInputStream
import scala.util.control.NonFatal

/** Which local files MCP clients may name by path: the outputs folder (incl. its tmp dir) plus extra folders
  * (`--allow-dir`, `SDFX_MCP_ALLOWED_DIRS`). Real paths are compared (symlinks / junctions cannot escape) and
  * UNC / network paths are rejected before touching the filesystem (no SMB credential leaks).
  */
final class PathPolicy(roots: () => Seq[Path]):
  def allowedRoots: Seq[Path] = roots().map(_.toAbsolutePath.normalize()).distinct

  private def denied(field: String): ToolError =
    ToolError(s"$field: no such file inside the allowed folders (${allowedRoots.mkString(", ")}); pass base64 or a data URI for other images")

  /** Resolves `spec` (absolute, or relative to `base`) to a real path inside an allowed root, or throws. The
    * message is the same whether the file is missing or outside the roots (no existence oracle).
    */
  def resolve(spec: String, field: String, base: Path): Path =
    val s = spec.trim
    if PathPolicy.isNetworkPath(s) then throw ToolError(s"$field: network (UNC) paths are not allowed")
    val p =
      try Paths.get(s)
      catch case NonFatal(_) => throw denied(field)
    val abs = (if p.isAbsolute then p else base.resolve(p)).normalize()
    val rs = allowedRoots
    if !abs.isAbsolute || !rs.exists(r => abs.startsWith(r)) then throw denied(field)
    val real =
      try abs.toRealPath()
      catch case NonFatal(_) => throw denied(field)
    val realRoots = rs.flatMap(r =>
      try Some(r.toRealPath())
      catch case NonFatal(_) => None
    )
    if !realRoots.exists(r => real.startsWith(r) && real != r) || !Files.isRegularFile(real) then throw denied(field)
    real

object PathPolicy:
  val EnvVar = "SDFX_MCP_ALLOWED_DIRS"

  def isNetworkPath(s: String): Boolean =
    val t = s.trim
    t.startsWith("\\\\") || t.startsWith("//") || t.startsWith("\\/") || t.startsWith("/\\") ||
    t.toLowerCase.startsWith("file:") || t.toLowerCase.startsWith("smb:")

  /** Extra allowed folders from the environment (path-separator separated). */
  def fromEnv(): Seq[Path] =
    Option(System.getenv(EnvVar)).toSeq.flatMap(_.split(File.pathSeparator)).map(_.trim).filter(_.nonEmpty)
      .filterNot(isNetworkPath).flatMap(d =>
        try Some(Paths.get(d).toAbsolutePath.normalize())
        catch case NonFatal(_) => None
      )

/** Image helpers: decoding tool inputs (path / base64 / data URI) and encoding downscaled previews. */
object Images:
  final case class Encoded(base64: String, mimeType: String, width: Int, height: Int)

  /** Input images larger than this per side are refused (checked from the header, before decoding). */
  val MaxInputSide = 4096
  val MaxInputBytes: Int = 48 * 1024 * 1024
  /** Images beyond this are never decoded for previews. */
  val MaxDecodeSide = 16384

  private val DataUri = """(?s)^data:([\w/+.-]*)(;[^,]*)?,(.*)$""".r

  /** Heuristic: base64 never contains '\', ':' or '.'; PNG base64 starts with "iVBOR", JPEG with "/9j/". */
  def looksLikePath(s: String): Boolean =
    s.length < 4096 && !s.contains('\n') &&
    (s.contains('\\') || s.contains(':') || s.contains('.') || (s.startsWith("/") && !s.startsWith("/9j/") && s.length < 1024))

  /** Decodes `spec` into an image file the engine can read. Paths must pass `policy` (relative paths resolve
    * against `base`); base64 / data URIs are written as PNG via `tempFile` (returned so the caller deletes them).
    */
  def materialize(spec: String, field: String, policy: PathPolicy, base: Path, tempFile: String => Path): (Path, Option[Path]) =
    val s = spec.trim
    if s.isEmpty then throw ToolError(s"$field is empty")
    s match
      case DataUri(_, meta, data) =>
        if !Option(meta).exists(_.contains("base64")) then throw ToolError(s"$field: only base64 data URIs are supported")
        writeDecoded(decodeB64(data, field), field, tempFile)
      case _ if PathPolicy.isNetworkPath(s) || looksLikePath(s) =>
        val p = policy.resolve(s, field, base)
        checkDims(dimensions(p), field)
        val img =
          try ImageIO.read(p.toFile)
          catch case NonFatal(_) => null
        if img == null then throw ToolError(s"$field: not a supported image file")
        (p, None)
      case _ => writeDecoded(decodeB64(s, field), field, tempFile)

  private def decodeB64(data: String, field: String): Array[Byte] =
    val clean = data.replaceAll("\\s", "")
    if clean.length.toLong * 3 / 4 > MaxInputBytes then throw ToolError(s"$field: image data larger than ${MaxInputBytes / 1024 / 1024} MB")
    try Base64.getDecoder.decode(clean)
    catch case NonFatal(_) => throw ToolError(s"$field is neither a file path nor valid base64 image data")

  private def checkDims(d: Option[(Int, Int)], field: String): Unit = d match
    case None => throw ToolError(s"$field: not a supported image (PNG/JPEG/BMP/GIF)")
    case Some((w, h)) =>
      if w <= 0 || h <= 0 || w > MaxInputSide || h > MaxInputSide then
        throw ToolError(s"$field: ${w}x$h is too large (max ${MaxInputSide}x$MaxInputSide)")

  private def writeDecoded(bytes: Array[Byte], field: String, tempFile: String => Path): (Path, Option[Path]) =
    checkDims(dimensions(bytes), field)
    val img =
      try ImageIO.read(new ByteArrayInputStream(bytes))
      catch case NonFatal(_) => null
    if img == null then throw ToolError(s"$field: decoded data is not a PNG/JPEG/BMP/GIF image")
    val f = tempFile(".png")
    ImageIO.write(toRgbOrArgb(img), "png", f.toFile)
    (f.toAbsolutePath, Some(f))

  /** Width/height from the image header without decoding the pixels. */
  def dimensions(p: Path): Option[(Int, Int)] =
    try withStream(ImageIO.createImageInputStream(p.toFile))
    catch case NonFatal(_) => None

  def dimensions(bytes: Array[Byte]): Option[(Int, Int)] =
    try withStream(ImageIO.createImageInputStream(new ByteArrayInputStream(bytes)))
    catch case NonFatal(_) => None

  private def withStream(iis: ImageInputStream): Option[(Int, Int)] =
    if iis == null then None
    else
      try
        val it = ImageIO.getImageReaders(iis)
        if !it.hasNext then None
        else
          val r = it.next()
          try
            r.setInput(iis, true, true)
            Some((r.getWidth(0), r.getHeight(0)))
          finally r.dispose()
      finally iis.close()

  def size(p: Path): (Int, Int) = dimensions(p).getOrElse(throw ToolError(s"not an image: $p"))

  private def toRgbOrArgb(img: BufferedImage): BufferedImage =
    if img.getType == BufferedImage.TYPE_INT_RGB || img.getType == BufferedImage.TYPE_INT_ARGB || img.getType == BufferedImage.TYPE_3BYTE_BGR || img.getType == BufferedImage.TYPE_4BYTE_ABGR then img
    else
      val out = new BufferedImage(img.getWidth, img.getHeight, if img.getColorModel.hasAlpha then BufferedImage.TYPE_INT_ARGB else BufferedImage.TYPE_INT_RGB)
      val g = out.createGraphics()
      try g.drawImage(img, 0, 0, null)
      finally g.dispose()
      out

  def mimeOf(p: Path): String =
    val n = p.getFileName.toString.toLowerCase
    if n.endsWith(".jpg") || n.endsWith(".jpeg") then "image/jpeg"
    else if n.endsWith(".webp") then "image/webp"
    else if n.endsWith(".gif") then "image/gif"
    else "image/png"

  /** Base64 of `p`; images larger than `maxSide` (0 = unlimited) are downscaled and re-encoded as PNG. The size is
    * read from the header first; files already small enough are sent as-is without decoding.
    */
  def encode(p: Path, maxSide: Int): Encoded =
    val (w, h) = dimensions(p).getOrElse(throw ToolError(s"not a readable image: $p"))
    if maxSide <= 0 || math.max(w, h) <= maxSide then
      Encoded(Base64.getEncoder.encodeToString(Files.readAllBytes(p)), mimeOf(p), w, h)
    else if math.max(w, h) > MaxDecodeSide then throw ToolError(s"$p is too large to preview (${w}x$h)")
    else
      val img = ImageIO.read(p.toFile)
      if img == null then throw ToolError(s"not a readable image: $p")
      val s = maxSide.toDouble / math.max(w, h)
      val nw = math.max(1, math.round(w * s).toInt)
      val nh = math.max(1, math.round(h * s).toInt)
      val out = new BufferedImage(nw, nh, if img.getColorModel.hasAlpha then BufferedImage.TYPE_INT_ARGB else BufferedImage.TYPE_INT_RGB)
      val g = out.createGraphics()
      try
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
        g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
        g.drawImage(img, 0, 0, nw, nh, null)
      finally g.dispose()
      val bos = new ByteArrayOutputStream()
      ImageIO.write(out, "png", bos)
      Encoded(Base64.getEncoder.encodeToString(bos.toByteArray), "image/png", nw, nh)

  /** Upper bound of the base64 size [[encode]] would produce, without reading the pixels. */
  def estimateBase64(p: Path, maxSide: Int): Long =
    dimensions(p) match
      case None => Long.MaxValue
      case Some((w, h)) =>
        if maxSide <= 0 || math.max(w, h) <= maxSide then (Files.size(p) + 2) / 3 * 4
        else
          val s = maxSide.toDouble / math.max(w, h)
          // re-encoded PNG: at most ~4 bytes per pixel plus overhead
          (math.round(w * s) * math.round(h * s) * 4 + 1024) * 4 / 3

  def deleteQuietly(p: Path): Unit =
    try Files.deleteIfExists(p)
    catch case NonFatal(_) => ()
