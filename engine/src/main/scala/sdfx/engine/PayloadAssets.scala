package sdfx.engine

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, IOException, InputStream}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.Base64
import javax.imageio.ImageIO
import javax.imageio.stream.{ImageOutputStreamImpl, MemoryCacheImageInputStream}
import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal
import sdfx.core.payload.PayloadJsonBudget

/** Optional, bounded diagnostics for the exact native GenerationRequest body.
  *
  * The body itself is never rewritten. Only the audited file-image fields below are considered;
  * model paths and arbitrary hook values are never searched. Call on a background dispatch thread
  * before temporary-input cleanup. The captured bytes/hash describe this diagnostic read, and do
  * not prove which bytes the inference worker will subsequently read from an external mutable file.
  */
object PayloadAssets:
  final case class Result(assetsJson: String, warnings: Vector[String])

  private[engine] final case class Limits(
      maxSourceBytes: Int = 16 * 1024 * 1024,
      maxSide: Int = 8192,
      maxPixels: Long = 32000000L,
      maxPngBytes: Int = 4 * 1024 * 1024,
      maxDocumentBytes: Int = 16 * 1024 * 1024,
      maxAssets: Int = 256,
      maxWarnings: Int = 32,
      maxWarningChars: Int = 512,
      maxInputBodyBytes: Int = 16 * 1024 * 1024
  )

  /** OS boundary: one bounded open supplies both hashing and decoding, including growth checks. */
  private[engine] trait SourceAccess:
    def isRegularFile(path: Path): Boolean
    def size(path: Path): Long
    def open(path: Path): InputStream

  private[engine] val fileSource: SourceAccess = new SourceAccess:
    def isRegularFile(path: Path): Boolean = Files.isRegularFile(path)
    def size(path: Path): Long = Files.size(path)
    def open(path: Path): InputStream = Files.newInputStream(path)

  def collect(bodyJson: String, includeImageContents: Boolean = false): Result =
    collectWith(bodyJson, includeImageContents, Limits(), fileSource)

  private[engine] def collectWith(bodyJson: String, includeImageContents: Boolean, limits: Limits, source: SourceAccess): Result =
    val capture = new Collector(includeImageContents, limits, source)
    try
      admitBodyResult(bodyJson, limits.maxInputBodyBytes)(ujson.read(bodyJson)) match
        case Right(parsed) => capture.enumerate(parsed)
        case Left(reason) => capture.rejectInputBody(reason)
    catch case NonFatal(_) => capture.warn("Native request image diagnostic unavailable: invalid JSON or unsupported request shape")
    capture.result()

  private class DiagnosticFailure(val reason: String) extends IOException(reason)

  private class Collector(include: Boolean, limits: Limits, source: SourceAccess):
    private val descriptors = ArrayBuffer.empty[String]
    private val warningText = ArrayBuffer.empty[String]
    private var descriptorBytes = 0L
    private var omittedAssets = 0L
    private var omittedWarnings = 0L
    private var documentLimit = false

    def rejectInputBody(reason: String): Unit =
      documentLimit = true
      warn(s"Image diagnostic unavailable: input body admission $reason")

    def warn(text: String): Unit =
      if warningText.size < limits.maxWarnings then warningText += text.take(limits.maxWarningChars)
      else omittedWarnings += 1

    private def shape(path: String, expected: String): Unit =
      warn(s"${path.take(240)}: expected $expected; configured input was not read")

    private def obj(value: ujson.Value, path: String): Option[ujson.Obj] = value match
      case o: ujson.Obj => Some(o)
      case _ => shape(path, "object"); None

    private def enabled(o: ujson.Obj, path: String): Option[Boolean] = o.obj.get("enabled") match
      case None => None
      case Some(ujson.Bool(b)) => Some(b)
      case Some(_) => shape(s"$path.enabled", "boolean"); None

    /** Fixed whitelist/order, preserving original array indices and duplicate paths. */
    def enumerate(root: ujson.Value): Unit =
      obj(root, "$").flatMap(_.obj.get("params")) match
        case None => shape("$.params", "object")
        case Some(p) => obj(p, "$.params").foreach { params =>
          val init = params.obj.get("img2img") match
            case Some(o: ujson.Obj) => o.obj.get("init_image").collect { case ujson.Str(s) if s.nonEmpty => s }
            case _ => None
          params.obj.get("img2img").foreach {
            case ujson.Null => ()
            case value => obj(value, "$.params.img2img").foreach { ip =>
              ip.obj.get("init_image").foreach(v => candidate("$.params.img2img.init_image", "init_image", v, None, initPolicy = true))
              ip.obj.get("mask").foreach {
                case ujson.Null => ()
                case v => candidate("$.params.img2img.mask", "inpaint_mask", v, None)
              }
            }
          }
          params.obj.get("controlnet").foreach {
            case a: ujson.Arr => a.value.zipWithIndex.foreach { case (value, i) =>
              val path = s"$$.params.controlnet[$i]"
              obj(value, path).foreach { unit =>
                val active = enabled(unit, path)
                val image = unit.obj.getOrElse("image", ujson.Null)
                val fallback = init.map(_ => "omitted_init_image_fallback").getOrElse("omitted_no_configured_image")
                candidate(s"$path.image", "controlnet_image", image, active, fallbackStatus = Some(fallback),
                  derived = init.map(_ => "$.params.img2img.init_image"))
              }
            }
            case _ => shape("$.params.controlnet", "array")
          }
          params.obj.get("hooks").foreach { value => obj(value, "$.params.hooks").foreach { hooks =>
            Vector("reference" -> "reference_image", "revision" -> "revision_image", "ip_adapter" -> "ip_adapter_image",
              "controllllite" -> "controllllite_image").foreach { case (hook, role) =>
              hooks.obj.get(hook).foreach { value =>
                val path = s"$$.params.hooks.$hook"
                obj(value, path).foreach { config =>
                  val active = enabled(config, path)
                  val fallback = Option.when(hook == "controllllite")("omitted_pipeline_input_fallback")
                  config.obj.get("image") match
                    case Some(v) => candidate(s"$path.image", role, v, active, fallbackStatus = fallback)
                    case None if fallback.isDefined => candidate(s"$path.image", role, ujson.Null, active, fallbackStatus = fallback)
                    case None => ()
                }
              }
            }
            hooks.obj.get("photomaker").foreach { value =>
              val path = "$.params.hooks.photomaker"
              obj(value, path).foreach { config =>
                val active = enabled(config, path)
                config.obj.get("images").foreach {
                  case a: ujson.Arr => a.value.zipWithIndex.foreach { case (v, i) =>
                    candidate(s"$path.images[$i]", "photomaker_id_image", v, active)
                  }
                  case _ => shape(s"$path.images", "array of nonempty path strings")
                }
              }
            }
          }}
        }

    private def candidate(jsonPath: String, role: String, value: ujson.Value, active: Option[Boolean],
        initPolicy: Boolean = false, fallbackStatus: Option[String] = None, derived: Option[String] = None): Unit =
      // Validate every explicit element for honest overflow counts, but never inspect overflow files.
      val validPath = value match
        case ujson.Str(s) if s.nonEmpty => Some(s)
        case ujson.Null if fallbackStatus.isDefined => None
        case ujson.Str("") if fallbackStatus.isDefined => None
        case _ => shape(jsonPath, "nonempty path string"); None
      if documentLimit || descriptors.size >= limits.maxAssets then
        omittedAssets += 1
        return
      val d = ujson.Obj("json_path" -> jsonPath, "role" -> role, "input_kind" -> "configured_input")
      validPath.foreach(s => d("path") = ujson.Str(s))
      active.foreach(b => d("enabled") = ujson.Bool(b))
      if initPolicy then d("status") = "omitted_source_init_images_policy"
      else if active.contains(false) then d("status") = "omitted_inactive_configuration"
      else if validPath.isEmpty then
        val isFallback = fallbackStatus.isDefined && (value == ujson.Null || value == ujson.Str(""))
        d("status") = if isFallback then fallbackStatus.get else "invalid_input"
        if isFallback then
          derived.foreach(p => d("derived_from_json_path") = ujson.Str(p))
          if fallbackStatus.contains("omitted_pipeline_input_fallback") then d("fallback") = "pipeline_input_pixels_not_captured"
      else if !include then d("status") = "omitted_disabled_by_preference"
      else
        // Paths/descriptors themselves can exhaust the document budget; check before file operations.
        if renderObject(d, remainingDescriptorBytes).isEmpty then
          hitDocumentLimit()
          omittedAssets += 1
          return
        try includeFile(d, validPath.get)
        catch
          case _: DocumentLimit =>
            hitDocumentLimit()
            omittedAssets += 1
            return
          case e: DiagnosticFailure => failed(d, jsonPath, e.reason)
          case NonFatal(_) => failed(d, jsonPath, "image_read_or_decode_failed")
      renderObject(d, remainingDescriptorBytes) match
        case Some((text, size)) =>
          descriptors += text
          descriptorBytes += size
        case None =>
          hitDocumentLimit()
          omittedAssets += 1

    private def failed(d: ujson.Obj, path: String, reason: String): Unit =
      d("status") = "unavailable"
      d("reason") = reason
      warn(s"${path.take(240)}: $reason; image diagnostic unavailable")

    // Reserve enough for the bounded envelope/counters and inter-descriptor separators.
    private def remainingDescriptorBytes: Int =
      math.max(0L, limits.maxDocumentBytes.toLong - descriptorBytes - descriptors.size - 512).toInt

    private def hitDocumentLimit(): Unit =
      if !documentLimit then
        documentLimit = true
        omittedAssets += descriptors.size
        descriptors.clear()
        descriptorBytes = 0
        warn("Image diagnostic unavailable: final UTF-8 asset document limit exceeded")

    private def includeFile(d: ujson.Obj, literal: String): Unit =
      val path = Paths.get(literal).toAbsolutePath.normalize()
      if !source.isRegularFile(path) then throw new DiagnosticFailure("not_a_regular_file")
      val size = source.size(path)
      if size < 0 || size > limits.maxSourceBytes then throw new DiagnosticFailure("source_file_limit")
      val bytes = readBounded(path)
      val input = new MemoryCacheImageInputStream(new ByteArrayInputStream(bytes))
      try
        val readers = ImageIO.getImageReaders(input)
        if !readers.hasNext then throw new DiagnosticFailure("unsupported_or_corrupt_image")
        val reader = readers.next()
        try
          reader.setInput(input, true, true)
          val w = reader.getWidth(0)
          val h = reader.getHeight(0)
          validateDimensions(w, h)
          val img = reader.read(0)
          if img == null then throw new DiagnosticFailure("unsupported_or_corrupt_image")
          validateDimensions(img.getWidth, img.getHeight)
          val output = new CappedImageOutput(limits.maxPngBytes)
          val writers = ImageIO.getImageWritersByFormatName("png")
          if !writers.hasNext then throw new DiagnosticFailure("png_encoder_unavailable")
          val writer = writers.next()
          val png = try
            writer.setOutput(output)
            try writer.write(img)
            catch
              case e: DiagnosticFailure => throw e
              case NonFatal(e) if output.exceeded => throw new DiagnosticFailure("encoded_png_limit")
            output.bytes
          finally
            writer.dispose()
            output.close()
          d("status") = "included"
          d("mime") = "image/png"
          d("source_sha256") = sha256(bytes)
          d("png_sha256") = sha256(png)
          d("source_bytes") = ujson.Num(bytes.length)
          d("png_bytes") = ujson.Num(png.length)
          d("width") = ujson.Num(img.getWidth)
          d("height") = ujson.Num(img.getHeight)
          // Check exact base64/JSON expansion before allocating the data URL string.
          val metadataBytes = renderObject(d, remainingDescriptorBytes).map(_._2).getOrElse(throw new DocumentLimit())
          val dataUrlBytes = 22L + 4L * ((png.length.toLong + 2) / 3)
          // The comma, quoted key, colon and quoted value add 18 bytes to this flat object.
          if metadataBytes.toLong + 18 + dataUrlBytes > remainingDescriptorBytes then throw new DocumentLimit()
          d("png_data_url") = "data:image/png;base64," + Base64.getEncoder.encodeToString(png)
        finally reader.dispose()
      finally input.close()

    private def validateDimensions(w: Int, h: Int): Unit =
      if w <= 0 || h <= 0 || w > limits.maxSide || h > limits.maxSide || w.toLong * h.toLong > limits.maxPixels then
        throw new DiagnosticFailure("image_dimensions_limit")

    private def readBounded(path: Path): Array[Byte] =
      val input = source.open(path)
      try
        val output = new ByteArrayOutputStream()
        val buffer = new Array[Byte](8192)
        var total = 0
        var done = false
        while !done do
          val n = input.read(buffer, 0, math.min(buffer.length, limits.maxSourceBytes - total + 1))
          if n < 0 then done = true
          else if n == 0 then
            val single = input.read()
            if single < 0 then done = true
            else
              total += 1
              if total > limits.maxSourceBytes then throw new DiagnosticFailure("source_file_limit")
              output.write(single)
          else
            total += n
            if total > limits.maxSourceBytes then throw new DiagnosticFailure("source_file_limit")
            output.write(buffer, 0, n)
        output.toByteArray
      finally input.close()

    def result(): Result =
      def document: Option[String] =
        val header = ujson.Obj("kind" -> "native_input_diagnostic", "byte_identity" -> "diagnostic_capture_only",
          "inference_input_identity" -> "not_proven", "include_image_contents" -> include,
          "status" -> (if documentLimit then "unavailable_document_limit" else "available"),
          "omitted_asset_count" -> ujson.Num(omittedAssets.toDouble),
          "omitted_warning_count" -> ujson.Num(omittedWarnings.toDouble))
        val output = new BoundedJsonWriter(limits.maxDocumentBytes)
        renderObject(header, limits.maxDocumentBytes).foreach { case (head, _) =>
          output.appendRaw(head.dropRight(1))
          output.appendRaw(",\"assets\":[")
          descriptors.zipWithIndex.foreach { case (d, i) =>
            if i > 0 then output.appendRaw(",")
            output.appendRaw(d)
          }
          output.appendRaw("]}")
        }
        output.result.filter(_.nonEmpty)
      val text = document.getOrElse {
        hitDocumentLimit()
        document.getOrElse("{\"status\":\"unavailable_document_limit\",\"assets\":[]}")
      }
      val warningOverflow = if omittedWarnings > 0 then Vector(s"$omittedWarnings additional image diagnostic warnings omitted".take(limits.maxWarningChars)) else Vector.empty
      val warnings = warningText.toVector ++ warningOverflow
      Result(text, warnings)

  private class DocumentLimit extends IOException("asset_document_limit")

  /** Admission runs before JSON parsing and never allocates a complete UTF-8 copy of the input. */
  private[engine] def admitBody(text: CharSequence, byteLimit: Int)(parse: => ujson.Value): Option[ujson.Value] =
    admitBodyResult(text, byteLimit)(parse).toOption

  private def admitBodyResult(text: CharSequence, byteLimit: Int)(parse: => ujson.Value): Either[String, ujson.Value] =
    PayloadJsonBudget.admit(text, PayloadJsonBudget.Limits(maxBytes = byteLimit))(parse)

  /** Flat descriptor/envelope serializer. It escapes strings incrementally, rather than asking a
    * general JSON renderer to allocate the complete escaped value before applying the budget.
    */
  private def renderObject(value: ujson.Obj, budget: Int): Option[(String, Int)] =
    val output = new BoundedJsonWriter(budget)
    output.appendRaw("{")
    value.obj.iterator.zipWithIndex.foreach { case ((key, v), i) =>
      if i > 0 then output.appendRaw(",")
      output.appendQuoted(key)
      output.appendRaw(":")
      v match
        case ujson.Str(s) => output.appendQuoted(s)
        // All other descriptor fields are bounded booleans or numeric counters/dimensions.
        case ujson.Bool(b) => output.appendRaw(if b then "true" else "false")
        case ujson.Num(n) => output.appendRaw(ujson.write(ujson.Num(n)))
        case ujson.Null => output.appendRaw("null")
        case _ => throw new IllegalArgumentException("Image diagnostic descriptors must be flat")
    }
    output.appendRaw("}")
    output.result.map(s => (s, output.bytesSeen))

  /** Direct package seam lets tests prove early termination using a non-materializable virtual
    * CharSequence. bytesSeen saturates at budget+1 and no character is read after rejection.
    */
  private[engine] class BoundedJsonWriter(byteLimit: Int) extends sdfx.core.payload.PayloadJsonBudget.Writer(byteLimit)

  private def sha256(bytes: Array[Byte]): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString

  /** Seekable capped PNG output. Writes are bounded directly, including writer seeks; ImageIO's
    * default disk/memory cache cannot grow beyond the cap before flushing to an underlying stream.
    */
  private class CappedImageOutput(limit: Int) extends ImageOutputStreamImpl:
    private var buffer = new Array[Byte](math.min(8192, limit))
    private var bufferedLength = 0
    private var overflow = false
    def exceeded: Boolean = overflow

    private def check(end: Long): Unit =
      if end < 0 || end > limit then
        overflow = true
        throw new DiagnosticFailure("encoded_png_limit")
      if end > buffer.length then buffer = java.util.Arrays.copyOf(buffer, math.min(limit.toLong, math.max(end, buffer.length.toLong * 2)).toInt)

    override def write(b: Int): Unit =
      checkClosed()
      flushBits()
      check(streamPos + 1)
      buffer(streamPos.toInt) = b.toByte
      streamPos += 1
      bufferedLength = math.max(bufferedLength, streamPos.toInt)

    override def write(b: Array[Byte], off: Int, len: Int): Unit =
      checkClosed()
      if off < 0 || len < 0 || off > b.length - len then throw new IndexOutOfBoundsException()
      flushBits()
      check(streamPos + len)
      System.arraycopy(b, off, buffer, streamPos.toInt, len)
      streamPos += len
      bufferedLength = math.max(bufferedLength, streamPos.toInt)

    override def read(): Int =
      checkClosed()
      bitOffset = 0
      if streamPos >= bufferedLength then -1
      else
        val b = buffer(streamPos.toInt) & 255
        streamPos += 1
        b

    override def read(b: Array[Byte], off: Int, len: Int): Int =
      checkClosed()
      if off < 0 || len < 0 || off > b.length - len then throw new IndexOutOfBoundsException()
      bitOffset = 0
      if len == 0 then 0
      else if streamPos >= bufferedLength then -1
      else
        val n = math.min(len, bufferedLength - streamPos.toInt)
        System.arraycopy(buffer, streamPos.toInt, b, off, n)
        streamPos += n
        n

    override def seek(pos: Long): Unit =
      check(pos)
      super.seek(pos)

    override def length(): Long = bufferedLength.toLong

    def bytes: Array[Byte] = java.util.Arrays.copyOf(buffer, bufferedLength)
