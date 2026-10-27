package sdfx.core.protocol

import sdfx.core.json.SnakePickle
import sdfx.core.json.SnakePickle.{ReadWriter as RW, *}

/** Native-only owned processing requests; their discriminators cannot be overridden by callers. */
sealed trait TemporalLabRequest:
  def params: ujson.Obj

private object TemporalLabEncoding:
  def withOperation[T: SnakePickle.Writer](value: T, operation: String): ujson.Obj =
    val obj = Json.toJs(value).obj
    ujson.Obj.from(obj.toSeq :+ ("operation" -> ujson.Str(operation)))

final case class TemporalLabDfi(
    referenceDir: String, generatedDir: String, outputParent: String, outputName: String,
    sourceDenoise: Int = 0, tolerance: Double = 5, expand: Int = 0, smooth: Int = 1,
    interDenoise: Int = 9, interDenoiseSize: Int = 9, interDenoiseFrequency: Int = 3,
    interBlur: Double = 3, refreshFrequency: Int = 0, preserve: Double = 0, frameLimit: Int = 0
) extends TemporalLabRequest derives RW:
  def params: ujson.Obj = TemporalLabEncoding.withOperation(this, "dfi")

final case class TemporalLabPreview(
    referenceDir: String, generatedDir: String, frameIndex: Int = 1,
    sourceDenoise: Int = 0, tolerance: Double = 5, expand: Int = 0, smooth: Int = 1
) extends TemporalLabRequest derives RW:
  def params: ujson.Obj = TemporalLabEncoding.withOperation(this, "preview")

final case class TemporalLabDeflicker(
    framesDir: String, outputParent: String, outputName: String,
    blend: Double = 0, overlay: Double = 0, normalize: Boolean = false
) extends TemporalLabRequest derives RW:
  def params: ujson.Obj = TemporalLabEncoding.withOperation(this, "deflicker")

final case class TemporalLabFuse(
    styleDir: String, videoDir: String, outputParent: String, outputName: String, strength: Double = 0.5
) extends TemporalLabRequest derives RW:
  def params: ujson.Obj = TemporalLabEncoding.withOperation(this, "fuse")

final case class TemporalLabExtract(
    videoPath: String, outputParent: String, outputName: String, fps: Double = 0
) extends TemporalLabRequest derives RW:
  def params: ujson.Obj = TemporalLabEncoding.withOperation(this, "extract")

/** Optional fields correspond to the selected operation; dry-run has no output. */
final case class TemporalLabResult(
    operation: String,
    dryRun: Boolean = false,
    outputDir: Option[String] = None,
    frameCount: Option[Int] = None,
    dimensions: Seq[Int] = Nil,
    outputBytes: Option[Long] = None,
    referenceAligned: Option[Boolean] = None,
    resizedStyle: Option[Boolean] = None,
    fps: Option[Double] = None,
    sourceFps: Option[Double] = None,
    duration: Option[Double] = None,
    maskPngBase64: Option[String] = None,
    frameIndex: Option[Int] = None,
    previewDimensions: Seq[Int] = Nil,
    generatedPrevious: Option[String] = None,
    generatedCurrent: Option[String] = None
) derives RW
