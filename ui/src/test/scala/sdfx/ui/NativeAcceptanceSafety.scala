package sdfx.ui

import javafx.scene.image.Image
import javafx.scene.input.{Clipboard, DataFormat}
import java.nio.ByteBuffer
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** Manual acceptance must leave the user's clipboard alone unless every advertised format is preservable. */
object NativeAcceptanceSafety:
  type Snapshot = java.util.HashMap[DataFormat, Object]
  def capture(formats: Seq[DataFormat], read: DataFormat => Object): Either[String, Snapshot] =
    val saved = new Snapshot()
    try
      val iterator = formats.iterator
      while iterator.hasNext do
        val format = iterator.next()
        val value = read(format)
        val safe = value match
          case null => false
          case _: String => true
          case _: ByteBuffer => true
          case image: Image => !image.isError && image.getProgress >= 1
          case files: java.util.List[?] if format == DataFormat.FILES => files.asScala.forall(_.isInstanceOf[java.io.File])
          case _ => false
        if !safe then return Left("an advertised clipboard format is null or cannot be safely restored")
        val stable: Object = value match
          case buffer: ByteBuffer =>
            val copy = buffer.duplicate()
            val bytes = new Array[Byte](copy.remaining())
            copy.get(bytes)
            ByteBuffer.wrap(bytes)
          case files: java.util.List[?] => new java.util.ArrayList[Object](files.asInstanceOf[java.util.List[Object]])
          case other => other
        saved.put(format, stable)
      Right(saved)
    catch case NonFatal(_) => Left("an advertised clipboard format could not be read")

  /** Call on the FX thread; no clipboard writes are used to probe safety. */
  def captureClipboard(): Either[String, Snapshot] =
    try
      val cb = Clipboard.getSystemClipboard
      capture(cb.getContentTypes.asScala.toSeq, cb.getContent)
    catch case NonFatal(_) => Left("clipboard formats could not be enumerated")

  def restore(saved: Snapshot): Unit =
    val cb = Clipboard.getSystemClipboard
    if saved.isEmpty then cb.clear()
    else require(cb.setContent(saved), "Original clipboard could not be restored")

  /** Every resource gets its cleanup even when an earlier cleanup throws. Preserve the original test failure. */
  def cleanup(primary: Throwable)(actions: (() => Unit)*): Unit =
    var failure = primary
    actions.foreach { action =>
      try action()
      catch case error: Throwable =>
        if failure == null then failure = error
        else if failure ne error then failure.addSuppressed(error)
    }
    if primary == null && failure != null then throw failure
