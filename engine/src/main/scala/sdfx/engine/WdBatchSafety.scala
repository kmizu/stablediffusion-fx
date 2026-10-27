package sdfx.engine

import sdfx.core.model.GenerationParams
import sdfx.core.protocol.{EngineError, Protocol, WdBatchResult}
import sdfx.core.protocol.{GeneratedImage, WdBatchItem, InterruptionDecision}
import java.nio.file.{Files, LinkOption, Path, Paths}
import java.nio.file.attribute.BasicFileAttributes
import java.util.concurrent.atomic.AtomicLong
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.*

private[engine] object WdBatchSafety:
  def timeoutAction(handle: WdBatchHandle, control: ExtrasBatchControl, supervisor: EngineSupervisor): Unit =
    control.abortAfterTimeout()
    // Target only this owner. Supervisor escalates only when the worker reports
    // this particular request active; a queued request must not kill another job.
    handle.interrupt()
    ()

  def markQueueCancelled(decision: InterruptionDecision, wd: Option[ExtrasBatchControl]): Boolean =
    decision.accepted && wd.forall(_.cancellationAccepted)

  def validateResolved(p: GenerationParams): Unit =
    if p.hires.enabled || p.hooks.nonEmpty || p.controlnet.exists(_.enabled) || p.img2img.exists(_.mask.nonEmpty) || p.loras.nonEmpty then
      throw EngineError(Protocol.ErrorCodes.Unsupported, "WD batch currently does not support hires, hooks, ControlNet, masks or LoRA")

  def validateResult(result: WdBatchResult): Unit =
    if !result.partial || result.rows.isEmpty then
      throw EngineError(Protocol.ErrorCodes.Internal, "WD batch returned an invalid partial result")
    result.rows.foreach { row =>
      row.status match
        case "error" =>
          if row.error.forall(_.trim.isEmpty) then throw EngineError(Protocol.ErrorCodes.Internal, "WD batch error has no message")
        case "complete" =>
          if row.positive.isEmpty || row.negative.isEmpty || row.result.forall(_.images.isEmpty) then
            throw EngineError(Protocol.ErrorCodes.Internal, "WD batch returned an incomplete success")
        case _ => throw EngineError(Protocol.ErrorCodes.Internal, "WD batch returned an unknown item status")
    }
    if !result.rows.exists(_.status == "complete") then
      throw EngineError(Protocol.ErrorCodes.InvalidParams,
        s"WD batch failed: ${result.rows.flatMap(_.error).headOption.getOrElse("no images generated")}")

  def validatePaths(result: WdBatchResult, outputRoot: Path): Seq[(WdBatchItem, GeneratedImage, Path)] =
    validateResult(result)
    val root = outputRoot.toAbsolutePath.normalize()
    if !Files.isDirectory(root, LinkOption.NOFOLLOW_LINKS) || Files.isSymbolicLink(root) then
      throw EngineError(Protocol.ErrorCodes.Internal, "WD batch output directory is unavailable")
    val seenPaths = scala.collection.mutable.HashSet.empty[Path]
    val seenKeys = scala.collection.mutable.HashSet.empty[AnyRef]
    val seenAll = scala.collection.mutable.ArrayBuffer.empty[Path]
    val seenWithoutKey = scala.collection.mutable.ArrayBuffer.empty[Path]
    result.rows.filter(_.status == "complete").flatMap { row =>
      row.result.toSeq.flatMap(_.images).map { image =>
        val path = try Paths.get(image.path).toAbsolutePath.normalize()
          catch case scala.util.control.NonFatal(_) => throw EngineError(Protocol.ErrorCodes.Internal, "WD batch returned an invalid image path")
        val item = path.getParent
        if item == null || item.getParent != root || !item.getFileName.toString.matches("item-[0-9]{4}") ||
            Files.isSymbolicLink(item) || !Files.isDirectory(item, LinkOption.NOFOLLOW_LINKS) ||
            Files.isSymbolicLink(path) || !Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) || !seenPaths.add(path) then
          throw EngineError(Protocol.ErrorCodes.Internal, "WD batch returned an invalid or duplicate image path")
        val key = try Files.readAttributes(path, classOf[BasicFileAttributes], LinkOption.NOFOLLOW_LINKS).fileKey()
          catch case scala.util.control.NonFatal(_) =>
            throw EngineError(Protocol.ErrorCodes.Internal, "WD batch could not inspect published image identity")
        // Some Windows providers do not return fileKey. Compare all prior
        // paths in that case; if a previous key was unavailable, compare it
        // even when this file has a key. Fail closed if comparison fails.
        val compare = if key == null then seenAll else seenWithoutKey
        val alias = try compare.exists(previous => Files.isSameFile(previous, path))
          catch case scala.util.control.NonFatal(_) =>
            throw EngineError(Protocol.ErrorCodes.Internal, "WD batch could not compare published image identities")
        if alias || (key != null && !seenKeys.add(key)) then
          throw EngineError(Protocol.ErrorCodes.Internal, "WD batch returned an aliased or unverifiable image")
        seenAll += path
        if key == null then seenWithoutKey += path
        (row, image, path)
      }
    }

  def awaitActive[A](future: Future[A], activity: AtomicLong, timeout: FiniteDuration, stopWorker: () => Unit): A =
    val poll = if timeout < 1.second then timeout else 1.second
    while true do
      try return Await.result(future, poll)
      catch
        case _: java.util.concurrent.TimeoutException =>
          if (System.nanoTime() - activity.get()).nanos > timeout then
            stopWorker()
            throw EngineError(EngineProcess.EngineTimeout, "WD batch timed out; targeted cancellation was requested and the output outcome is unknown. Check the destination before retrying.")
    throw new IllegalStateException("unreachable")
