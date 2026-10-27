package sdfx.engine

import sdfx.core.protocol.{EngineError, InterruptionDecision, Protocol, WdBatchResult}
import scala.concurrent.{ExecutionContext, Future}
import scala.util.control.NonFatal

/** One WD-only prepare/execute pair sharing an owned cancellation gate. */
private[engine] object WdBatchFlow:
  private given ExecutionContext = ExecutionContext.parasitic

  def run(transport: ExtrasBatchHandle, prepare: ujson.Value): Future[ujson.Value] =
    val frozen = try Right(ujson.read(ujson.write(prepare)))
      catch case NonFatal(e) => Left(e)
    val work = frozen match
      case Left(error) => Future.failed(error)
      case Right(request) =>
        transport.request("batch_interrogate_prepare", request).flatMap { result =>
          if transport.control.cancellationAccepted then
            Future.failed(EngineError(Protocol.ErrorCodes.Interrupted, "WD batch interrupted after preparation"))
          else
            val token = result.objOpt.flatMap(_.get("token")).flatMap(_.strOpt)
            if !result.objOpt.flatMap(_.get("partial")).contains(ujson.Bool(true)) ||
                !token.exists(_.matches("[0-9a-f]{32}")) then
              Future.failed(new IllegalArgumentException("invalid WD batch preparation response"))
            else transport.request("batch_interrogate_execute", ujson.Obj("token" -> token.get))
        }
    work.onComplete(_ => transport.finish())
    work

/** A partial WD job and its exact-owner cancellation handle. */
final class WdBatchHandle private[engine](val result: Future[WdBatchResult], transport: ExtrasBatchHandle,
                                        process: java.util.concurrent.atomic.AtomicReference[EngineProcess]):
  def interrupt(): Future[InterruptionDecision] = transport.cancel()
  private[engine] def control: ExtrasBatchControl = transport.control
  private[engine] def processSnapshot: Option[EngineProcess] = Option(process.get())
