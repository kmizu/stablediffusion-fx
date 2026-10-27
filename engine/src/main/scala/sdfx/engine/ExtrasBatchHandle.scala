package sdfx.engine

import scala.concurrent.Future
import sdfx.core.protocol.InterruptionDecision

final class ExtrasBatchHandle private[engine](val control: ExtrasBatchControl,
    requestFn: (String, ujson.Value) => Future[ujson.Value]):
  def request(kind: String, params: ujson.Value): Future[ujson.Value] = requestFn(kind, params)
  def cancel(): Future[InterruptionDecision] =
    Future(control.cancel())(using scala.concurrent.ExecutionContext.global).flatMap(identity)(using scala.concurrent.ExecutionContext.parasitic)
  def finish(): Unit = control.finish()
