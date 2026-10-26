package sdfx.ui

import sdfx.engine.JobEvent

/** Payload documents are already observed through the bounded store subscription. Keep diagnostic
  * event bodies and completion JSON out of pending FX callbacks altogether.
  */
private[ui] object QueueEventFxDispatch:
  def forward(event: JobEvent, postFx: (() => Unit) => Unit,
      onImage: JobEvent.ImageSaved => Unit, onOrdinary: JobEvent => Unit): Unit = event match
    case _: JobEvent.PayloadCaptured | _: JobEvent.PayloadLifecycle => ()
    case saved: JobEvent.ImageSaved => onImage(saved)
    case other => postFx(() => onOrdinary(other))
