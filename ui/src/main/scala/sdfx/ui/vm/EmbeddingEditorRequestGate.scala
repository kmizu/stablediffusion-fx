package sdfx.ui.vm

/** Own one editor request; only its matching reply may mutate the draft or unlock controls. */
final class EmbeddingEditorRequestGate(setBusy: Boolean => Unit):
  private var serial = 0L
  private var active: Option[Long] = None

  def begin(): Option[Long] =
    if active.nonEmpty then None
    else
      serial += 1
      active = Some(serial)
      setBusy(true)
      active

  def settle(token: Long)(applyReply: => Unit): Boolean =
    if !active.contains(token) then false
    else
      try applyReply
      finally
        active = None
        setBusy(false)
      true
