package sdfx.engine

import sdfx.core.modelcards.{CardPreviewMode, CardPreviewPolicy}
import java.util.concurrent.CopyOnWriteArrayList
import scala.collection.mutable
import scala.util.control.NonFatal

/** Process-service scoped temporary display choice; it never writes settings. */
final class CardPreviewSession(initialDefault: String):
  @volatile private var mode = CardPreviewMode.Hide
  private val listeners = new CopyOnWriteArrayList[CardPreviewMode => Unit]()
  private val pending = mutable.Queue.empty[CardPreviewMode]
  private var dispatching = false
  mode = CardPreviewPolicy.startup(initialDefault)

  def current: CardPreviewMode = mode

  def setTemporary(next: CardPreviewMode): Unit = synchronized {
    mode = next
    pending.enqueue(next)
    if !dispatching then
      dispatching = true
      try
        while pending.nonEmpty do
          val event = pending.dequeue()
          listeners.forEach(l => try l(event) catch case NonFatal(_) => ())
      finally dispatching = false
  }

  def addListener(l: CardPreviewMode => Unit): () => Unit =
    listeners.add(l)
    () => { listeners.remove(l); () }
