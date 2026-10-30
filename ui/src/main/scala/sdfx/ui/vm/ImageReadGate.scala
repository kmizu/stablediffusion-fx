package sdfx.ui.vm

import sdfx.core.png.{ImageScrambleSecret, ImageScrambleSession}
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicLong
import scala.concurrent.{ExecutionContext, Future}
import scala.util.Try

/** Delivers only the newest image read for the current session revision. */
final class ImageReadGate[A](session: ImageScrambleSession, deliver: (() => Unit) => Unit,
    read: (Path, Option[ImageScrambleSecret]) => A)(using ec: ExecutionContext):
  private val generation = new AtomicLong(0L)
  def invalidate(): Unit = { generation.incrementAndGet(); () }
  def request(path: Path)(done: Try[A] => Unit): Unit =
    val ticket = generation.incrementAndGet()
    val state = session.state
    Future(read(path, state.secret)).onComplete { result =>
      deliver(() => if generation.get() == ticket && session.state.revision == state.revision then done(result))
    }(using ExecutionContext.parasitic)
