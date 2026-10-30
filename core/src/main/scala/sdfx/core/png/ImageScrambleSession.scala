package sdfx.core.png

/** In-memory session only; do not include snapshots in settings, diagnostics, or image metadata. */
final class ImageScrambleSecret private[png] (private val value: String, val plaintextVideoAllowed: Boolean = false):
  def use[A](f: String => A): A = f(value)
  /** A job-local choice; it is not stored in the session or generation parameters. */
  def withPlaintextVideoAllowed: ImageScrambleSecret = new ImageScrambleSecret(value, true)
  def withoutPlaintextVideoAllowed: ImageScrambleSecret = new ImageScrambleSecret(value, false)
  override def toString: String = "[image password]"

final case class ImageScrambleState(revision: Long, secret: Option[ImageScrambleSecret])

/** Snapshot at submission time, so changing or locking the UI does not alter queued jobs. */
final class ImageScrambleSession:
  private var current: Option[ImageScrambleSecret] = None
  private var revision = 0L
  def unlock(password: String): Unit = synchronized {
    require(password != null && password.nonEmpty, "Image password must not be empty")
    current = Some(new ImageScrambleSecret(password))
    revision += 1
  }
  def lock(): Unit = synchronized { current = None; revision += 1 }
  def setPlaintextVideoAllowed(allowed: Boolean): Unit = synchronized {
    require(current.nonEmpty, "Image scrambling is not enabled")
    current = current.map(secret => if allowed then secret.withPlaintextVideoAllowed else secret.withoutPlaintextVideoAllowed)
    revision += 1
  }
  def snapshot: Option[ImageScrambleSecret] = synchronized { current }
  def state: ImageScrambleState = synchronized { ImageScrambleState(revision, current) }
