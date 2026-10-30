package sdfx.ui.vm

/** Owns the lifetime of one preview dialog's asynchronous image selection. */
final class PreviewChoiceSession(resources: Map[String, String]):
  private var revision = 0L
  private var closed = false
  def begin(name: String): Option[(Long, String)] =
    if closed then None
    else
      revision += 1
      resources.get(name).map(revision -> _)
  def accept(candidate: Long): Boolean = !closed && candidate == revision
  def close(): Unit =
    closed = true
    revision += 1

object PreviewChoiceSession:
  def filter(options: Seq[String], query: String): Seq[String] =
    val needle = query.trim.toLowerCase(java.util.Locale.ROOT)
    options.filter(_.toLowerCase(java.util.Locale.ROOT).contains(needle))
