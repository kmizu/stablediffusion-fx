package sdfx.ui.vm

import java.nio.file.Path

/** Per-window only. A batch/script job may publish multiple images; it advances history once. */
final class ComparisonHistory:
  private val results = scala.collection.mutable.Map.empty[Mode, Path]
  def recordResult(job: String, path: Path, mode: Mode): Unit =
    if mode == Mode.Txt2Img then record(job, path)
    else results(mode) = path
  def latest(mode: Mode): Option[Path] = results.get(mode)
  private var lastJob: Option[String] = None
  private var previous: Option[Path] = None
  private var current: Option[Path] = None
  def record(job: String, path: Path): Unit =
    if !lastJob.contains(job) then previous = current
    lastJob = Some(job)
    current = Some(path)
  def pair: Option[(Path, Path)] = for a <- previous; b <- current yield a -> b
