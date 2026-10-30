package sdfx.ui.vm

import sdfx.core.infotext.{HiresSeedMetadata, Infotext}

/** Effective seeds from completed Hires images, independent of first-pass seed history. */
final class HiresSeedHistory(initialSeed: Option[Long] = None, initialSubseed: Option[Long] = None):
  private var lastSeed = initialSeed.filter(_ >= 0)
  private var lastSubseed = initialSubseed.filter(_ >= 0)
  def seed: Option[Long] = lastSeed
  def subseed: Option[Long] = lastSubseed
  def record(infotext: String): Unit =
    val fields = Infotext.parse(infotext).fields
    fields.get("Hires seed").flatMap(HiresSeedMetadata.parse).foreach { h =>
      val s = HiresSeedMetadata.seedValue(h("seed")).get
      val sub = HiresSeedMetadata.seedValue(h("subseed")).get
      if s >= 0 then lastSeed = Some(s)
      if sub >= 0 then lastSubseed = Some(sub)
    }
