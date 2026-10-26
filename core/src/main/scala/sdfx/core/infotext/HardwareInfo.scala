package sdfx.core.infotext

import sdfx.core.json.SnakePickle.{ReadWriter => RW, *}
import scala.collection.immutable.ListMap

/** Image provenance only: none of these values are sampling or device-selection instructions. */
object HardwareInfo:
  case class Options(enabled: Boolean = false, remark: String = "") derives RW
  case class Record(hardwareInfo: String, timeTaken: String, remark: Option[String] = None) derives RW
  case class Difference(currentDevice: String, imageDevice: String)

  val fieldNames: Set[String] = Set("Hardware Info", "Remark", "Time taken")
  def fields(record: Option[Record]): ListMap[String, String] = record.fold(ListMap.empty[String, String]) { r =>
    ListMap.from(Seq("Hardware Info" -> Some(r.hardwareInfo), "Remark" -> r.remark.filter(_.nonEmpty),
      "Time taken" -> Some(r.timeTaken)).collect { case (key, Some(value)) => key -> value })
  }

  /** Shared by all native Restore/Send-to entrypoints in one window; never changes local hardware/settings. */
  final class PasteNotices:
    private var shown = 0
    private def device(text: String): Option[String] =
      Option(text).map(_.takeWhile(_ != ',').trim).filter(s => s.nonEmpty && !s.equalsIgnoreCase("unknown"))
    def pasted(local: String, imported: Option[String]): Option[Difference] = synchronized {
      val difference = for
        current <- device(local)
        image <- imported.flatMap(device)
        if image != current && shown < 2
      yield Difference(current, image)
      if difference.nonEmpty then shown += 1
      difference
    }
