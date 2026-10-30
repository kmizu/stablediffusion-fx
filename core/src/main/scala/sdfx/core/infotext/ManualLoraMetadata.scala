package sdfx.core.infotext

import sdfx.core.model.LoraRef
import sdfx.core.json.SnakePickle
import scala.util.Try

/** Manual adapters are not prompt tags: their identity must survive network-tag removal. */
object ManualLoraMetadata:
  val key = "NUI Manual LoRAs"
  def encode(refs: Seq[LoraRef]): String = SnakePickle.write(refs)
  def parse(text: String): Option[Seq[LoraRef]] =
    if text.length > 65536 then None
    else Try {
      val values = ujson.read(text).arr
      require(values.size <= 256)
      val refs = values.toSeq.map { value =>
        require(value.obj.keySet.subsetOf(Set("name", "weight", "unet", "lbw", "start", "stop")))
        val ref = SnakePickle.read[LoraRef](value.render())
        require(ref.name.nonEmpty && ref.name.length <= 4096)
        require((Seq(ref.weight) ++ ref.unet ++ ref.start ++ ref.stop).forall(_.isFinite))
        require(ref.lbw.forall(_.length <= 16384))
        ref
      }
      refs
    }.toOption
