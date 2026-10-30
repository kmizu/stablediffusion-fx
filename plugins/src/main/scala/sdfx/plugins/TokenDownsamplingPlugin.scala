package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object TokenDownsamplingPlugin:
  def replay(fields: Map[String, String]): Option[Values] =
    val keys = Seq(("ToDo factor", "factor", 1.0, 5.0), ("ToDo factor2", "factor2", 1.0, 5.0),
      ("ToDo disable after", "disable_after", 0.0, 1.0))
    if !keys.exists(k => fields.contains(k._1)) then Some(Values(Map("enabled" -> "false")))
    else
      val parsed = keys.map { (label, key, low, high) =>
        fields.getOrElse(label, "1").toDoubleOption.filter(v => v.isFinite && v >= low && v <= high).map(v => key -> v.toString)
      }
      if parsed.forall(_.isDefined) then Some(Values(parsed.flatten.toMap.updated("enabled", "true"))) else None

class TokenDownsamplingPlugin extends HookPlugin:
  def id = "token_downsampling"
  def name = "Token Downsampling"
  def hook = id
  override def description = "Reduce self-attention keys/values to trade detail for speed. 1 disables a depth. " +
    "The cutoff counts UNet evaluations and resets each pass. Cannot combine with PAG or HyperTile."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Slider("factor", "Downsampling factor", 1, 5, .5, 1),
    Slider("factor2", "Deeper-layer factor", 1, 5, .5, 1),
    Slider("disable_after", "Disable after (fraction)", 0, 1, .1, 1))
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "factor" -> v.double("factor", 1), "factor2" -> v.double("factor2", 1),
    "disable_after" -> v.double("disable_after", 1))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq(
    "ToDo factor" -> field(o, "factor"), "ToDo factor2" -> field(o, "factor2"),
    "ToDo disable after" -> field(o, "disable_after"))
  override def xyzAxes: Seq[XyzAxis] = Seq(
    numAxis("[ToDo] Downsampling factor", "factor"),
    numAxis("[ToDo] Downsampling factor 2", "factor2"),
    numAxis("[ToDo] Disable after", "disable_after"))
