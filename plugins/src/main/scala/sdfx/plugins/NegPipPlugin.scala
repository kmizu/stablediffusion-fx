package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** CLIP NegPiP, adapted from hako-mikan/sd-webui-negpip (AGPL-3.0). */
class NegPipPlugin extends HookPlugin:
  def id = "negpip"
  def name = "NegPiP"
  def hook = "negpip"
  override def description = "Negative weights suppress a concept in the positive prompt and boost it in the negative prompt."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("syntax", "Example: a portrait, (black hair:-2). Keys keep the concept; attention values reverse its contribution. " +
      "Ordinary positive weights are unchanged. Prompt editing and Weight Gradient can be combined."),
    Note("strength", "The effect depends on the checkpoint. Try -1, then -2 or -3 for SDXL; excessive weights may damage the image."),
    Note("compatibility", "For SD1.x, SD2.x and SDXL. Cannot process negative weights with PhotoMaker.")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj()
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("NegPiP Active" -> "True")
