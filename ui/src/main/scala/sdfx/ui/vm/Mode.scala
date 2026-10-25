package sdfx.ui.vm

import sdfx.core.plugins.api.Tab

/** The three generation tabs. Inpaint is img2img with a mask, so it shares the img2img plugins. */
enum Mode(val key: String, val label: String):
  case Txt2Img extends Mode("txt2img", "txt2img")
  case Img2Img extends Mode("img2img", "img2img")
  case Inpaint extends Mode("inpaint", "Inpaint")

  def pluginTab: Tab = if this == Mode.Txt2Img then Tab.Txt2Img else Tab.Img2Img
  def usesInitImage: Boolean = this != Mode.Txt2Img

object Mode:
  def fromKey(k: String): Option[Mode] = values.find(_.key == k)
