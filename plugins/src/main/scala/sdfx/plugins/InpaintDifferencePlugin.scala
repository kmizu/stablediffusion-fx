package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Native inpaint mode: the ordinary init image is the altered image. */
class InpaintDifferencePlugin extends HookPlugin:
  def id = "inpaint_difference"
  def name = "Inpaint Difference"
  def hook = id
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Create an inpaint mask from a base and an altered image."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Use image difference"),
    ImageFile("base_path", "Base image"),
    Slider("threshold", "Difference threshold", 0, 1, .01, 1),
    Check("contours", "Contours only"),
    IntField("dilation", "Mask dilation", 0, 100, 0),
    IntField("erosion", "Mask erosion", 0, 100, 0),
    Text("color", "Preview color", "#ffffff"),
    Check("show_under", "Show altered image under mask", true),
    Text("painted_path", "Saved painted input", ""),
    Text("altered_path", "Saved altered input", ""))

  def hookFields(v: Values): ujson.Obj =
    val base = v.str("base_path").trim
    require(base.nonEmpty, "Choose a base image for Inpaint Difference")
    val threshold = v.str("threshold", "1").toDoubleOption.filter(x => x.isFinite && x >= 0 && x <= 1)
      .getOrElse(throw new IllegalArgumentException("Difference threshold must be 0..1"))
    def amount(key: String): Int = v.str(key, "0").toIntOption.filter(n => n >= 0 && n <= 100)
      .getOrElse(throw new IllegalArgumentException(s"$key must be 0..100"))
    ujson.Obj("base_path" -> base, "threshold" -> threshold, "contours" -> v.bool("contours"),
      "dilation" -> amount("dilation"), "erosion" -> amount("erosion"))

  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Nil
