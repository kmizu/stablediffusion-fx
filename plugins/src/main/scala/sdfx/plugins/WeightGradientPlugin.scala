package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Weight Gradient, reimplemented from the unlicensed upstream project's documented syntax and examples. */
class WeightGradientPlugin extends HookPlugin:
  def id = "weight_gradient"
  def name = "Weight Gradient"
  def hook = "weight_gradient"
  override def description = "Changes the weight of words during sampling. Works in positive and negative prompts; " +
    "gradients restart in each pass, including Hires. fix."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("syntax", "Use {words : first-last step : first-last[-return] weight : curve}. " +
      "Example: {red scarf : 0.1-0.8 : 0-1.5 : e}. Leave the step range empty for the whole pass: {red scarf::0-1.5}."),
    Note("curves", "Curves: a = linear; e = exponential, c = circular (ease in/out). " +
      "Use ei/ci for a slow start, eo/co for a slow finish. Fractions such as 0.1, or 10%, refer to a fraction of the steps."),
    Note("compatibility", "Cannot be combined with PhotoMaker.")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj()
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Weight Gradient" -> "True")
