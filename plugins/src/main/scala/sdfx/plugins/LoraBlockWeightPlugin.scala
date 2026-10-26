package sdfx.plugins

import sdfx.core.model.{GenerationParams, LoraSyntax}
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** LoRA Block Weight (hako-mikan/sd-webui-lora-block-weight), re-implemented.
  *
  * The engine understands `<lora:name:w:lbw=PRESET|w1,w2,...:start=N:stop=N>` (python/sdfx/lora_blocks.py); this
  * plugin adds the same arguments from the UI, per LoRA name or for every LoRA (`*`), so prompts stay readable and
  * X/Y/Z can compare presets. Tags written in the prompt by hand win over the rules.
  */
class LoraBlockWeightPlugin extends Plugin:
  import LoraBlockWeightPlugin.*
  def id = "lora_block_weight"
  def name = "LoRA Block Weight"
  def kind = PluginKind.AlwaysOn
  override def description =
    "Weights each U-Net block of a LoRA separately (presets or explicit weights) and limits a LoRA to a step range."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Text("rules", "Rules", multiline = true,
      placeholder = "one per line:  LoRA name or *  :  preset or weights  [: start=N] [: stop=N]\ne.g.  * : OUTALL\n      myStyle : 1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0 : stop=0.5"),
    Section("help", "Presets and block layouts", expanded = false),
    Note("presets", s"Presets: ${Presets.mkString(", ")}. Own presets: config/lbwpresets.txt, one NAME:w1,w2,... per line."),
    Note("layout15", "SD1.x (17): BASE IN01 IN02 IN04 IN05 IN07 IN08 MID OUT03 OUT04 OUT05 OUT06 OUT07 OUT08 OUT09 OUT10 OUT11"),
    Note("layoutxl", "SDXL (12): BASE IN04 IN05 IN07 IN08 MID OUT00 OUT01 OUT02 OUT03 OUT04 OUT05"),
    Note("steps", "start / stop: a step number, or a fraction of the steps between 0 and 1 (stop=0.4 = 40 %). BASE = text encoder.")
  )

  override def isActive(v: Values): Boolean = v.bool("enabled") && parseRules(v.str("rules")).nonEmpty

  override def process(p: GenerationParams, v: Values): GenerationParams =
    val rules = parseRules(v.str("rules"))
    if rules.isEmpty then p
    else p.copy(prompt = apply(p.prompt, rules), hires = p.hires.copy(prompt = apply(p.hires.prompt, rules)))

  override def xyzAxes: Seq[XyzAxis] = Seq(
    // values are preset names, or weights separated by spaces (commas separate the axis values); works without
    // enabling the plugin and overrides lbw= arguments already in the prompt
    XyzAxis("LoRA block weight (all LoRAs)") { (p, s) =>
      val r = Seq(Rule("*", s.trim.split("\\s+").mkString(","), None, None))
      p.copy(prompt = apply(p.prompt, r, force = true), hires = p.hires.copy(prompt = apply(p.hires.prompt, r, force = true)))
    }
  )

object LoraBlockWeightPlugin:
  val Presets: Seq[String] = Seq("NONE", "ALL", "ALL0.5", "INS", "IND", "INALL", "MIDD", "OUTD", "OUTS", "OUTALL")

  final case class Rule(target: String, weights: String, start: Option[String], stop: Option[String])

  def parseRules(text: String): Seq[Rule] =
    text.linesIterator.map(_.trim).filter(l => l.nonEmpty && !l.startsWith("#")).flatMap { l =>
      val parts = l.split(':').map(_.trim).toSeq
      if parts.size < 2 || parts.head.isEmpty || parts(1).isEmpty then None
      else
        val named = parts.drop(2).flatMap { s =>
          val i = s.indexOf('=')
          Option.when(i > 0)(s.take(i).trim.toLowerCase -> s.drop(i + 1).trim)
        }.toMap
        Some(Rule(parts.head, parts(1).replace(" ", ""), named.get("start"), named.get("stop")))
    }.toSeq

  /** Adds lbw/start/stop to matching tags that do not already carry an lbw= argument. */
  def apply(prompt: String, rules: Seq[Rule], force: Boolean = false): String =
    LoraSyntax.rewriteTags(prompt, (name, args) =>
      if !force && args.toLowerCase.contains(":lbw=") then args
      else
        rules.find(_.target == name).orElse(rules.find(_.target == "*")) match
          case None    => args
          case Some(r) => LoraSyntax.withNamed(args, Seq("lbw" -> r.weights) ++ r.start.map("start" -> _) ++ r.stop.map("stop" -> _)*)
    )
