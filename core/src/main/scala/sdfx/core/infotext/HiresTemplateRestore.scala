package sdfx.core.infotext

import sdfx.core.model.*
import scala.util.Try

/** Restore raw templates only when the current style catalog reproduces the recorded prompts. */
object HiresTemplateRestore:
  private val promptKeys = Set("prompt_mode", "negative_mode", "remove_networks", "append_separator",
    "prepend_separator", "separator", "marker", "styles_enabled", "styles", "empty_prompt", "empty_negative_prompt")

  def restore(p: GenerationParams, fields: Map[String, String], styles: Seq[PromptStyle],
              enabled: Boolean = true): GenerationParams =
    if !p.hires.enabled || p.isImg2Img then p
    else
      val previous = p.hooks.get("hires_fix_tweaks").collect { case o: ujson.Obj => o }.getOrElse(ujson.Obj())
      val hook = ujson.Obj.from(previous.value.toSeq.filterNot((k, _) =>
        promptKeys(k) || Set("prompts_resolved", "prompt_template")(k)))
      hook("enabled") = (previous.value.nonEmpty && !previous.value.get("enabled").contains(ujson.Bool(false))) ||
        (enabled && (fields.contains("NUI Hires template") || fields.contains("Hires prompt mode")))
      hook("prompt_mode") = "Default"
      hook("negative_mode") = "Default"
      hook("remove_networks") = false
      hook("styles_enabled") = false
      hook("styles") = ujson.Arr()
      val emptyPositive = p.hires.prompt.isEmpty && fields.get("Hires empty prompt").exists(_.equalsIgnoreCase("true"))
      val emptyNegative = p.hires.negativePrompt.isEmpty && fields.get("Hires empty negative prompt").exists(_.equalsIgnoreCase("true"))
      hook("empty_prompt") = emptyPositive
      hook("empty_negative_prompt") = emptyNegative
      if emptyPositive || emptyNegative then hook("enabled") = true
      val clean = p.copy(styles = Nil, hooks = p.hooks.updated("hires_fix_tweaks", hook))
      if !enabled then clean
      else native(clean, fields.get("NUI Hires template"), styles).getOrElse {
        fields.get("Hires prompt mode").flatMap(HiresPromptMetadata.parse).fold(clean) { m =>
          val matched = m.matching(p.prompt, p.negativePrompt, p.hires.prompt, p.hires.negativePrompt)
          val h = ujson.Obj.from(hook.value.toSeq)
          h("append_separator") = m.appendSeparator
          h("prepend_separator") = m.prependSeparator
          h("marker") = m.marker
          matched.positive.foreach { t => h("prompt_mode") = t.mode; h("remove_networks") = t.removeNetworks; h("empty_prompt") = false }
          matched.negative.foreach { t => h("negative_mode") = t.mode; h("empty_negative_prompt") = false }
          clean.copy(hires = clean.hires.copy(prompt = matched.positive.fold(p.hires.prompt)(_.prompt),
            negativePrompt = matched.negative.fold(p.hires.negativePrompt)(_.prompt)),
            hooks = clean.hooks.updated("hires_fix_tweaks", h))
        }
      }

  private def native(p: GenerationParams, text: Option[String], styles: Seq[PromptStyle]): Option[GenerationParams] =
    text.filter(_.length <= 262144).flatMap { s => Try {
      val o = ujson.read(s).obj
      require(o("version").num == 1)
      val config = o("config").obj
      require(config.keySet.subsetOf(promptKeys))
      config.foreach { (key, value) =>
        if Set("remove_networks", "styles_enabled", "empty_prompt", "empty_negative_prompt")(key) then value.bool
        else if key == "styles" then value.arr.foreach(_.str)
        else value.str
      }
      val selected = o("styles").arr.toSeq.map(_.str)
      val hiresSelected = config.get("styles").toSeq.flatMap(_.arr.toSeq.map(_.str))
      require((selected ++ hiresSelected).forall(name => styles.exists(_.name == name)))
      val h = ujson.Obj.from(p.hooks("hires_fix_tweaks").obj.toSeq ++ config.toSeq)
      for key <- Seq("empty_prompt", "empty_negative_prompt") do h(key) = config.getOrElse(key, ujson.Bool(false))
      for key <- Seq("append_separator", "prepend_separator") do
        if !config.contains(key) then h(key) = config.getOrElse("separator", ujson.Str("{newline}"))
      val candidate = p.copy(prompt = o("prompt").str, negativePrompt = o("negative_prompt").str,
        styles = selected, hires = p.hires.copy(prompt = o("hires_prompt").str,
          negativePrompt = o("hires_negative_prompt").str), hooks = p.hooks.updated("hires_fix_tweaks", h))
      val resolved = HiresPreparation.compose(candidate, styles, saveTemplate = false)
      require(resolved.prompt == p.prompt && resolved.negativePrompt == p.negativePrompt &&
        resolved.hires.prompt == p.hires.prompt && resolved.hires.negativePrompt == p.hires.negativePrompt)
      candidate
    }.toOption }
