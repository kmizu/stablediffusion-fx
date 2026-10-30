package sdfx.core.model

/** Resolve raw prompt modes before expanding independent base/hires styles. */
object HiresPreparation:
  def active(p: GenerationParams): Option[ujson.Obj] =
    Option.when(p.hires.enabled && p.img2img.isEmpty)(p.hooks.get("hires_fix_tweaks")).flatten
      .collect { case o: ujson.Obj if !o.value.get("enabled").contains(ujson.Bool(false)) => o }

  def compose(p: GenerationParams, styles: Seq[PromptStyle], saveTemplate: Boolean = true): GenerationParams =
    active(p) match
      case Some(c) if c.value.get("prompts_resolved").contains(ujson.Bool(true)) =>
        if saveTemplate then p
        else p.copy(hooks = p.hooks.updated("hires_fix_tweaks",
          ujson.Obj.from(c.value.toSeq.filterNot(_._1 == "prompt_template"))))
      case config =>
        val raw = config match
          case None => (p.prompt, p.negativePrompt, p.hires.prompt, p.hires.negativePrompt)
          case Some(c) =>
            def text(key: String, default: String): String = c.value.get(key).map(_.str).getOrElse(default)
            def separator(mode: String): String = text(if mode == "Prepend" then "prepend_separator" else "append_separator",
              text("separator", "{newline}"))
            val marker = text("marker", "@")
            val remove = c.value.get("remove_networks").contains(ujson.Bool(true))
            val positiveMode = text("prompt_mode", "Default")
            val negativeMode = text("negative_mode", "Default")
            val (base, hr) = HiresPromptModes.transform(p.prompt, p.hires.prompt, positiveMode, remove, separator(positiveMode), marker)
            val (negative, hrNegative) = HiresPromptModes.transform(p.negativePrompt, p.hires.negativePrompt,
              negativeMode, false, separator(negativeMode), marker)
            // Append/Prepend with an empty input retain the source fallback to base.
            val emptyPositive = p.hires.prompt.isEmpty && c.value.get("empty_prompt").contains(ujson.Bool(true))
            val emptyNegative = p.hires.negativePrompt.isEmpty && c.value.get("empty_negative_prompt").contains(ujson.Bool(true))
            (base, negative, if emptyPositive then "" else if hr.isEmpty && !remove then base else hr,
              if emptyNegative then "" else if hrNegative.isEmpty then negative else hrNegative)
        val (base, negative) = Styles.apply(styles, p.styles, raw._1, raw._2)
        config match
          case None => p.copy(prompt = base, negativePrompt = negative, styles = Nil)
          case Some(c) =>
            val selected = if c.value.get("styles_enabled").contains(ujson.Bool(true)) then
              c.value.get("styles").map(_.arr.toSeq.map(_.str)).getOrElse(Nil)
            else p.styles
            val (hr, hrNegative) = Styles.apply(styles, selected, raw._3, raw._4)
            val resolved = ujson.Obj.from(c.value.toSeq)
            resolved.value.remove("prompt_template")
            if saveTemplate then resolved("prompt_template") = sdfx.core.infotext.HiresPromptMetadata.capture(p, c)
            resolved("prompts_resolved") = true
            p.copy(prompt = base, negativePrompt = negative, styles = Nil,
              hires = p.hires.copy(prompt = hr, negativePrompt = hrNegative),
              hooks = p.hooks.updated("hires_fix_tweaks", resolved))
