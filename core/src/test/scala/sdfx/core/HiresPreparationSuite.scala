package sdfx.core

import sdfx.core.model.*

class HiresPreparationSuite extends munit.FunSuite:
  test("template capture retains raw prompts and styles separately from resolved text") {
    val p = params(ujson.Obj("prompt_mode" -> "Append", "styles_enabled" -> true,
      "styles" -> ujson.Arr("hires")))
    val q = HiresPreparation.compose(p, styles)
    val t = q.hooks("hires_fix_tweaks")("prompt_template")
    assertEquals(t("prompt").str, p.prompt)
    assertEquals(t("hires_prompt").str, "detail")
    assertEquals(t("styles").arr.toSeq.map(_.str), Seq("base"))
    assertEquals(t("config")("styles").arr.toSeq.map(_.str), Seq("hires"))
    assertEquals(sdfx.core.infotext.HiresPromptMetadata.parse(t("mode").str).get.positive.get.prompt, "detail")
    assertEquals(HiresPreparation.compose(q, styles), q)
  }

  test("disabling template saving removes even previously captured templates") {
    val p = HiresPreparation.compose(params(ujson.Obj("prompt_mode" -> "Append")), styles)
    val q = HiresPreparation.compose(p, styles, saveTemplate = false)
    assert(!q.hooks("hires_fix_tweaks").obj.contains("prompt_template"))
    assertEquals(q.prompt, p.prompt)
    assertEquals(q.hires, p.hires)
  }

  test("append and prepend use independent source separators") {
    val p = GenerationParams(prompt = "base", negativePrompt = "bad",
      hires = HiresParams(enabled = true, prompt = "detail", negativePrompt = "noise"),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("prompt_mode" -> "Append", "negative_mode" -> "Prepend",
        "append_separator" -> " + ", "prepend_separator" -> " / ")))
    val q = HiresPreparation.compose(p, Nil)
    assertEquals(q.hires.prompt, "base + detail")
    assertEquals(q.hires.negativePrompt, "noise / bad")
  }
  private val styles = Seq(PromptStyle("base", "base({prompt}) <lora:style:0.5>", "base-negative"),
    PromptStyle("hires", "hr({prompt})", "hr-negative"))
  private def params(h: ujson.Obj) = GenerationParams(prompt = "cat <lora:first:1> embedding", negativePrompt = "blur",
    styles = Seq("base"), hires = HiresParams(enabled = true, prompt = "detail"), hooks = Map("hires_fix_tweaks" -> h))

  test("remove first pass networks before inherited styles, preserve embedding and manual LoRAs") {
    val p = params(ujson.Obj("prompt_mode" -> "Append", "remove_networks" -> true))
      .copy(loras = Seq(LoraRef("manual")))
    val q = HiresPreparation.compose(p, styles)
    assert(q.prompt.contains("<lora:first:1>"))
    assert(!q.hires.prompt.contains("<lora:first:1>"))
    assert(q.hires.prompt.contains("<lora:style:0.5>"))
    assert(q.hires.prompt.contains("embedding\ndetail"))
    assertEquals(q.loras, p.loras)
    assertEquals(q.styles, Nil)
  }

  test("independent empty styles remove inherited styles; ordered selection applies once") {
    val p = params(ujson.Obj("styles_enabled" -> true, "styles" -> ujson.Arr()))
    assertEquals(HiresPreparation.compose(p, styles).hires.prompt, "detail")
    assertEquals(HiresPreparation.compose(p, styles).hires.negativePrompt, "blur")
    val q = HiresPreparation.compose(params(ujson.Obj("styles_enabled" -> true, "styles" -> ujson.Arr("hires"))), styles)
    assertEquals(q.hires.prompt, "hr(detail)")
    assertEquals(q.hires.negativePrompt, "blur, hr-negative")
  }

  test("insertion marker removed before styles; negative mode independent") {
    val p = params(ujson.Obj("prompt_mode" -> "Prompt S/R", "negative_mode" -> "Append", "separator" -> ", "))
      .copy(prompt = "cat @light@", hires = HiresParams(enabled = true, prompt = "@light@sun", negativePrompt = "noise"))
    val q = HiresPreparation.compose(p, styles)
    assertEquals(q.prompt, "base(cat ) <lora:style:0.5>")
    assertEquals(q.hires.prompt, "base(cat sun) <lora:style:0.5>")
    assertEquals(q.hires.negativePrompt, "blur, noise, base-negative")
  }

  test("disabled leaves existing hires behavior, resolved result is idempotent") {
    val p = params(ujson.Obj("enabled" -> false))
    val q = HiresPreparation.compose(p, styles)
    assertEquals(q.hires, p.hires)
    val active = HiresPreparation.compose(params(ujson.Obj("prompt_mode" -> "Append")), styles)
    assertEquals(HiresPreparation.compose(active, styles), active)
  }
