package sdfx.core

import sdfx.core.infotext.*
import sdfx.core.model.*

class HiresTemplateRestoreSuite extends munit.FunSuite:
  test("empty resolved prompt survives PNG and disabled template restore without inheriting base") {
    val raw = GenerationParams(prompt = "<lora:only:1>", hires = HiresParams(enabled = true),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("remove_networks" -> true)))
    val effective = HiresPreparation.compose(raw, Nil)
    assertEquals(effective.hires.prompt, "")
    val parsed = Infotext.parse(Infotext.write(effective, ImageFacts(seed = 1)))
    val restored = HiresTemplateRestore.restore(Infotext.toParams(parsed), parsed.fields, Nil, enabled = false)
    val replay = HiresPreparation.compose(restored, Nil)
    assertEquals(replay.hires.prompt, "")
    assert(HiresPreparation.active(replay).exists(_.value.get("prompts_resolved").contains(ujson.Bool(true))))
    assertEquals(parsed.fields("Hires empty prompt"), "True")
    val edited = restored.copy(hires = restored.hires.copy(prompt = "new prompt"))
    assertEquals(HiresPreparation.compose(edited, Nil).hires.prompt, "new prompt")
  }
  private val styles = Seq(PromptStyle("frame", "frame({prompt})", "bad frame"))
  private val raw = GenerationParams(prompt = "cat @light@", negativePrompt = "blur", styles = Seq("frame"),
    hires = HiresParams(enabled = true, prompt = "@light@sun", negativePrompt = "noise"),
    hooks = Map("hires_fix_tweaks" -> ujson.Obj("prompt_mode" -> "Prompt S/R", "negative_mode" -> "Append")))
  private val resolved = HiresPreparation.compose(raw, styles)
  private val fields = Map("NUI Hires template" -> resolved.hooks("hires_fix_tweaks")("prompt_template").render())

  test("ordinary hires metadata does not opt into tweaks") {
    val p = GenerationParams(prompt = "cat", hires = HiresParams(enabled = true, prompt = "detail"))
    val q = HiresTemplateRestore.restore(p, Map.empty, Nil)
    assertEquals(q.hooks("hires_fix_tweaks")("enabled").bool, false)
  }

  test("native restore recovers markers and style selections only after full recomposition") {
    val restored = HiresTemplateRestore.restore(resolved, fields, styles)
    assertEquals(restored.prompt, raw.prompt)
    assertEquals(restored.hires, raw.hires)
    assertEquals(restored.styles, raw.styles)
    val replay = HiresPreparation.compose(restored, styles)
    assertEquals(replay.prompt, resolved.prompt)
    assertEquals(replay.hires, resolved.hires)
  }
  test("changed or missing styles and disabled restore retain resolved text without old modes") {
    Seq(Nil, Seq(PromptStyle("frame", "changed({prompt})", "bad frame"))).foreach { current =>
      val p = HiresTemplateRestore.restore(resolved, fields, current)
      assertEquals(p.prompt, resolved.prompt)
      assertEquals(p.styles, Nil)
      assertEquals(p.hooks("hires_fix_tweaks")("prompt_mode").str, "Default")
    }
    val disabled = HiresTemplateRestore.restore(resolved, fields, styles, enabled = false)
    assertEquals(disabled.hires, resolved.hires)
    assertEquals(disabled.styles, Nil)
  }
  test("source-only restore is independent by side and preserves effective seed hooks") {
    val p = GenerationParams(prompt = "cat", negativePrompt = "bad", hires = HiresParams(enabled = true,
      prompt = "cat + detail", negativePrompt = "different"), hooks = Map("hires_fix_tweaks" -> ujson.Obj("seed" -> 123)))
    val f = Map("Hires prompt mode" -> "{'h':['Append','detail'],'n':['Append','noise'],'a':' + '}")
    val q = HiresTemplateRestore.restore(p, f, Nil)
    assertEquals(q.hires.prompt, "detail")
    assertEquals(q.hires.negativePrompt, "different")
    assertEquals(q.hooks("hires_fix_tweaks")("seed").num, 123.0)
    assertEquals(q.hooks("hires_fix_tweaks")("negative_mode").str, "Default")
  }
  test("malformed, wrong version and oversized native snapshots leave resolved inputs intact") {
    Seq("[]", "{}", "x" * 262145, fields("NUI Hires template").replace("\"version\":1", "\"version\":2")).foreach { text =>
      val p = HiresTemplateRestore.restore(resolved, Map("NUI Hires template" -> text), styles)
      assertEquals(p.prompt, resolved.prompt)
      assertEquals(p.hires, resolved.hires)
    }
  }
