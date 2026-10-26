package sdfx.plugins

import sdfx.core.model.{GenerationParams, LoraRef, LoraSyntax}
import sdfx.core.plugins.api.Values
import LoraBlockWeightPlugin.*

class LoraBlockWeightSuite extends munit.FunSuite:
  test("tag arguments: te/unet positional and named, lbw, start/stop and step ranges"):
    val ex = LoraSyntax.extract("a <lora:x:0.8:0.5:lbw=OUTS:stop=0.4>, b <lora:y:lbw=1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0:step=5-10> <lora:z>")
    assertEquals(ex.prompt, "a, b")
    assertEquals(ex.loras, Seq(
      LoraRef("x", 0.8, Some(0.5), Some("OUTS"), None, Some(0.4)),
      LoraRef("y", 1.0, None, Some("1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0"), Some(5.0), Some(10.0)),
      LoraRef("z", 1.0)))
    assertEquals(LoraSyntax.extract("<lora:x:te=0.3:unet=0.7>").loras, Seq(LoraRef("x", 0.3, Some(0.7))))

  test("the same LoRA twice sums weights and keeps the first tag's arguments"):
    assertEquals(LoraSyntax.extract("<lora:x:0.5:lbw=INS> <lora:x:0.25>").loras, Seq(LoraRef("x", 0.75, None, Some("INS"))))

  test("rules add lbw to matching tags, * matches the rest, hand-written lbw wins"):
    val rules = parseRules("style : OUTALL : stop=10\n* : INS\n# comment\nbroken")
    assertEquals(rules.size, 2)
    val out = LoraBlockWeightPlugin.apply("<lora:style:0.7> <lora:char:1> <lora:own:1:lbw=MIDD>", rules)
    assertEquals(out, "<lora:style:0.7:lbw=OUTALL:stop=10> <lora:char:1:lbw=INS> <lora:own:1:lbw=MIDD>")
    assertEquals(LoraBlockWeightPlugin.apply("<lora:own:1:lbw=MIDD>", rules, force = true), "<lora:own:1:lbw=INS>")

  test("process rewrites the prompt and the hires prompt; the XYZ axis works without the plugin"):
    val p = new LoraBlockWeightPlugin
    val gp = GenerationParams(prompt = "cat <lora:a:1>")
    val v = Values(Map("enabled" -> "true", "rules" -> "* : OUTS"))
    assert(p.isActive(v))
    assertEquals(p.process(gp, v).prompt, "cat <lora:a:1:lbw=OUTS>")
    assertEquals(p.xyzAxes.head.apply(gp, "1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1").prompt,
      "cat <lora:a:1:lbw=1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1>")
    assert(!p.isActive(Values(Map("enabled" -> "true", "rules" -> ""))))
