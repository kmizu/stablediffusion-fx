package sdfx.mcp

import java.nio.file.Path

class ExtendedGenerationSuite extends munit.FunSuite:
  private def args(o: ujson.Obj) = Args(o, Seq("subseed", "subseed_strength", "eta", "manual_loras", "controlnet"))
  private def lora(name: String): String = if name == "known" then name else throw ToolError("LoRA not found")
  private def control(name: String): String = if name == "cn" then name else throw ToolError("ControlNet not found")
  private def prep(name: String): String = if Set("none", "canny")(name) then name else throw ToolError("unknown preprocessor")
  private def image(spec: String, field: String): Path = if spec == "inside" then Path.of("inside.png") else throw ToolError("image denied")

  test("typed variation, manual LoRA and ControlNet retain their independent values") {
    val input = ujson.Obj("subseed" -> 42, "subseed_strength" -> 0.4, "eta" -> 0.2,
      "manual_loras" -> ujson.Arr(ujson.Obj("name" -> "known", "weight" -> 0.8, "unet" -> 0.5, "lbw" -> "ALL", "start" -> 0.1, "stop" -> 0.8, "dyn" -> 4)),
      "controlnet" -> ujson.Arr(ujson.Obj("model" -> "cn", "image" -> "inside", "preprocessor" -> "canny", "weight" -> 0.6,
        "guidance_start" -> 0.2, "guidance_end" -> 0.7, "preprocessor_args" -> ujson.Obj("low_threshold" -> 100))))
    val p = ExtendedGeneration.parse(args(input), lora, control, prep, image)
    assertEquals(p.subseed, 42L); assertEquals(p.subseedStrength, 0.4); assertEquals(p.eta, Some(0.2))
    assertEquals(p.loras.head.dyn, Some(4)); assertEquals(p.loras.head.unet, Some(0.5))
    assertEquals(p.controlnet.head.image, Some("inside.png")); assertEquals(p.controlnet.head.guidanceEnd, 0.7)
    assertEquals(p.controlnet.head.preprocessorArgs("low_threshold"), 100.0)
  }

  test("rejects unknown models, denied images, invalid bounds and extra fields") {
    for input <- Seq(
      ujson.Obj("manual_loras" -> ujson.Arr(ujson.Obj("name" -> "bad"))),
      ujson.Obj("manual_loras" -> ujson.Arr(ujson.Obj("name" -> "known", "dyn" -> 0))),
      ujson.Obj("manual_loras" -> ujson.Arr(ujson.Obj("name" -> "known", "start" -> 0.9, "stop" -> 0.1))),
      ujson.Obj("controlnet" -> ujson.Arr(ujson.Obj("model" -> "bad", "image" -> "inside"))),
      ujson.Obj("controlnet" -> ujson.Arr(ujson.Obj("model" -> "cn", "image" -> "outside"))),
      ujson.Obj("controlnet" -> ujson.Arr(ujson.Obj("model" -> "cn", "image" -> "inside", "hook" -> "unsafe")))
    ) do intercept[ToolError](ExtendedGeneration.parse(args(input), lora, control, prep, image))
  }

  test("independent Hires prompt and sampler settings round trip") {
    val input = ujson.Obj("prompt" -> "detail", "negative_prompt" -> "noise", "cfg_scale" -> 4.5,
      "sampler_name" -> "Euler a", "scheduler" -> "Karras")
    val hires = ExtendedGeneration.withHires(Args(input, ExtendedGeneration.hiresKeys), sdfx.core.model.HiresParams(), identity, identity)
    assertEquals(hires.prompt, "detail"); assertEquals(hires.negativePrompt, "noise")
    assertEquals(hires.cfgScale, Some(4.5)); assertEquals(hires.samplerName, Some("Euler a")); assertEquals(hires.scheduler, Some("Karras"))
  }
