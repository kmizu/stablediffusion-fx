package sdfx.core

import sdfx.core.json.SnakePickle
import sdfx.core.model.*

class GenerationParamsSuite extends munit.FunSuite:

  val full: GenerationParams = GenerationParams(
    prompt = "masterpiece, 1girl, <lora:foo:0.8>",
    negativePrompt = "lowres",
    seed = 12345L,
    subseed = 999L,
    subseedStrength = 0.25,
    samplerName = "Euler a",
    scheduler = "Karras",
    steps = 28,
    cfgScale = 6.5,
    width = 832,
    height = 1216,
    batchSize = 2,
    nIter = 3,
    clipSkip = 2,
    eta = Some(0.67),
    hires = HiresParams(enabled = true, scale = 1.5, upscaler = "R-ESRGAN 4x+", secondPassSteps = 10,
      denoisingStrength = 0.4, samplerName = Some("DPM++ 2M"), scheduler = Some("Exponential"),
      prompt = "hires, detailed", negativePrompt = "blurry", cfgScale = Some(5.0)),
    img2img = Some(Img2ImgParams("C:/in/init.png", mask = Some("C:/in/mask.png"), denoisingStrength = 0.6,
      resizeMode = 1, maskBlur = 8, inpaintingFill = 2, inpaintFullRes = true, inpaintFullResPadding = 48,
      inpaintingMaskInvert = true)),
    loras = Seq(LoraRef("foo", 0.8), LoraRef("bar")),
    controlnet = Seq(ControlNetUnit(model = "canny", image = Some("c.png"), preprocessor = "canny",
      weight = 0.9, guidanceStart = 0.1, guidanceEnd = 0.8, preprocessorArgs = Map("low_threshold" -> 100.0))),
    hooks = Map("freeu" -> ujson.Obj("b1" -> 1.3, "s1" -> 0.9)),
    styles = Seq("Style A", "Style B"),
    checkpoint = Some("animagineXL.safetensors"),
    vae = Some("sdxl_vae.safetensors"),
    extra = Map("FreeU Stages" -> "\"[{...}]\"")
  )

  test("JSON round trip of a fully populated params") {
    val s = SnakePickle.write(full)
    assertEquals(SnakePickle.read[GenerationParams](s), full)
  }

  test("JSON round trip of defaults") {
    assertEquals(SnakePickle.read[GenerationParams](SnakePickle.write(GenerationParams())), GenerationParams())
    assertEquals(GenerationParams.default, GenerationParams())
  }

  test("A1111-style snake_case field names") {
    val js = SnakePickle.writeJs(full)
    Seq("prompt", "negative_prompt", "seed", "subseed", "subseed_strength", "sampler_name", "scheduler", "steps",
      "cfg_scale", "width", "height", "batch_size", "n_iter", "clip_skip", "eta", "hires", "img2img", "loras",
      "controlnet", "hooks", "styles", "checkpoint", "vae", "extra").foreach(k => assert(js.obj.contains(k), k))
    assertEquals(js.obj.size, 24)
    val hr = js("hires")
    Seq("enabled", "scale", "upscaler", "second_pass_steps", "resize_x", "resize_y", "denoising_strength",
      "sampler_name", "scheduler", "prompt", "negative_prompt", "cfg_scale").foreach(k => assert(hr.obj.contains(k), k))
    assertEquals(hr("second_pass_steps").num, 10.0)
    val i2i = js("img2img")
    Seq("init_image", "mask", "denoising_strength", "resize_mode", "mask_blur", "inpainting_fill", "inpaint_full_res",
      "inpaint_full_res_padding", "inpainting_mask_invert").foreach(k => assert(i2i.obj.contains(k), k))
    assertEquals(i2i("init_image").str, "C:/in/init.png")
    val cn = js("controlnet")(0)
    Seq("guidance_start", "guidance_end", "preprocessor_args").foreach(k => assert(cn.obj.contains(k), k))
    assertEquals(cn("preprocessor_args").obj.keys.toSeq, Seq("low_threshold"))
    assertEquals(js("loras")(1)("weight").num, 1.0)
    assertEquals(js("extra").obj.keys.toSeq, Seq("FreeU Stages"))
  }

  test("seeds survive JSON exactly (no precision loss)") {
    val p = GenerationParams(seed = 4294967295L, subseed = 1234567890123L)
    val back = SnakePickle.read[GenerationParams](SnakePickle.write(p))
    assertEquals(back.seed, 4294967295L)
    assertEquals(back.subseed, 1234567890123L)
  }

  test("reads a minimal / partial JSON with defaults") {
    val p = SnakePickle.read[GenerationParams]("""{"prompt":"cat","cfg_scale":5,"hires":{"enabled":true},"img2img":{"init_image":"x.png"}}""")
    assertEquals(p.prompt, "cat")
    assertEquals(p.cfgScale, 5.0)
    assertEquals(p.steps, 20)
    assertEquals(p.hires, HiresParams(enabled = true))
    assertEquals(p.img2img, Some(Img2ImgParams("x.png")))
    assertEquals(SnakePickle.read[GenerationParams]("{}"), GenerationParams())
  }

  test("img2img requires init_image") {
    intercept[Exception](SnakePickle.read[GenerationParams]("""{"img2img":{"mask":"m.png"}}"""))
  }

  test("img2img null means txt2img") {
    assertEquals(SnakePickle.read[GenerationParams]("""{"img2img":null}""").img2img, None)
  }

  test("isImg2Img / isInpaint / imagesPerJob") {
    assert(!GenerationParams().isImg2Img)
    assert(!GenerationParams().isInpaint)
    val i2i = GenerationParams(img2img = Some(Img2ImgParams("a.png")))
    assert(i2i.isImg2Img && !i2i.isInpaint)
    assert(full.isImg2Img && full.isInpaint)
    assertEquals(full.imagesPerJob, 6)
  }

  test("validate: defaults and a sane full config are valid") {
    assertEquals(GenerationParams().validate, Nil)
    assertEquals(full.validate, Nil)
  }

  test("validate: boundaries accepted") {
    val ok = GenerationParams(steps = 150, width = 4096, height = 64, batchSize = 16, nIter = 100, cfgScale = 30,
      clipSkip = 12, subseedStrength = 1.0)
    assertEquals(ok.validate, Nil)
    assertEquals(GenerationParams(steps = 1, cfgScale = 0, subseedStrength = 0).validate, Nil)
  }

  test("validate: each violation reported") {
    def one(p: GenerationParams, frag: String) =
      val v = p.validate
      assertEquals(v.size, 1, v)
      assert(v.head.contains(frag), v)
    one(GenerationParams(steps = 0), "steps")
    one(GenerationParams(steps = 151), "steps")
    one(GenerationParams(width = 513), "width")
    one(GenerationParams(width = 56), "width")
    one(GenerationParams(width = 4104), "width")
    one(GenerationParams(height = 1001), "height")
    one(GenerationParams(batchSize = 0), "batch size")
    one(GenerationParams(batchSize = 17), "batch size")
    one(GenerationParams(nIter = 0), "batch count")
    one(GenerationParams(nIter = 101), "batch count")
    one(GenerationParams(cfgScale = -0.5), "CFG")
    one(GenerationParams(cfgScale = 30.5), "CFG")
    one(GenerationParams(clipSkip = 0), "clip skip")
    one(GenerationParams(clipSkip = 13), "clip skip")
    one(GenerationParams(subseedStrength = 1.1), "variation")
    one(GenerationParams(subseedStrength = -0.1), "variation")
    one(GenerationParams(img2img = Some(Img2ImgParams("a", denoisingStrength = 1.5))), "denoising")
    one(GenerationParams(hires = HiresParams(enabled = true, denoisingStrength = -1)), "hires denoising")
  }

  test("validate: disabled hires is not validated") {
    assertEquals(GenerationParams(hires = HiresParams(enabled = false, denoisingStrength = 5)).validate, Nil)
  }

  test("validate: multiple problems are all reported") {
    assertEquals(GenerationParams(steps = 0, width = 1, height = 1, batchSize = 0).validate.size, 4)
  }

  // A1111 calculate_target_resolution: int(w * scale) / integer division, latent = size // 8.
  test("HiresParams.targetSize: scale") {
    assertEquals(HiresParams(scale = 2.0).targetSize(512, 512), (1024, 1024))
    assertEquals(HiresParams(scale = 1.5).targetSize(832, 1216), (1248, 1824))
    assertEquals(HiresParams(scale = 1.25).targetSize(512, 768), (640, 960))
    // int(520 * 1.3) = 676 -> latent 84 -> 672 (rounding to nearest would give 680)
    assertEquals(HiresParams(scale = 1.3).targetSize(520, 520), (672, 672))
    // int(512 * 1.1) = 563 -> 560
    assertEquals(HiresParams(scale = 1.1).targetSize(512, 512), (560, 560))
    assertEquals(HiresParams(scale = 1.0).targetSize(512, 768), (512, 768))
  }

  test("HiresParams.targetSize: explicit width only keeps aspect ratio") {
    assertEquals(HiresParams(resizeX = 1024).targetSize(512, 768), (1024, 1536))
    // 1000 * 768 // 512 = 1500 -> 1496
    assertEquals(HiresParams(resizeX = 1000).targetSize(512, 768), (1000, 1496))
  }

  test("HiresParams.targetSize: explicit height only keeps aspect ratio") {
    assertEquals(HiresParams(resizeY = 1536).targetSize(512, 768), (1024, 1536))
    // 1200 * 832 // 1216 = 821 -> 816
    assertEquals(HiresParams(resizeY = 1200).targetSize(832, 1216), (816, 1200))
  }

  test("HiresParams.targetSize: both sides -> exact (cropped) size; explicit resize wins over scale") {
    assertEquals(HiresParams(resizeX = 1920, resizeY = 1080, scale = 4).targetSize(512, 512), (1920, 1080))
    assertEquals(HiresParams(resizeX = 1023, resizeY = 777).targetSize(512, 512), (1016, 776))
  }

  test("HiresParams.targetSize: never below 64 and never divides by zero") {
    assertEquals(HiresParams(scale = 0.05).targetSize(512, 512), (64, 64))
    assertEquals(HiresParams(resizeX = 1024).targetSize(0, 0)._1, 1024)
  }

  test("LoraRef default weight") {
    assertEquals(SnakePickle.read[LoraRef]("""{"name":"x"}"""), LoraRef("x", 1.0))
  }
