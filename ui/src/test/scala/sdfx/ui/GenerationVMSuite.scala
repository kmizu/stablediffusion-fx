package sdfx.ui

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.*
import sdfx.ui.vm.*

class GenerationVMSuite extends munit.FunSuite:
  test("difference inpaint can submit without a hand-painted mask") {
    val vm = new GenerationVM(Mode.Inpaint)
    vm.initImage.set("altered.png")
    val exists: String => Boolean = _ => true
    assert(vm.validate(exists).exists(_.contains("Paint a mask")))
    assert(!vm.validate(exists, differenceActive = true).exists(_.contains("Paint a mask")))
  }

  test("disabled_gate_keeps_drafts_but_submits_zero_strength") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      vm.variationEnabled.set(false)
      assertEquals(vm.subseed.get, 99L)
      assertEquals(vm.subseedStrength.get, 0.4)
      assertEquals(vm.toParams.subseed, 99L)
      assertEquals(vm.toParams.subseedStrength, 0.0)
      assertEquals(vm.snapshot.subseed, 99L)
      assertEquals(vm.snapshot.subseedStrength, 0.4)
    }
  }

  test("gate_true_restores_original_effective_values") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      assert(vm.variationEnabled.get)
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      vm.variationEnabled.set(false)
      vm.variationEnabled.set(true)
      assertEquals(vm.toParams.subseed, 99L)
      assertEquals(vm.toParams.subseedStrength, 0.4)
    }
  }

  test("load_params_reenables_by_default") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.variationEnabled.set(false)
      vm.load(GenerationParams(subseed = 99L, subseedStrength = 0.4))
      assert(vm.variationEnabled.get)
      assertEquals(vm.toParams.subseedStrength, 0.4)
      vm.load(GenerationParams(subseed = 99L, subseedStrength = 0.4), variationEnabledState = false)
      assert(!vm.variationEnabled.get)
      assertEquals(vm.snapshot.subseedStrength, 0.4)
    }
  }

  private def roundTrip(mode: Mode, p: GenerationParams): GenerationParams =
    val vm = new GenerationVM(mode)
    vm.load(p, includeModel = true)
    vm.toParams

  test("txt2img load/toParams round trip keeps every field") {
    val p = GenerationParams(
      prompt = "a cat, (sitting:1.2)",
      negativePrompt = "blurry",
      seed = 1234,
      subseed = 99,
      subseedStrength = 0.3,
      samplerName = "Euler a",
      scheduler = "Karras",
      steps = 28,
      cfgScale = 6.5,
      width = 832,
      height = 1216,
      batchSize = 2,
      nIter = 3,
      clipSkip = 2,
      checkpoint = Some("sdxl_base"),
      vae = Some("Automatic")
    )
    assertEquals(roundTrip(Mode.Txt2Img, p), p)
  }

  test("hires params round trip and target size") {
    val h = HiresParams(enabled = true, scale = 1.5, upscaler = "R-ESRGAN 4x+", secondPassSteps = 12, denoisingStrength = 0.4,
      samplerName = Some("DPM++ SDE"), scheduler = Some("Exponential"), prompt = "sharp", negativePrompt = "soft", cfgScale = Some(5.0))
    val p = GenerationParams(width = 512, height = 768, hires = h)
    val out = roundTrip(Mode.Txt2Img, p)
    assertEquals(out.hires, h)
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.load(p)
    assertEquals(vm.hiresTarget, (768, 1152))
    vm.hiresResizeX.set(1000)
    assertEquals(vm.hiresTarget, h.copy(resizeX = 1000).targetSize(512, 768))
  }

  test("hires is never sent from img2img / inpaint") {
    val vm = new GenerationVM(Mode.Img2Img)
    vm.hiresEnabled.set(true)
    assert(!vm.toParams.hires.enabled)
  }

  test("img2img round trip") {
    val i = Img2ImgParams("C:/x/init.png", None, denoisingStrength = 0.42, resizeMode = 2)
    val p = GenerationParams(prompt = "p", img2img = Some(i))
    val out = roundTrip(Mode.Img2Img, p)
    assertEquals(out.img2img, Some(i))
    assert(!out.isInpaint)
  }

  test("inpaint round trip with mask options") {
    val i = Img2ImgParams("init.png", Some("mask.png"), denoisingStrength = 0.6, resizeMode = 1, maskBlur = 8, inpaintingFill = 2,
      inpaintFullRes = true, inpaintFullResPadding = 48, inpaintingMaskInvert = true)
    val out = roundTrip(Mode.Inpaint, GenerationParams(img2img = Some(i)))
    assertEquals(out.img2img, Some(i))
    assert(out.isInpaint)
  }

  test("txt2img never carries img2img inputs") {
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.initImage.set("x.png")
    assertEquals(vm.toParams.img2img, None)
  }

  test("controlnet: only enabled units with a model are sent, snapshot keeps all") {
    val units = Seq(
      ControlNetUnit(enabled = true, model = "canny", image = Some("c.png"), preprocessor = "canny", weight = 0.8, guidanceStart = 0.1, guidanceEnd = 0.9),
      ControlNetUnit(enabled = false, model = "depth"),
      ControlNetUnit(enabled = true, model = "None")
    )
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.load(GenerationParams(controlnet = units))
    assertEquals(vm.toParams.controlnet, Seq(units.head))
    assertEquals(vm.snapshot.controlnet.size, 3)
    assertEquals(vm.snapshot.controlnet(1), units(1))
  }

  test("manual LoRAs retain full settings separately from same-name prompt tags") {
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.load(GenerationParams(prompt = "girl, <lora:detail:0.8>"))
    assertEquals(vm.toParams.prompt, "girl, <lora:detail:0.8>")
    assertEquals(vm.toParams.loras, Nil)
    val manual = Seq(LoraRef("detail", 0.8, Some(.4), Some("CUSTOM"), Some(.1), Some(.9)), LoraRef("style", 1.0))
    vm.load(GenerationParams(prompt = "girl, <lora:detail:0.8>", loras = manual))
    assertEquals(vm.prompt.get, "girl, <lora:detail:0.8>")
    assertEquals(vm.toParams.loras, manual)
    assertEquals(vm.snapshot.loras, manual)
    val extracted = LoraSyntax.extract(vm.toParams.prompt)
    assertEquals(extracted.loras.map(_.name), Seq("detail"))
  }

  test("restore from infotext text keeps the init image") {
    val src = GenerationParams(
      prompt = "castle on a hill",
      negativePrompt = "lowres",
      steps = 30,
      samplerName = "DPM++ 2M",
      scheduler = "Karras",
      cfgScale = 5.5,
      width = 640,
      height = 896,
      clipSkip = 2,
      hires = HiresParams(enabled = true, scale = 2, upscaler = "Latent", secondPassSteps = 10, denoisingStrength = 0.5)
    )
    val text = Infotext.write(src, ImageFacts(seed = 777, modelName = Some("anything"), modelHash = Some("abcdef1234")))
    val t2i = new GenerationVM(Mode.Txt2Img)
    t2i.restoreInfotext(text)
    val r = t2i.toParams
    assertEquals(r.prompt, "castle on a hill")
    assertEquals(r.negativePrompt, "lowres")
    assertEquals(r.steps, 30)
    assertEquals(r.scheduler, "Karras")
    assertEquals(r.cfgScale, 5.5)
    assertEquals((r.width, r.height), (640, 896))
    assertEquals(r.seed, 777L)
    assertEquals(r.clipSkip, 2)
    assert(r.hires.enabled)
    assertEquals(r.hires.secondPassSteps, 10)

    val i2i = new GenerationVM(Mode.Img2Img)
    i2i.initImage.set("keep.png")
    i2i.denoise.set(0.3)
    i2i.restoreInfotext("a dog\nSteps: 12, Sampler: Euler a, CFG scale: 4, Seed: 5, Size: 512x512, Denoising strength: 0.65")
    assertEquals(i2i.initImage.get, "keep.png")
    assertEquals(i2i.denoise.get, 0.65)
    assertEquals(i2i.sampler.get, "Euler a")
    assertEquals(i2i.prompt.get, "a dog")
  }

  test("restore splits legacy combined sampler names") {
    val vm = new GenerationVM(Mode.Txt2Img)
    vm.restoreInfotext("x\nSteps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1, Size: 512x512")
    assertEquals(vm.sampler.get, "DPM++ 2M")
    assertEquals(vm.scheduler.get, "Karras")
  }

  test("validation messages") {
    val vm = new GenerationVM(Mode.Txt2Img)
    assertEquals(vm.validate(_ => true), Nil)
    vm.steps.set(0)
    vm.width.set(500)
    val probs = vm.validate(_ => true)
    assert(probs.exists(_.startsWith("steps must be 1..150")), probs)
    assert(probs.exists(_.startsWith("width must be a multiple of 8")), probs)
    assertEquals(vm.problems.size, probs.size)

    val i2i = new GenerationVM(Mode.Img2Img)
    assert(i2i.validate(_ => true).exists(_.contains("init image")))
    i2i.initImage.set("missing.png")
    assert(i2i.validate(_ => false).exists(_.startsWith("Init image not found")))
    assertEquals(i2i.validate(_ => true), Nil)

    val inp = new GenerationVM(Mode.Inpaint)
    inp.initImage.set("a.png")
    assert(inp.validate(_ => true).exists(_.contains("mask")))
    inp.hasMask.set(true)
    assertEquals(inp.validate(_ => true), Nil)

    val cn = new GenerationVM(Mode.Txt2Img)
    cn.controlNet.head.enabled.set(true)
    assert(cn.validate(_ => true).exists(_.contains("without a model")))
    cn.controlNet.head.model.set("canny")
    assert(cn.validate(_ => true).exists(_.contains("control image")))
    cn.controlNet.head.image.set("c.png")
    assertEquals(cn.validate(_ => true), Nil)
    vm.steps.set(20)
    vm.width.set(512)
    assertEquals(vm.validate(_ => true), Nil)
    assert(vm.problems.isEmpty)
  }

  test("shared model VM feeds checkpoint, vae and clip skip of every tab") {
    val shared = new ModelVM
    val a = new GenerationVM(Mode.Txt2Img, shared)
    val b = new GenerationVM(Mode.Img2Img, shared)
    shared.checkpoint.set("model.safetensors")
    shared.clipSkip.set(2)
    assertEquals(a.toParams.checkpoint, Some("model.safetensors"))
    assertEquals(b.toParams.clipSkip, 2)
    shared.checkpoint.set("")
    assertEquals(a.toParams.checkpoint, None)
  }
