package sdfx.core

import sdfx.core.infotext.*
import sdfx.core.model.*

import scala.collection.immutable.ListMap

class InfotextSuite extends munit.FunSuite:
  import Infotext.*

  val V: String = Version.label

  test("upstream Hires CFG scale spelling restores and is consumed") {
    val text = "cat\nSteps: 12, CFG scale: 7, Hires upscale: 1.5, Hires CFG scale: 3"
    val p = toParams(parse(text))
    assertEquals(p.hires.cfgScale, Some(3.0))
    assert(!p.extra.contains("Hires CFG scale"))
    assertEquals(toParams(parse(text + ", Hires CFG Scale: 4")).hires.cfgScale, Some(4.0))
  }

  // Exactly as reForge (f1.0.0v2-v1.10.1RC) writes it.
  val reforgeHires: String =
    "masterpiece, 1girl\nNegative prompt: lowres, bad hands\n" +
      "Steps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 12345, Size: 832x1216, " +
      "Model hash: 4496b36d48, Model: animagineXL, Denoising strength: 0.4, Hires upscale: 1.5, " +
      "Hires upscaler: R-ESRGAN 4x+, Lora hashes: \"foo: abcdef123456, bar: 0123456789ab\", Version: f1.0.0v2-v1.10.1RC"

  // A1111 < 1.9: scheduler folded into the sampler name, no Schedule type.
  val legacy: String =
    "a photo of a cat\nNegative prompt: ugly\n" +
      "Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 965400086, Size: 512x768, Model hash: 45dee52b, " +
      "Model: v1-5-pruned-emaonly, Clip skip: 2, ENSD: 31337, Version: v1.6.0"

  val hiresParams: GenerationParams = GenerationParams(
    prompt = "masterpiece, 1girl",
    negativePrompt = "lowres, bad hands",
    steps = 28,
    samplerName = "DPM++ 2M",
    scheduler = "Karras",
    cfgScale = 7,
    width = 832,
    height = 1216,
    hires = HiresParams(enabled = true, scale = 1.5, upscaler = "R-ESRGAN 4x+", denoisingStrength = 0.4)
  )
  val hiresFacts: ImageFacts = ImageFacts(
    seed = 12345L,
    modelName = Some("animagineXL"),
    modelHash = Some("4496b36d48"),
    loraHashes = Seq("foo" -> "abcdef123456", "bar" -> "0123456789ab")
  )

  // ---------------------------------------------------------------- quote / unquote

  test("quote: like infotext_utils.quote + json.dumps(ensure_ascii=False)") {
    assertEquals(quote("Euler a"), "Euler a")
    assertEquals(quote(""), "")
    assertEquals(quote("a, b"), "\"a, b\"")
    assertEquals(quote("a:b"), "\"a:b\"")
    assertEquals(quote("line1\nline2"), "\"line1\\nline2\"")
    assertEquals(quote("say \"hi\", ok"), "\"say \\\"hi\\\", ok\"")
    assertEquals(quote("tab\t,"), "\"tab\\t,\"")
    assertEquals(quote("cr\r,"), "\"cr\\r,\"")
    assertEquals(quote("back\\slash,"), "\"back\\\\slash,\"")
    assertEquals(quote("ctrl\u0001,"), "\"ctrl\\u0001,\"")
    assertEquals(quote("日本語, テスト"), "\"日本語, テスト\"")
    assertEquals(quote("emoji \uD83D\uDE00,"), "\"emoji \uD83D\uDE00,\"")
    assertEquals(quote("slash/,"), "\"slash/,\"")
    // only , \n : trigger quoting; quotes alone do not
    assertEquals(quote("say \"hi\""), "say \"hi\"")
  }

  test("unquote") {
    assertEquals(unquote("plain"), "plain")
    assertEquals(unquote("\""), "\"")
    assertEquals(unquote("\"\""), "")
    assertEquals(unquote("\"a, b\""), "a, b")
    assertEquals(unquote("\"a\\nb \\\"c\\\"\""), "a\nb \"c\"")
    assertEquals(unquote("\"\\u65e5\""), "日")
    assertEquals(unquote("\"broken \\x\""), "\"broken \\x\"") // invalid JSON stays as is
  }

  test("quote -> unquote round trip") {
    Seq("a, b", "x:y", "multi\nline, with \"quotes\" and \\ backslash", "日本語: テスト, \t", "").foreach { s =>
      assertEquals(unquote(quote(s)), s)
    }
  }

  // ---------------------------------------------------------------- write exactness

  test("write: txt2img + hires + lora hashes is byte-identical to reForge (modulo Version)") {
    val expected = reforgeHires.replace("Version: f1.0.0v2-v1.10.1RC", s"Version: $V")
    assertEquals(write(hiresParams, hiresFacts), expected)
  }

  test("write: plain txt2img, A1111 order") {
    val p = GenerationParams(prompt = "a cat", negativePrompt = "", steps = 20, samplerName = "Euler a",
      scheduler = "Automatic", cfgScale = 7.5, width = 512, height = 768)
    assertEquals(write(p, ImageFacts(seed = 1)),
      s"a cat\nSteps: 20, Sampler: Euler a, Schedule type: Automatic, CFG scale: 7.5, Seed: 1, Size: 512x768, Version: $V")
  }

  test("write: engine-reported scheduler wins over requested one") {
    val p = GenerationParams(prompt = "x", scheduler = "Automatic")
    assert(write(p, ImageFacts(seed = 1, scheduler = Some("Karras"))).contains("Schedule type: Karras,"))
  }

  test("write: empty negative prompt -> no Negative prompt line") {
    val t = write(GenerationParams(prompt = "p"), ImageFacts(seed = 3))
    assert(!t.contains("Negative prompt"), t)
    assertEquals(t.linesIterator.size, 2)
  }

  test("write: empty prompt -> params only") {
    val t = write(GenerationParams(), ImageFacts(seed = 3))
    assert(t.startsWith("Steps: 20,"), t)
  }

  test("write: multi-line prompt and negative are kept verbatim") {
    val p = GenerationParams(prompt = "line one,\nline two\nBREAK\nline three", negativePrompt = "n1,\nn2")
    val t = write(p, ImageFacts(seed = 5))
    assert(t.startsWith("line one,\nline two\nBREAK\nline three\nNegative prompt: n1,\nn2\nSteps: 20,"), t)
  }

  test("write: full field order (variation seed, VAE, clip skip, hires options)") {
    val p = GenerationParams(prompt = "p", negativePrompt = "n", steps = 30, samplerName = "Euler", scheduler = "Karras",
      cfgScale = 5, width = 512, height = 512, subseedStrength = 0.3, clipSkip = 2,
      hires = HiresParams(enabled = true, resizeX = 1024, resizeY = 1536, upscaler = "Latent", secondPassSteps = 15,
        denoisingStrength = 0.55, samplerName = Some("DPM++ SDE"), scheduler = Some("Exponential"),
        prompt = "hr, prompt\nwith \"quotes\"", negativePrompt = "hr neg", cfgScale = Some(4.5)))
    val f = ImageFacts(seed = 10, subseed = 20, modelName = Some("m"), modelHash = Some("h"), vaeName = Some("v.safetensors"),
      vaeHash = Some("vh"), loraHashes = Seq("l" -> "abc"))
    val keys = fields(p, f).keys.toSeq
    assertEquals(keys, Seq("Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size", "Model hash", "Model",
      "VAE hash", "VAE", "Variation seed", "Variation seed strength", "Denoising strength", "Clip skip", "Hires sampler",
      "Hires prompt", "Hires negative prompt", "Hires CFG Scale", "Hires schedule type", "Hires resize", "Hires steps",
      "Hires upscaler", "Lora hashes", "Version"))
    val t = write(p, f)
    assert(t.contains("Hires prompt: \"hr, prompt\\nwith \\\"quotes\\\"\", "), t)
    assert(t.contains("Hires resize: 1024x1536, Hires steps: 15, Hires upscaler: Latent, Lora hashes: \"l: abc\""), t)
    assert(t.contains("Variation seed: 20, Variation seed strength: 0.3, Denoising strength: 0.55, Clip skip: 2,"), t)
  }

  test("write: hires sampler / schedule equal to base are omitted") {
    val p = hiresParams.copy(hires = hiresParams.hires.copy(samplerName = Some("DPM++ 2M"), scheduler = Some("Karras")))
    val ks = fields(p, hiresFacts).keySet
    assert(!ks.contains("Hires sampler"))
    assert(!ks.contains("Hires schedule type"))
    // compared against the effective (engine-reported) schedule
    val p2 = p.copy(scheduler = "Automatic")
    assert(!fields(p2, hiresFacts.copy(scheduler = Some("Karras"))).contains("Hires schedule type"))
  }

  test("write: hires is ignored for img2img and when disabled") {
    val i2i = hiresParams.copy(img2img = Some(Img2ImgParams("in.png", denoisingStrength = 0.3)))
    val f = fields(i2i, hiresFacts)
    assert(!f.keys.exists(_.startsWith("Hires")), f)
    assertEquals(f("Denoising strength"), "0.3")
    val off = fields(hiresParams.copy(hires = hiresParams.hires.copy(enabled = false)), hiresFacts)
    assert(!off.keys.exists(_.startsWith("Hires")))
    assert(!off.contains("Denoising strength"))
  }

  test("write: inpaint fields in A1111 order, after Clip skip") {
    val p = GenerationParams(prompt = "p", clipSkip = 2,
      img2img = Some(Img2ImgParams("i.png", mask = Some("m.png"), denoisingStrength = 0.75, maskBlur = 4,
        inpaintFullRes = true, inpaintFullResPadding = 32, inpaintingFill = 0, inpaintingMaskInvert = true)))
    val f = fields(p, ImageFacts(seed = 1))
    assertEquals(f.keys.toSeq.dropWhile(_ != "Denoising strength"),
      Seq("Denoising strength", "Clip skip", "Mask mode", "Mask blur", "Inpaint area", "Masked area padding",
        "Masked content", "Version"))
    assertEquals(f("Mask mode"), "Inpaint not masked")
    assertEquals(f("Inpaint area"), "Only masked")
    assertEquals(f("Masked content"), "fill")
  }

  test("write: default inpaint (whole picture, original, blur 0) writes no mask fields") {
    val p = GenerationParams(img2img = Some(Img2ImgParams("i.png", mask = Some("m.png"), maskBlur = 0)))
    val f = fields(p, ImageFacts(seed = 1))
    Seq("Mask mode", "Mask blur", "Inpaint area", "Masked area padding", "Masked content").foreach(k => assert(!f.contains(k), k))
  }

  test("write: plain img2img has no mask fields") {
    val f = fields(GenerationParams(img2img = Some(Img2ImgParams("i.png", maskBlur = 4))), ImageFacts(seed = 1))
    assert(!f.contains("Mask blur"))
  }

  test("write: number formatting like Python str() for floats that matter") {
    val p = GenerationParams(cfgScale = 7.0, subseedStrength = 0.05,
      hires = HiresParams(enabled = true, scale = 2.0, denoisingStrength = 0.7))
    val f = fields(p, ImageFacts(seed = -1, subseed = 7))
    assertEquals(f("CFG scale"), "7")
    assertEquals(f("Variation seed strength"), "0.05")
    assertEquals(f("Hires upscale"), "2")
    assertEquals(f("Denoising strength"), "0.7")
    assertEquals(fields(GenerationParams(cfgScale = 0.1 + 0.2), ImageFacts(seed = 1))("CFG scale"), "0.30000000000000004")
    assertEquals(fields(GenerationParams(cfgScale = Double.NaN), ImageFacts(seed = 1))("CFG scale"), "NaN")
  }

  test("write: plugin extras (params + facts) appear before Version, facts override params") {
    val p = GenerationParams(prompt = "p", extra = ListMap("FreeU Stages" -> "[1, 2]", "Plugin flag" -> "on"))
    val f = ImageFacts(seed = 1, extra = ListMap("Plugin flag" -> "off", "Engine" -> "diffusers"))
    val ks = fields(p, f)
    assertEquals(ks.keys.toSeq.takeRight(4), Seq("FreeU Stages", "Plugin flag", "Engine", "Version"))
    assertEquals(ks("Plugin flag"), "off")
    assert(write(p, f).contains("FreeU Stages: \"[1, 2]\", Plugin flag: off, Engine: diffusers, Version:"))
  }

  test("render: key == value is written bare (A1111 flag style)") {
    assertEquals(render("p", "", ListMap("Steps" -> "1", "Tiling" -> "Tiling")), "p\nSteps: 1, Tiling")
  }

  // ---------------------------------------------------------------- parse

  test("parse: reForge hires fixture") {
    val pi = parse(reforgeHires)
    assertEquals(pi.prompt, "masterpiece, 1girl")
    assertEquals(pi.negativePrompt, "lowres, bad hands")
    assertEquals(pi.fields.keys.toSeq, Seq("Steps", "Sampler", "Schedule type", "CFG scale", "Seed", "Size", "Model hash",
      "Model", "Denoising strength", "Hires upscale", "Hires upscaler", "Lora hashes", "Version"))
    assertEquals(pi.get("Lora hashes"), Some("foo: abcdef123456, bar: 0123456789ab"))
    assertEquals(pi.get("Hires upscaler"), Some("R-ESRGAN 4x+"))
    assertEquals(pi.get("Version"), Some("f1.0.0v2-v1.10.1RC"))
    assertEquals(pi.get("nope"), None)
  }

  test("parse -> render reproduces the reForge text exactly") {
    val pi = parse(reforgeHires)
    assertEquals(render(pi.prompt, pi.negativePrompt, pi.fields), reforgeHires)
  }

  test("parse -> toParams: reForge hires fixture") {
    val p = toParams(parse(reforgeHires))
    assertEquals(p.prompt, "masterpiece, 1girl")
    assertEquals(p.negativePrompt, "lowres, bad hands")
    assertEquals(p.steps, 28)
    assertEquals(p.samplerName, "DPM++ 2M")
    assertEquals(p.scheduler, "Karras")
    assertEquals(p.cfgScale, 7.0)
    assertEquals(p.seed, 12345L)
    assertEquals((p.width, p.height), (832, 1216))
    assertEquals(p.hires.enabled, true)
    assertEquals(p.hires.scale, 1.5)
    assertEquals(p.hires.upscaler, "R-ESRGAN 4x+")
    assertEquals(p.hires.denoisingStrength, 0.4)
    assertEquals(p.hires.secondPassSteps, 0)
    assertEquals(p.clipSkip, 1)
    assertEquals(p.subseedStrength, 0.0)
    assertEquals(p.extra, Map.empty[String, String])
  }

  test("legacy combined sampler is split") {
    val pi = parse(legacy)
    val p = toParams(pi)
    assertEquals(p.samplerName, "DPM++ 2M")
    assertEquals(p.scheduler, "Karras")
    assertEquals(p.cfgScale, 7.5)
    assertEquals(p.clipSkip, 2)
    assertEquals(p.extra, Map("ENSD" -> "31337"))
    assertEquals(p.hires.enabled, false)
  }

  test("splitSampler") {
    assertEquals(splitSampler("DPM++ 2M Karras", None), ("DPM++ 2M", Some("Karras")))
    assertEquals(splitSampler("DPM++ 2M SDE Exponential", None), ("DPM++ 2M SDE", Some("Exponential")))
    assertEquals(splitSampler("DPM++ 2M SGM Uniform", None), ("DPM++ 2M", Some("SGM Uniform")))
    assertEquals(splitSampler("Euler SGMUniform", None), ("Euler", Some("SGM Uniform")))
    assertEquals(splitSampler("Euler Uniform", None), ("Euler", Some("Uniform")))
    assertEquals(splitSampler("DPM2 a karras", None), ("DPM2 a", Some("Karras")))
    assertEquals(splitSampler("Euler a", None), ("Euler a", None))
    assertEquals(splitSampler("Karras", None), ("Karras", None))
    // explicit Schedule type wins, sampler untouched
    assertEquals(splitSampler("DPM++ 2M Karras", Some("Exponential")), ("DPM++ 2M Karras", Some("Exponential")))
  }

  test("parse: multi-line prompt and negative") {
    val t = "line 1,\nline 2\n\nline 4\nNegative prompt: bad,\nworse\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 1, Size: 512x512"
    val pi = parse(t)
    assertEquals(pi.prompt, "line 1,\nline 2\n\nline 4")
    assertEquals(pi.negativePrompt, "bad,\nworse")
    assertEquals(pi.fields.size, 5)
  }

  test("parse: lines are trimmed like A1111") {
    val pi = parse("  a  \n  Negative prompt:   b  \nSteps: 1, Sampler: Euler, Seed: 2")
    assertEquals(pi.prompt, "a")
    assertEquals(pi.negativePrompt, "b")
  }

  test("parse: empty negative prompt") {
    val pi = parse("cat\nNegative prompt: \nSteps: 1, Sampler: Euler, Seed: 2")
    assertEquals(pi.negativePrompt, "")
    assertEquals(pi.prompt, "cat")
    val pi2 = parse("cat\nSteps: 1, Sampler: Euler, Seed: 2")
    assertEquals(pi2.negativePrompt, "")
  }

  test("parse: prompt containing colons and parentheses stays in the prompt") {
    val t = "(masterpiece:1.2), style: anime, artist: someone\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512"
    val pi = parse(t)
    assertEquals(pi.prompt, "(masterpiece:1.2), style: anime, artist: someone")
    assertEquals(pi.fields("Sampler"), "Euler a")
  }

  test("parse: last line with fewer than 3 params is treated as prompt (A1111 rule)") {
    val pi = parse("hello\nworld: yes, foo: bar")
    assertEquals(pi.prompt, "hello\nworld: yes, foo: bar")
    assert(pi.fields.isEmpty)
  }

  test("parse: params only / empty text") {
    val pi = parse("Steps: 20, Sampler: Euler, Seed: 5")
    assertEquals(pi.prompt, "")
    assertEquals(pi.fields("Seed"), "5")
    val e = parse("")
    assertEquals(e, ParsedInfotext("", "", ListMap.empty))
    assertEquals(parse("   \n  "), ParsedInfotext("", "", ListMap.empty))
  }

  test("parse: quoted hires prompt with commas, newlines and escaped quotes") {
    val t = "p\nSteps: 20, Sampler: Euler, Seed: 1, Hires prompt: \"a, b\\nc \\\"quoted, inner\\\" d\", " +
      "Hires negative prompt: \"x: y\", Hires upscaler: Latent, Version: v1.10.1"
    val pi = parse(t)
    assertEquals(pi.fields("Hires prompt"), "a, b\nc \"quoted, inner\" d")
    assertEquals(pi.fields("Hires negative prompt"), "x: y")
    assertEquals(pi.fields("Hires upscaler"), "Latent")
    assertEquals(pi.fields("Version"), "v1.10.1")
  }

  test("parse: Japanese prompt, negative and field values") {
    val t = "桜の木の下で微笑む少女, 着物\nNegative prompt: 低品質, ぼやけ\n" +
      "Steps: 20, Sampler: Euler a, Schedule type: Automatic, CFG scale: 6.5, Seed: 987654321, Size: 512x768, " +
      "Model: 日本語モデル, ユーザー設定: \"値, その他\", Version: v1.10.1"
    val pi = parse(t)
    assertEquals(pi.prompt, "桜の木の下で微笑む少女, 着物")
    assertEquals(pi.negativePrompt, "低品質, ぼやけ")
    assertEquals(pi.fields("Model"), "日本語モデル")
    // Python's \w is Unicode-aware, so non-ASCII keys parse too
    assertEquals(pi.fields("ユーザー設定"), "値, その他")
    val p = toParams(pi)
    assertEquals(p.extra.get("ユーザー設定"), Some("値, その他"))
    assertEquals(p.cfgScale, 6.5)
  }

  test("parse: Windows line endings") {
    val pi = parse("a cat\r\nNegative prompt: dog\r\nSteps: 20, Sampler: Euler, Seed: 3\r\n")
    assertEquals(pi.prompt, "a cat")
    assertEquals(pi.negativePrompt, "dog")
    assertEquals(pi.fields("Seed"), "3")
  }

  test("parse: bad Size is ignored by toParams") {
    val p = toParams(parse("x\nSteps: 20, Sampler: Euler, Seed: 3, Size: big"))
    assertEquals((p.width, p.height), (512, 512))
    assertEquals(toParams(parse("x\nSteps: abc, Sampler: Euler, Seed: x")).steps, 20)
  }

  test("toParams uses base for missing values, resets per-image ones") {
    val base = GenerationParams(steps = 33, cfgScale = 4, width = 640, height = 640, clipSkip = 2, subseed = 5,
      subseedStrength = 0.5, samplerName = "Euler", scheduler = "Simple", checkpoint = Some("ck"), styles = Seq("s"),
      hires = HiresParams(enabled = true, upscaler = "Keep me"))
    val p = toParams(parse("only prompt\nSeed: 9, Foo: 1, Bar: 2"), base)
    assertEquals(p.steps, 33)
    assertEquals(p.cfgScale, 4.0)
    assertEquals((p.width, p.height), (640, 640))
    assertEquals(p.samplerName, "Euler")
    assertEquals(p.scheduler, "Simple")
    assertEquals(p.seed, 9L)
    assertEquals(p.clipSkip, 1) // missing Clip skip means 1 (A1111)
    assertEquals(p.subseed, -1L)
    assertEquals(p.subseedStrength, 0.0)
    assertEquals(p.hires.enabled, false)
    assertEquals(p.hires.upscaler, "Keep me")
    assertEquals(p.checkpoint, Some("ck"))
    assertEquals(p.styles, Seq("s"))
  }

  // ---------------------------------------------------------------- write -> parse -> toParams round trips

  private def roundTrip(p: GenerationParams, f: ImageFacts, base: GenerationParams = GenerationParams()): GenerationParams =
    toParams(parse(write(p, f)), base)

  test("round trip: txt2img") {
    val p = GenerationParams(prompt = "a, b: c\n(d:1.2)", negativePrompt = "neg, ative", seed = 77, steps = 25,
      samplerName = "DPM++ SDE", scheduler = "Karras", cfgScale = 5.5, width = 768, height = 1344)
    assertEquals(roundTrip(p, ImageFacts(seed = 77)), p)
  }

  test("round trip: hires with everything") {
    val p = GenerationParams(prompt = "p", negativePrompt = "n", seed = 1, samplerName = "Euler a", scheduler = "Automatic",
      hires = HiresParams(enabled = true, scale = 1.75, upscaler = "4x-UltraSharp", secondPassSteps = 12,
        denoisingStrength = 0.35, samplerName = Some("DPM++ 2M"), scheduler = Some("Karras"),
        prompt = "hi, \"res\"\nline", negativePrompt = "hr: neg", cfgScale = Some(3.5)))
    assertEquals(roundTrip(p, ImageFacts(seed = 1)), p)
  }

  test("round trip: hires resize") {
    val p = GenerationParams(prompt = "p", seed = 1, scheduler = "Karras",
      hires = HiresParams(enabled = true, resizeX = 1024, resizeY = 0, upscaler = "Latent (nearest-exact)", denoisingStrength = 0.6))
    val back = roundTrip(p, ImageFacts(seed = 1))
    assertEquals(back.hires, p.hires)
  }

  test("round trip: img2img (base supplies the init image)") {
    val i2i = Img2ImgParams("C:/init.png", denoisingStrength = 0.45)
    val p = GenerationParams(prompt = "p", seed = 2, scheduler = "Karras", img2img = Some(i2i))
    val back = roundTrip(p, ImageFacts(seed = 2), GenerationParams(img2img = Some(Img2ImgParams("C:/init.png"))))
    assertEquals(back, p)
  }

  test("round trip: inpaint (only masked, invert, latent noise)") {
    val i2i = Img2ImgParams("C:/init.png", mask = Some("C:/mask.png"), denoisingStrength = 0.8, maskBlur = 6,
      inpaintFullRes = true, inpaintFullResPadding = 64, inpaintingFill = 2, inpaintingMaskInvert = true)
    val p = GenerationParams(prompt = "p", seed = 3, scheduler = "Karras", img2img = Some(i2i))
    val base = GenerationParams(img2img = Some(Img2ImgParams("C:/init.png", mask = Some("C:/mask.png"))))
    assertEquals(roundTrip(p, ImageFacts(seed = 3), base), p)
  }

  test("round trip: inpaint defaults reset base inpaint options") {
    val p = GenerationParams(prompt = "p", seed = 3, scheduler = "Karras",
      img2img = Some(Img2ImgParams("i.png", mask = Some("m.png"), maskBlur = 4)))
    val base = GenerationParams(img2img = Some(Img2ImgParams("i.png", mask = Some("m.png"), inpaintFullRes = true,
      inpaintFullResPadding = 8, inpaintingFill = 0, inpaintingMaskInvert = true)))
    assertEquals(roundTrip(p, ImageFacts(seed = 3), base), p)
  }

  test("round trip: variation seed") {
    val p = GenerationParams(prompt = "p", seed = 10, subseed = 4242, subseedStrength = 0.15, scheduler = "Karras")
    assertEquals(roundTrip(p, ImageFacts(seed = 10, subseed = 4242)), p)
  }

  test("round trip: clip skip") {
    val p = GenerationParams(prompt = "p", seed = 10, clipSkip = 2, scheduler = "Karras")
    assertEquals(roundTrip(p, ImageFacts(seed = 10)), p)
  }

  test("round trip: Japanese text everywhere") {
    val p = GenerationParams(prompt = "桜、少女\n二行目: 夜景", negativePrompt = "低品質, ぼやけ", seed = 5, scheduler = "Karras",
      hires = HiresParams(enabled = true, prompt = "高解像度, 詳細", denoisingStrength = 0.5),
      extra = ListMap("プラグイン" -> "有効, 強"))
    assertEquals(roundTrip(p, ImageFacts(seed = 5, modelName = Some("日本語モデル"))), p)
  }

  test("round trip: extra keys preserved in order and re-written") {
    val t = "p\nSteps: 20, Sampler: Euler, Schedule type: Karras, CFG scale: 7, Seed: 1, Size: 512x512, " +
      "ADetailer model: face_yolov8n.pt, ADetailer confidence: 0.3, ADetailer version: 24.1.2, " +
      "FreeU Stages: \"[{\\\"backbone_factor\\\": 1.2, \\\"skip_factor\\\": 0.9}]\", Version: v1.10.1"
    val p = toParams(parse(t))
    assertEquals(p.extra.keys.toSeq, Seq("ADetailer model", "ADetailer confidence", "ADetailer version", "FreeU Stages"))
    assertEquals(p.extra("FreeU Stages"), "[{\"backbone_factor\": 1.2, \"skip_factor\": 0.9}]")
    val again = write(p, ImageFacts(seed = 1))
    assert(again.contains("Size: 512x512, ADetailer model: face_yolov8n.pt, ADetailer confidence: 0.3, " +
      "ADetailer version: 24.1.2, FreeU Stages: \"[{\\\"backbone_factor\\\": 1.2, \\\"skip_factor\\\": 0.9}]\", Version: "), again)
    assertEquals(toParams(parse(again)), p)
  }

  test("known keys are not leaked into extra") {
    val p = toParams(parse(reforgeHires))
    assert(p.extra.keySet.intersect(knownKeys).isEmpty)
  }
