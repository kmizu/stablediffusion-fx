package sdfx.plugins

import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.*

import javax.imageio.ImageIO

class ScriptsSuite extends munit.FunSuite:
  def defaults(p: Plugin, kv: (String, String)*) = Values(Values.defaults(p.controls).map ++ kv)
  val base = GenerationParams(prompt = "a cat", seed = 100, width = 64, height = 64)

  // ------------------------------------------------------------- prompt matrix
  test("prompt matrix combinations are 2^n in A1111 order") {
    val c = PromptMatrixPlugin.combinations("a cat|red|big|fluffy", false, ", ")
    assertEquals(c.size, 8)
    assertEquals(c.map(_.prompt).take(4), Seq("a cat", "a cat, red", "a cat, big", "a cat, red, big"))
    assertEquals(c.last.prompt, "a cat, red, big, fluffy")
    val s = PromptMatrixPlugin.combinations("a cat|red", true, " ")
    assertEquals(s.map(_.prompt), Seq("a cat", "red a cat"))
    assertEquals(PromptMatrixPlugin.layout(3), (4, 2))
    assertEquals(PromptMatrixPlugin.layout(2), (2, 2))
  }

  test("prompt matrix run: one image per combo, seeds, grid") {
    val pm = new PromptMatrixPlugin
    val ctx = FakeContext()
    val out = pm.run(base.copy(prompt = "a cat|red|big"), defaults(pm, "different_seeds" -> "true"), ctx)
    assertEquals(ctx.generated.map(_.prompt), Seq("a cat", "a cat, red", "a cat, big", "a cat, red, big"))
    assertEquals(ctx.generated.map(_.seed), Seq(100L, 101L, 102L, 103L))
    assertEquals(out.images.size, 4)
    assertEquals(out.grids.size, 1)
    val g = ctx.saved.head._2
    assert(g.getWidth >= 128 && g.getHeight >= 128)
    val ctx2 = FakeContext()
    pm.run(base.copy(negativePrompt = "ugly|blurry"), defaults(pm, "prompt_type" -> "negative"), ctx2)
    assertEquals(ctx2.generated.map(_.negativePrompt), Seq("ugly", "ugly, blurry"))
    assertEquals(ctx2.generated.map(_.seed).distinct, Seq(100L))
  }

  test("prompt matrix interrupted mid-generation still saves a partial grid") {
    val pm = new PromptMatrixPlugin
    val ctx = FakeContext(interruptAfter = 3)
    val out = pm.run(base.copy(prompt = "a cat|red|big"), defaults(pm), ctx)
    assertEquals(ctx.generated.size, 3)
    assertEquals(ctx.interruptedCalls, 1)
    assertEquals(out.images.size, 3)
    assertEquals(out.grids.size, 1)
  }

  // ------------------------------------------------------------- prompts from file
  test("shlex splitting") {
    assertEquals(PromptsFromFilePlugin.shlex("""--prompt "a, b \"c\"" --steps 20 'x y'"""), Seq("--prompt", "a, b \"c\"", "--steps", "20", "x y"))
    intercept[IllegalArgumentException](PromptsFromFilePlugin.shlex("\"open"))
  }

  test("cmdargs parses options; prompt swallows bare words") {
    val r = PromptsFromFilePlugin.cmdargs("--prompt a red cat --steps 30 --cfg_scale 6.5 --negative_prompt \"bad, ugly\" --seed 7 --width 768 --sampler_name \"euler a\"")
    assertEquals(
      r,
      Right(Map("prompt" -> "a red cat", "steps" -> "30", "cfg_scale" -> "6.5", "negative_prompt" -> "bad, ugly", "seed" -> "7", "width" -> "768", "sampler_name" -> "euler a"))
    )
    assert(PromptsFromFilePlugin.cmdargs("--bogus 1").isLeft)
    assert(PromptsFromFilePlugin.cmdargs("--steps").isLeft)
    assert(PromptsFromFilePlugin.cmdargs("--steps many").isLeft)
    assertEquals(PromptsFromFilePlugin.parseLine("plain -- dashes"), Map("prompt" -> "plain -- dashes"))
    assertEquals(PromptsFromFilePlugin.parseLine("just a prompt"), Map("prompt" -> "just a prompt"))
  }

  test("prompts from file run: per-line options, sampler mapping, insert position, iterate seed") {
    val pf = new PromptsFromFilePlugin
    val ctx = FakeContext()
    val text = "dog\n\n--prompt bird --steps 5 --sampler_name \"euler a\" --batch_size 2\n--negative_prompt blurry --cfg_scale 3"
    val out = pf.run(base.copy(negativePrompt = "bad"), defaults(pf, "prompts" -> text, "iterate_seed" -> "true", "prompt_position" -> "end"), ctx)
    assertEquals(ctx.generated.size, 3)
    assertEquals(out.images.size, 4)
    assertEquals(ctx.generated.map(_.prompt), Seq("a cat dog", "a cat bird", "a cat"))
    assertEquals(ctx.generated(1).steps, 5)
    assertEquals(ctx.generated(1).samplerName, "Euler a")
    assertEquals(ctx.generated(2).negativePrompt, "bad blurry")
    assertEquals(ctx.generated(2).cfgScale, 3.0)
    assertEquals(ctx.generated.map(_.seed), Seq(100L, 101L, 102L))
  }

  test("prompts from file: same random seed for all lines") {
    val pf = new PromptsFromFilePlugin
    val ctx = FakeContext()
    pf.run(base.copy(prompt = "", seed = -1), defaults(pf, "prompts" -> "a\nb\nc", "same_seed" -> "true"), ctx)
    assertEquals(ctx.generated.map(_.prompt), Seq("a", "b", "c"))
    assert(ctx.generated.head.seed >= 0)
    assertEquals(ctx.generated.map(_.seed).distinct.size, 1)
  }

  test("prompts from file stops when interrupted") {
    val pf = new PromptsFromFilePlugin
    val ctx = FakeContext(interruptAfter = 1)
    val out = pf.run(base, defaults(pf, "prompts" -> "a\nb\nc"), ctx)
    assertEquals(ctx.generated.size, 1)
    assertEquals(ctx.interruptedCalls, 1)
    assertEquals(out.images.size, 1)
  }

  // ------------------------------------------------------------- loopback
  test("loopback denoising curves") {
    def d(i: Int, c: String) = LoopbackPlugin.denoising(i, 5, 0.8, 0.4, c)
    assertEqualsDouble(d(0, "Linear"), 0.8, 1e-9)
    assertEqualsDouble(d(2, "Linear"), 0.6, 1e-9)
    assertEqualsDouble(d(4, "Linear"), 0.4, 1e-9)
    assertEqualsDouble(d(1, "Aggressive"), 0.8 - 0.4 * math.sin(0.25 * math.Pi * 0.5), 1e-9)
    assertEqualsDouble(d(1, "Lazy"), 0.8 - 0.4 * (1 - math.cos(0.25 * math.Pi * 0.5)), 1e-9)
    assert(d(1, "Aggressive") < d(1, "Linear") && d(1, "Linear") < d(1, "Lazy"))
    assertEqualsDouble(LoopbackPlugin.denoising(0, 1, 0.7, 0.1, "Linear"), 0.7, 1e-9)
  }

  test("loopback chains init images, seeds and denoising; grid of iterations") {
    val lb = new LoopbackPlugin
    assertEquals(lb.tabs, Set(Tab.Img2Img))
    val ctx = FakeContext()
    val init = ctx.initImage(64, 64)
    val p = base.copy(img2img = Some(Img2ImgParams(init, denoisingStrength = 0.8)))
    val out = lb.run(p, defaults(lb, "loops" -> "3", "final_denoising_strength" -> "0.2"), ctx)
    assertEquals(ctx.generated.size, 3)
    assertEquals(ctx.generated.head.img2img.get.initImage, init)
    assertEquals(ctx.generated(1).img2img.get.initImage, out.images(0).path.toString)
    assertEquals(ctx.generated(2).img2img.get.initImage, out.images(1).path.toString)
    assertEquals(ctx.generated.map(_.img2img.get.denoisingStrength).map(x => math.round(x * 100)), Seq(80L, 50L, 20L))
    assertEquals(ctx.generated.map(_.seed), Seq(100L, 101L, 102L))
    assertEquals(out.grids.size, 1)
    assertEquals(ctx.saved.head._2.getWidth, 192)
    assertEquals(ctx.generated.head.extra("Denoising curve"), "Linear")
  }

  test("loopback batch semantics: n_iter outer, every iteration runs the whole batch with seed + k") {
    val lb = new LoopbackPlugin
    val ctx = FakeContext()
    val init = ctx.initImage(64, 64)
    val p = base.copy(batchSize = 2, nIter = 2, img2img = Some(Img2ImgParams(init, mask = Some(init), inpaintingFill = 0)))
    val out = lb.run(p, defaults(lb, "loops" -> "2"), ctx)
    assertEquals(ctx.generated.size, 8)
    assert(ctx.generated.forall(g => g.batchSize == 1 && g.nIter == 1))
    assertEquals(ctx.generated.map(_.seed), Seq(100L, 101L, 101L, 102L, 102L, 103L, 103L, 104L))
    // member k of iteration 2 continues from member k of iteration 1
    assertEquals(ctx.generated(2).img2img.get.initImage, out.images(0).path.toString)
    assertEquals(ctx.generated(3).img2img.get.initImage, out.images(2).path.toString)
    // a new repetition restarts from the original init image
    assertEquals(ctx.generated(4).img2img.get.initImage, init)
    assertEquals(ctx.generated.map(_.img2img.get.inpaintingFill), Seq(0, 0, 1, 1, 0, 0, 1, 1))
    assertEquals(ctx.saved.head._2.getWidth, 128) // 4 chains x 2 iterations
    assertEquals(ctx.saved.head._2.getHeight, 256)
  }

  test("loopback stops on interrupt, keeps a partial grid, and requires img2img") {
    val lb = new LoopbackPlugin
    val ctx = FakeContext(interruptAfter = 2)
    val p = base.copy(img2img = Some(Img2ImgParams(ctx.initImage(64, 64))))
    val out = lb.run(p, defaults(lb, "loops" -> "10"), ctx)
    assertEquals(ctx.generated.size, 2)
    assertEquals(ctx.interruptedCalls, 1)
    assertEquals(out.images.size, 2)
    assertEquals(out.grids.size, 1)
    intercept[IllegalArgumentException](lb.run(base, defaults(lb), FakeContext()))
  }

  // ------------------------------------------------------------- SD upscale
  test("SD upscale tiles the upscaled image and reassembles at full size") {
    val su = new SdUpscalePlugin
    val ctx = FakeContext(echoInit = true)
    val init = ctx.gradientInit(100, 80)
    val p = base.copy(width = 64, height = 64, img2img = Some(Img2ImgParams(init, denoisingStrength = 0.3)))
    val out = su.run(p, defaults(su, "upscaler" -> "Lanczos", "scale" -> "2", "overlap" -> "16"), ctx)
    assertEquals(ctx.upscales.map(u => (u._2, u._3)).toSeq, Seq(("Lanczos", 2.0)))
    // 200x160 with 64px tiles, 16 overlap -> ceil(184/48)=4 cols, ceil(144/48)=3 rows
    // tiles go through generateTransient only: zero regular generations, exactly one saved final image
    assertEquals(ctx.generated.size, 0)
    assertEquals(ctx.outputPaths.size, 0)
    assertEquals(ctx.transientCalls.size, 12)
    assertEquals(ctx.saved.size, 1)
    val tiles = ctx.transientCalls
    assert(tiles.forall(g => g.width == 64 && g.height == 64 && g.img2img.get.denoisingStrength == 0.3))
    assertEquals(tiles.map(_.seed), (100L until 112L)) // seed + tile index, like A1111
    // transient tile files, tile temp files and the upscaled temp are removed
    assert(ctx.transientPaths.forall(f => !java.nio.file.Files.exists(f)))
    assert(tiles.forall(g => !java.nio.file.Files.exists(java.nio.file.Paths.get(g.img2img.get.initImage))))
    assert(ctx.dir.resolve("upscaled").toFile.listFiles().isEmpty)
    assertEquals(out.images.size, 1)
    val (kind, merged, gp, _) = ctx.saved.head
    assertEquals(kind, "img2img")
    assertEquals((merged.getWidth, merged.getHeight), (200, 160))
    assertEquals((gp.width, gp.height), (200, 160))
    // identity img2img -> merged equals the upscaled image
    val up = sdfx.plugins.imageops.ImageOps.resize(ImageIO.read(new java.io.File(init)), 200, 160)
    for y <- 0 until 160 by 3; x <- 0 until 200 by 3 do assertEquals(merged.getRGB(x, y), up.getRGB(x, y), s"$x,$y")
  }

  test("SD upscale interrupt leaves no merged image") {
    val su = new SdUpscalePlugin
    val ctx = FakeContext(interruptAfter = 3)
    val p = base.copy(img2img = Some(Img2ImgParams(ctx.initImage(100, 100))))
    val out = su.run(p, defaults(su, "scale" -> "2"), ctx)
    assertEquals(ctx.transientCalls.size, 3)
    assertEquals(ctx.generated.size, 0)
    assertEquals(ctx.interruptedCalls, 1)
    assertEquals(out.images, Nil)
    assertEquals(ctx.saved.size, 0)
    assert(ctx.transientPaths.forall(f => !java.nio.file.Files.exists(f)))
  }

  // ------------------------------------------------------------- outpainting
  test("outpaint builds expanded canvas + mask and inpaints once") {
    val op = new OutpaintPlugin
    val ctx = FakeContext()
    val p = base.copy(img2img = Some(Img2ImgParams(ctx.initImage(64, 64), maskBlur = 1)))
    op.run(p, defaults(op, "pixels" -> "32", "down" -> "false", "up" -> "false", "fill" -> "latent noise", "mask_blur" -> "8"), ctx)
    assertEquals(ctx.generated.size, 1)
    val g = ctx.generated.head
    assertEquals((g.width, g.height), (128, 64))
    val ii = g.img2img.get
    assertEquals(ii.inpaintingFill, 2)
    assertEquals(ii.maskBlur, 8)
    val (canvas, maskOpt) = ctx.inputs.head
    val mask = maskOpt.get
    assert(!java.nio.file.Files.exists(java.nio.file.Paths.get(ii.initImage)), "temp canvas removed")
    assert(!java.nio.file.Files.exists(java.nio.file.Paths.get(ii.mask.get)), "temp mask removed")
    assertEquals((canvas.getWidth, canvas.getHeight), (128, 64))
    assertEquals((mask.getWidth, mask.getHeight), (128, 64))
    assertEquals(mask.getRaster.getSample(10, 30, 0), 255)
    assertEquals(mask.getRaster.getSample(64, 30, 0), 0)
    assertEquals(mask.getRaster.getSample(120, 30, 0), 255)
    assertEquals(g.extra("Outpaint directions"), "left right")
  }

  // ------------------------------------------------------------- upscale extras
  test("upscale extras: img2img upscales the init image only") {
    val ue = new UpscaleExtrasPlugin
    val ctx = FakeContext()
    val p = base.copy(img2img = Some(Img2ImgParams(ctx.initImage(50, 40))))
    val out = ue.run(p, defaults(ue, "upscaler" -> "R-ESRGAN 4x+", "scale" -> "2"), ctx)
    assertEquals(ctx.generated.size, 0)
    assertEquals(out.images.size, 1)
    assertEquals(ctx.saved.head._1, "extras")
    assertEquals(ctx.saved.head._2.getWidth, 100)
    assert(ctx.saved.head._4.contains("Postprocess upscaler: R-ESRGAN 4x+"))
  }

  test("upscale extras: txt2img generates then upscales every image") {
    val ue = new UpscaleExtrasPlugin
    val ctx = FakeContext()
    val out = ue.run(base.copy(batchSize = 2), defaults(ue, "upscaler" -> "Lanczos", "scale" -> "1.5"), ctx)
    assertEquals(ctx.generated.size, 1)
    assertEquals(out.images.size, 2)
    assert(ctx.saved.forall(s => s._1 == "extras" && s._2.getWidth == 96))
    assert(ctx.saved.head._4.startsWith("a cat"))
  }

  // ------------------------------------------------------------- registry
  test("standard plugins have unique ids in a stable order") {
    val ps = StandardPlugins.all(() => Nil)
    assertEquals(
      ps.map(_.id),
      Seq("xyz_grid", "prompt_matrix", "prompts_from_file", "loopback", "sd_upscale", "outpaint", "outpaint_mk2", "img2img_alt",
        "svd", "zero123", "custom_code", "upscale_extras", "face_restore_extras", "caption", "flipped_copies", "split_oversized",
        "autosized_crop", "focal_crop", "freeu", "dynamic_thresholding", "apg", "mahiro", "rescale_cfg", "pag", "sag", "hypertile",
        "kohya_hrfix", "multidiffusion", "model_sampling", "latent_modifier", "random_resolutions", "reference", "revision",
        "ip_adapter", "controllllite", "photomaker", "soft_inpainting", "fooocus_inpaint", "face_restore", "neveroom")
    )
    assert(ps.forall(p => p.controls.map(_.key).distinct.size == p.controls.size), "duplicate control keys")
  }
