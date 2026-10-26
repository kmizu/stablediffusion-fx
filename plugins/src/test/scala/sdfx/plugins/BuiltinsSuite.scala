package sdfx.plugins

import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.*
import sdfx.plugins.imageops.ImageOps

import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO

/** The reForge built-in rewrites added on top of the first standard plugin set. */
class BuiltinsSuite extends munit.FunSuite:

  private def defaults(p: Plugin, extra: (String, String)*): Values =
    Values(Values.defaults(p.controls).map ++ extra.toMap)

  private def hookOf(p: Plugin, v: Values, base: GenerationParams = GenerationParams()): ujson.Obj =
    p.process(base, v).hooks(p.asInstanceOf[HookPlugin].hook).obj

  test("every reForge built-in in docs/PLUGINS.md has a plugin id"):
    val ids = StandardPlugins.withOwnAxes.map(_.id).toSet
    val expected = Set("xyz_grid", "prompt_matrix", "prompts_from_file", "loopback", "sd_upscale", "outpaint", "outpaint_mk2",
      "img2img_alt", "custom_code", "autosized_crop", "upscale_extras", "split_oversized", "flipped_copies", "focal_crop",
      "face_restore", "face_restore_extras", "caption", "reference", "revision", "ip_adapter", "controllllite", "photomaker",
      "fooocus_inpaint", "freeu", "dynamic_thresholding", "hypertile", "sag", "kohya_hrfix", "latent_modifier",
      "multidiffusion", "neveroom", "svd", "zero123", "soft_inpainting", "random_resolutions", "apg", "rescale_cfg",
      "model_sampling", "mahiro", "pag")
    assertEquals(expected -- ids, Set.empty[String])
    assertEquals(ids.size, StandardPlugins.withOwnAxes.size, "ids must be unique")
    val doc = Files.readString(Paths.get("docs/PLUGINS.md"))
    expected.foreach(id => assert(doc.contains(s"`$id`"), s"docs/PLUGINS.md misses $id"))

  test("every extension-index entry marked done names a real plugin (docs/extensions-status.json)"):
    val ids = StandardPlugins.withOwnAxes.map(_.id).toSet
    val status = ujson.read(Files.readString(Paths.get("docs/extensions-status.json"))).arr
    assertEquals(status.size, 353)
    val done = status.map(_("status").str).filter(_.startsWith("done:")).map(_.stripPrefix("done:")).filterNot(_.startsWith("core:"))
    assert(done.nonEmpty)
    assertEquals(done.filterNot(ids.contains).toSet, Set.empty[String])
    assert(status.forall(e => Seq("done:", "todo:", "na:").exists(e("status").str.startsWith)))

  test("every hook plugin declares its engine feature and emits JSON with numeric/bool types"):
    StandardPlugins.alwaysOn.collect { case h: HookPlugin => h }.foreach { h =>
      assertEquals(h.requiresEngineFeatures, Seq(h.hook))
      val v = defaults(h, "enabled" -> "true")
      val o = hookOf(h, v)
      h.controls.foreach {
        case c: Control.Slider if o.value.contains(c.key) => assert(o(c.key).isInstanceOf[ujson.Num], s"${h.id}.${c.key}")
        case c: Control.Check if o.value.contains(c.key) => assert(o(c.key).isInstanceOf[ujson.Bool], s"${h.id}.${c.key}")
        case _ => ()
      }
    }

  test("NegPiP is opt-in and preserves its replay key"):
    val p = new NegPipPlugin
    assert(!p.isActive(defaults(p)))
    assert(p.isActive(defaults(p, "enabled" -> "true")))
    val out = p.process(GenerationParams(), defaults(p, "enabled" -> "true"))
    assertEquals(out.hooks("negpip"), ujson.Obj())
    assertEquals(out.extra("NegPiP Active"), "True")
    assert(StandardPlugins.alwaysOn.exists(_.id == "negpip"))

  test("APG / RescaleCFG / Kohya / MultiDiffusion hook fields match the engine contract"):
    val apg = hookOf(new ApgPlugin, defaults(new ApgPlugin, "enabled" -> "true"))
    assertEquals(apg.value.keySet.toSet, Set("momentum", "adaptive_momentum", "norm_threshold", "eta", "guidance_limiter",
      "guidance_sigma_start", "guidance_sigma_end"))
    assertEquals(apg("momentum").num, 0.5)
    assertEquals(hookOf(new RescaleCfgPlugin, defaults(new RescaleCfgPlugin))("multiplier").num, 0.7)
    val k = hookOf(new KohyaHrFixPlugin, defaults(new KohyaHrFixPlugin))
    assertEquals(k("block_number").num, 3.0)
    assertEquals(k("downscale_after_skip").bool, true)
    val md = hookOf(new MultiDiffusionPlugin, defaults(new MultiDiffusionPlugin))
    assertEquals(md("tile_width").num, 768.0)

  test("X/Y/Z axes of the new hook plugins override their slider values"):
    val p = new ApgPlugin
    val axis = p.xyzAxes.find(_.name == "APG eta").get
    val base = axis(GenerationParams(), "0.25")
    assertEquals(hookOf(p, defaults(p, "enabled" -> "true"), base)("eta").num, 0.25)

  test("image plugins stay inactive until an image (and a model) is chosen"):
    val ref = new ReferencePlugin
    assert(!ref.isActive(defaults(ref, "enabled" -> "true")))
    val img = Files.createTempFile("ref", ".png")
    ImageIO.write(new java.awt.image.BufferedImage(8, 8, 1), "png", img.toFile)
    assert(ref.isActive(defaults(ref, "enabled" -> "true", "image" -> img.toString)))
    assertEquals(hookOf(ref, defaults(ref, "enabled" -> "true", "image" -> img.toString))("image").str, img.toString)
    val ip = new IpAdapterPlugin
    assert(!ip.isActive(defaults(ip, "enabled" -> "true", "image" -> img.toString)))
    assert(ip.isActive(defaults(ip, "enabled" -> "true", "image" -> img.toString, "model" -> "ip-adapter_sd15")))
    val pm = new PhotoMakerPlugin
    val pmv = defaults(pm, "enabled" -> "true", "model" -> "photomaker-v1", "images" -> s"${img}\n\n$img")
    assertEquals(hookOf(pm, pmv)("images").arr.size, 2)
    val fr = new FaceRestorePlugin
    assert(!fr.isActive(defaults(fr, "enabled" -> "true")))
    assert(fr.isActive(defaults(fr, "enabled" -> "true", "model" -> "GFPGANv1.4")))

  test("never OOM is active when either toggle is on, and has no 'enabled' control"):
    val n = new NeverOomPlugin
    assert(!n.isActive(defaults(n)))
    assert(n.isActive(defaults(n, "vae_tiling" -> "true")))
    assert(!n.controls.exists(_.key == "enabled"))

  test("random resolutions picks reproducibly from the list with the seed"):
    val r = new RandomResolutionsPlugin
    assertEquals(r.parse("1024x1024, 832 x 1216\n1216×832 nonsense 13x7"), Seq((1024, 1024), (832, 1216), (1216, 832), (64, 64)))
    val v = defaults(r, "enabled" -> "true")
    val a = r.process(GenerationParams(seed = 42), v)
    val b = r.process(GenerationParams(seed = 42), v)
    assertEquals((a.width, a.height), (b.width, b.height))
    assert(r.parse(v.str("resolutions")).contains((a.width, a.height)))
    assertEquals(a.extra("Random resolution"), s"${a.width}x${a.height}")
    val seen = (0 until 40).map(s => r.process(GenerationParams(seed = s), v)).map(p => (p.width, p.height)).toSet
    assert(seen.size > 3, "should vary with the seed")

  test("extras ops: flip, split boxes, crop box, auto size, focal point"):
    val ctx = new FakeContext()
    val img = Support.readImage(ctx.gradientInit(40, 20))
    val f = ExtrasOps.flip(img, horizontal = true, vertical = false)
    assertEquals(f.getRGB(0, 5), img.getRGB(39, 5))
    val boxes = ExtrasOps.splitBoxes(1000, 200, 0.5, 0.2)
    assert(boxes.size >= 2)
    assert(boxes.forall(b => b.w == 200 && b.h == 200))
    assertEquals(boxes.last.x + boxes.last.w, 1000)
    assertEquals(ExtrasOps.splitBoxes(500, 400, 0.5, 0.2), Seq(ExtrasOps.Box(0, 0, 500, 400)))
    val cb = ExtrasOps.cropBox(1000, 500, 512, 512, 900, 250)
    assertEquals((cb.w, cb.h), (500, 500))
    assertEquals(cb.x, 500)
    val (w, h) = ExtrasOps.autoSize(1600, 900, 384, 768, 512 * 512)
    assert(w % 64 == 0 && h % 64 == 0 && w > h)
    val sharp = new java.awt.image.BufferedImage(64, 64, java.awt.image.BufferedImage.TYPE_INT_RGB)
    for y <- 40 until 64; x <- 40 until 64 do sharp.setRGB(x, y, if (x + y) % 2 == 0 then 0xffffff else 0)
    val (fx, fy) = ExtrasOps.focalPoint(sharp, 1.0, 0.0)
    assert(fx > 40 && fy > 40, s"focal ($fx,$fy) should be in the detailed corner")

  test("outpainting mk2 fills the new border deterministically and inpaints the expanded canvas"):
    val ctx = new FakeContext()
    val p = new OutpaintMk2Plugin
    val init = ctx.initImage(64, 64, 0x808080)
    val gp = GenerationParams(width = 64, height = 64, seed = 7, img2img = Some(Img2ImgParams(initImage = init, denoisingStrength = 0.8)))
    val out = p.run(gp, defaults(p, "pixels" -> "32"), ctx)
    assertEquals(out.images.size, 1)
    val sent = ctx.generated.head
    assertEquals((sent.width, sent.height), (128, 64))
    assertEquals(sent.img2img.get.inpaintingFill, 1)
    val e = ImageOps.expansion(64, 64, 32, true, true, false, false)
    val a = ExtrasOps.noiseFill(Support.readImage(init), e, 0.5, 1.0, 1L)
    val b = ExtrasOps.noiseFill(Support.readImage(init), e, 0.5, 1.0, 1L)
    assertEquals(a.getRGB(0, 0), b.getRGB(0, 0))
    assertEquals(a.getRGB(64, 32), 0x808080 | 0xff000000)

  test("img2img alternative test sets the img2imgalt hook"):
    val ctx = new FakeContext()
    val p = new Img2ImgAltPlugin
    val gp = GenerationParams(prompt = "edited prompt", negativePrompt = "edited negative", steps = 23,
      img2img = Some(Img2ImgParams(initImage = ctx.initImage(64, 64))))
    p.run(gp, defaults(p, "randomness" -> "0.3", "override_prompt" -> "original prompt",
      "override_negative" -> "original negative"), ctx)
    val h = ctx.generated.head.hooks("img2imgalt").obj
    assertEquals(h("randomness").num, 0.3)
    assertEquals(h("inversion_steps").num, 50.0)
    assertEquals(ctx.generated.head.extra("Decode prompt"), "original prompt")
    assertEquals(ctx.generated.head.extra("Decode negative prompt"), "original negative")
    assertEquals(ctx.generated.head.extra("Decode steps"), "50")
    p.run(gp, defaults(p, "inversion_steps" -> "0"), ctx)
    assertEquals(ctx.generated.last.extra("Decode prompt"), gp.prompt)
    assertEquals(ctx.generated.last.extra("Decode negative prompt"), gp.negativePrompt)
    assertEquals(ctx.generated.last.extra("Decode steps"), "23")
    intercept[IllegalArgumentException](p.run(GenerationParams(), defaults(p), ctx))

  test("custom code saves returned images and logs stdout"):
    val ctx = new FakeContext()
    val p = new CustomCodePlugin
    val out = p.run(GenerationParams(), defaults(p, "code" -> "result = 42"), ctx)
    assertEquals(ctx.engineCalls.head._1, "run_code")
    assertEquals(ctx.engineCalls.head._2("code").str, "result = 42")
    assertEquals(out.images.size, 1)
    assert(ctx.logs.exists(_.contains("hello")))
    assert(ctx.logs.exists(_.contains("42")))

  test("SVD saves the first frame with infotext and copies the GIF next to it"):
    val ctx = new FakeContext()
    val p = new SvdPlugin
    val gp = GenerationParams(seed = 5, img2img = Some(Img2ImgParams(initImage = ctx.initImage(32, 16))))
    val out = p.run(gp, defaults(p, "model" -> "svd_xt"), ctx)
    assertEquals(out.images.size, 1)
    val (kind, params) = ctx.engineCalls.head
    assertEquals(kind, "img2vid")
    assertEquals(params("model").str, "/models/SvdModels/svd_xt")
    assertEquals(params("frames").num, 14.0)
    val png = out.images.head.path
    assert(Files.exists(png.resolveSibling(png.getFileName.toString.replace(".png", ".gif"))))
    assertEquals(ctx.saved.head._1, "videos")
    intercept[IllegalArgumentException](p.run(gp, defaults(p, "model" -> "missing"), ctx))

  test("zero123 calls novel_view and saves the result"):
    val ctx = new FakeContext()
    val p = new Zero123Plugin
    val gp = GenerationParams(img2img = Some(Img2ImgParams(initImage = ctx.initImage(32, 32))))
    val out = p.run(gp, defaults(p, "model" -> "zero123-xl", "azimuth" -> "45"), ctx)
    assertEquals(out.images.size, 1)
    assertEquals(ctx.engineCalls.head._2("azimuth").num, 45.0)
    assert(ctx.saved.head._4.contains("Azimuth: 45"))

  test("extras on txt2img generate first, on img2img use the init image"):
    val ctx = new FakeContext()
    val p = new FlippedCopiesPlugin
    val outT = p.run(GenerationParams(width = 16, height = 16, batchSize = 2), defaults(p, "vertical" -> "true"), ctx)
    assertEquals(ctx.generated.size, 1)
    assertEquals(outT.images.size, 4) // 2 images x (horizontal + vertical)
    val ctx2 = new FakeContext()
    val outI = p.run(GenerationParams(img2img = Some(Img2ImgParams(initImage = ctx2.initImage(16, 8)))), defaults(p), ctx2)
    assertEquals(ctx2.generated.size, 0)
    assertEquals(outI.images.size, 1)
    assertEquals(ctx2.saved.head._1, "extras")
    assert(ctx2.saved.head._4.contains("Flipped: horizontal"))

  test("face restore and caption extras go through the engine; caption writes a .txt"):
    val ctx = new FakeContext()
    val gp = GenerationParams(img2img = Some(Img2ImgParams(initImage = ctx.initImage(16, 16))))
    val fr = new FaceRestoreExtrasPlugin
    fr.run(gp, defaults(fr, "model" -> "GFPGANv1.4"), ctx)
    assertEquals(ctx.engineCalls.last._1, "restore_faces")
    assertEquals(ctx.engineCalls.last._2("model").str, "/models/FaceRestorers/GFPGANv1.4")
    val cap = new CaptionPlugin
    val out = cap.run(gp, defaults(cap), ctx)
    val txt = out.images.head.path.resolveSibling(out.images.head.path.getFileName.toString.replace(".png", ".txt"))
    assertEquals(Files.readString(txt), "a fake caption")
    assert(ctx.saved.last._4.contains("Caption: a fake caption"))

  test("split oversized and auto-sized crop resize to the requested size"):
    val ctx = new FakeContext()
    val sp = new SplitOversizedPlugin
    val wide = ctx.initImage(300, 100)
    val out = sp.run(GenerationParams(width = 64, height = 64, img2img = Some(Img2ImgParams(initImage = wide))), defaults(sp), ctx)
    assert(out.images.size >= 3)
    assert(ctx.saved.forall(s => s._2.getWidth == 64 && s._2.getHeight == 64))
    val ac = new AutosizedCropPlugin
    val ctx2 = new FakeContext()
    ac.run(GenerationParams(width = 512, height = 512, img2img = Some(Img2ImgParams(initImage = ctx2.initImage(400, 200)))),
      defaults(ac, "min_dim" -> "128", "max_dim" -> "512"), ctx2)
    val img = ctx2.saved.head._2
    assert(img.getWidth % 64 == 0 && img.getHeight % 64 == 0 && img.getWidth > img.getHeight)

  test("an interrupt stops extras loops and keeps what was saved"):
    val ctx = new FakeContext(interruptAfter = 0)
    val p = new FlippedCopiesPlugin
    val out = p.run(GenerationParams(width = 16, height = 16), defaults(p), ctx)
    assertEquals(out.images, Nil)
