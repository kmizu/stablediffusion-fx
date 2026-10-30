package sdfx.plugins

import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.*
import java.nio.file.{Files, Paths}

class SeamlessTileSuite extends munit.FunSuite:
  private def values(extra: (String, String)*): Values =
    Values(Values.defaults(new SeamlessTilePlugin().controls).map ++ extra.toMap)

  test("circular offset round-trips odd sizes and only moves the selected axis"):
    val ctx = new FakeContext
    val src = Support.readImage(ctx.gradientInit(7, 5))
    val moved = SeamlessTilePlugin.offset(src, 3, 0)
    assertEquals(moved.getRGB(0, 2), src.getRGB(4, 2))
    val restored = SeamlessTilePlugin.offset(SeamlessTilePlugin.offset(src, 3, 2), -3, -2)
    for y <- 0 until 5; x <- 0 until 7 do assertEquals(restored.getRGB(x, y), src.getRGB(x, y))

  test("seam masks cover exact central bands and respect direction and width limits"):
    val h = SeamlessTilePlugin.mask(8, 6, "Horizontal", 2)
    val v = SeamlessTilePlugin.mask(8, 6, "Vertical", 2)
    val both = SeamlessTilePlugin.mask(8, 6, "Both", 2)
    for y <- 0 until 6; x <- 0 until 8 do
      assertEquals(h.getRaster.getSample(x, y, 0), if x == 3 || x == 4 then 255 else 0)
      assertEquals(v.getRaster.getSample(x, y, 0), if y == 2 || y == 3 then 255 else 0)
      assertEquals(both.getRaster.getSample(x, y, 0), if x == 3 || x == 4 || y == 2 || y == 3 then 255 else 0)
    val full = SeamlessTilePlugin.mask(8, 6, "Horizontal", 100)
    assertEquals(full.getRaster.getSample(0, 0, 0), 255)
    assertEquals(full.getRaster.getSample(7, 5, 0), 255)

  test("script builds full-image inpainting, restores batches, records settings and removes temp files"):
    val ctx = new FakeContext(echoInit = true)
    val source = ctx.gradientInit(64, 48)
    val p = GenerationParams(width = 64, height = 48, batchSize = 2, seed = 5,
      img2img = Some(Img2ImgParams(source, mask = Some("ignored"), inpaintFullRes = true, inpaintingMaskInvert = true)))
    val result = new SeamlessTilePlugin().run(p, values("direction" -> "Horizontal", "mask_width" -> "12"), ctx)
    assertEquals(result.images.size, 2)
    assertEquals(ctx.generated.size, 0)
    val sent = ctx.transientCalls.head
    assertEquals(sent.batchSize, 2)
    assertEquals(sent.img2img.get.inpaintFullRes, false)
    assertEquals(sent.img2img.get.inpaintingMaskInvert, false)
    assertEquals(sent.extra("Seamless tile direction"), "Horizontal")
    assert(!Files.exists(Paths.get(sent.img2img.get.initImage)))
    assert(!Files.exists(Paths.get(sent.img2img.get.mask.get)))
    assert(ctx.transientPaths.forall(p => !Files.exists(p)))
    val original = Support.readImage(source)
    result.images.foreach { o =>
      val img = Support.readImage(o.path)
      assertEquals(img.getRGB(0, 4), original.getRGB(0, 4))
      assertEquals(img.getRGB(63, 40), original.getRGB(63, 40))
    }
    assertEquals(result.images.map(_.seed), Seq(5L, 6L))

  test("optional shifted output, missing init image, cancellation and registration"):
    val ctx = new FakeContext(echoInit = true)
    val source = ctx.gradientInit(64, 48)
    val p = GenerationParams(width = 64, height = 48, img2img = Some(Img2ImgParams(source)))
    val out = new SeamlessTilePlugin().run(p, values("restore" -> "false", "direction" -> "Vertical"), ctx)
    assertEquals(Support.readImage(out.images.head.path).getRGB(3, 0), Support.readImage(source).getRGB(3, 24))
    intercept[IllegalArgumentException](new SeamlessTilePlugin().run(GenerationParams(), values(), ctx))
    val cancelled = new FakeContext(interruptAfter = 0)
    assertEquals(new SeamlessTilePlugin().run(p, values(), cancelled).images.size, 0)
    assert(scala.util.Using.resource(Files.list(cancelled.dir))(_.noneMatch(p => p.getFileName.toString.startsWith("tmp"))))
    assertEquals(StandardPlugins.withOwnAxes.count(_.id == "seamless_tile"), 1)

  test("final images retain engine infotext including model hashes and actual variation seed"):
    val rich = "texture\nSteps: 20, Seed: 5, Model: actual-checkpoint, Model hash: abc123, VAE hash: def456, Variation seed: 42"
    val ctx = new FakeContext(echoInit = true):
      override def generateTransient(p: GenerationParams): Seq[OutputImage] =
        super.generateTransient(p).map(_.copy(infotext = rich))
    val p = GenerationParams(width = 64, height = 48, seed = 5, subseedStrength = 0.5,
      img2img = Some(Img2ImgParams(ctx.gradientInit(64, 48))))
    val out = new SeamlessTilePlugin().run(p, values(), ctx)
    assertEquals(out.images.head.infotext, rich)

  test("a final save failure still deletes transient outputs, shifted input and mask"):
    val ctx = new FakeContext(echoInit = true):
      override def saveImage(img: java.awt.image.BufferedImage, kind: String, p: GenerationParams, text: String): OutputImage =
        throw new IllegalStateException("save failed")
    val p = GenerationParams(width = 64, height = 48, img2img = Some(Img2ImgParams(ctx.gradientInit(64, 48))))
    intercept[IllegalStateException](new SeamlessTilePlugin().run(p, values(), ctx))
    assert(ctx.transientPaths.nonEmpty && ctx.transientPaths.forall(p => !Files.exists(p)))
    val sent = ctx.transientCalls.head.img2img.get
    assert(!Files.exists(Paths.get(sent.initImage)))
    assert(!Files.exists(Paths.get(sent.mask.get)))


  test("scrambled input is decoded through the job context before tile processing"):
    val password = "session secret"
    val ctx = new FakeContext(echoInit = true):
      override def readImage(path: java.nio.file.Path): java.awt.image.BufferedImage =
        sdfx.core.png.ScrambledPng.read(path, Some(password)).image
    val source = Paths.get(ctx.gradientInit(64, 48))
    val original = Support.readImage(source)
    sdfx.core.png.ScrambledPng.write(original, source, Nil, Some(password))
    val p = GenerationParams(width = 64, height = 48, seed = 5, img2img = Some(Img2ImgParams(source.toString)))
    val result = new SeamlessTilePlugin().run(p, values(), ctx)
    val decoded = Support.readImage(result.images.head.path)
    assert(original.getRGB(0, 0, 64, 48, null, 0, 64).sameElements(decoded.getRGB(0, 0, 64, 48, null, 0, 64)))
