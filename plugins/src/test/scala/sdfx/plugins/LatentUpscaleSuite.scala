package sdfx.plugins

import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.*

class LatentUpscaleSuite extends munit.FunSuite:
  test("script selects source-resolution interpolation and full steps without changing model inputs"):
    val plugin = new LatentUpscalePlugin
    val ctx = new FakeContext()
    val source = ctx.gradientInit(64, 48)
    val p = GenerationParams(steps = 7, scheduler = "Karras", batchSize = 2,
      img2img = Some(Img2ImgParams(source, denoisingStrength = .4)))
    plugin.run(p, Values(Map("method" -> "area", "scheduler" -> "automatic")), ctx)
    val sent = ctx.generated.head
    assertEquals(sent.img2img, p.img2img)
    assertEquals(sent.steps, 7)
    assertEquals(sent.hooks("latent_upscale")("method").str, "area")
    assertEquals(sent.hooks("latent_upscale")("scheduler").str, "automatic")
    assertEquals(sent.extra("Latent upscale method"), "area")
    assertEquals(sent.extra("Latent upscale scheduler"), "automatic")
    assertEquals(plugin.tabs, Set(Tab.Img2Img))
    assertEquals(plugin.requiresEngineFeatures, Seq("latent_upscale"))
    assertEquals(StandardPlugins.withOwnAxes.count(_.id == "latent_upscale"), 1)
    val fields = sdfx.core.infotext.Infotext.parse(Support.infotext(sent, 1)).fields
    assertEquals(LatentUpscalePlugin.replay(fields), Some(Values(Map("method" -> "area", "scheduler" -> "automatic"))))

  test("invalid dimensional interpolation and missing init image fail before engine call"):
    val plugin = new LatentUpscalePlugin
    intercept[IllegalArgumentException](plugin.run(GenerationParams(), Values.empty, new FakeContext()))
    intercept[IllegalArgumentException](plugin.run(GenerationParams(img2img = Some(Img2ImgParams("x"))),
      Values(Map("method" -> "linear")), new FakeContext()))
