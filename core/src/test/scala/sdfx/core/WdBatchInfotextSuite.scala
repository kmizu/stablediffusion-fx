package sdfx.core

import sdfx.core.infotext.Infotext
import sdfx.core.model.{GenerationParams, Img2ImgParams}

class WdBatchInfotextSuite extends munit.FunSuite:
  test("persisted WD fields restore exact image settings without changing selected image or mask") {
    val provenance = """[{"backend":"wd_ext","model_sha256":"abcdef"}]"""
    val text = "猫, effective\nNegative prompt: bad\n" +
      "Steps: 11, Sampler: Euler, Schedule type: Karras, CFG scale: 4.5, Seed: 123, Size: 64x80, " +
      "Batch size: 2, Batch count: 3, Eta: 0.2, Denoising strength: 0.4, Resize mode: 2, " +
      "Variation seed: 456, Variation seed strength: 0.1, Clip skip: 2, " +
      "WD interrogators: " + Infotext.quote(provenance) + ", WD input: /original.png"
    val base = GenerationParams(img2img = Some(Img2ImgParams("/selected.png", mask = Some("/mask.png"))))
    val p = Infotext.toParams(Infotext.parse(text), base)
    assertEquals(p.prompt, "猫, effective")
    assertEquals(p.negativePrompt, "bad")
    assertEquals((p.seed, p.subseed), (123L, 456L))
    assertEquals((p.width, p.height, p.batchSize, p.nIter), (64, 80, 2, 3))
    assertEquals(p.eta, Some(0.2))
    assertEquals(p.img2img.get.resizeMode, 2)
    assertEquals(p.img2img.get.denoisingStrength, 0.4)
    assertEquals(p.img2img.get.initImage, "/selected.png")
    assertEquals(p.img2img.get.mask, Some("/mask.png"))
    assertEquals(p.extra("WD interrogators"), provenance)
  }

  test("WD default eta clears an earlier explicit eta, legacy imports preserve it") {
    val base = GenerationParams(eta = Some(0.4))
    assertEquals(Infotext.toParams(Infotext.parse("p\nSteps: 2, Seed: 1, Size: 64x64, Eta: Automatic"), base).eta, None)
    assertEquals(Infotext.toParams(Infotext.parse("p\nSteps: 2, Seed: 1, Size: 64x64"), base).eta, Some(0.4))
  }
