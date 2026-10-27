package sdfx.ui

import sdfx.ui.vm.{GenerationVM, Mode}

class WdBatchRestoreSuite extends munit.FunSuite:
  test("fresh form restores persisted WD parameters and retains imported eta on resubmission") {
    val vm = new GenerationVM(Mode.Img2Img)
    vm.initImage.set("selected.png")
    vm.restoreInfotext("猫, caption\nNegative prompt: bad\n" +
      "Steps: 12, Sampler: Euler, Schedule type: Karras, CFG scale: 4.5, Seed: 987, Size: 64x80, " +
      "Batch size: 2, Batch count: 3, Eta: 0.2, Denoising strength: 0.4, Resize mode: 2")
    val p = vm.toParams
    assertEquals(p.prompt, "猫, caption")
    assertEquals(p.seed, 987L)
    assertEquals((p.batchSize, p.nIter), (2, 3))
    assertEquals(p.eta, Some(0.2))
    assertEquals(p.img2img.get.initImage, "selected.png")
    assertEquals(p.img2img.get.resizeMode, 2)
    assertEquals(p.img2img.get.denoisingStrength, 0.4)
    vm.restoreInfotext("caption\nSteps: 12, Seed: 987, Size: 64x80, Eta: Automatic")
    assertEquals(vm.toParams.eta, None)
  }
