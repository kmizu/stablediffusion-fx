package sdfx.ui

import munit.FunSuite
import sdfx.ui.vm.WdBatchUiState
import sdfx.ui.vm.{GenerationVM, Mode}

class WdBatchUiStateSuite extends FunSuite:
  test("disabled batch leaves ordinary img2img unchanged") {
    assertEquals(WdBatchUiState.validate(false, false, "", ""), Right(None))
  }

  test("enabled batch requires partial worker capability and explicit safe inputs") {
    assert(WdBatchUiState.validate(true, false, "/images", "result").isLeft)
    assert(WdBatchUiState.validate(true, true, "", "result").isLeft)
    assert(WdBatchUiState.validate(true, true, "/images", "../overwrite").isLeft)
    assert(WdBatchUiState.validate(true, true, "/images", "folder.").isLeft)
    assert(WdBatchUiState.validate(true, true, "/images", "result").isRight)
  }

  test("folder batch only relaxes the single image check") {
    val vm = new GenerationVM(Mode.Img2Img)
    assert(vm.validate(_ => false).exists(_.contains("init image")))
    assertEquals(vm.validate(_ => false, batchImages = true), Nil)
    vm.steps.set(0)
    assert(vm.validate(_ => false, batchImages = true).exists(_.contains("steps")))
    assert(vm.initImage.get.isEmpty)
  }
