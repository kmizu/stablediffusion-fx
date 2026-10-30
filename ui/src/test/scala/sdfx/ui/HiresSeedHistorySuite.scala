package sdfx.ui

import sdfx.ui.vm.HiresSeedHistory

class HiresSeedHistorySuite extends munit.FunSuite:
  test("large seeds survive metadata history and saved UI state") {
    val history = new HiresSeedHistory
    history.record("cat\nSteps: 2, Sampler: Euler, Seed: 3, Hires seed: \"{'Seed':'9007199254740993','Subseed':'9223372036854775807'}\"")
    assertEquals(history.seed, Some(9007199254740993L))
    assertEquals(history.subseed, Some(Long.MaxValue))
    val saved = sdfx.ui.vm.UiState(lastHiresSeed = history.seed, lastHiresSubseed = history.subseed)
    val loaded = sdfx.ui.vm.UiState.fromJson(sdfx.ui.vm.UiState.toJson(saved))
    assertEquals(loaded.lastHiresSeed, history.seed)
    assertEquals(loaded.lastHiresSubseed, history.subseed)
  }
  test("seed history persists independently and legacy UI state defaults to empty") {
    val saved = sdfx.ui.vm.UiState(lastSeed = 3, lastHiresSeed = Some(100), lastHiresSubseed = Some(201))
    val loaded = sdfx.ui.vm.UiState.fromJson(sdfx.ui.vm.UiState.toJson(saved))
    val history = new HiresSeedHistory(loaded.lastHiresSeed, loaded.lastHiresSubseed)
    assertEquals(history.seed, Some(100L))
    assertEquals(history.subseed, Some(201L))
    assertEquals(loaded.lastSeed, 3L)
    val legacy = sdfx.ui.vm.UiState.fromJson("{\"last_seed\":3}")
    assertEquals(legacy.lastHiresSeed, None)
    assertEquals(legacy.lastHiresSubseed, None)
  }
  test("reuse tracks completed hires output, not ordinary images or malformed fields") {
    val history = new HiresSeedHistory
    assertEquals(history.seed, None)
    history.record("cat\nSteps: 2, Sampler: Euler, Seed: 3, Hires seed: \"{'Seed':100,'Subseed':201,'Strength':0.4}\"")
    assertEquals(history.seed, Some(100L))
    assertEquals(history.subseed, Some(201L))
    history.record("cat\nSteps: 2, Sampler: Euler, Seed: 9")
    history.record("cat\nSteps: 2, Sampler: Euler, Seed: 9, Hires seed: \"{'Seed':-2}\"")
    assertEquals(history.seed, Some(100L))
    assertEquals(history.subseed, Some(201L))
  }
