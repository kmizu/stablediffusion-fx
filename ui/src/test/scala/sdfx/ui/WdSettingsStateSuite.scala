package sdfx.ui

import munit.FunSuite
import sdfx.core.config.Settings
import sdfx.ui.vm.{SettingsMerge, WdSettingsState}

class WdSettingsStateSuite extends FunSuite:
  test("WD settings report partial setup and survive a settings merge") {
    assertEquals(WdSettingsState.status("", ""), "Not configured")
    assertEquals(WdSettingsState.status("model", ""), "Incomplete: choose both files")
    assertEquals(WdSettingsState.status("model", "tags"), "Files selected")
    val current = Settings(lastModel = Some("latest"))
    val edited = Settings(wdV3Model = Some("model"), wdV3Tags = Some("tags"))
    val merged = SettingsMerge(current, edited)
    assertEquals(merged.lastModel, Some("latest"))
    assertEquals(merged.wdV3Model, Some("model"))
    assertEquals(merged.wdV3Tags, Some("tags"))
  }

  test("output directory restart follows configured WD temp location") {
    val configured = Settings(wdV3Model = Some("model"), wdV3Tags = Some("tags"))
    assert(WdSettingsState.restartRequired(configured, configured.copy(outputDir = "elsewhere")))
    assert(!WdSettingsState.restartRequired(Settings(), Settings(outputDir = "elsewhere")))
    assert(WdSettingsState.restartRequired(configured, configured.copy(wdV3Model = None)))
  }

  test("worker capability is separate from unsaved selections") {
    assert(WdSettingsState.workerStatus(Some(true)).contains("last start"))
    assert(WdSettingsState.workerStatus(Some(false)).contains("not configured"))
    assert(WdSettingsState.workerStatus(None).contains("unknown"))
  }
