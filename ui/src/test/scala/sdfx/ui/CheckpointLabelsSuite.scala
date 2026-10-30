package sdfx.ui

import sdfx.core.protocol.ModelEntry
import sdfx.ui.vm.CheckpointLabels

class CheckpointLabelsSuite extends munit.FunSuite:
  private val entries = Seq(ModelEntry("a/model", "D:/a/model.safetensors", "sd15"),
    ModelEntry("b/model", "D:/b/model.safetensors", "sdxl"), ModelEntry("other", "D:/c/other.safetensors", "sdxl"))
  test("label scope never changes model identities and disambiguates equal basenames") {
    assertEquals(CheckpointLabels.label(entries.head, entries, "Default"), "a/model")
    assertEquals(CheckpointLabels.label(entries.head, entries, "With paths (Global)"), entries.head.path)
    assertEquals(CheckpointLabels.label(entries.head, entries, "With paths (Hires fix)"), "a/model")
    assertEquals(CheckpointLabels.label(entries.head, entries, "With paths (Hires fix)", true), entries.head.path)
    val labels = entries.map(CheckpointLabels.label(_, entries, "Without paths (Global)"))
    assertEquals(labels.distinct.size, entries.size)
    assertEquals(labels.last, "other.safetensors")
    assertEquals(entries.map(_.name), Seq("a/model", "b/model", "other"))
  }
