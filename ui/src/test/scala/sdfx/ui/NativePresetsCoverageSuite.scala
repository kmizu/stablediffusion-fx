package sdfx.ui

import java.nio.file.{Files, Paths}
import sdfx.plugins.StandardPlugins

/** Post-evidence ledger gate. This deliberately fails before the two source entries are promoted. */
class NativePresetsCoverageSuite extends munit.FunSuite:
  private val root = Paths.get("").toAbsolutePath.normalize()
  private def source = Files.readString(root.resolve("scripts/extensions_matrix.py"))
  private def declaration(name: String): String =
    val pattern = ("""(?ms)^""" + java.util.regex.Pattern.quote(name) + """ = \{\R(.*?)^\}""").r
    pattern.findFirstMatchIn(source).map(_.group(1))
      .getOrElse(sys.error(s"Missing complete $name declaration in extensions_matrix.py"))
  private def ledger: Map[Int, String] =
    "([0-9]+): \"([^\"]+)\"".r.findAllMatchIn(declaration("S"))
      .map(m => m.group(1).toInt -> m.group(2)).toMap
  private def implemented: Set[String] =
    "\"([^\"]+)\"".r.findAllMatchIn(declaration("IMPLEMENTED")).map(_.group(1)).toSet
  private def effective(raw: String): String =
    if raw.startsWith("todo:") && implemented(raw.stripPrefix("todo:"))
    then "done:" + raw.stripPrefix("todo:") else raw

  test("only source indices 30 and 31 become the two distinct native preset core targets"):
    val statuses = ledger
    assertEquals(statuses.keySet, (0 until 353).toSet)
    assertEquals(statuses(30), "todo:core:config-presets")
    assertEquals(statuses(31), "todo:core:preset-utilities")
    assertEquals(implemented.intersect(Set("core:config-presets", "core:preset-utilities")),
      Set("core:config-presets", "core:preset-utilities"))
    assertEquals(statuses.collect { case (index, status) if
      Set("core:config-presets", "core:preset-utilities").exists(status.endsWith) => index }.toSet,
      Set(30, 31))
    assertEquals(effective(statuses(30)), "done:core:config-presets")
    assertEquals(effective(statuses(31)), "done:core:preset-utilities")

  test("other similarly named sources remain distinct pending targets"):
    val statuses = ledger
    assertEquals(statuses(203), "todo:controlnet_fastload")
    assertEquals(statuses(221), "todo:scenario_loader")
    assertEquals(statuses(262), "todo:core:state-manager")
    assertEquals(statuses(146), "todo:model_presets")
    Set(203, 221, 262, 146).foreach(i => assertEquals(effective(statuses(i)), statuses(i)))
    assert(!statuses.values.exists(_ == "todo:core:presets"))
    assert(!implemented("core:presets"))

  test("generated status preserves every source identity and real non-core done ids are registered"):
    val entries = ujson.read(Files.readString(root.resolve("docs/extensions-status.json"))).arr.toVector
    assertEquals(entries.size, 353)
    val statuses = ledger
    val ids = StandardPlugins.withOwnAxes.map(_.id).toSet
    entries.zipWithIndex.foreach { (entry, index) =>
      assertEquals(entry("index").num.toInt, index)
      assertEquals(entry("status").str, effective(statuses(index)), s"index $index")
      val status = entry("status").str
      if status.startsWith("done:") && !status.startsWith("done:core:") then
        assert(ids(status.stripPrefix("done:")), s"unregistered done plugin at index $index: $status")
    }

  test("real native MainWindow entry points and three mode buttons remain wired"):
    val main = Files.readString(root.resolve("ui/src/main/scala/sdfx/ui/MainWindow.scala"))
    val panel = Files.readString(root.resolve("ui/src/main/scala/sdfx/ui/view/GenerationPanel.scala"))
    val pane = Files.readString(root.resolve("ui/src/main/scala/sdfx/ui/view/NativePresetsPane.scala"))
    assert(main.contains("def openPresets(mode: Mode): Unit"))
    assert(main.contains("def presetsInspectorStage: Option[Stage]"))
    assert(main.contains("new GenerationPanel(ctx, vms(m), plugins(m), initial.expanded,"))
    assert(main.contains("lastSeed, openPayload, openPresets"))
    assert(panel.contains("presets-open-${mode.key}"))
    assert(panel.contains("presets-entry-${mode.key}"))
    Seq("presets-profile", "presets-select", "presets-utilities-quick-select",
      "presets-utilities-detailed-select", "presets-quick-save", "presets-save",
      "presets-apply", "presets-reload-tools").foreach(id => assert(pane.contains(id), id))
