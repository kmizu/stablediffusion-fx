package sdfx.core

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class ValuesSuite extends munit.FunSuite:

  val v: Values = Values(Map(
    "i" -> "42", "neg" -> "-3", "d" -> "0.75", "di" -> "3.9", "sp" -> "  7 ", "bad" -> "abc", "empty" -> "",
    "t1" -> "true", "t2" -> "TRUE", "t3" -> "1", "t4" -> " yes ", "f1" -> "false", "f2" -> "0", "f3" -> "no", "f4" -> "on"
  ))

  test("str") {
    assertEquals(v.str("i"), "42")
    assertEquals(v.str("missing"), "")
    assertEquals(v.str("missing", "dflt"), "dflt")
    assertEquals(v.str("empty", "dflt"), "")
  }

  test("int: lenient, doubles truncated, bad -> default") {
    assertEquals(v.int("i"), 42)
    assertEquals(v.int("neg"), -3)
    assertEquals(v.int("di"), 3)
    assertEquals(v.int("sp"), 7)
    assertEquals(v.int("bad", 5), 5)
    assertEquals(v.int("empty", 5), 5)
    assertEquals(v.int("missing"), 0)
    assertEquals(v.int("missing", -1), -1)
  }

  test("double") {
    assertEquals(v.double("d"), 0.75)
    assertEquals(v.double("i"), 42.0)
    assertEquals(v.double("sp"), 7.0)
    assertEquals(v.double("bad", 1.5), 1.5)
    assertEquals(v.double("missing"), 0.0)
  }

  test("bool") {
    Seq("t1", "t2", "t3", "t4").foreach(k => assert(v.bool(k), k))
    Seq("f1", "f2", "f3", "f4", "bad", "empty").foreach(k => assert(!v.bool(k, d = true), k))
    assert(!v.bool("missing"))
    assert(v.bool("missing", d = true))
  }

  test("updated is persistent") {
    val v2 = Values.empty.updated("a", "1")
    assertEquals(v2.str("a"), "1")
    assertEquals(Values.empty.map, Map.empty[String, String])
    assertEquals(v2.updated("a", "2").str("a"), "2")
    assertEquals(v2.str("a"), "1")
  }

  test("fmt") {
    assertEquals(Values.fmt(1.0), "1")
    assertEquals(Values.fmt(-2.0), "-2")
    assertEquals(Values.fmt(0.5), "0.5")
    assertEquals(Values.fmt(1.25), "1.25")
    assertEquals(Values.fmt(0.1 + 0.2), "0.30000000000000004")
    assertEquals(Values.fmt(1e-7), "0.0000001")
    assertEquals(Values.fmt(1e20), "100000000000000000000")
    assertEquals(Values.fmt(Double.NaN), "NaN")
    assertEquals(Values.fmt(Double.PositiveInfinity), "Infinity")
  }

  test("fmt output parses back with double") {
    Seq(0.0, 1.0, -0.35, 2.5e-3, 12345.678, 1e20).foreach { d =>
      assertEquals(Values(Map("k" -> Values.fmt(d))).double("k"), d)
    }
  }

  val controls: Seq[Control] = Seq(
    Control.Check("enabled", "Enabled"),
    Control.Slider("b1", "B1", 0, 2, 0.01, 1.3),
    Control.Slider("steps", "Steps", 1, 150, 1, 20),
    Control.IntField("n", "N", 1, 10, 3),
    Control.Choice("mode", "Mode", Seq("a", "b"), "b"),
    Control.DynamicChoice("model", "Model", ChoiceSource.Checkpoints),
    Control.Text("text", "Text", "hello", multiline = true),
    Control.Note("note", "Just a note")
  )

  test("Control defaults") {
    assertEquals(controls.map(_.default), Seq("false", "1.3", "20", "3", "b", "", "hello", ""))
  }

  test("Values.defaults skips notes and is typed-readable") {
    val d = Values.defaults(controls)
    assertEquals(d.map.keySet, Set("enabled", "b1", "steps", "n", "mode", "model", "text"))
    assert(!d.bool("enabled", d = true))
    assertEquals(d.double("b1"), 1.3)
    assertEquals(d.int("steps"), 20)
    assertEquals(d.int("n"), 3)
    assertEquals(d.str("mode"), "b")
  }

  test("Plugin defaults: isActive / process / run") {
    object AlwaysOnP extends Plugin:
      def id = "a"
      def name = "A"
      def kind = PluginKind.AlwaysOn
    object ScriptP extends Plugin:
      def id = "s"
      def name = "S"
      def kind = PluginKind.Script
    assert(!AlwaysOnP.isActive(Values.empty))
    assert(AlwaysOnP.isActive(Values(Map("enabled" -> "true"))))
    assert(ScriptP.isActive(Values.empty))
    val p = GenerationParams(prompt = "x")
    assertEquals(AlwaysOnP.process(p, Values.empty), p)
    assertEquals(AlwaysOnP.tabs, Set(Tab.Txt2Img, Tab.Img2Img))
    assertEquals(AlwaysOnP.controls, Nil)
    assertEquals(AlwaysOnP.xyzAxes, Nil)
    assertEquals(AlwaysOnP.requiresEngineFeatures, Nil)
    assertEquals(AlwaysOnP.description, "")
  }

  test("XyzAxis.apply builder") {
    val ax = XyzAxis("Steps", isNumeric = true)((p, s) => p.copy(steps = s.trim.toInt))
    assertEquals(ax.name, "Steps")
    assert(ax.numeric)
    assertEquals(ax.choices, None)
    assertEquals(ax(GenerationParams(), " 33 ").steps, 33)
    val ax2 = XyzAxis("Sampler", source = Some(ChoiceSource.Samplers))((p, s) => p.copy(samplerName = s))
    assert(!ax2.numeric)
    assertEquals(ax2.choices, Some(ChoiceSource.Samplers))
  }

  test("JobOutput.all lists grids first") {
    import java.nio.file.Paths
    val p = GenerationParams()
    val i1 = OutputImage(Paths.get("1.png"), p, "t", 1)
    val i2 = OutputImage(Paths.get("2.png"), p, "t", 2)
    val g = OutputImage(Paths.get("g.png"), p, "t", 1)
    assertEquals(JobOutput(Seq(i1, i2), Seq(g)).all, Seq(g, i1, i2))
    assertEquals(JobOutput(Seq(i1)).all, Seq(i1))
  }
