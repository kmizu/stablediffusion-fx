package sdfx.ui

import sdfx.core.plugins.api.Control.*
import sdfx.ui.view.PluginForm

class PluginFormGroupsSuite extends munit.FunSuite:
  test("controls are grouped at Section markers"):
    val a = Check("enabled", "Enabled")
    val s1 = Section("s1", "One")
    val b = Slider("b", "B", 0, 1, 0.1, 0.5)
    val s2 = Section("s2", "Two", expanded = false)
    val c = Text("c", "C")
    val g = PluginForm.groups(Seq(a, s1, b, s2, c)).map((s, cs) => (s.map(_.key), cs.map(_.key)))
    assertEquals(g, Seq((None, Seq("enabled")), (Some("s1"), Seq("b")), (Some("s2"), Seq("c"))))
    assertEquals(PluginForm.groups(Seq(b)).map(_._1), Seq(None))
