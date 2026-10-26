package sdfx.plugins

import sdfx.core.plugins.api.Control

class EnabledBySuite extends munit.FunSuite:
  test("every enabledBy entry of a built-in plugin names a control and a checkbox of that plugin"):
    StandardPlugins.withOwnAxes.foreach { p =>
      val byKey = p.controls.map(c => c.key -> c).toMap
      p.enabledBy.foreach { (k, gate) =>
        assert(byKey.contains(k), s"${p.id}: $k")
        assert(byKey.get(gate).exists(_.isInstanceOf[Control.Check]), s"${p.id}: gate $gate")
      }
    }
