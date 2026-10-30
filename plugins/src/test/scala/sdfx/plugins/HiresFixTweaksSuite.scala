package sdfx.plugins

import sdfx.core.model.*
import sdfx.core.plugins.api.Values

class HiresFixTweaksSuite extends munit.FunSuite:
  test("large Hires seeds remain exact through hook JSON") {
    val plugin = new HiresFixTweaksPlugin
    val values = Values(Values.defaults(plugin.controls).map ++ Map("seed" -> "9007199254740993", "subseed" -> Long.MaxValue.toString))
    val hook = plugin.hookFields(values)
    assertEquals(hook("seed"), ujson.Str("9007199254740993"))
    assertEquals(hook("subseed"), ujson.Str(Long.MaxValue.toString))
  }
  test("opt-in form retains seed modes and independent styles without stale resolution flags") {
    val plugin = new HiresFixTweaksPlugin
    val defaults = Values.defaults(plugin.controls)
    assert(!plugin.isActive(defaults))
    val v = Values(defaults.map ++ Map("enabled" -> "true", "batch_count" -> "3", "seed_enabled" -> "true",
      "seed" -> "42", "prompt_mode" -> "Append", "styles_enabled" -> "true", "styles" -> "[\"one\",\"two\"]"))
    val p = GenerationParams(hires = HiresParams(enabled = true, cfgScale = Some(4.5)),
      hooks = Map("hires_fix_tweaks" -> ujson.Obj("prompts_resolved" -> true)))
    val q = plugin.process(p, v)
    assertEquals(q.hires.cfgScale, p.hires.cfgScale)
    val h = q.hooks(plugin.id)
    assertEquals(h("batch_count").num, 3.0)
    assertEquals(h("styles").arr.map(_.str).toSeq, Seq("one", "two"))
    assert(!h.obj.contains("prompts_resolved"))
    assertEquals(plugin.xyzAxes.head.name, "[Hires fix tweaks] Hires CFG Scale")
    assertEquals(plugin.xyzAxes.head(p, "0").hires.cfgScale, None)
    assertEquals(plugin.xyzAxes.head(p, "6").hires.cfgScale, Some(6.0))
  }
