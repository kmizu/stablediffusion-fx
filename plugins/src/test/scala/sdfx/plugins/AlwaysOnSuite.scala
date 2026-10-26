package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class AlwaysOnSuite extends munit.FunSuite:
  def on(p: Plugin, kv: (String, String)*) = Values(Values.defaults(p.controls).map ++ kv + ("enabled" -> "true"))
  val NeedsInput = Set("reference", "revision", "ip_adapter", "controllllite", "fooocus_inpaint", "photomaker", "face_restore",
    "neveroom", "lora_block_weight")

  test("all AlwaysOn plugins are disabled by default and declare their engine feature") {
    StandardPlugins.alwaysOn.foreach { p =>
      assertEquals(p.kind, PluginKind.AlwaysOn)
      assert(!p.isActive(Values.defaults(p.controls)), p.id)
      // image / model based plugins additionally need an input (covered in BuiltinsSuite)
      if !NeedsInput.contains(p.id) then assert(p.isActive(on(p)), p.id)
    }
    Seq("freeu", "dynamic_thresholding").foreach { id =>
      assertEquals(StandardPlugins.alwaysOn.find(_.id == id).get.requiresEngineFeatures, Seq(id))
    }
    StandardPlugins.alwaysOn.foreach {
      case h: HookPlugin => assertEquals(h.requiresEngineFeatures, Seq(h.hook), h.id)
      case t: TokenizerPlugin => assertEquals(t.requiresEngineFeatures, Seq("tokenizer"))
      case t: VaeToolsPlugin => assertEquals(t.requiresEngineFeatures, Seq("vae_tools"))
      case other         => assertEquals(other.requiresEngineFeatures, Nil, other.id)
    }
  }

  test("FreeU hook and infotext") {
    val f = new FreeUPlugin
    val p = f.process(GenerationParams(), on(f, "b1" -> "1.3", "s2" -> "0.2"))
    assertEquals(p.hooks("freeu"), ujson.Obj("b1" -> 1.3, "b2" -> 1.02, "s1" -> 0.99, "s2" -> 0.2))
    val stages = ujson.read(p.extra("FreeU Stages"))
    assertEquals(stages(0)("backbone_factor").num, 1.3)
    assertEquals(stages(1)("skip_factor").num, 0.2)
    // survives an infotext round trip
    val parsed = Infotext.parse(Infotext.write(p, ImageFacts(seed = 1)))
    assertEquals(ujson.read(parsed.get("FreeU Stages").get), stages)
    assertEquals(f.xyzAxes.map(_.name), Seq("FreeU B1", "FreeU B2", "FreeU S1", "FreeU S2"))
  }

  test("Dynamic thresholding hook has every engine field; simple mode uses defaults for advanced ones") {
    val d = new DynamicThresholdingPlugin
    val p = d.process(GenerationParams(cfgScale = 20), on(d, "mimic_scale" -> "5", "cfg_mode" -> "Cosine Down", "separate_feature_channels" -> "false"))
    val h = p.hooks("dynamic_thresholding").obj
    assertEquals(
      h.keys.toSeq,
      Seq("mimic_scale", "threshold_percentile", "mimic_mode", "mimic_scale_min", "cfg_mode", "cfg_scale_min", "sched_val",
        "separate_feature_channels", "scaling_startpoint", "variability_measure", "interpolate_phi")
    )
    assertEquals(h("mimic_scale").num, 5.0)
    assertEquals(h("cfg_mode").str, "Cosine Down")
    assertEquals(h("separate_feature_channels"), ujson.Bool(false))
    assertEquals(p.extra("Dynamic thresholding separate feature channels"), "False")
    // JSON types: numbers are numbers, the channel switch is a boolean, enums are strings
    Seq("mimic_scale", "threshold_percentile", "mimic_scale_min", "cfg_scale_min", "sched_val", "interpolate_phi").foreach(k =>
      assert(h(k).isInstanceOf[ujson.Num], k)
    )
    val legacy = d.process(GenerationParams(), on(d, "separate_feature_channels" -> "disable")).hooks("dynamic_thresholding")
    assertEquals(legacy("separate_feature_channels"), ujson.Bool(false))
    val dflt = d.process(GenerationParams(), on(d)).hooks("dynamic_thresholding")
    assertEquals(dflt("separate_feature_channels"), ujson.Bool(true))
    assertEquals(p.extra("Dynamic thresholding mimic scale"), "5")
    val simple = d.process(GenerationParams(), on(d, "simple_mode" -> "true", "cfg_mode" -> "Cosine Down"))
    assertEquals(simple.hooks("dynamic_thresholding")("cfg_mode").str, "Constant")
    assertEquals(d.xyzAxes.map(_.name), Seq("DynThres Mimic Scale", "DynThres Threshold Percentile"))
  }

  test("HyperTile hook uses ints and a boolean") {
    val h = new HyperTilePlugin
    val p = h.process(GenerationParams(), on(h, "tile_size" -> "384", "scale_depth" -> "true"))
    assertEquals(p.hooks("hypertile"), ujson.Obj("tile_size" -> 384, "swap_size" -> 2, "max_depth" -> 0, "scale_depth" -> true))
    assertEquals(ujson.write(p.hooks("hypertile")), """{"tile_size":384,"swap_size":2,"max_depth":0,"scale_depth":true}""")
    assertEquals(p.extra("Hypertile tile size"), "384")
    assertEquals(p.extra("Hypertile scale depth"), "True")
  }

  test("PAG hook and infotext") {
    val g = new PagPlugin
    val p = g.process(GenerationParams(), on(g, "scale" -> "2.5", "layers" -> " mid, up.block_0 "))
    assertEquals(p.hooks("pag"), ujson.Obj("scale" -> 2.5, "adaptive_scale" -> 0.0, "layers" -> ujson.Arr("mid", "up.block_0")))
    assertEquals(p.extra("PAG scale"), "2.5")
    assertEquals(p.extra("PAG applied layers"), "mid, up.block_0")
    assertEquals(g.process(GenerationParams(), on(g)).hooks("pag")("layers"), ujson.Arr("mid"))
    assertEquals(PagPlugin.parseLayers("middle"), Seq("mid"))
    assertEquals(PagPlugin.parseLayers("down"), Seq("down"))
    assertEquals(PagPlugin.parseLayers("up"), Seq("up"))
    assertEquals(PagPlugin.parseLayers("  "), Seq("mid"))
  }

  test("XYZ overrides win over control values and the override key never reaches the engine") {
    val f = new FreeUPlugin
    val withOverride = f.xyzAxes.find(_.name == "FreeU S1").get.apply(GenerationParams(), "1.7")
    assert(withOverride.hooks.contains("freeu_xyz"))
    val once = f.process(withOverride, on(f, "s1" -> "0.5"))
    assertEquals(once.hooks("freeu")("s1").num, 1.7)
    assertEquals(once.hooks("freeu")("b1").num, 1.01)
    assertEquals(once.hooks.keySet, Set("freeu"))
  }

  test("process preserves existing hooks and extra") {
    val p0 = GenerationParams(hooks = Map("other" -> ujson.Obj("x" -> 1)), extra = Map("Keep" -> "me"))
    val p = StandardPlugins.alwaysOn.foldLeft(p0)((p, pl) => pl.process(p, on(pl)))
    val hookNames = StandardPlugins.alwaysOn.collect { case h: HookPlugin => h.hook }.toSet
    assertEquals(p.hooks.keySet, hookNames + "other")
    assertEquals(p.extra("Keep"), "me")
  }
