package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class TokenizerSuite extends munit.FunSuite:
  test("tokenizer is a registered tool capability without generation controls or mutation"):
    val p = StandardPlugins.withOwnAxes.find(_.id == "tokenizer").get
    assertEquals(p.tabs.size, 0)
    assertEquals(p.controls.size, 0)
    assertEquals(p.requiresEngineFeatures, Seq("tokenizer"))
    val params = GenerationParams(prompt = "cat")
    assertEquals(p.process(params, Values.empty), params)
