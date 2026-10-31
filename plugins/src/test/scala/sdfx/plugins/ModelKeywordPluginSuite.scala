package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values

class ModelKeywordPluginSuite extends munit.FunSuite:
  test("model keyword starts enabled to match upstream, but an explicit opt-out remains effective") {
    val plugin = new ModelKeywordPlugin
    val defaults = sdfx.core.plugins.api.Values.defaults(plugin.controls)
    assert(plugin.isActive(defaults))
    assert(!plugin.isActive(Values(defaults.map.updated("enabled", "false"))))
  }
  test("embedding selector uses the local embedding catalog") {
    val controls = new ModelKeywordPlugin().controls
    val selected = controls.collectFirst { case c: sdfx.core.plugins.api.Control.DynamicChoice if c.key == "embedding" => c }
    assertEquals(selected.map(_.source), Some(sdfx.core.plugins.api.ChoiceSource.Embeddings))
  }
  test("model keyword freezes the selection without modifying prompt before worker resolution") {
    val plugin = new ModelKeywordPlugin
    val values = Values(Map("enabled" -> "true", "placement" -> "prompt, keyword",
      "model_choice" -> "iterate", "embedding" -> "detail", "order" -> "model keyword first",
      "lora" -> "folder/style", "lora_weight" -> "0.7", "lora_choice" -> "< iterate >"))
    val result = plugin.process(GenerationParams(prompt = "cat"), values)
    assertEquals(result.prompt, "cat")
    val hook = result.hooks("model_keyword")
    assertEquals(hook("placement").str, "prompt, keyword")
    assertEquals(hook("model_choice").str, "iterate")
    assertEquals(hook("lora").str, "folder/style")
    assertEquals(hook("lora_weight").num, 0.7)
  }
