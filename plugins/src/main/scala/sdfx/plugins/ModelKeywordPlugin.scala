package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Model keyword selection; mappings are edited in the Model Keywords tab. */
class ModelKeywordPlugin extends HookPlugin:
  def id = "model_keyword"
  def name = "Model Keyword"
  def hook = "model_keyword"
  override def description = "Insert checkpoint and selected LoRA keywords, with optional textual inversion."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled", defaultValue = true),
    Choice("placement", "Keyword placement", Seq("keyword prompt", "prompt keyword", "keyword, prompt", "prompt, keyword"), "keyword prompt"),
    Choice("model_choice", "Checkpoint keywords", Seq("keyword1, keyword2", "random", "iterate", "keyword1", "keyword2"), "keyword1, keyword2", allowCustom = true),
    DynamicChoice("embedding", "Textual inversion embedding", ChoiceSource.Embeddings),
    Choice("order", "Keyword order", Seq("textual inversion first", "model keyword first"), "textual inversion first"),
    DynamicChoice("lora", "LoRA model", ChoiceSource.Loras),
    Slider("lora_weight", "LoRA multiplier", 0, 2, 0.01, 0.7),
    Choice("lora_choice", "LoRA keywords", Seq("None", "< iterate >", "< random >"), "None", allowCustom = true)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "placement" -> v.str("placement", "keyword prompt"), "model_choice" -> v.str("model_choice", "keyword1, keyword2"),
    "embedding" -> v.str("embedding"), "order" -> v.str("order", "textual inversion first"),
    "lora" -> v.str("lora"), "lora_weight" -> v.double("lora_weight", 0.7), "lora_choice" -> v.str("lora_choice", "None"))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Model keyword settings" -> o.render())
