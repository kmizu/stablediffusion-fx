package sdfx.plugins

import sdfx.core.plugins.api.*

/** Native tool tab, deliberately absent from generation forms. */
final class TokenizerPlugin extends Plugin:
  val id = "tokenizer"
  val name = "Tokenizer"
  override val description = "Inspect the loaded model's CLIP token IDs, BPE pieces and decoded text in the Tokenizer tab."
  val kind = PluginKind.AlwaysOn
  override val tabs: Set[Tab] = Set.empty
  override val requiresEngineFeatures = Seq("tokenizer")
