package sdfx.plugins

import sdfx.core.plugins.api.*

/** File-preserving native tool, deliberately absent from image generation forms. */
final class VaeToolsPlugin extends Plugin:
  val id = "vae_tools"
  val name = "VAE Tools"
  override val description = "Encode, decode and compare VAE latent representations in the native VAE Tools tab."
  val kind = PluginKind.AlwaysOn
  override val tabs: Set[Tab] = Set.empty
  override val requiresEngineFeatures = Seq("vae_tools")
