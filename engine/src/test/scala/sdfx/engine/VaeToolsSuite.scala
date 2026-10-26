package sdfx.engine

import sdfx.core.protocol.*
import sdfx.core.model.GenerationParams

class VaeToolsSuite extends munit.FunSuite:
  test("recovery artifacts survive protocol round trip separately from final images") {
    val recovery = ujson.Obj("path" -> "outputs/vae-recovery/latent.png", "latent_path" -> "outputs/vae-recovery/latent.safetensors", "error" -> "decode failed")
    val result = GenerationResult(Seq(GeneratedImage("broken-preview.png", 1)), vaeRecovery = Seq(recovery))
    assertEquals(Json.fromJs[GenerationResult](Json.toJs(result)).vaeRecovery, Seq(recovery))
    assertEquals(Json.fromJs[GenerationResult](ujson.Obj("images" -> ujson.Arr())).vaeRecovery, Nil)
    val request = GenerationRequest(GenerationParams(), ".tmp", vaeRecoveryDir = "outputs")
    assertEquals(Json.toJs(request).apply("vae_recovery_dir").str, "outputs")
    assertEquals(Protocol.Types.VaeTools, "vae_tools")
  }
