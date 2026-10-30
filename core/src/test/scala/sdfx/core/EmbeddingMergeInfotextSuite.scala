package sdfx.core

import munit.FunSuite
import sdfx.core.infotext.*
import sdfx.core.model.*

class EmbeddingMergeInfotextSuite extends FunSuite:
  test("upstream ephemeral expression provenance restores base negative and Hires prompts") {
    val provenance = "<'EM_1'>=<'cat' * 2>, {'EM_2'}={'dog':1}"
    val text = "portrait <'EM_1'>\nNegative prompt: {'EM_2'}\nSteps: 20, Seed: 17, CFG scale: 7, " +
      "Hires upscale: 1.5, Hires prompt: <'EM_1'>, Hires negative prompt: {'EM_2'}, EmbeddingMerge: " + Infotext.quote(provenance)
    val params = Infotext.toParams(Infotext.parse(text))
    assertEquals(params.prompt, "portrait <'cat' * 2>")
    assertEquals(params.negativePrompt, "{'dog':1}")
    assertEquals(params.hires.prompt, "<'cat' * 2>")
    assertEquals(params.hires.negativePrompt, "{'dog':1}")
    assert(!params.extra.contains("EmbeddingMerge"))
    assertEquals(params.seed, 17L)
  }

  test("native expressions retain apostrophes delimiters and algebra through PNG infotext") {
    val prompt = "portrait (< 'ordinary':1), <'artist''s red hair':+1 * 0.7 + 'blue hair' * 0.3>"
    val params = GenerationParams(prompt = prompt, negativePrompt = "{'bad', 000}:1", seed = 173L)
    val restored = Infotext.toParams(Infotext.parse(Infotext.write(params, ImageFacts(173L))))
    assertEquals(restored.prompt, prompt)
    assertEquals(restored.negativePrompt, params.negativePrompt)
  }

  test("malformed provenance remains inspectable instead of silently replacing a different expression") {
    val text = "<'EM_1'>\nSteps: 20, Seed: 17, CFG scale: 7, EmbeddingMerge: malformed"
    val params = Infotext.toParams(Infotext.parse(text))
    assertEquals(params.prompt, "<'EM_1'>")
    assertEquals(params.extra.get("EmbeddingMerge"), Some("malformed"))
  }
