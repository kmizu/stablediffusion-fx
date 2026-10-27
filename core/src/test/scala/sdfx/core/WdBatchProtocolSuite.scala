package sdfx.core

import munit.FunSuite
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.*

class WdBatchProtocolSuite extends FunSuite:
  test("typed preparation uses worker field names and preserves frozen generation") {
    val request = WdBatchPrepareRequest("/images", GenerationRequest(GenerationParams(prompt = "base"), ""),
      "/outputs", "batch-1", "checkpoint-token")
    val body = Json.toJs(request)
    assertEquals(body("images_root").str, "/images")
    assertEquals(body("generation")("params")("prompt").str, "base")
    assertEquals(body("checkpoint_token").str, "checkpoint-token")
  }

  test("typed partial outcome retains effective prompt, model identity and failures") {
    val wire = ujson.read("""{"partial":true,"rows":[{"status":"complete","input":"/in.png","positive":"wd, base","negative":"bad","result":{"images":[]},"interrogators":[{"backend":"wd_ext","model_id":"candidate","model_sha256":"abc","tags_sha256":"def","threshold":0.35,"rating_threshold":0.5,"remove_underscores":true,"append_ratings":false,"preprocessing":"policy"}]},{"status":"error","input":"/bad.png","error":"bad file"}]}""")
    val result = Json.fromJs[WdBatchResult](wire)
    assert(result.partial)
    assertEquals(result.rows.head.positive, Some("wd, base"))
    assertEquals(result.rows.head.interrogators.head.modelId, Some("candidate"))
    assertEquals(result.rows(1).error, Some("bad file"))
  }
