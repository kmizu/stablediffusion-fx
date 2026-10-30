package sdfx.core

import munit.FunSuite
import sdfx.core.protocol.*

class EmbeddingMergeProtocolSuite extends FunSuite:
  test("merge request retains expression, file catalog and explicit safe export controls") {
    val request = EmbeddingMergeRequest(text = "<'雪' * 2>", operation = "save", grouping = "tokens",
      name = "snow", outputDir = "D:/embeddings", embeddingPaths = Seq("D:/embeddings/style.pt"), safetensors = false)
    val wire = Json.toJs(request)
    assertEquals(wire("output_dir").str, "D:/embeddings")
    assertEquals(wire("embedding_paths").arr.head.str, "D:/embeddings/style.pt")
    assertEquals(wire("safetensors").bool, false)
    assertEquals(Json.fromJs[EmbeddingMergeRequest](wire), request)
    assertEquals(Json.fromJs[EmbeddingMergeRequest](ujson.Obj("text" -> "cat")), EmbeddingMergeRequest(text = "cat"))
  }

  test("merge rows preserve mixed numeric and TI token labels plus nullable sample deviation") {
    val wire = ujson.read("""{"text":"cat", "streams":[{"name":"clip_l","count":1,"rows":[{"start":1,"end":1,"count":1,"text":"cat","tokens":[2,"*_0"],"statistics":{"min":1,"max":1,"sum":1,"abs_sum":1,"norm":1,"std":null}}],"total":{"min":1,"max":1,"sum":1,"abs_sum":1,"norm":1,"std":null}}],"paths":["D:/embeddings/cat.pt"],"active":false,"warning":"Reload checkpoint"}""")
    val result = Json.fromJs[EmbeddingMergeResult](wire)
    assertEquals(result.streams.head.rows.head.tokens, Seq(ujson.Num(2), ujson.Str("*_0")))
    assertEquals(result.streams.head.total.std, None)
    assert(!result.active)
    assertEquals(result.warning, "Reload checkpoint")
    assertEquals(Json.fromJs[EmbeddingMergeResult](Json.toJs(result)), result)
  }
