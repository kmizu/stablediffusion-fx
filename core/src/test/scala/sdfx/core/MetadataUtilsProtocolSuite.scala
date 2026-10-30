package sdfx.core

import munit.FunSuite
import sdfx.core.protocol.*

class MetadataUtilsProtocolSuite extends FunSuite:
  test("metadata edit request preserves Unicode JSON and defaults across wire") {
    val request = MetadataUtilsRequest("write", "D:/models/a.safetensors", "{\"note\":\"日本語\"}", "copy", "opaque-read-token")
    val wire = Json.toJs(request)
    assertEquals(wire("metadata_json").str, "{\"note\":\"日本語\"}")
    assertEquals(wire("new_name").str, "copy")
    assertEquals(wire("source_token").str, "opaque-read-token")
    assertEquals(Json.fromJs[MetadataUtilsRequest](ujson.read(ujson.write(wire))), request)
    assertEquals(Json.fromJs[MetadataUtilsRequest](ujson.Obj("path" -> "a.safetensors")),
      MetadataUtilsRequest(path = "a.safetensors"))
  }

  test("read and write result variants retain metadata and large byte counts") {
    val read = Json.fromJs[MetadataUtilsResult](ujson.Obj("path" -> "a", "metadata" -> ujson.Obj("x" -> "{\"a\":1}"),
      "editable" -> "{\"x\":{\"a\":1}}", "tensor_bytes" -> 7000000000L, "source_token" -> "opaque-read-token"))
    assertEquals(read.metadata, Map("x" -> "{\"a\":1}"))
    assertEquals(read.tensorBytes, 7000000000L)
    assertEquals(read.sourceToken, "opaque-read-token")
    val write = Json.fromJs[MetadataUtilsResult](ujson.Obj("path" -> "b", "tensor_bytes" -> 8, "tensor_sha256" -> "abc"))
    assertEquals(write.tensorSha256, "abc")
    assertEquals(write.metadata, Map.empty[String, String])
    assertEquals(Json.fromJs[MetadataUtilsResult](Json.toJs(write)), write)
  }
