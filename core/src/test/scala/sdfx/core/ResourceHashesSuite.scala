package sdfx.core

import sdfx.core.infotext.{ResourceHashes, Infotext}

class ResourceHashesSuite extends munit.FunSuite:
  test("merge preserves existing resources and current facts replace stale hashes"):
    val value = ResourceHashes.merge(Some("""{"model":"aaaaaaaaaa","external":"bbbbbbbbbb"}"""),
      Seq("model" -> ("c" * 64), "lora:日本語, \"name\"" -> ("D" * 64)))
    val json = ujson.read(value)
    assertEquals(json("model").str, "cccccccccc")
    assertEquals(json("external").str, "bbbbbbbbbb")
    assertEquals(json("lora:日本語, \"name\"").str, "dddddddddd")
    val text = Infotext.render("test", "", scala.collection.immutable.ListMap("Steps" -> "8", "Sampler" -> "Euler", "Hashes" -> value))
    assertEquals(ujson.read(Infotext.parse(text).fields("Hashes")), json)

  test("invalid existing JSON and invalid hashes are explicit errors"):
    intercept[IllegalArgumentException](ResourceHashes.merge(Some("not JSON"), Nil))
    intercept[IllegalArgumentException](ResourceHashes.merge(None, Seq("model" -> "unknown")))

  test("resource order does not change output"):
    val entries = Seq("model" -> ("a" * 64), "vae" -> ("b" * 64))
    assertEquals(ResourceHashes.merge(None, entries), ResourceHashes.merge(None, entries.reverse))
