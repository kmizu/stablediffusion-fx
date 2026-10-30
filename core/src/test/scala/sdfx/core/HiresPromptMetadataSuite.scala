package sdfx.core

import sdfx.core.infotext.HiresPromptMetadata

class HiresPromptMetadataSuite extends munit.FunSuite:
  test("restore accepts each side only when recomposition matches stored resolved text") {
    val m = HiresPromptMetadata.parse("{'h':['Append','detail'],'n':['Prepend','noise'],'a':' + ','p':' / '}").get
    val restored = m.matching("base", "bad", "base + detail", "changed")
    assertEquals(restored.positive, m.positive)
    assertEquals(restored.negative, None)
    assertEquals(m.matching("base", "bad", "base + detail", "noise / bad"), m)
    assertEquals(m.matching("base", "bad", "already expanded twice", "noise / bad").positive, None)
  }

  test("restore strips only comparison edges and preserves original template bytes") {
    val m = HiresPromptMetadata.parse("{'h':['Append','  detail  '],'a':' + '}").get
    assertEquals(m.matching(" base ", "", " base + detail ", "").positive, m.positive)
  }

  test("source format preserves independent modes, separators, quotes and Unicode") {
    val text = "{'h':['Append','猫 \"quoted\"',true],'n':['Prepend','noise'],'a':' + ','p':' / ','c':'※'}"
    val m = HiresPromptMetadata.parse(text).get
    assertEquals(m.positive.get.prompt, "猫 'quoted'")
    assert(m.positive.get.removeNetworks)
    assertEquals(m.negative.get.mode, "Prepend")
    assertEquals(m.appendSeparator, " + ")
    assertEquals(m.prependSeparator, " / ")
    assertEquals(HiresPromptMetadata.parse(m.encode), Some(m))
  }

  test("omitted fields use source defaults and ordinary JSON is accepted") {
    val m = HiresPromptMetadata.parse("""{"n":["Prompt S/R","a\nb"]}""").get
    assertEquals(m.positive, None)
    assertEquals(m.appendSeparator, "{newline}")
    assertEquals(m.marker, "@")
    assertEquals(m.negative.get.prompt, "a\nb")
    assertEquals(HiresPromptMetadata.parse(m.encode), Some(m))
  }

  test("invalid structures are rejected atomically") {
    Seq("{}", "[]", "{'h':['Unknown','x']}", "{'h':['Append']}",
      "{'h':['Append',42]}", "{'h':['Append','x','true']}",
      "{'n':['Append','x',true]}", "{'h':['Append','x'],'a':42}",
      "{'h':['Append','x'],'c':''}", "x" * 65537).foreach { s =>
      assertEquals(HiresPromptMetadata.parse(s), None, s.take(100))
    }
  }
