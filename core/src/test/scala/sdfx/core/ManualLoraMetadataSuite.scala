package sdfx.core

import sdfx.core.infotext.*
import sdfx.core.model.*

class ManualLoraMetadataSuite extends munit.FunSuite:
  test("PNG roundtrip keeps structured manual refs independent of tagged refs") {
    val refs = Seq(LoraRef("detail", .6, Some(.4), Some("CUSTOM"), Some(.1), Some(.9)))
    val p = GenerationParams(prompt = "cat <lora:detail:0.2>", loras = refs)
    val parsed = Infotext.parse(Infotext.write(p, ImageFacts(seed = 7)))
    val restored = Infotext.toParams(parsed)
    assertEquals(restored.loras, refs)
    assertEquals(restored.prompt, p.prompt)
    assertEquals(Infotext.toParams(ParsedInfotext("cat", "", scala.collection.immutable.ListMap.empty), p).loras, Nil)
  }
  test("malformed manual metadata is rejected atomically") {
    Seq("{}", "[{}]", "[{\"name\":\"a\",\"weight\":\"x\"}]", "x" * 65537).foreach { value =>
      val p = Infotext.toParams(ParsedInfotext("cat", "", scala.collection.immutable.ListMap("NUI Manual LoRAs" -> value)))
      assertEquals(p.loras, Nil)
    }
  }
