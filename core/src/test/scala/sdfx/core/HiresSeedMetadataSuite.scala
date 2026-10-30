package sdfx.core

import sdfx.core.infotext.*
import sdfx.core.model.*

class HiresSeedMetadataSuite extends munit.FunSuite:
  test("large source integer and native string seed metadata remain exact") {
    for text <- Seq("{\"Seed\":9007199254740993,\"Subseed\":9223372036854775807}",
        "{'Seed':'9007199254740993','Subseed':'9223372036854775807'}") do
      val h = HiresSeedMetadata.parse(text).get
      assertEquals(h("seed"), ujson.Str("9007199254740993"))
      assertEquals(h("subseed"), ujson.Str(Long.MaxValue.toString))
    assertEquals(HiresSeedMetadata.parse("{'Seed':9223372036854775808}"), None)
  }
  test("upstream quote-swapped seed metadata restores effective hires settings") {
    val text = "cat\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 9, Size: 512x512, Hires upscale: 2, " +
      "Hires seed: \"{'Seed':100,'Subseed':200,'Strength':0.4,'Resize':[512,768]}\""
    val p = Infotext.toParams(Infotext.parse(text))
    val h = p.hooks("hires_fix_tweaks")
    assertEquals(p.seed, 9L)
    assertEquals(h("seed").num, 100.0)
    assertEquals(h("subseed").num, 200.0)
    assertEquals(h("strength").num, .4)
    assertEquals(h("resize_width").num, 512.0)
    assertEquals(h("resize_height").num, 768.0)
    assert(h("seed_enabled").bool && h("extras").bool)
    assertEquals(h("batch_count").num, 1.0)
  }

  test("omitted source fields use inheritance and explicit zero variation") {
    val h = HiresSeedMetadata.parse("{'Seed':123}").get
    assertEquals(h("subseed").num, 0.0)
    assertEquals(h("strength").num, 0.0)
    assertEquals(h("resize_width").num, 0.0)
    assert(HiresSeedMetadata.parse("{\"Seed\":123}").nonEmpty)
  }

  test("malformed, oversized and out-of-range data is rejected atomically") {
    Seq("[]", "{'Seed':-2}", "{'Seed':1.5}", "{'Strength':2}", "{'Strength':'x'}",
      "{'Resize':[512]}", "{'Resize':[513,512]}", "{'Resize':[32768,512]}",
      "{'Resize':[512,512,512]}", "{'Unknown':1}", "x" * 4097).foreach { text =>
      assertEquals(HiresSeedMetadata.parse(text), None, text)
    }
    val base = GenerationParams(hooks = Map("unrelated" -> ujson.Obj("enabled" -> true)))
    val parsed = ParsedInfotext("cat", "", scala.collection.immutable.ListMap("Hires seed" -> "{'Seed':-2}"))
    assertEquals(Infotext.toParams(parsed, base).hooks, base.hooks)
  }
