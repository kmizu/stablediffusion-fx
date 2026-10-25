package sdfx.core

import sdfx.core.model.*

import java.util.Locale

class LoraSyntaxSuite extends munit.FunSuite:
  import LoraSyntax.*

  test("explicit weights") {
    val e = extract("1girl, <lora:foo:0.8>, <lora:bar:-0.5>")
    assertEquals(e.loras, Seq(LoraRef("foo", 0.8), LoraRef("bar", -0.5)))
  }

  test("missing / empty / invalid weight defaults to 1") {
    assertEquals(extract("<lora:a>").loras, Seq(LoraRef("a", 1.0)))
    assertEquals(extract("<lora:b:>").loras, Seq(LoraRef("b", 1.0)))
    assertEquals(extract("<lora:c: >").loras, Seq(LoraRef("c", 1.0)))
    assertEquals(extract("<lora:d:unet=0.5>").loras, Seq(LoraRef("d", 1.0)))
  }

  test("lyco tags are recognized") {
    assertEquals(extract("x <lyco:loha_style:0.6>").loras, Seq(LoraRef("loha_style", 0.6)))
  }

  test("te:unet weights use the first (te) value") {
    assertEquals(extract("<lora:foo:0.7:0.3>").loras, Seq(LoraRef("foo", 0.7)))
  }

  test("names with spaces, dots, dashes and non-ASCII") {
    val e = extract("<lora:my style v1.5-final:0.9> <lora:  spaced  :1.2> <lora:日本語ロラ:0.3>")
    assertEquals(e.loras, Seq(LoraRef("my style v1.5-final", 0.9), LoraRef("spaced", 1.2), LoraRef("日本語ロラ", 0.3)))
  }

  test("duplicates are summed, first-occurrence order") {
    val e = extract("<lora:b:0.5>, <lora:a:0.25>, <lora:b:0.25>, <lora:a>")
    assertEquals(e.loras, Seq(LoraRef("b", 0.75), LoraRef("a", 1.25)))
  }

  test("other extra networks and plain angle brackets are left alone") {
    val e = extract("<hypernet:h:1>, a <b> c, <lorax:z:1>")
    assertEquals(e.loras, Nil)
    assertEquals(e.prompt, "<hypernet:h:1>, a <b> c, <lorax:z:1>")
  }

  test("no tags: prompt only trimmed") {
    assertEquals(extract("  masterpiece, 1girl  ").prompt, "masterpiece, 1girl")
    assertEquals(extract("").prompt, "")
    assertEquals(extract("").loras, Nil)
  }

  test("stripping: middle tag leaves a single separator") {
    assertEquals(extract("1girl, <lora:foo:0.8>, smile").prompt, "1girl, smile")
    assertEquals(extract("1girl <lora:foo:0.8> smile").prompt, "1girl smile")
    assertEquals(extract("1girl,<lora:foo:0.8>,smile").prompt, "1girl,smile")
  }

  test("stripping: leading / trailing tags leave no dangling comma") {
    assertEquals(extract("<lora:a:1>, 1girl, smile").prompt, "1girl, smile")
    assertEquals(extract("1girl, smile, <lora:a:1>").prompt, "1girl, smile")
    assertEquals(extract("1girl, smile, <lora:a:1>, <lora:b:0.5>").prompt, "1girl, smile")
    assertEquals(extract("<lora:a:1>").prompt, "")
  }

  test("stripping keeps newlines (BREAK / multi-line prompts)") {
    val e = extract("1girl, <lora:a:1>\nBREAK\nsky, <lora:b:1>")
    assertEquals(e.prompt, "1girl\nBREAK\nsky")
    assertEquals(e.loras.map(_.name), Seq("a", "b"))
  }

  test("stripping keeps emphasis and weights syntax") {
    assertEquals(extract("(masterpiece:1.2), <lora:a:0.5>, [bad:good:0.5]").prompt, "(masterpiece:1.2), [bad:good:0.5]")
  }

  test("tag formatting") {
    assertEquals(tag("foo"), "<lora:foo:1.0>")
    assertEquals(tag("foo", 2), "<lora:foo:2.0>")
    assertEquals(tag("foo", 0.8), "<lora:foo:0.8>")
    assertEquals(tag("foo", 0.75), "<lora:foo:0.75>")
    assertEquals(tag("foo", -0.5), "<lora:foo:-0.5>")
    assertEquals(tag("foo", 0), "<lora:foo:0.0>")
  }

  test("tag formatting is locale independent") {
    val old = Locale.getDefault
    try
      Locale.setDefault(Locale.GERMANY)
      assertEquals(tag("foo", 1.0), "<lora:foo:1.0>")
      assertEquals(tag("foo", 0.5), "<lora:foo:0.5>")
    finally Locale.setDefault(old)
  }

  test("tag -> extract round trip") {
    Seq(1.0, 0.8, 0.125, -1.0, 1.5).foreach { w =>
      assertEquals(extract(tag("n", w)).loras, Seq(LoraRef("n", w)))
    }
  }
