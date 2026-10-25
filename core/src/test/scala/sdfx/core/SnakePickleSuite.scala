package sdfx.core

import sdfx.core.json.SnakePickle
import sdfx.core.json.SnakePickle.{ReadWriter => RW, *}

object SnakePickleSuite:
  case class Inner(innerValue: Int = 3, label: Option[String] = None) derives RW
  case class Outer(
      someName: String,
      maxSizeMb: Double = 1.5,
      inner: Inner = Inner(),
      innerList: Seq[Inner] = Nil,
      maybeLong: Option[Long] = None,
      hooks: Map[String, ujson.Value] = Map.empty,
      extraKeys: Map[String, String] = Map.empty
  ) derives RW

class SnakePickleSuite extends munit.FunSuite:
  import SnakePickleSuite.*

  test("camelToSnake") {
    assertEquals(SnakePickle.camelToSnake("negativePrompt"), "negative_prompt")
    assertEquals(SnakePickle.camelToSnake("cfgScale"), "cfg_scale")
    assertEquals(SnakePickle.camelToSnake("jpegB64"), "jpeg_b64")
    assertEquals(SnakePickle.camelToSnake("etaS"), "eta_s")
    assertEquals(SnakePickle.camelToSnake("prompt"), "prompt")
    assertEquals(SnakePickle.camelToSnake(""), "")
    assertEquals(SnakePickle.camelToSnake("inpaintFullResPadding"), "inpaint_full_res_padding")
  }

  test("snakeToCamel") {
    assertEquals(SnakePickle.snakeToCamel("negative_prompt"), "negativePrompt")
    assertEquals(SnakePickle.snakeToCamel("jpeg_b64"), "jpegB64")
    assertEquals(SnakePickle.snakeToCamel("second_pass_steps"), "secondPassSteps")
    assertEquals(SnakePickle.snakeToCamel("seed"), "seed")
    assertEquals(SnakePickle.snakeToCamel(""), "")
  }

  test("camel -> snake -> camel is identity for lower camel identifiers") {
    Seq("a", "abc", "abcDef", "vramMb", "timingS", "hrSecondPassSteps", "x1Y2").foreach { s =>
      assertEquals(SnakePickle.snakeToCamel(SnakePickle.camelToSnake(s)), s)
    }
  }

  test("writes snake_case keys and all defaults") {
    val js = SnakePickle.writeJs(Outer("n"))
    val keys = js.obj.keys.toSet
    assertEquals(keys, Set("some_name", "max_size_mb", "inner", "inner_list", "maybe_long", "hooks", "extra_keys"))
    assertEquals(js("max_size_mb").num, 1.5)
    assertEquals(js("inner")("inner_value").num, 3.0)
    assert(js("inner").obj.contains("label"))
  }

  test("Option is written as null / bare value") {
    val none = SnakePickle.writeJs(Outer("n"))
    assertEquals(none("maybe_long"), ujson.Null)
    assertEquals(none("inner")("label"), ujson.Null)
    val some = SnakePickle.writeJs(Outer("n", maybeLong = Some(42L), inner = Inner(1, Some("x"))))
    assertEquals(some("maybe_long").num, 42.0)
    assertEquals(some("inner")("label").str, "x")
  }

  test("reads snake_case keys, null Options and missing defaults") {
    val o = SnakePickle.read[Outer]("""{"some_name":"a","maybe_long":null,"inner":{"label":"q"}}""")
    assertEquals(o, Outer("a", inner = Inner(3, Some("q"))))
    val o2 = SnakePickle.read[Outer]("""{"some_name":"a","maybe_long":7}""")
    assertEquals(o2.maybeLong, Some(7L))
  }

  test("unknown keys are ignored") {
    val o = SnakePickle.read[Outer]("""{"some_name":"a","future_field":{"x":[1,2]},"another":null}""")
    assertEquals(o, Outer("a"))
  }

  test("missing required field fails") {
    intercept[Exception](SnakePickle.read[Outer]("""{"max_size_mb":2}"""))
  }

  test("nested case classes and lists round trip") {
    val o = Outer("n", 2.25, Inner(9, Some("i")), Seq(Inner(1), Inner(2, Some("b"))), Some(Long.MaxValue / 2))
    val s = SnakePickle.write(o)
    assert(s.contains("\"inner_list\""), s)
    assert(s.contains("\"inner_value\""), s)
    assertEquals(SnakePickle.read[Outer](s), o)
  }

  test("Map keys are NOT renamed (neither on write nor read)") {
    val hooks = Map[String, ujson.Value](
      "freeU" -> ujson.Obj("b1" -> 1.1, "enabledFlag" -> true),
      "Dynamic Thresholding" -> ujson.Arr(1, "two", ujson.Null),
      "snake_key" -> ujson.Str("v")
    )
    val extra = Map("Hires CFG Scale" -> "3", "camelKey" -> "x", "a_b" -> "y")
    val o = Outer("n", hooks = hooks, extraKeys = extra)
    val js = SnakePickle.writeJs(o)
    assertEquals(js("hooks").obj.keys.toSet, Set("freeU", "Dynamic Thresholding", "snake_key"))
    // nested ujson values keep their own keys verbatim
    assert(js("hooks")("freeU").obj.contains("enabledFlag"))
    assertEquals(js("extra_keys").obj.keys.toSet, Set("Hires CFG Scale", "camelKey", "a_b"))
    val back = SnakePickle.read[Outer](SnakePickle.write(o))
    assertEquals(back.hooks, hooks)
    assertEquals(back.extraKeys, extra)
  }

  test("unicode strings survive") {
    val o = Outer("桜, \"quoted\"\nline2 \u0000 tab\t")
    assertEquals(SnakePickle.read[Outer](SnakePickle.write(o)), o)
  }

  test("indent output is still readable") {
    val o = Outer("n", innerList = Seq(Inner()))
    assertEquals(SnakePickle.read[Outer](SnakePickle.write(o, indent = 2)), o)
  }
