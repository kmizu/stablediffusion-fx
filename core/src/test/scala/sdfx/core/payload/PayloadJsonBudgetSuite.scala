package sdfx.core.payload

import java.nio.charset.StandardCharsets.UTF_8

class PayloadJsonBudgetSuite extends munit.FunSuite:
  test("budget guard rejects huge non-materializable input before parser invocation") {
    var reads = 0
    var parsed = false
    val input = new CharSequence:
      def length(): Int = Int.MaxValue
      def charAt(index: Int): Char = { reads += 1; ' ' }
      def subSequence(start: Int, end: Int): CharSequence = throw new AssertionError("must not slice")
      override def toString: String = throw new AssertionError("must not materialize")
    val result = PayloadJsonBudget.admit(input, PayloadJsonBudget.Limits(maxBytes = 1024)) {
      parsed = true
      ujson.read(input.toString)
    }
    assert(result.isLeft)
    assert(!parsed)
    assert(reads <= 1025, reads.toString)
  }

  test("guard counts scalar and key tokens plus depth before parsing small but hostile JSON") {
    val wide = "[" + Vector.fill(1000)("0").mkString(",") + "]"
    val deep = ("[" * 200) + "0" + ("]" * 200)
    Seq(wide -> PayloadJsonBudget.Limits(maxNodes = 100), deep -> PayloadJsonBudget.Limits(maxDepth = 128)).foreach { (json, limits) =>
      var parsed = false
      assert(PayloadJsonBudget.admit(json, limits) { parsed = true; ujson.read(json) }.isLeft)
      assert(!parsed)
    }
    val literal = "{\"[]{}\\\"\":\"[this is only a string]\",\"n\":-1.25e-3}"
    assert(PayloadJsonBudget.guard(literal, PayloadJsonBudget.Limits(maxNodes = 5, maxDepth = 1)).isRight)
    assert(PayloadJsonBudget.guard(literal, PayloadJsonBudget.Limits(maxNodes = 4, maxDepth = 1)).isLeft)
  }

  test("iterative writer preserves literal JSON values and charges escaped Unicode output bytes") {
    val value = ujson.Obj("literal" -> ("<b>猫😀\"\\\u0000" + '\ud800'), "list" -> ujson.Arr(ujson.Null, true, 1.25))
    val result = PayloadJsonBudget.write(value, PayloadJsonBudget.Limits(maxBytes = 1024)).toOption.get
    assertEquals(ujson.read(result), value)
    val exact = result.getBytes(UTF_8).length
    assertEquals(PayloadJsonBudget.write(value, PayloadJsonBudget.Limits(maxBytes = exact)).toOption, Some(result))
    assert(PayloadJsonBudget.write(value, PayloadJsonBudget.Limits(maxBytes = exact - 1)).isLeft)
    assert(PayloadJsonBudget.write(ujson.Str("\u0000" * 1000), PayloadJsonBudget.Limits(maxBytes = 128)).isLeft)
  }

  test("shared quoted writer stops before reading a complete escaped value") {
    var reads = 0
    val input = new CharSequence:
      def length(): Int = Int.MaxValue
      def charAt(index: Int): Char = { reads += 1; '\u0000' }
      def subSequence(start: Int, end: Int): CharSequence = throw new AssertionError("must not slice")
      override def toString: String = throw new AssertionError("must not materialize")
    val out = new PayloadJsonBudget.Writer(128)
    out.appendQuoted(input)
    assertEquals(out.result, None)
    assertEquals(out.bytesSeen, 129)
    assert(reads < 30)
  }

  test("writer uses bounded iterative container work for wide and deeply nested values") {
    var deep: ujson.Value = ujson.Null
    (1 to 10000).foreach(_ => deep = ujson.Arr(deep))
    assert(PayloadJsonBudget.write(deep, PayloadJsonBudget.Limits(maxDepth = 128)).isLeft)
    val wide = ujson.Arr.from((1 to 10000).map(ujson.Num(_)))
    assert(PayloadJsonBudget.write(wide, PayloadJsonBudget.Limits(maxNodes = 100)).isLeft)
  }

  test("admission turns ordinary malformed diagnostic JSON into an explicit unavailable result") {
    assert(PayloadJsonBudget.admit("{not json", PayloadJsonBudget.Limits()) { ujson.read("{not json") }.isLeft)
    assert(PayloadJsonBudget.guard("[" * 140 + "0" + "]" * 140, PayloadJsonBudget.Limits()).isRight)
    assertEquals(PayloadJsonBudget.Limits().maxDepth, 512)
  }
