package sdfx.plugins

import ValueKind.*

class XyzValuesSuite extends munit.FunSuite:
  def p(t: String, k: ValueKind) = XyzValues.parse(t, k)

  test("plain int list") {
    assertEquals(p("1, 2,3", IntNum), Seq("1", "2", "3"))
  }
  test("int range inclusive") {
    assertEquals(p("1-5", IntNum), Seq("1", "2", "3", "4", "5"))
  }
  test("int range with step") {
    assertEquals(p("1-10 (+3)", IntNum), Seq("1", "4", "7", "10"))
    assertEquals(p("1-5 (+2)", IntNum), Seq("1", "3", "5"))
  }
  test("descending int range follows python range(start, end + 1, step) like A1111") {
    assertEquals(p("10-1 (-3)", IntNum), Seq("10", "7", "4"))
    assertEquals(p("5-1 (-1)", IntNum), Seq("5", "4", "3"))
    assertEquals(p("5-1", IntNum), Nil)
  }
  test("ranges beyond Int (large seeds)") {
    assertEquals(p("3000000000-3000000002", IntNum), Seq("3000000000", "3000000001", "3000000002"))
    assertEquals(p("4000000000-4000000010 [2]", IntNum), Seq("4000000000", "4000000010"))
  }
  test("huge ranges and permutation counts fail with a clear error") {
    val e = intercept[IllegalArgumentException](p("1-2000000000", IntNum))
    assert(e.getMessage.contains("too many values"))
    intercept[IllegalArgumentException](p("0-1000 (+0.001)", FloatNum))
    intercept[IllegalArgumentException](p("1-2 [20000]", FloatNum))
    intercept[IllegalArgumentException](p("a,b,c,d,e,f,g", Permutations))
    assertEquals(p("a,b,c,d,e,f", Permutations).size, 720)
  }
  test("int counted range uses truncated linspace") {
    assertEquals(p("1-10 [5]", IntNum), Seq("1", "3", "5", "7", "10"))
    assertEquals(p("0-100 [3]", IntNum), Seq("0", "50", "100"))
  }
  test("ranges mixed with singles, spaces tolerated") {
    assertEquals(p("1 - 3, 8, 20-22", IntNum), Seq("1", "2", "3", "8", "20", "21", "22"))
  }
  test("negative single value is not a range") {
    assertEquals(p("-1, 5", IntNum), Seq("-1", "5"))
  }
  test("float range with step is exact") {
    assertEquals(p("0.1-0.5 (+0.1)", FloatNum), Seq("0.1", "0.2", "0.3", "0.4", "0.5"))
    assertEquals(p("5-7 (+0.5)", FloatNum), Seq("5", "5.5", "6", "6.5", "7"))
  }
  test("float range default step 1 and descending") {
    assertEquals(p("1-3", FloatNum), Seq("1", "2", "3"))
    assertEquals(p("2-1 (-0.5)", FloatNum), Seq("2", "1.5", "1"))
  }
  test("float counted range") {
    assertEquals(p("1-2 [3]", FloatNum), Seq("1", "1.5", "2"))
    assertEquals(p("0-1 [4]", FloatNum), Seq("0", "0.3333333333333333", "0.6666666666666666", "1"))
  }
  test("float singles normalised") {
    assertEquals(p("7, 7.50, 12", FloatNum), Seq("7", "7.5", "12"))
  }
  test("invalid numbers throw") {
    intercept[IllegalArgumentException](p("abc", IntNum))
    intercept[IllegalArgumentException](p("1, x", FloatNum))
  }
  test("quoted commas in string values") {
    assertEquals(p("cat, \"dog, running\", \"say \"\"hi\"\"\" ", Str),Seq("cat", "dog, running", "say \"hi\""))
  }
  test("empty items dropped, newlines separate") {
    assertEquals(p("a,, b\nc", Str), Seq("a", "b", "c"))
  }
  test("permutations produce n! csv values that round-trip") {
    val v = p("a, b, c", Permutations)
    assertEquals(v.size, 6)
    assertEquals(v.head, "a, b, c")
    assert(v.contains("c, b, a"))
    assertEquals(v.map(XyzValues.csv).toSet.size, 6)
  }
  test("toCsv quotes when needed and round-trips") {
    val vals = Seq("x", "a, b", "q\"t")
    assertEquals(XyzValues.csv(XyzValues.toCsv(vals)), vals)
  }
