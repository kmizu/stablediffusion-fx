package sdfx.core.payload

class PayloadTreeBudgetSuite extends munit.FunSuite:
  private def rejected(json: String, reason: String): Unit =
    val error = intercept[IllegalArgumentException](RequestPayload.tree(json))
    assert(error.getMessage.contains(reason), error.getMessage)

  private def rejected(json: String, limits: RequestPayload.TreeLimits, reason: String): Unit =
    val error = intercept[RequestPayload.TreeUnavailable](RequestPayload.tree(json, limits))
    assertEquals(error.reason, reason)

  test("tiny tree budgets include every exact path, label and scalar at inclusive boundaries") {
    val json = """{"a":{"b":null}}"""
    val exact = RequestPayload.TreeLimits(maxNodes = 5, maxDepth = 2, maxPathBytes = 11,
      maxTotalPathBytes = 18, maxTextBytes = 7)
    val tree = RequestPayload.tree(json, exact)
    assertEquals(tree.children.head.children.head.path, "$[\"a\"][\"b\"]")
    rejected(json, exact.copy(maxPathBytes = 10), "path_limit")
    rejected(json, exact.copy(maxTotalPathBytes = 17), "total_path_limit")
    rejected(json, exact.copy(maxTextBytes = 6), "text_limit")
    rejected(json, exact.copy(maxNodes = 4), "nodes_limit")
    rejected(json, exact.copy(maxDepth = 1), "depth_limit")
  }

  test("tiny cumulative path budget rejects wide children under a shared long prefix") {
    val json = """{"prefix":[0,1,2,3]}"""
    val exact = RequestPayload.TreeLimits(maxPathBytes = 14, maxTotalPathBytes = 68, maxTextBytes = 15)
    rejected(json, exact.copy(maxTotalPathBytes = 67), "total_path_limit")
    assertEquals(RequestPayload.tree(json, exact).children.head.children.last.path, "$[\"prefix\"][3]")
  }

  test("tiny cumulative path budget rejects tall prefixes independently of the per-path limit") {
    val json = """{"kk":{"kk":{"kk":0}}}"""
    val exact = RequestPayload.TreeLimits(maxPathBytes = 19, maxTotalPathBytes = 40)
    rejected(json, exact.copy(maxTotalPathBytes = 39), "total_path_limit")
    assertEquals(RequestPayload.tree(json, exact).children.head.children.head.children.head.path,
      "$[\"kk\"][\"kk\"][\"kk\"]")
  }

  test("escaped Unicode key budgets count UTF-8 and preserve ujson short control escapes") {
    val json = """{"雪\n":[0]}"""
    val exact = RequestPayload.TreeLimits(maxPathBytes = 13, maxTotalPathBytes = 24, maxTextBytes = 7)
    val tree = RequestPayload.tree(json, exact)
    assertEquals(tree.children.head.path, "$[\"雪\\n\"]")
    assertEquals(tree.children.head.children.head.path, "$[\"雪\\n\"][0]")
    rejected(json, exact.copy(maxTotalPathBytes = 23), "total_path_limit")
    rejected(json, exact.copy(maxPathBytes = 9), "path_limit")
    rejected("""{"雪雪雪雪雪雪":0}""", exact.copy(maxPathBytes = 8), "path_limit")
  }

  test("scalar JSON is capped before rendering escaped text without changing its spelling") {
    val json = """["\b\t\n\f\r\u0000雪"]"""
    // Root label 1, index label 1; scalar quotes 2 + five short escapes 10 + NUL escape 6 + 雪 3.
    val exact = RequestPayload.TreeLimits(maxTextBytes = 23)
    val tree = RequestPayload.tree(json, exact)
    assertEquals(tree.children.head.valueJson, Some("\"\\b\\t\\n\\f\\r\\u0000雪\""))
    rejected(json, exact.copy(maxTextBytes = 22), "text_limit")
    rejected("\"" + ("雪" * 1000) + "\"", exact.copy(maxTextBytes = 4), "text_limit")
  }

  test("tree rejects a long Unicode key before constructing an oversized escaped path") {
    val json = "{\"" + ("雪" * 22000) + "\":null}"
    val snapshot = PayloadSnapshot(PayloadKey("path-budget", 1), "txt2img", Some(json), 0L)
    rejected(json, "path_limit")
    assertEquals(snapshot.bodyJson, Some(json))
  }

  test("tree rejects tall long-key prefix amplification within admitted JSON depth") {
    val json = ("{\"" + ("k" * 700) + "\":") * 100 + "null" + ("}" * 100)
    rejected(json, "path_limit")
  }

  test("tree guards excessive document depth before parsing malformed trailing input") {
    rejected(("[" * 513) + "0" + ("]" * 513) + "!", "depth_limit")
  }

  test("tree guards raw token count before parsing repeated keys and malformed trailing input") {
    val json = "{" + Vector.fill(50000)("\"x\":0").mkString(",") + "}!"
    rejected(json, "nodes_limit")
  }

  test("default tree admits native depth-140 continuation and ordinary 1000-element arrays") {
    val deep = RequestPayload.tree(("[" * 140) + "0" + ("]" * 140))
    var leaf = deep
    for _ <- 0 until 140 do leaf = leaf.children.head
    assertEquals(leaf.depth, 140)
    assertEquals(leaf.path, "$" + ("[0]" * 140))
    val stop = PayloadTreeState().subtree(deep, expand = true, maxDepth = 128)
    assert(stop.depthLimited)
    val continued = stop.state.resume(stop.remaining, expand = true, maxDepth = 140)
    assertEquals(continued.remaining, Nil)
    val broad = RequestPayload.tree("[" + (0 until 1000).mkString(",") + "]")
    assertEquals(broad.childCount, 1000)
    assertEquals(broad.children(999).path, "$[999]")
    assertEquals(broad.children(999).valueJson, Some("999"))
  }

  test("tree retains exact source paths, lexical keys, scalar JSON, marks and selected paths") {
    val frozen = """{"z":["<script>雪\n</script>",null],"a.b\"/[雪]":{"items":[false]}}"""
    val tree = RequestPayload.tree(frozen)
    assertEquals(tree.children.map(_.label), Vector("a.b\"/[雪]", "z"))
    val leaf = tree.children.head.children.head.children.head
    assertEquals(leaf.path, "$[\"a.b\\\"/[雪]\"][\"items\"][0]")
    assertEquals(leaf.valueJson, Some("false"))
    assertEquals(tree.children(1).children.head.valueJson, Some("\"<script>雪\\n</script>\""))
    val marked = PayloadTreeState.initial(tree).mark(leaf).showPath(leaf)
    val first = marked.subtree(tree, expand = true, maxNodes = 1)
    val finished = first.state.resume(first.remaining, expand = true)
    assertEquals(finished.state.marked, Set("$[\"a.b\\\"/[雪]\"][\"items\"][0]"))
    assertEquals(finished.state.selectedPath, Some("$[\"a.b\\\"/[雪]\"][\"items\"][0]"))
  }
