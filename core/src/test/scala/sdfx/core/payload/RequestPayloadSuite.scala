package sdfx.core.payload

class RequestPayloadSuite extends munit.FunSuite:
  test("frozen documents retain literal values independently of mutable input") {
    val input = ujson.Obj("z" -> ujson.Arr(ujson.Null, true, 1.25, "<script>雪\n</script>"), "a" -> ujson.Obj("one" -> false))
    val frozen = RequestPayload.freeze(input)
    input("z")(3) = "changed"
    val tree = RequestPayload.tree(frozen)
    assertEquals(tree.children.map(_.label), Vector("a", "z"))
    assertEquals(tree.children(1).children.map(_.kind), Vector("null", "boolean", "number", "string"))
    assertEquals(tree.children(1).children.map(_.valueJson), Vector(Some("null"), Some("true"), Some("1.25"), Some("\"<script>雪\\n</script>\"")))
    assertEquals(ujson.read(frozen)("z")(3).str, "<script>雪\n</script>")
  }

  test("JSON paths quote hostile object keys and preserve array indices") {
    val tree = RequestPayload.tree("""{"a.b\"/[雪]":{"items":[null]}}""")
    val leaf = tree.children.head.children.head.children.head
    assertEquals(tree.path, "$")
    assertEquals(leaf.path, "$[\"a.b\\\"/[雪]\"][\"items\"][0]")
  }

  test("root and small containers expand initially and leaves never expand") {
    val tree = RequestPayload.tree("""{"large":[1,2],"small":[{"x":true}],"empty":{}}""")
    val initial = PayloadTreeState.initial(tree)
    assert(initial.expanded("$"))
    assert(!initial.expanded("$[\"large\"]"))
    assert(initial.expanded("$[\"small\"]"))
    assert(initial.expanded("$[\"empty\"]"))
    val leaf = tree.children.find(_.label == "large").get.children.head
    assertEquals(initial.toggle(leaf), initial)
    val large = tree.children.find(_.label == "large").get
    assert(initial.toggle(large).expanded(large.path))
  }

  test("recursive toggling and marks use paths without changing the document") {
    val tree = RequestPayload.tree("""{"a":{"b":[1,2]},"c":null}""")
    val initial = PayloadTreeState.initial(tree)
    val expanded = initial.subtree(tree, expand = true).state
    assertEquals(expanded.expanded, Set("$", "$[\"a\"]", "$[\"a\"][\"b\"]"))
    assertEquals(expanded.subtree(tree, expand = false).state.expanded, Set.empty[String])
    val marked = expanded.mark(tree.children.head).showPath(tree.children.head)
    assertEquals(marked.marked, Set("$[\"a\"]"))
    assertEquals(marked.selectedPath, Some("$[\"a\"]"))
    assertEquals(marked.mark(tree.children.head).marked, Set.empty[String])
  }

  test("subtree work stops visibly at its node budget and can resume") {
    val tree = RequestPayload.tree("""{"a":{"b":{}},"c":{}}""")
    val first = PayloadTreeState().subtree(tree, expand = true, maxNodes = 2)
    assert(first.remaining.nonEmpty)
    assertEquals(first.visited, 2)
    val done = first.state.resume(first.remaining, expand = true, maxNodes = 10)
    assertEquals(done.remaining.size, 0)
    assertEquals(done.state.expanded, Set("$", "$[\"a\"]", "$[\"a\"][\"b\"]", "$[\"c\"]"))
  }

  test("terminal payload states cannot be replaced by a later job failure") {
    val completed = PayloadSnapshot(PayloadKey("job", 1), "txt2img", Some("{}"), 1L, state = PayloadState.Succeeded)
    assertEquals(completed.transition(PayloadState.Cancelled), completed)
    val pending = completed.copy(key = PayloadKey("job", 2), state = PayloadState.Submitted)
    assertEquals(pending.transition(PayloadState.Failed).state, PayloadState.Failed)
    assertEquals(SubmittedPayloadContext("mcp").configurationJson, None)
    assertEquals(SubmittedPayloadContext("mcp").uiVariationEnabled, None)
    assertEquals(SubmittedPayloadContext("ui", Some("txt2img"), None).uiVariationEnabled, None)
  }

  test("initial expansion exposes pending work and depth limits survive continuation") {
    val tree = RequestPayload.tree("""{"a":{"b":{}},"c":{}}""")
    val initial = PayloadTreeState.initial(tree, maxNodes = 1)
    assert(initial.initialLimitReached)
    assertEquals(initial.expanded, Set("$"))
    val finished = initial.resumeInitial(maxNodes = 20).state
    assert(!finished.initialLimitReached)
    assert(finished.expanded("$[\"a\"][\"b\"]"))
    val bounded = PayloadTreeState().subtree(tree, expand = true, maxDepth = 1)
    assert(bounded.depthLimited)
    assertEquals(bounded.remaining.filter(_.nextChildIndex < 0).map(_.path), List("$[\"a\"][\"b\"]"))
    val stillBounded = bounded.state.resume(bounded.remaining, expand = true, maxDepth = 1)
    assert(stillBounded.depthLimited)
    assert(!stillBounded.state.expanded("$[\"a\"][\"b\"]"))
    assert(bounded.state.resume(bounded.remaining, expand = true, maxDepth = 2).state.expanded("$[\"a\"][\"b\"]"))
  }

  test("wide subtree slices keep a compact frontier instead of enumerating all siblings") {
    val children = Vector.tabulate(20000)(i => PayloadNode(i.toString, s"$$[$i]", 1, "object", None, Vector.empty))
    val tree = PayloadNode("$", "$", 0, "array", None, children)
    var work = PayloadTreeState().mark(children.head).showPath(children.head).subtree(tree, expand = true, maxNodes = 1)
    assertEquals(work.visited, 1)
    assert(work.processedFrames <= 3)
    assert(work.remaining.size <= 2, s"One root visit enumerated ${work.remaining.size} pending siblings")
    for _ <- 0 until 20 do
      work = work.state.resume(work.remaining, expand = true, maxNodes = 1)
      assert(work.visited <= 1)
      assert(work.processedFrames <= 3)
      assert(work.remaining.size <= 2, s"One slice retained ${work.remaining.size} pending siblings")
    assertEquals(work.state.marked, Set("$[0]"))
    assertEquals(work.state.selectedPath, Some("$[0]"))
    while work.remaining.nonEmpty do
      work = work.state.resume(work.remaining, expand = true, maxNodes = 1000)
      assert(work.processedFrames <= 3000)
      assert(work.remaining.size <= 2)
    assertEquals(work.state.expanded.size, 20001)
    assert(work.state.expanded("$[19999]"))

    val initial = PayloadTreeState.initial(tree, maxNodes = 1)
    assert(initial.initialLimitReached)
    assert(initial.initialRemaining.size <= 2, s"Initial root visit enumerated ${initial.initialRemaining.size} pending siblings")
    val complete = Iterator.iterate(initial)(_.resumeInitial(maxNodes = 1000).state).dropWhile(_.initialLimitReached).next()
    assertEquals(complete.expanded.size, 20001)

    val depthStop = PayloadTreeState().subtree(tree, expand = true, maxDepth = 0)
    assert(depthStop.depthLimited)
    assert(depthStop.remaining.size <= 2, "Depth-limited siblings must also remain behind a cursor")
    val retry = depthStop.state.resume(depthStop.remaining, expand = true, maxDepth = 0, maxNodes = 1)
    assert(retry.depthLimited)
    assertEquals(retry.remaining, depthStop.remaining)
    assert(retry.processedFrames <= 3)
  }
