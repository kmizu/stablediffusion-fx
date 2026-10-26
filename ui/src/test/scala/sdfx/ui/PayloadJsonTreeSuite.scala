package sdfx.ui

import sdfx.core.payload.*
import sdfx.ui.view.PayloadJsonTree
import sdfx.ui.view.PayloadJsonTree.{Action, Modifiers}
import scala.collection.mutable.ArrayDeque

/** Pure adapter tests: no Toolkit startup, engine, clipboard, or native resources. */
class PayloadJsonTreeSuite extends munit.FunSuite:
  private def fixture = RequestPayload.tree("""{"z":[3,2,1],"a":{"one":true},"empty":{},"null":null}""")

  test("source modifier priority is Alt, Shift, Ctrl/Meta, ordinary toggle"):
    assertEquals(PayloadJsonTree.action(Modifiers(alt = true, shift = true, control = true)), Action.Mark)
    assertEquals(PayloadJsonTree.action(Modifiers(shift = true, meta = true)), Action.ShowPath)
    assertEquals(PayloadJsonTree.action(Modifiers(control = true)), Action.ToggleSubtree)
    assertEquals(PayloadJsonTree.action(Modifiers(meta = true)), Action.ToggleSubtree)
    assertEquals(PayloadJsonTree.action(Modifiers()), Action.Toggle)

  test("ordinary, mark, path and accessible actions preserve independent state"):
    val root = fixture
    val child = root.children.head
    val initial = PayloadTreeState.initial(root)
    assert(initial.expanded(root.path))
    assert(initial.expanded(child.path))
    val collapsed = PayloadJsonTree.change(initial, child, Action.Toggle)
    assert(!collapsed.expanded(child.path))
    val marked = PayloadJsonTree.change(collapsed, child, Action.Mark)
    assert(marked.marked(child.path))
    val selected = PayloadJsonTree.change(marked, child, Action.ShowPath)
    assertEquals(selected.selectedPath, Some(child.path))
    assertEquals(selected.expanded, collapsed.expanded)
    assertEquals(PayloadJsonTree.change(selected, child, Action.Mark).marked, Set.empty[String])
    val leaf = root.children.find(_.kind == "null").get
    assertEquals(PayloadJsonTree.change(selected, leaf, Action.Toggle), selected)

  test("child pages preserve lexical objects and stable arrays without widening the slice"):
    val root = fixture
    val page = PayloadJsonTree.children(root, 0, 10000, 128)
    assertEquals(page.nodes.map(_.label), Vector("a", "empty", "null", "z"))
    val array = root.children.last
    assertEquals(PayloadJsonTree.children(array, 0, 10000, 128).nodes.flatMap(_.valueJson), Vector("3", "2", "1"))
    val wide = RequestPayload.tree((0 until 12000).mkString("[", ",", "]"))
    val first = PayloadJsonTree.children(wide, 0, 10000, 128)
    assertEquals(first.nodes.size, 1000)
    assertEquals(first.nextOffset, Some(1000))
    val limited = PayloadJsonTree.children(wide, 9500, 500, 128)
    assertEquals(limited.nodes.size, 500)
    assertEquals(limited.nextOffset, Some(10000))
    val exhausted = PayloadJsonTree.children(wide, 10000, 0, 128)
    assert(exhausted.nodes.isEmpty)
    assertEquals(exhausted.nextOffset, Some(10000))
    assert(exhausted.boundary.nonEmpty)

  test("absolute depth boundary offers exact continuation rather than dropping a child"):
    val leaf = PayloadNode("deep", "$[0]", 129, "null", Some("null"), Vector.empty)
    val root = PayloadNode("parent", "$", 128, "array", None, Vector(leaf))
    val blocked = PayloadJsonTree.children(root, 0, 10000, 128)
    assert(blocked.nodes.isEmpty)
    assertEquals(blocked.nextOffset, Some(0))
    assert(blocked.boundary.exists(_.contains("depth")))
    assertEquals(PayloadJsonTree.children(root, 0, 10000, 256).nodes, Vector(leaf))

  test("live materialization budget never grows across slices and resets only for a replacement window"):
    val budget = new PayloadJsonTree.MaterializationBudget()
    var reserved = 0
    for _ <- 0 until 20 do
      val n = budget.reserve(5000)
      assert(n <= 1000)
      reserved += n
      assert(reserved <= 10000)
      assertEquals(budget.used, reserved)
    assertEquals(reserved, 10000)
    assertEquals(budget.reserve(1), 0)
    budget.reset()
    assertEquals(budget.used, 0)
    assertEquals(budget.reserve(1), 1)
    assertEquals(budget.reserve(1500), 1000)

  test("recursive adapter uses compact cursors with 1000-node work slices and preserves marks/path"):
    val root = RequestPayload.tree((0 until 11000).map(i => s"[$i]").mkString("[", ",", "]"))
    var state = PayloadTreeState(marked = Set("$[0]"), selectedPath = Some("$[0]"))
    var cursor = List(PayloadTreeFrame(root))
    var visited = 0
    while cursor.nonEmpty && visited < 10000 do
      val work = PayloadJsonTree.subtreeSlice(state, cursor, true, 128, math.min(1000, 10000 - visited))
      assert(work.visited <= 1000)
      assert(work.processedFrames <= 3000)
      assert(work.remaining.size <= 4)
      visited += work.visited
      state = work.state
      cursor = work.remaining
    assertEquals(visited, 10000)
    assert(cursor.nonEmpty)
    assertEquals(state.marked, Set("$[0]"))
    assertEquals(state.selectedPath, Some("$[0]"))

  test("literal hostile Unicode labels retain text, type/count and bounded scalar rendering"):
    val root = RequestPayload.tree("""{"<script>猫.\"/[]🧪":"<b>日本語</b>","long":"""" + ("🧪" * 900) + "\"}")
    val hostile = root.children.head
    val rendered = PayloadJsonTree.label(hostile)
    assert(rendered.contains("<script>"))
    assert(rendered.contains("<b>日本語</b>"))
    assertEquals(PayloadJsonTree.change(PayloadTreeState(), hostile, Action.ShowPath).selectedPath, Some(hostile.path))
    assert(PayloadJsonTree.label(root).contains("object (2)"))
    val shortened = PayloadJsonTree.label(root.children.last)
    assert(shortened.length < 1100)
    assert(shortened.contains("…"))
    assert(!shortened.sliding(2).exists(p => p.length == 2 && Character.isHighSurrogate(p(0)) && !Character.isLowSurrogate(p(1))))
    val hugeKey = root.children.last.copy(label = "猫🧪" * 10000)
    assert(PayloadJsonTree.label(hugeKey).length < 1100)
    val hugeContainer = root.copy(label = "猫🧪" * 10000)
    assert(PayloadJsonTree.label(hugeContainer).contains("object (2)"),
      "A long key must leave room for the container type and child count")
    val hugePath = "$" + ("[\"猫🧪\"]" * 10000)
    assert(PayloadJsonTree.pathLabel(hugePath).length < 1100)
    assert(Set("object", "array", "string", "number", "boolean", "null").forall(PayloadJsonTree.typeColor(_).nonEmpty))

  private final class Queues:
    val work = ArrayDeque.empty[() => Unit]
    val ui = ArrayDeque.empty[() => Unit]
    var onUi = true
    def worker(): Unit =
      onUi = false
      try work.removeHead()() finally onUi = true
    def deliver(): Unit = ui.removeHead()()
    def gate(parser: String => PayloadNode = json => RequestPayload.tree(json)) =
      new PayloadJsonTree.AsyncPresentation(work.append(_), ui.append(_), () => onUi, parser)

  test("parse is scheduled off UI, exact frozen input is retained, and newest selection wins"):
    val q = new Queues
    var parsed = Vector.empty[String]
    val gate = q.gate(s => { assert(!q.onUi); parsed :+= s; RequestPayload.tree(s) })
    var shown = Vector.empty[String]
    gate.showJson("{\"a\":1}")(n => shown :+= n.children.head.label, e => fail(e.toString))
    assert(parsed.isEmpty)
    q.worker()
    gate.showJson("{\"b\":2}")(n => shown :+= n.children.head.label, e => fail(e.toString))
    q.deliver()
    assert(shown.isEmpty)
    q.worker(); q.deliver()
    assertEquals(shown, Vector("b"))
    assertEquals(parsed, Vector("{\"a\":1}", "{\"b\":2}"))

  test("invalid selection and disposal cancel queued work and already queued UI callbacks"):
    val q = new Queues
    var parses = 0
    var changes = 0
    val gate = q.gate(s => { parses += 1; RequestPayload.tree(s) })
    gate.showJson("[1]")(_ => changes += 1, _ => changes += 1)
    gate.invalidate()
    q.worker()
    assertEquals(parses, 0)
    assert(q.ui.isEmpty)
    gate.showJson("[2]")(_ => changes += 1, _ => changes += 1)
    q.worker()
    gate.dispose()
    q.deliver()
    assertEquals(changes, 0)
    gate.showJson("[3]")(_ => changes += 1, _ => changes += 1)
    assert(q.work.isEmpty)

  test("parse failure is visible and guard rejects accidental UI execution"):
    val q = new Queues
    var errors = Vector.empty[String]
    val gate = q.gate()
    gate.showJson("invalid")(_ => fail("invalid JSON was accepted"), e => errors :+= e.getClass.getSimpleName)
    q.worker(); q.deliver()
    assertEquals(errors.size, 1)
    gate.showJson("[]")(_ => fail("UI parse was accepted"), e => errors :+= e.getMessage)
    q.work.removeHead()()
    q.deliver()
    assert(errors.last.contains("UI thread"))

  test("a selection burst retains one queued parse drain and parses only the newest document"):
    val q = new Queues
    var parsed = Vector.empty[String]
    var shown = Vector.empty[String]
    val gate = q.gate(s => { parsed :+= s; RequestPayload.tree(s) })
    for i <- 0 until 100 do
      gate.showJson(s"[$i]")(n => shown :+= n.children.head.valueJson.get, e => fail(e.toString))
    assertEquals(q.work.size, 1)
    q.worker(); q.deliver()
    assertEquals(parsed, Vector("[99]"))
    assertEquals(shown, Vector("99"))

  test("parsed-result bursts retain one queued UI drain publishing only the newest root"):
    val q = new Queues
    var shown = Vector.empty[String]
    val gate = q.gate()
    for i <- 0 until 100 do
      gate.showJson(s"[$i]")(n => shown :+= n.children.head.valueJson.get, e => fail(e.toString))
      q.worker()
    assertEquals(q.ui.size, 1)
    q.deliver()
    assertEquals(shown, Vector("99"))

  test("invalidate and dispose clear pending parse and publication slots even during bursts"):
    val q = new Queues
    var parsed = 0
    var shown = 0
    val gate = q.gate(s => { parsed += 1; RequestPayload.tree(s) })
    for i <- 0 until 20 do gate.showJson(s"[$i]")(_ => shown += 1, _ => shown += 1)
    gate.invalidate()
    q.worker()
    assertEquals(parsed, 0)
    gate.showJson("[21]")(_ => shown += 1, _ => shown += 1)
    q.worker()
    gate.dispose()
    while q.ui.nonEmpty do q.deliver()
    assertEquals(shown, 0)

  test("structural preflight rejects node/depth excess before invoking the parser"):
    val cases = Vector(
      (Vector.fill(100001)("0").mkString("[", ",", "]"), "nodes_limit"),
      (("[" * 513) + "null" + ("]" * 513), "depth_limit"))
    cases.foreach { (frozen, expected) =>
      val q = new Queues
      var parsed = 0
      var shown = 0
      var errors = Vector.empty[String]
      val gate = q.gate(s => { parsed += 1; fixture })
      gate.showJson(frozen)(_ => shown += 1, e => errors :+= e.getMessage)
      q.worker(); q.deliver()
      assertEquals(parsed, 0, expected)
      assertEquals(shown, 0)
      assert(errors.exists(_.contains(expected)))
      // The exact frozen document remains the caller's retained body. Tree rejection does not rewrite it.
      assertEquals(frozen.head, '[')
      assertEquals(frozen.last, ']')
    }

  test("structural preflight treats hostile quoted punctuation as literal string content"):
    val frozen = ujson.write(ujson.Obj("text" -> (("[" * 100001) + ("," * 100001) + "猫🧪\"\\")))
    val q = new Queues
    var parsed = 0
    var shown = 0
    val gate = q.gate(s => { assertEquals(s, frozen); parsed += 1; fixture })
    gate.showJson(frozen)(_ => shown += 1, e => fail(e.toString))
    q.worker(); q.deliver()
    assertEquals(parsed, 1)
    assertEquals(shown, 1)

  test("newer ordinary expansion supersedes a pending recursive publication without losing marks/path"):
    val q = new Queues
    val publication = new PayloadJsonTree.ExpansionPublication(q.ui.append(_))
    val root = RequestPayload.tree("""{"A":{"one":1,"two":2},"B":{"leaf":true}}""")
    val a = root.children.head
    val b = root.children.last
    var state = PayloadTreeState.initial(root).mark(b).showPath(b)
    val ticket = publication.supersede()
    var updates = 0
    publication.publish(ticket, Iterator(a.path), true, () => true)(
      (path, expand) => { updates += 1; state = state.copy(expanded = if expand then state.expanded + path else state.expanded - path) },
      _ => (), () => ())
    state = state.toggle(b)
    publication.supersede() // the same ordering policy used by ordinary native toggle/disclosure
    q.deliver()
    assertEquals(updates, 0)
    assert(!state.expanded(b.path))
    assert(!state.expanded(a.path))
    assertEquals(state.marked, Set(b.path))
    assertEquals(state.selectedPath, Some(b.path))

  test("recursive deltas merge into live expansion state after a lazy small-container publication"):
    val q = new Queues
    val publication = new PayloadJsonTree.ExpansionPublication(q.ui.append(_))
    var state = PayloadTreeState(marked = Set("$[1]"), selectedPath = Some("$[1]"))
    val ticket = publication.supersede()
    publication.publish(ticket, Iterator("$[0]"), true, () => true)(
      (path, expand) => state = state.copy(expanded = if expand then state.expanded + path else state.expanded - path),
      _ => (), () => ())
    state = state.copy(expanded = state.expanded + "$[1]") // lazy initial policy, not a newer explicit input
    q.deliver()
    assertEquals(state.expanded, Set("$[0]", "$[1]"))
    assertEquals(state.marked, Set("$[1]"))
    assertEquals(state.selectedPath, Some("$[1]"))

  test("FX recursive publication consumes a compact iterator in at most1000 update slices"):
    val q = new Queues
    val publication = new PayloadJsonTree.ExpansionPublication(q.ui.append(_))
    var consumed = 0
    var updates = 0
    var slices = Vector.empty[Int]
    var complete = 0
    val changes = Iterator.range(0, 2501).map { i => consumed += 1; s"$$[$i]" }
    val ticket = publication.supersede()
    publication.publish(ticket, changes, false, () => true)(
      (_, _) => updates += 1, n => slices :+= n, () => complete += 1)
    assertEquals(consumed, 0)
    assertEquals(q.ui.size, 1)
    q.deliver()
    assertEquals(updates, 1000)
    assertEquals(consumed, 1000)
    assertEquals(slices, Vector(1000))
    assertEquals(q.ui.size, 1)
    q.deliver()
    assertEquals(updates, 2000)
    q.deliver()
    assertEquals(updates, 2501)
    assertEquals(slices, Vector(1000, 1000, 501))
    assertEquals(complete, 1)
    assert(q.ui.isEmpty)

  test("FX continuation checks newer input and replaced-window guards before consuming more paths"):
    for replaceWindow <- Vector(false, true) do
      val q = new Queues
      val publication = new PayloadJsonTree.ExpansionPublication(q.ui.append(_))
      var currentWindow = true
      var updates = 0
      var complete = 0
      val ticket = publication.supersede()
      publication.publish(ticket, Iterator.fill(2501)("$"), true, () => currentWindow)(
        (_, _) => updates += 1, _ => (), () => complete += 1)
      q.deliver()
      assertEquals(updates, 1000)
      if replaceWindow then currentWindow = false else publication.supersede()
      q.deliver()
      assertEquals(updates, 1000)
      assertEquals(complete, 0)
      assert(q.ui.isEmpty)

  test("valid JSON rejected by derived-tree limits retains its structured reason without reparsing"):
    val cases = Vector(
      RequestPayload.TreeLimits(maxPathBytes = 2) -> "path_limit",
      RequestPayload.TreeLimits(maxTotalPathBytes = 2) -> "total_path_limit",
      RequestPayload.TreeLimits(maxTextBytes = 1) -> "text_limit")
    cases.foreach { (limits, expected) =>
      val q = new Queues
      var parses = 0
      var shown = 0
      var errors = Vector.empty[String]
      val gate = q.gate(json => { parses += 1; RequestPayload.tree(json, limits) })
      gate.showJson("""{"key":0}""")(_ => shown += 1, e => errors :+= e.getMessage)
      q.worker(); q.deliver()
      assertEquals(parses, 1)
      assertEquals(shown, 0)
      assert(errors.exists(_.contains(expected)), expected)
      assert(!errors.exists(_.contains("invalid_json")))
    }
