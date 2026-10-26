package sdfx.engine

import sdfx.core.payload.*
import java.nio.charset.StandardCharsets.UTF_8
import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean
import scala.jdk.CollectionConverters.*

class RequestPayloadStoreSuite extends munit.FunSuite:
  private def bytes(s: String): Long = s.getBytes(UTF_8).length.toLong
  private def measured(s: PayloadSnapshot): Long =
    Vector(s.key.jobId, s.requestType, s.state.wireName).map(bytes).sum +
      (s.bodyJson.toVector ++ s.contextJson ++ s.assetsJson ++ s.sourceMode ++ s.warnings).map(bytes).sum
  private def snapshot(id: String, sequence: Int = 1, mode: String = "txt2img", body: String = "{}") =
    PayloadSnapshot(PayloadKey(id, sequence), "txt2img", Some(body), 123L,
      contextJson = Some(ujson.Obj("submitted" -> ujson.Obj("origin" -> "ui", "source_mode" -> mode)).render()), sourceMode = Some(mode))

  test("defaults enforce 50 captures, 8 MiB body, 24 MiB entry and 32 MiB total") {
    val l = RequestPayloadStore.Limits()
    assertEquals(l.maxCaptures, 50)
    assertEquals(l.maxBodyBytes, 8L * 1024 * 1024)
    assertEquals(l.maxEntryBytes, 24L * 1024 * 1024)
    assertEquals(l.maxStoreBytes, 32L * 1024 * 1024)
    val store = new RequestPayloadStore()
    (1 to 51).foreach(i => store.put(snapshot(s"job-$i")))
    assertEquals(store.snapshots.size, 50)
    assertEquals(store.snapshots.head.key.jobId, "job-2")
    assertEquals(store.latestForMode("txt2img").map(_.key.jobId), Some("job-51"))
  }

  test("identity retains identical bodies and filters by submitted mode rather than request type") {
    val store = new RequestPayloadStore()
    store.put(snapshot("same", 1, "inpaint"))
    store.put(snapshot("same", 2, "inpaint"))
    store.put(snapshot("other", 1, "img2img"))
    assertEquals(store.snapshots.map(_.key), Vector(PayloadKey("same", 1), PayloadKey("same", 2), PayloadKey("other", 1)))
    assertEquals(store.latestForMode("inpaint").map(_.key), Some(PayloadKey("same", 2)))
    assertEquals(store.latestForMode("txt2img"), None)
  }

  test("UTF-8 accounting includes all documents, identity, lifecycle and bounded warnings") {
    val store = new RequestPayloadStore()
    val s = snapshot("日本語", body = "\"猫\"").copy(assetsJson = Some("{\"data\":\"é\"}"),
      warnings = Vector.fill(200)("診断" * 1500))
    store.put(s)
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
    assert(store.snapshots.head.warnings.size <= 33)
    assert(store.snapshots.head.warnings.exists(_.contains("omitted")))
    assert(store.snapshots.head.warnings.forall(w => bytes(w) <= 1024))
    store.transition(s.key, PayloadState.Submitted)
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
    store.transition(s.key, PayloadState.Succeeded, Some("{\"model_name\":\"実測\"}"))
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
    assert(ujson.read(store.snapshots.head.contextJson.get).obj.contains("reported_at_completion"))
  }

  test("oversized body or context becomes explicit unavailable entry without truncated documents") {
    val store = new RequestPayloadStore(RequestPayloadStore.Limits(maxBodyBytes = 512, maxEntryBytes = 1500, maxStoreBytes = 5000))
    val body = snapshot("body", body = "\"" + ("x" * 512) + "\"")
    store.put(body)
    val b = store.snapshots.head
    assertEquals(b.bodyJson, None)
    assert(b.warnings.exists(_.contains("unavailable")))
    val context = snapshot("context").copy(contextJson = Some("\"" + ("猫" * 1000) + "\""))
    store.put(context)
    val c = store.snapshots.last
    assertEquals(c.contextJson, None)
    assert(c.warnings.exists(_.contains("unavailable")))
    assertEquals(c.sourceMode, Some("txt2img"))
    assertEquals(store.latestForMode("txt2img").map(_.key.jobId), Some("context"))
    assert(store.snapshots.forall(s => measured(s) <= 1500))
  }

  test("oversized job identity is rejected instead of retained with saturated byte accounting") {
    val limits = RequestPayloadStore.Limits(maxEntryBytes = 128, maxStoreBytes = 512)
    val store = new RequestPayloadStore(limits)
    store.put(snapshot("safe"))
    val seen = new ConcurrentLinkedQueue[(Vector[PayloadSnapshot], Long)]()
    store.subscribe(all => { seen.add((all, store.rejectedCaptures)); () })
    val hugeId = "i" * 1024
    store.put(snapshot(hugeId))
    assert(!store.snapshots.exists(_.key.jobId == hugeId))
    assert(store.snapshots.forall(s => measured(s) <= limits.maxEntryBytes))
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
    assertEquals(store.rejectedCaptures, 1L)
    assertEquals(seen.asScala.toVector.lastOption, Some((store.snapshots, 1L)))
  }

  test("oversized request type update leaves the prior valid same-key body unchanged") {
    val limits = RequestPayloadStore.Limits(maxEntryBytes = 128, maxStoreBytes = 512)
    val store = new RequestPayloadStore(limits)
    val key = PayloadKey("same-key", 1)
    store.put(snapshot(key.jobId))
    val before = store.snapshots
    store.put(snapshot(key.jobId).copy(requestType = "t" * 1024, state = PayloadState.Submitted))
    assertEquals(store.snapshots, before)
    assertEquals(store.snapshots.find(_.key == key).flatMap(_.bodyJson), Some("{}"))
    assert(store.snapshots.forall(s => measured(s) <= limits.maxEntryBytes))
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
    assertEquals(store.rejectedCaptures, 1L)
  }

  test("routing mode accepts only exact known native modes without retaining hostile metadata") {
    val store = new RequestPayloadStore()
    store.put(snapshot("unknown", mode = "other-mode"))
    assertEquals(store.snapshots.head.sourceMode, None)
    assertEquals(store.latestForMode("other-mode"), None)
    store.put(snapshot("inpaint", mode = "inpaint").copy(contextJson = None))
    assertEquals(store.latestForMode("inpaint").map(_.key.jobId), Some("inpaint"))
  }

  test("aggregate budget evicts oldest atomically on insert and enlarged lifecycle documents") {
    val store = new RequestPayloadStore(RequestPayloadStore.Limits(maxBodyBytes = 2000, maxEntryBytes = 3500, maxStoreBytes = 5000))
    (1 to 3).foreach(i => store.put(snapshot(s"b$i", body = "\"" + ("a" * 1400) + "\"")))
    store.transition(PayloadKey("b3", 1), PayloadState.Succeeded, Some("{\"model_name\":\"" + ("é" * 900) + "\"}"))
    assert(store.retainedBytes <= 5000)
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
    assert(!store.snapshots.exists(_.key.jobId == "b1"))
    assertEquals(store.snapshots.last.state, PayloadState.Succeeded)
  }

  test("completed subrequests remain terminal when enclosing script fails or cancels") {
    val store = new RequestPayloadStore()
    store.put(snapshot("script", 1).copy(state = PayloadState.Succeeded))
    store.put(snapshot("script", 2).copy(state = PayloadState.Submitted))
    store.finishJob("script", PayloadState.Failed)
    assertEquals(store.snapshots.map(_.state), Vector(PayloadState.Succeeded, PayloadState.Failed))
    store.finishJob("script", PayloadState.Cancelled)
    assertEquals(store.snapshots.map(_.state), Vector(PayloadState.Succeeded, PayloadState.Failed))
  }

  test("subscribe removal and clear release retained history; observer failure cannot reject capture") {
    val store = new RequestPayloadStore()
    val seen = new ConcurrentLinkedQueue[Vector[PayloadSnapshot]]()
    val remove = store.subscribe(s => { seen.add(s); () })
    val broken = store.subscribe(_ => throw new IllegalStateException("viewer closed"))
    store.put(snapshot("a"))
    assert(seen.asScala.exists(_.exists(_.key.jobId == "a")))
    remove()
    val n = seen.size()
    store.put(snapshot("b"))
    assertEquals(seen.size(), n)
    broken()
    store.clear()
    assertEquals(store.snapshots, Vector.empty)
    assertEquals(store.retainedBytes, 0L)
  }

  test("concurrent captures and transitions preserve one immutable entry per key and bounded accounting") {
    val store = new RequestPayloadStore()
    val done = new CountDownLatch(8)
    (1 to 8).foreach { worker =>
      val thread = new Thread(() =>
        try (1 to 40).foreach { i =>
          val s = snapshot(s"$worker-$i")
          store.put(s)
          store.transition(s.key, PayloadState.Succeeded)
        }
        finally done.countDown()
      )
      thread.start()
    }
    assert(done.await(10, TimeUnit.SECONDS))
    val all = store.snapshots
    assertEquals(all.map(_.key).distinct.size, all.size)
    assert(all.size <= 50)
    assertEquals(store.retainedBytes, all.map(measured).sum)
  }

  test("close rejects late capture work and new subscriptions without retaining documents") {
    val store = new RequestPayloadStore()
    store.put(snapshot("existing"))
    store.close()
    val callbacks = new java.util.concurrent.atomic.AtomicInteger(0)
    val remove = store.subscribe(_ => { callbacks.incrementAndGet(); () })
    store.put(snapshot("late"))
    store.transition(PayloadKey("late", 1), PayloadState.Succeeded, Some("{\"model_name\":\"late\"}"))
    store.finishJob("late", PayloadState.Failed)
    assertEquals(store.snapshots, Vector.empty)
    assertEquals(store.retainedBytes, 0L)
    assertEquals(callbacks.get(), 0)
    remove()
  }

  test("UTF-8 counter stops at limit plus one without materializing a huge diagnostic input") {
    val reads = new java.util.concurrent.atomic.AtomicInteger(0)
    val huge = new CharSequence:
      def length(): Int = Int.MaxValue
      def charAt(index: Int): Char = { reads.incrementAndGet(); 'a' }
      def subSequence(start: Int, end: Int): CharSequence = throw new AssertionError("must not slice the input")
      override def toString: String = throw new AssertionError("must not materialize the input")
    assertEquals(RequestPayloadStore.utf8Size(huge, 1024), 1025L)
    assert(reads.get() <= 1025, s"counter read ${reads.get()} characters")
    val literal = "a猫😀" + '\ud800' + "z"
    assertEquals(RequestPayloadStore.utf8Size(literal, 100), bytes(literal))
    assertEquals(RequestPayloadStore.utf8Size(literal, 2), 3L)
  }

  test("a blocked subscriber cannot prevent another thread from completing a store mutation") {
    val store = new RequestPayloadStore()
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val secondDone = new CountDownLatch(1)
    val first = new Thread(() => store.put(snapshot("first")))
    val second = new Thread(() =>
      try store.put(snapshot("second"))
      finally secondDone.countDown()
    )
    val remove = store.subscribe { all =>
      if all.exists(_.key.jobId == "first") && !all.exists(_.key.jobId == "second") then
        entered.countDown()
        release.await(10, TimeUnit.SECONDS)
        ()
    }
    try
      first.start()
      assert(entered.await(10, TimeUnit.SECONDS), "first callback never entered")
      second.start()
      assert(secondDone.await(5, TimeUnit.SECONDS), "second put was blocked by the unrelated callback")
      assert(store.snapshots.exists(_.key.jobId == "second"))
    finally
      release.countDown()
      first.join(10000)
      second.join(10000)
      remove()
  }

  test("reentrant transition never delivers prepared after succeeded to a later observer") {
    val store = new RequestPayloadStore()
    val key = PayloadKey("reentrant", 1)
    val transitionOnce = new AtomicBoolean(false)
    store.subscribe { all =>
      if all.exists(s => s.key == key && s.state == PayloadState.Prepared) && transitionOnce.compareAndSet(false, true) then
        store.transition(key, PayloadState.Succeeded)
    }
    val seen = new ConcurrentLinkedQueue[PayloadState]()
    store.subscribe { all => all.find(_.key == key).foreach(s => { seen.add(s.state); () }) }
    store.put(snapshot(key.jobId))
    val states = seen.asScala.toVector
    assertEquals(store.snapshots.find(_.key == key).map(_.state), Some(PayloadState.Succeeded))
    assertEquals(states.lastOption, Some(PayloadState.Succeeded))
    assert(!states.sliding(2).exists(_ == Vector(PayloadState.Succeeded, PayloadState.Prepared)), states.toString)
  }

  test("concurrent transition coalesces pending callback and cannot resurrect an older history") {
    val store = new RequestPayloadStore()
    val key = PayloadKey("parallel", 1)
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val putDone = new CountDownLatch(1)
    val transitionDone = new CountDownLatch(1)
    store.subscribe { all =>
      if all.exists(s => s.key == key && s.state == PayloadState.Prepared) then
        entered.countDown()
        release.await(10, TimeUnit.SECONDS)
        ()
    }
    val seen = new ConcurrentLinkedQueue[PayloadState]()
    store.subscribe { all => all.find(_.key == key).foreach(s => { seen.add(s.state); () }) }
    val first = new Thread(() => try store.put(snapshot(key.jobId)) finally putDone.countDown())
    val second = new Thread(() => try store.transition(key, PayloadState.Succeeded) finally transitionDone.countDown())
    try
      first.start()
      assert(entered.await(10, TimeUnit.SECONDS), "prepared callback never entered")
      second.start()
      assert(transitionDone.await(5, TimeUnit.SECONDS), "transition waited for a blocked subscriber")
    finally
      release.countDown()
      first.join(10000)
      second.join(10000)
    assert(putDone.getCount == 0, "initial publisher did not finish")
    val states = seen.asScala.toVector
    assertEquals(store.snapshots.find(_.key == key).map(_.state), Some(PayloadState.Succeeded))
    assertEquals(states.lastOption, Some(PayloadState.Succeeded))
    assert(!states.sliding(2).exists(_ == Vector(PayloadState.Succeeded, PayloadState.Prepared)), states.toString)
  }

  test("concurrent mutation cannot make the initial subscription snapshot arrive last") {
    val store = new RequestPayloadStore()
    val entered = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val subscribed = new CountDownLatch(1)
    val putDone = new CountDownLatch(1)
    val seen = new ConcurrentLinkedQueue[Vector[PayloadSnapshot]]()
    val subscribing = new Thread(() => try
      store.subscribe { all =>
        if all.isEmpty then
          entered.countDown()
          release.await(10, TimeUnit.SECONDS)
        seen.add(all)
        ()
      }
      ()
    finally subscribed.countDown())
    val putting = new Thread(() => try store.put(snapshot("new")) finally putDone.countDown())
    try
      subscribing.start()
      assert(entered.await(10, TimeUnit.SECONDS), "initial callback never entered")
      putting.start()
      assert(putDone.await(5, TimeUnit.SECONDS), "put waited for the initial callback")
    finally
      release.countDown()
      subscribing.join(10000)
      putting.join(10000)
    assert(subscribed.getCount == 0, "subscription did not finish")
    assertEquals(seen.asScala.toVector.lastOption.flatMap(_.lastOption.map(_.key.jobId)), Some("new"))
  }

  test("context depth and token guards reject allocation-heavy diagnostic documents while body stays exact") {
    val store = new RequestPayloadStore()
    val deep = ("[" * 600) + "null" + ("]" * 600)
    val wide = "[" + Vector.fill(100001)("0").mkString(",") + "]"
    Seq("deep" -> deep, "wide" -> wide).foreach { (id, context) =>
      store.put(snapshot(id).copy(contextJson = Some(context)))
      val saved = store.snapshots.last
      assertEquals(saved.bodyJson, Some("{}"))
      assertEquals(saved.contextJson, None)
      assertEquals(saved.sourceMode, Some("txt2img"))
      assert(saved.warnings.exists(_.contains("unavailable")))
    }
  }

  test("completion merge budgets both documents and final JSON overhead without replacing capture facts") {
    val store = new RequestPayloadStore(RequestPayloadStore.Limits(maxEntryBytes = 1500, maxStoreBytes = 5000))
    store.put(snapshot("completion").copy(contextJson = Some("{\"reported_at_capture\":\"" + ("a" * 600) + "\"}")))
    store.transition(PayloadKey("completion", 1), PayloadState.Succeeded,
      Some("{\"model_name\":\"" + ("\\u0000" * 200) + "\"}"))
    val saved = store.snapshots.head
    assertEquals(saved.contextJson, None)
    assertEquals(saved.bodyJson, Some("{}"))
    assertEquals(saved.sourceMode, Some("txt2img"))
    assert(saved.warnings.exists(_.contains("unavailable")))
    assertEquals(store.retainedBytes, store.snapshots.map(measured).sum)
  }
