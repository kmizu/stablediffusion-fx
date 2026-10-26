package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.infotext.Infotext
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.engine.testplugins.{MarkerAlwaysOn, PartialGridScript, RepeatScript, ThrowingScript}

import java.util.concurrent.{ConcurrentLinkedQueue, CountDownLatch, TimeUnit}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class JobQueueSuite extends munit.FunSuite:
  import FakeEngine.*

  override val munitTimeout: Duration = 120.seconds

  private lazy val home = tempHome()
  private lazy val services =
    val s = new AppServices(home, Settings(lastModel = Some("foo")), dryRun = true, builtins = Seq(new RepeatScript, new MarkerAlwaysOn),
      launchOverride = Some(() => launch()), persistSettings = false)
    s.start()
    s

  override def afterAll(): Unit =
    services.shutdown()
    deleteTree(home.root)

  private def queue = services.queue
  private def base = GenerationParams(prompt = "q", seed = 5, steps = 2, width = 64, height = 64)
  private def slow(stepDelay: String) = base.copy(steps = 10, extra = Map("step_delay" -> stepDelay))

  private def recording(): (ConcurrentLinkedQueue[JobEvent], () => Unit) =
    val evs = new ConcurrentLinkedQueue[JobEvent]()
    (evs, queue.addListener(evs.add(_)))

  test("jobs run in FIFO order and report events") {
    val (evs, remove) = recording()
    try
      val jobs = (1 to 3).map(i => Job(Job.newId(), base.copy(seed = i.toLong)))
      val fs = jobs.map(queue.submit)
      fs.foreach(f => assertEquals(await(f).images.size, 1))
      val started = evs.asScala.collect { case JobEvent.JobStarted(j) => j.id }.toList
      assertEquals(started, jobs.map(_.id).toList)
      val finished = evs.asScala.collect { case JobEvent.JobFinished(id, _) => id }.toList
      assertEquals(finished, jobs.map(_.id).toList)
      assert(evs.asScala.exists(_.isInstanceOf[JobEvent.Progress]))
      assert(evs.asScala.exists(_.isInstanceOf[JobEvent.Preview]))
      assertEquals(evs.asScala.collect { case e: JobEvent.ImageSaved => e }.size, 3)
      assert(evs.asScala.collect { case e: JobEvent.JobQueued => e }.size == 3)
    finally remove()
  }

  test("cancel a pending job and clear the queue") {
    val (evs, remove) = recording()
    val first = Job(Job.newId(), base)
    val started = new CountDownLatch(1)
    val release = new CountDownLatch(1)
    val removeGate = queue.addListener {
      case JobEvent.JobStarted(j) if j.id == first.id =>
        started.countDown()
        release.await(20, TimeUnit.SECONDS)
        ()
      case _ => ()
    }
    try
      val a = queue.submit(first)
      assert(started.await(20, TimeUnit.SECONDS), "first job did not start")
      val b = Job(Job.newId(), base)
      val c = Job(Job.newId(), base)
      val d = Job(Job.newId(), base)
      val fb = queue.submit(b)
      val fc = queue.submit(c)
      val fd = queue.submit(d)
      assertEquals(queue.pending.map(_.id), Seq(b.id, c.id, d.id))
      assert(queue.cancel(b.id))
      val eb = intercept[JobCancelledException](await(fb))
      assertEquals(eb.jobId, b.id)
      assertEquals(queue.clear(), 2)
      intercept[JobCancelledException](await(fc))
      intercept[JobCancelledException](await(fd))
      release.countDown()
      assertEquals(await(a).images.size, 1)
      assert(evs.asScala.exists { case JobEvent.JobCancelled(id, _) => id == b.id; case _ => false })
    finally
      release.countDown()
      removeGate()
      remove()
  }

  test("script job: every sub-generation gets AlwaysOn processing, grids are separate") {
    val (evs, remove) = recording()
    try
      val script = new RepeatScript
      val marker = new MarkerAlwaysOn
      val disabledMarker = new MarkerAlwaysOn
      val job = Job(Job.newId(), base, Some(script -> Values(Map("n" -> "3", "grid" -> "true"))),
        Seq(marker -> Values(Map("enabled" -> "true", "value" -> "hello")), disabledMarker -> Values(Map("enabled" -> "false", "value" -> "nope"))))
      val out = await(queue.submit(job))
      assertEquals(out.images.map(_.seed), Seq(1000L, 1001L, 1002L))
      assertEquals(out.grids.size, 1)
      assert(out.grids.head.path.toString.replace('\\', '/').contains("/grids/"))
      out.images.foreach { img =>
        assertEquals(Infotext.parse(img.infotext).get("Marker"), Some("hello"))
        assertEquals(img.params.hooks.get("marker"), Some(ujson.Str("hello")))
      }
      val st = await(services.client.raw("get_state"))
      assertEquals(st("last_generation")("request")("params")("hooks")("marker").str, "hello")
      val statuses = evs.asScala.collect { case JobEvent.Status(_, t, d, n) => (t, d, n) }.toList
      assertEquals(statuses, List(("repeat 1/3", 0, 3), ("repeat 2/3", 1, 3), ("repeat 3/3", 2, 3)))
      assert(evs.asScala.exists { case JobEvent.Progress(_, _, t, _, 3) => t.startsWith("repeat"); case _ => false })
      assert(evs.asScala.exists { case JobEvent.ImageSaved(_, _, true) => true; case _ => false })
    finally remove()
  }

  test("plain job applies AlwaysOn once") {
    val job = Job(Job.newId(), base, None, Seq(new MarkerAlwaysOn -> Values(Map("enabled" -> "true", "value" -> "once"))))
    val out = await(queue.submit(job))
    assertEquals(Infotext.parse(out.images.head.infotext).get("Marker"), Some("once"))
  }

  test("interrupt stops the script loop and keeps partial outputs") {
    val (evs, remove) = recording()
    try
      val job = Job(Job.newId(), slow("0.03"), Some(new RepeatScript -> Values(Map("n" -> "20"))))
      val f = queue.submit(job)
      assert(eventually()(evs.asScala.exists(_.isInstanceOf[JobEvent.ImageSaved])))
      assert(queue.interruptCurrent())
      val e = intercept[JobCancelledException](await(f))
      assert(e.partial.images.nonEmpty && e.partial.images.size < 20, e.partial.images.size.toString)
      assert(evs.asScala.exists { case JobEvent.JobCancelled(id, p) => id == job.id && p.images.nonEmpty; case _ => false })
      // queue keeps working afterwards
      assertEquals(await(queue.submit(Job(base))).images.size, 1)
      assert(eventually()(queue.isIdle), s"queue not idle: current=${queue.current} pending=${queue.pending}")
    finally remove()
  }

  test("fatal errors from a job fail that job but keep the worker alive") {
    val (evs, remove) = recording()
    try
      val oom = Job(Job.newId(), base, Some(new ThrowingScript(new OutOfMemoryError("test oom")) -> Values.empty))
      // scala.concurrent boxes fatal errors in an ExecutionException
      def root(t: Throwable): Throwable = if t.getCause != null && (t.getCause ne t) then root(t.getCause) else t
      val e1 = intercept[Throwable](await(queue.submit(oom)))
      assert(root(e1).isInstanceOf[OutOfMemoryError] && root(e1).getMessage == "test oom", e1.toString)
      val linkage = Job(Job.newId(), base, Some(new ThrowingScript(new NoClassDefFoundError("missing/Dep")) -> Values.empty))
      val e2 = intercept[Throwable](await(queue.submit(linkage)))
      assert(root(e2).isInstanceOf[NoClassDefFoundError], e2.toString)
      assert(evs.asScala.exists { case JobEvent.JobFailed(id, _) => id == oom.id; case _ => false })
      assertEquals(await(queue.submit(Job(base))).images.size, 1)
      assert(eventually()(queue.isIdle))
    finally remove()
  }

  test("generateTransient: AlwaysOn applied, events forwarded, files stay in .tmp, no ImageSaved") {
    val (evs, remove) = recording()
    try
      val job = Job(Job.newId(), base, Some(new testplugins.TransientScript -> Values.empty),
        Seq(new MarkerAlwaysOn -> Values(Map("enabled" -> "true", "value" -> "tr"))))
      val out = await(queue.submit(job))
      assertEquals(out.images.map(_.seed), Seq(77L, 78L))
      val tmp = home.tmpDir(services.settings).toAbsolutePath.normalize()
      out.images.foreach { img =>
        assert(java.nio.file.Files.isRegularFile(img.path))
        assertEquals(img.path.toAbsolutePath.normalize().getParent, tmp)
        assertEquals(img.params.extra.get("Marker"), Some("tr"))
        java.nio.file.Files.deleteIfExists(img.path)
      }
      assert(!evs.asScala.exists(_.isInstanceOf[JobEvent.ImageSaved]))
      assert(evs.asScala.exists { case p: JobEvent.Progress => p.jobId == job.id; case _ => false })
      assertEquals(await(services.client.raw("get_state"))("last_generation")("request")("params")("hooks")("marker").str, "tr")
    finally remove()
  }

  test("JobLimits: a script looping past maxGenerations fails the job after the allowed generations") {
    val (evs, remove) = recording()
    try
      val job = Job(Job.newId(), base, Some(new RepeatScript -> Values(Map("n" -> "5"))), limits = JobLimits(maxGenerations = 3))
      val e = intercept[JobLimitExceeded](await(queue.submit(job)))
      assert(e.getMessage.contains("3 generations"), e.getMessage)
      assertEquals(evs.asScala.count { case JobEvent.ImageSaved(id, _, _) => id == job.id; case _ => false }, 3)
      assert(evs.asScala.exists { case JobEvent.JobFailed(id, _) => id == job.id; case _ => false })
    finally remove()
  }

  test("JobLimits: hires target over maxSide is rejected before reaching the engine") {
    val before = await(services.client.raw("get_state"))("last_generation")
    val p = base.copy(prompt = "hires-limit", hires = sdfx.core.model.HiresParams(enabled = true, scale = 4.0))
    val e = intercept[JobLimitExceeded](await(queue.submit(Job(Job.newId(), p, limits = JobLimits(maxSide = 200)))))
    assert(e.getMessage.contains("hires target 256x256"), e.getMessage)
    assertEquals(await(services.client.raw("get_state"))("last_generation"), before)
    // within limits it runs
    assertEquals(await(queue.submit(Job(Job.newId(), p, limits = JobLimits(maxSide = 256)))).images.size, 1)
  }

  test("JobLimits.check: images and pixels per generation") {
    val l = JobLimits(maxImagesPerGeneration = 4, maxPixelsPerGeneration = 64L * 64 * 3)
    l.check(base.copy(batchSize = 2, nIter = 2), 1)
    intercept[JobLimitExceeded](l.check(base.copy(batchSize = 1, nIter = 5), 1))
    intercept[JobLimitExceeded](l.check(base.copy(batchSize = 4), 1))
    intercept[JobLimitExceeded](JobLimits(maxSide = 32).check(base, 1))
    l.check(base.copy(batchSize = 3), 1)
  }

  test("per-job temp files are deleted when the job ends, except files in its output") {
    import java.nio.file.Files
    val ok = new testplugins.TempUsingScript(fail = false)
    val out = await(queue.submit(Job(Job.newId(), base, Some(ok -> Values.empty))))
    assertEquals(ok.created.size, 4)
    val kept = ok.kept.get
    assertEquals(out.images.map(_.path), Seq(kept))
    assert(Files.exists(kept), "file in the job output must survive")
    ok.created.filterNot(_ == kept).foreach(p => assert(!Files.exists(p), s"not cleaned: $p"))
    Files.deleteIfExists(kept)

    val bad = new testplugins.TempUsingScript(fail = true)
    intercept[IllegalStateException](await(queue.submit(Job(Job.newId(), base, Some(bad -> Values.empty)))))
    assertEquals(bad.created.size, 4)
    bad.created.foreach(p => assert(!Files.exists(p), s"not cleaned after failure: $p"))
  }

  test("engine-side interrupt is visible via ctx.isInterrupted; partial grid is kept") {
    val script = new PartialGridScript
    val job = Job(Job.newId(), base, Some(script -> Values.empty))
    val e = intercept[JobCancelledException](await(queue.submit(job)))
    assert(script.sawInterrupted)
    assertEquals(e.partial.images.size, 1)
    assertEquals(e.partial.grids.size, 1)
  }

  test("cancel of the running plain job ends as cancelled") {
    val f = queue.submit(Job(Job.newId(), slow("0.1")))
    assert(eventually()(queue.current.isDefined))
    Thread.sleep(200)
    assert(queue.cancel(queue.current.get.id))
    intercept[JobCancelledException](await(f))
  }

  test("failures are reported") {
    val (evs, remove) = recording()
    try
      val job = Job(Job.newId(), base.copy(checkpoint = Some("does-not-exist")))
      val e = intercept[sdfx.core.protocol.EngineError](await(queue.submit(job)))
      assert(e.message.contains("does-not-exist"))
      assert(evs.asScala.exists { case JobEvent.JobFailed(id, _) => id == job.id; case _ => false })
    finally remove()
  }
