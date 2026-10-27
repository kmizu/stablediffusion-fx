package sdfx.engine

import sdfx.core.config.*
import sdfx.core.modelcards.CardPreviewMode
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.*

class CardPreviewSessionSuite extends munit.FunSuite:
  import FakeEngine.{deleteTree, tempHome}
  private given ExecutionContext = ExecutionContext.global

  private def fixture(body: AppHome => Unit): Unit =
    val home = tempHome()
    try body(home) finally deleteTree(home.root)

  private def service(home: AppHome, settings: Settings, recovery: Option[SettingsRecovery] = None,
      persistent: Boolean = true, writer: (Path, Settings) => Unit = Settings.save,
      starts: AtomicInteger = new AtomicInteger()): AppServices =
    new AppServices(home, settings, dryRun = true, builtins = Nil,
      launchOverride = Some(() => { starts.incrementAndGet(); throw new IllegalStateException("unexpected engine start") }),
      persistSettings = persistent, installShutdownHook = false, initialRecovery = recovery, settingsWriter = writer)

  test("temporary mode is shared, listeners isolated and removable, and never saves or starts engine") {
    fixture { home =>
      val writes = new AtomicInteger()
      val starts = new AtomicInteger()
      val s = service(home, Settings(nsfwCardBlurDefault = "Blur"), writer = (_, _) => { writes.incrementAndGet(); () }, starts = starts)
      try
        val one = s.cardPreviews
        val otherWindow = s.cardPreviews
        assert(one eq otherWindow)
        assertEquals(one.current, CardPreviewMode.Blur)
        var seen = List.empty[CardPreviewMode]
        val removeBad = one.addListener(_ => throw new RuntimeException("listener"))
        val remove = otherWindow.addListener(m => seen = seen :+ m)
        one.setTemporary(CardPreviewMode.Show)
        assertEquals(otherWindow.current, CardPreviewMode.Show)
        assertEquals(seen, List(CardPreviewMode.Show))
        remove()
        one.setTemporary(CardPreviewMode.Hide)
        assertEquals(seen, List(CardPreviewMode.Show))
        removeBad()
        assertEquals(writes.get(), 0)
        assertEquals(starts.get(), 0)
      finally s.shutdown()
    }
  }

  test("saved default applies only to a new session, invalid defaults start hidden") {
    fixture { home =>
      val s = service(home, Settings(), persistent = false)
      try
        s.cardPreviews.setTemporary(CardPreviewMode.Show)
        assert(s.updateSettingsChecked(_.copy(nsfwCardBlurDefault = "Hide")).isRight)
        assertEquals(s.cardPreviews.current, CardPreviewMode.Show)
        val next = service(home, s.settings, persistent = false)
        try assertEquals(next.cardPreviews.current, CardPreviewMode.Hide) finally next.shutdown()
        assertEquals(new CardPreviewSession("bad").current, CardPreviewMode.Hide)
        assertEquals(new CardPreviewSession(null).current, CardPreviewMode.Hide)
      finally s.shutdown()
    }
  }

  test("reentrant mode changes reach later listeners in order and finish at current mode") {
    val session = new CardPreviewSession("Blur")
    var observed = List.empty[CardPreviewMode]
    val removeA = session.addListener { mode =>
      if mode == CardPreviewMode.Show then session.setTemporary(CardPreviewMode.Hide)
    }
    val removeThrowing = session.addListener(_ => throw new IllegalStateException("listener"))
    val removeB = session.addListener(mode => observed = observed :+ mode)
    session.setTemporary(CardPreviewMode.Show)
    assertEquals(observed, List(CardPreviewMode.Show, CardPreviewMode.Hide))
    assertEquals(observed.last, session.current)
    removeA(); removeThrowing(); removeB()
    session.setTemporary(CardPreviewMode.Blur)
    assertEquals(observed, List(CardPreviewMode.Show, CardPreviewMode.Hide))
  }

  test("recovery save retains independently valid preview map after the other field fails") {
    fixture { home =>
      Files.createDirectories(home.settingsFile.getParent)
      val pairs = Seq(
        ("\"model_preview_overrides\":null", "\"model_preview_folders\":{\"hyper\":[\"D:/models\"]}", false),
        ("\"model_preview_folders\":null", "\"model_preview_overrides\":{\"id\":\"D:/kept.png\"}", true)
      )
      pairs.foreach { (bad, good, overrideValid) =>
        Seq(Seq(bad, good), Seq(good, bad)).foreach { fields =>
          val bytes = s"""{"theme":"light",${fields.mkString(",")}}""".getBytes(StandardCharsets.UTF_8)
          Files.write(home.settingsFile, bytes)
          val loaded = Settings.loadResult(home.settingsFile)
          assert(loaded.recovery.nonEmpty)
          assertEquals(Files.readAllBytes(home.settingsFile).toSeq, bytes.toSeq)
          val s = service(home, loaded.settings, loaded.recovery)
          try
            assert(s.updateSettingsChecked(identity, SettingsSaveIntent.Recover).isRight)
            assertEquals(s.settingsRecovery, None)
            val saved = Settings.loadResult(home.settingsFile)
            assertEquals(saved.recovery, None)
            assertEquals(saved.settings.theme, "light")
            if overrideValid then assertEquals(saved.settings.modelPreviewOverrides, Map("id" -> "D:/kept.png"))
            else assertEquals(saved.settings.modelPreviewFolders, Map("hyper" -> Seq("D:/models")))
          finally s.shutdown()
        }
      }
    }
  }

  test("recovered normalized settings carry provenance and suppress automatic writes through shutdown") {
    fixture { home =>
      Files.createDirectories(home.settingsFile.getParent)
      val original = "{broken settings".getBytes(StandardCharsets.UTF_8)
      Files.write(home.settingsFile, original)
      val loaded = Settings.loadResult(home.settingsFile)
      assert(loaded.recovery.flatMap(_.originalSha256).nonEmpty)
      // Mirrors a normalization copy; this exercises provenance handoff, not the UI entry point.
      val normalized = loaded.settings.copy(vramMode = "model_offload")
      val writes = new AtomicInteger()
      val s = service(home, normalized, loaded.recovery, writer = (_, _) => { writes.incrementAndGet(); () })
      try
        assertEquals(s.settingsRecovery, loaded.recovery)
        s.updateSettings(_.copy(lastModel = Some("model")))
        s.updateSettings(_.copy(lastVae = Some("vae")))
        s.updateSettings(_.copy(pluginValues = Map("p" -> Map("x" -> "y"))))
        s.updateSettings(_.copy(pythonExe = Some("python")))
        assertEquals(s.settings.lastModel, Some("model"))
        assertEquals(s.settings.pluginValues, Map("p" -> Map("x" -> "y")))
        assert(s.updateSettingsChecked(_.copy(theme = "light")).isLeft)
        assertEquals(writes.get(), 0)
      finally s.shutdown()
      assertEquals(writes.get(), 0)
      assertEquals(Files.readAllBytes(home.settingsFile).toSeq, original.toSeq)
    }
  }

  test("failed checked write retains memory, file and recovery; successful recovery saves current memory") {
    fixture { home =>
      Files.createDirectories(home.settingsFile.getParent)
      val original = "{bad".getBytes(StandardCharsets.UTF_8)
      Files.write(home.settingsFile, original)
      val loaded = Settings.loadResult(home.settingsFile)
      var fail = true
      val writes = new AtomicInteger()
      val s = service(home, loaded.settings, loaded.recovery, writer = (path, value) => {
        writes.incrementAndGet()
        if fail then throw new java.io.IOException("injected failure")
        Settings.save(path, value)
      })
      try
        s.updateSettings(_.copy(lastModel = Some("remembered"), pluginValues = Map("p" -> Map("x" -> "y"))))
        val before = s.settings
        assert(s.updateSettingsChecked(_.copy(theme = "light"), SettingsSaveIntent.Recover).isLeft)
        assertEquals(s.settings, before)
        assertEquals(s.settingsRecovery, loaded.recovery)
        assertEquals(Files.readAllBytes(home.settingsFile).toSeq, original.toSeq)
        fail = false
        val result = s.updateSettingsChecked(_.copy(theme = "light"), SettingsSaveIntent.Recover)
        assert(result.isRight)
        assertEquals(s.settingsRecovery, None)
        assertEquals(Settings.load(home.settingsFile), s.settings)
        assertEquals(s.settings.lastModel, Some("remembered"))
        assertEquals(s.settings.pluginValues, Map("p" -> Map("x" -> "y")))
        assertEquals(s.settings.theme, "light")
        assertEquals(writes.get(), 2)
      finally s.shutdown()
    }
  }

  test("nonpersistent Recover cannot clear hold; checked write failure without hold does not publish") {
    fixture { home =>
      val recovery = Some(SettingsRecovery("bad", None))
      val held = service(home, Settings(), recovery, persistent = false)
      try
        assert(held.updateSettingsChecked(_.copy(theme = "light"), SettingsSaveIntent.Recover).isLeft)
        assertEquals(held.settingsRecovery, recovery)
        assertEquals(held.settings.theme, "dark")
      finally held.shutdown()
      Settings.save(home.settingsFile, Settings())
      val bytes = Files.readAllBytes(home.settingsFile)
      val s = service(home, Settings(), writer = (_, _) => throw new java.io.IOException("cannot save"))
      try
        assert(s.updateSettingsChecked(_.copy(theme = "light")).isLeft)
        assertEquals(s.settings, Settings())
        assertEquals(Files.readAllBytes(home.settingsFile).toSeq, bytes.toSeq)
      finally s.shutdown()
    }
  }

  test("aggregate bounds reject before writer, including escaped UTF-8 serialization") {
    fixture { home =>
      val writes = new AtomicInteger()
      val s = service(home, Settings(), writer = (_, _) => { writes.incrementAndGet(); () })
      try
        val tooManyUnits = (0 until 130).map(i => i.toString -> ("x" * 8000)).toMap
        val tooManyEncodedBytes = (0 until 90).map(i => i.toString -> ("\u0000" * 8000)).toMap
        Seq(tooManyUnits, tooManyEncodedBytes).foreach { overrides =>
          assert(s.updateSettingsChecked(_.copy(modelPreviewOverrides = overrides)).isLeft)
          assertEquals(s.settings.modelPreviewOverrides, Map.empty)
        }
        assertEquals(writes.get(), 0)
      finally s.shutdown()
    }
  }

  test("checked writes serialize with legacy updates and preserve concurrent fields") {
    fixture { home =>
      val entered = new CountDownLatch(1)
      val release = new CountDownLatch(1)
      val s = service(home, Settings(), writer = (path, value) => {
        entered.countDown()
        if !release.await(5, TimeUnit.SECONDS) then throw new IllegalStateException("writer timed out")
        Settings.save(path, value)
      })
      try
        val checked = Future(s.updateSettingsChecked(_.copy(nsfwCardBlurDefault = "Show")))
        assert(entered.await(5, TimeUnit.SECONDS))
        val legacy = Future(s.updateSettings(_.copy(lastModel = Some("concurrent"))))
        release.countDown()
        assert(Await.result(checked, 10.seconds).isRight)
        Await.result(legacy, 10.seconds)
        assertEquals(s.settings.nsfwCardBlurDefault, "Show")
        assertEquals(s.settings.lastModel, Some("concurrent"))
        assertEquals(Settings.load(home.settingsFile), s.settings)
      finally
        release.countDown()
        s.shutdown()
    }
  }
