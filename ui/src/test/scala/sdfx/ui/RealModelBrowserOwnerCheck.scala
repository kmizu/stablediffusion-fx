package sdfx.ui

import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.control.ListView
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.modelcards.ModelCardRecord
import sdfx.engine.{AppServices, ModelCardCatalog}
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{CatalogBrowserView, ModelBrowser, PreviewImageLoader}
import sdfx.ui.vm.UiState

/** Manual ROOT-only native shared-owner check. One output directory argument; no engine or clipboard. */
object RealModelBrowserOwnerCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String)(condition: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    var done = false
    while !done && System.nanoTime() < end do
      done = condition
      if !done then Thread.sleep(20)
    require(done, s"Timed out: $label")

  private def sha(path: Path): String =
    MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 0xff}%02x").mkString

  private def fieldByType[A](instance: AnyRef, kind: Class[A]): A =
    val field = instance.getClass.getDeclaredFields.find(f => kind.isAssignableFrom(f.getType))
      .getOrElse(throw new IllegalStateException(s"No ${kind.getSimpleName} field in ${instance.getClass.getName}"))
    field.setAccessible(true)
    kind.cast(field.get(instance))

  private def visibleOwner(browser: ModelBrowser): Option[CatalogBrowserView] =
    val field = browser.getClass.getDeclaredFields.find(_.getName == "sdfx$ui$view$ModelBrowser$$current")
      .getOrElse(throw new IllegalStateException("No actual browser current-view field"))
    field.setAccessible(true)
    field.get(browser).asInstanceOf[Option[CatalogBrowserView]]

  private def browser(window: MainWindow): ModelBrowser = fieldByType(window, classOf[ModelBrowser])
  private def stage(owner: Stage): Stage =
    Window.getWindows.asScala.collectFirst {
      case s: Stage if s.isShowing && s.getOwner == owner && s.getTitle == "Models and networks" => s
    }.getOrElse(throw new IllegalStateException("Actual browser stage not visible"))

  private def hasModel(s: Stage, path: String): Boolean =
    s.getScene.getRoot.lookupAll(".list-view").asScala.exists {
      case list: ListView[?] => list.getItems.asScala.exists {
        case r: ModelCardRecord => r.id.modelPath == path
        case _ => false
      }
      case _ => false
    }

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val run = Files.createTempDirectory(output, "browser-owner-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    val modelDir = Files.createDirectories(home.root.resolve("fixture-models"))
    val model = Files.write(modelDir.resolve("fixture.safetensors"), Array[Byte](1, 2, 3))
    val originalHash = sha(model)
    val starts = new AtomicInteger()
    val checks = ArrayBuffer.empty[String]
    val screenshots = ArrayBuffer.empty[String]
    val cleanupErrors = ArrayBuffer.empty[String]
    var failure: Throwable = null
    var services: AppServices = null
    var one: MainWindow = null
    var two: MainWindow = null
    var startedFx = false
    def cleanup(label: String)(action: => Unit): Unit =
      try action
      catch case e: Throwable => cleanupErrors += s"$label: ${Option(e.getMessage).getOrElse(e.toString)}"
    try
      val dirs = ModelDirs(checkpoints = Seq(modelDir.toString), vae = Nil, lora = Nil, controlnet = Nil,
        embeddings = Nil, upscalers = Nil, faceRestore = Nil, svd = Nil, zero123 = Nil, photomaker = Nil)
      services = new AppServices(home, Settings(modelDirs = dirs), true, StandardPlugins.withOwnAxes,
        launchOverride = Some(() => { starts.incrementAndGet(); throw new IllegalStateException("unexpected engine launch") }),
        persistSettings = false, installShutdownHook = false)
      val ready = new CountDownLatch(1)
      Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
      require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
      startedFx = true
      fx {
        one = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        two = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        one.show(); two.show()
        one.ctx.openModelBrowser("checkpoint", None)
      }
      eventually("first actual browser renders fixture") { fx { hasModel(stage(one.stage), model.toAbsolutePath.normalize.toString) } }
      fx { two.ctx.openModelBrowser("checkpoint", None) }
      eventually("second actual browser renders fixture") { fx { hasModel(stage(two.stage), model.toAbsolutePath.normalize.toString) } }
      fx {
        Seq(one, two).zipWithIndex.foreach { (window, i) =>
          val browserStage = stage(window.stage)
          browserStage.getScene.getRoot.applyCss(); browserStage.getScene.getRoot.layout()
          val out = run.resolve(s"browser-${i + 1}.png")
          require(ImageIO.write(SwingFXUtils.fromFXImage(browserStage.getScene.snapshot(null), null), "png", out.toFile))
          screenshots += out.toString
        }
      }
      val identities = fx {
        val b1 = browser(one); val b2 = browser(two)
        (fieldByType(b1, classOf[ModelCardCatalog]) eq services.modelCards,
          fieldByType(b2, classOf[ModelCardCatalog]) eq services.modelCards)
      }
      require(identities == ((true, true)), s"Actual browser catalog owners differ from AppServices: $identities")
      checks += "both actual browsers use AppServices.modelCards"
      eventually("both browser views show shared catalog revision") { fx {
        val revision = services.modelCards.snapshotInfo.map(_.revision)
        revision.nonEmpty && visibleOwner(browser(one)).exists(v => revision.contains(v.revision)) &&
          visibleOwner(browser(two)).exists(v => revision.contains(v.revision))
      }}
      checks += "both visible views consume current shared revision"
      val previous = services.modelCards.snapshotInfo.get.revision
      fx { one.dispose(); one.stage.close() }
      val extraLease = PreviewImageLoader.acquire()
      try
        val unsubscribe = extraLease.loader.addCapacityListener(() => ())
        unsubscribe()
      finally extraLease.close()
      checks += "first window disposal retains process loader"
      fx { stage(two.stage).getScene.getRoot.lookupAll(".button").asScala.collectFirst {
        case b: javafx.scene.control.Button if b.getText == "Refresh previews" => b
      }.getOrElse(throw new IllegalStateException("Missing actual refresh button")).fire() }
      eventually("remaining browser refreshes shared catalog") { fx {
        services.modelCards.snapshotInfo.exists(_.revision > previous) &&
          visibleOwner(browser(two)).exists(v => services.modelCards.snapshotInfo.exists(_.revision == v.revision)) &&
          hasModel(stage(two.stage), model.toAbsolutePath.normalize.toString)
      }}
      checks += "remaining browser uses shared catalog after first disposal"
      require(starts.get() == 0 && services.supervisor.pid.isEmpty && services.queue.isIdle, "engine/queue activity")
      require(sha(model) == originalHash, "fixture model changed")
      checks += "no engine launch and fixture bytes unchanged"
    catch case e: Throwable => failure = e
    finally
      if startedFx then
        cleanup("windows") { fx {
          if one != null then { one.dispose(); one.stage.close() }
          if two != null then { two.dispose(); two.stage.close() }
        }}
      cleanup("services") { if services != null then services.shutdown() }
      cleanup("process preview loader") {
        PreviewImageLoader.shutdownProcess()
        val probe = PreviewImageLoader.acquire()
        try
          val rejected = try
            val unsubscribe = probe.loader.addCapacityListener(() => ())
            unsubscribe()
            false
          catch case _: IllegalArgumentException => true
          require(rejected, "process loader remained usable after shutdown")
        finally probe.close()
      }
      if services != null then cleanup("catalog closed") {
        require(services.modelCards.acquireSnapshot().isLeft, "shared catalog remained open")
      }
      if startedFx then cleanup("JavaFX") { Platform.exit() }
      val sourceFiles = Seq("ui/src/main/scala/sdfx/ui/MainWindow.scala", "ui/src/main/scala/sdfx/ui/view/ModelBrowser.scala",
        "engine/src/main/scala/sdfx/engine/AppServices.scala", "ui/src/test/scala/sdfx/ui/RealModelBrowserOwnerCheck.scala")
      val hashes = ujson.Obj.from(sourceFiles.map(path => path -> ujson.Str(sha(Paths.get(path)))))
      val report = ujson.Obj("status" -> (if failure == null && cleanupErrors.isEmpty then "passed" else "failed"),
        "checks" -> ujson.Arr.from(checks), "screenshots" -> ujson.Arr.from(screenshots),
        "sourceHashes" -> hashes, "fixtureSha256" -> originalHash, "engineStarts" -> starts.get(),
        "failure" -> Option(failure).map(_.toString).getOrElse(""), "cleanupErrors" -> ujson.Arr.from(cleanupErrors))
      Files.writeString(run.resolve("browser-owner-report.json"), ujson.write(report, indent = 2))
    if failure != null || cleanupErrors.nonEmpty then
      throw new IllegalStateException(s"Owner check failed: ${Option(failure).map(_.toString).getOrElse("")} ${cleanupErrors.mkString("; ")}", failure)
