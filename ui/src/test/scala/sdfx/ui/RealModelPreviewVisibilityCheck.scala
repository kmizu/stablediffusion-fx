package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javafx.animation.AnimationTimer
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Group, Node, Parent, Scene}
import javafx.scene.image.ImageView
import javafx.scene.shape.Rectangle
import javafx.stage.Stage
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*
import sdfx.core.modelcards.*
import sdfx.engine.CardPreviewSession
import sdfx.ui.view.{ModelPreviewCard, PreviewImageLoader, PreviewLoaderLease}

/** ROOT-only native_preview_visibility regression helper. Arg: isolated output-directory.
  * No engine, AppServices, settings, model loading, clipboard or existing user-state operations.
  */
object RealModelPreviewVisibilityCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(15, TimeUnit.SECONDS)

  private def await(label: String, seconds: Int = 10)(condition: => Boolean): Unit =
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds.toLong)
    var met = condition
    while !met && System.nanoTime() < deadline do
      Thread.sleep(20) // bounded observation only; product must subscribe to invalidations, not poll
      met = condition
    require(met, s"Timed out observing $label")

  private def pulses(): Unit =
    val done = new CountDownLatch(1)
    val timer = fx {
      val value = new AnimationTimer:
        private var count = 0
        def handle(now: Long): Unit =
          count += 1
          if count == 3 then
            stop()
            done.countDown()
      value.start()
      value
    }
    try require(done.await(10, TimeUnit.SECONDS), "Three native pulses did not complete")
    finally fx { timer.stop() }

  private def descendants(node: Node): Vector[Node] = node match
    case parent: Parent => Vector(node) ++ parent.getChildrenUnmodifiable.asScala.toVector.flatMap(descendants)
    case _ => Vector(node)

  private def digest(path: Path): String =
    MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 0xff}%02x").mkString

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val realOutput = output.toRealPath()
    val artifacts = Files.createTempDirectory(realOutput, "native-preview-visibility-").toRealPath()
    require(artifacts.startsWith(realOutput), "Artifact directory escaped isolated output")
    def artifact(name: String): Path =
      val path = artifacts.resolve(name).toAbsolutePath.normalize()
      require(path.startsWith(artifacts) && path.getParent.toRealPath().startsWith(realOutput), "Artifact path escaped output")
      path

    val phases = ArrayBuffer.empty[ujson.Value]
    val screenshots = ArrayBuffer.empty[String]
    val cleanupErrors = ArrayBuffer.empty[String]
    val sourceHashes = ArrayBuffer.empty[(Path, String)]
    var failure: Throwable = null
    var started = false
    var stage: Stage = null
    var scene: Scene = null
    var parent: Group = null
    var card: ModelPreviewCard = null
    var lease: PreviewLoaderLease = null
    var revision = 0L
    var disposed = false
    var actualScale = (0.0, 0.0)
    var finalOwner: ujson.Value = ujson.Null

    def stateJson(): ujson.Value = fx {
      val metrics = PreviewImageLoader.metrics
      val image = if card == null then None else descendants(card.node).collectFirst { case view: ImageView => view }
      val clip = Option(parent).flatMap(p => Option(p.getClip)).map(_.getBoundsInParent)
      val clipBounds = clip.map(b => ujson.Arr(b.getMinX, b.getMinY, b.getWidth, b.getHeight)).getOrElse(ujson.Null)
      val rootTransform = Option(card).map(_.node.getLocalToSceneTransform)
        .map(t => ujson.Arr(t.getMxx, t.getMxy, t.getTx, t.getMyx, t.getMyy, t.getTy)).getOrElse(ujson.Null)
      ujson.Obj("visibleConsumers" -> metrics.visibleConsumers, "activeWorkers" -> metrics.activeWorkers,
        "ownedImages" -> metrics.ownedImages, "completed" -> metrics.completed,
        "imageAttached" -> image.exists(_.getImage != null),
        "sceneWidth" -> Option(scene).map(_.getWidth).getOrElse(0.0),
        "sceneHeight" -> Option(scene).map(_.getHeight).getOrElse(0.0),
        "clipBoundsInParent" -> clipBounds, "rootTransform" -> rootTransform,
        "accessibleText" -> Option(card).map(_.node.getAccessibleText).getOrElse(""))
    }

    def screenshot(name: String): Unit =
      if stage != null && fx { stage.isShowing } then
        val path = artifact(name + ".png")
        fx { require(ImageIO.write(SwingFXUtils.fromFXImage(scene.snapshot(null), null), "png", path.toFile)) }
        screenshots += path.toString

    def phase(name: String)(body: => Unit): Unit =
      val phaseStarted = System.nanoTime()
      var problem: Option[Throwable] = None
      try body
      catch case error: Throwable =>
        problem = Some(error)
        if failure == null then failure = error else failure.addSuppressed(error)
      var observed: ujson.Value = ujson.Null
      try observed = stateJson()
      catch case error: Throwable =>
        if problem.isEmpty then problem = Some(error)
        if failure == null then failure = error else if failure ne error then failure.addSuppressed(error)
      try screenshot(f"${phases.size + 1}%02d-$name")
      catch case error: Throwable =>
        if problem.isEmpty then problem = Some(error)
        if failure == null then failure = error else if failure ne error then failure.addSuppressed(error)
      phases += ujson.Obj("name" -> name, "status" -> (if problem.isEmpty then "passed" else "failed"),
        "error" -> problem.map(_.toString).getOrElse(""), "observed" -> observed,
        "elapsedMillis" -> ((System.nanoTime() - phaseStarted) / 1000000L))
      println(s"NATIVE PREVIEW VISIBILITY ${if problem.isEmpty then "PASS" else "FAIL"}: $name")

    def transform(): Vector[Double] = fx {
      val t = card.node.getLocalToSceneTransform
      Vector(t.getMxx, t.getMxy, t.getMxz, t.getTx, t.getMyx, t.getMyy, t.getMyz, t.getTy,
        t.getMzx, t.getMzy, t.getMzz, t.getTz)
    }
    def stationary(before: Vector[Double]): Unit =
      val after = transform()
      require(before.zip(after).forall((a, b) => math.abs(a - b) <= 0.001),
        s"Fixture invalid: card root transform moved: $before -> $after")

    def extent(width: Double, height: Double): Unit =
      fx {
        stage.setWidth(width + stage.getWidth - scene.getWidth)
        stage.setHeight(height + stage.getHeight - scene.getHeight)
      }
      await("requested native scene extent") { fx {
        math.abs(scene.getWidth - width) <= 2 && math.abs(scene.getHeight - height) <= 2
      }}

    def attached(expected: Boolean): Boolean = fx {
      val image = descendants(card.node).collectFirst { case view: ImageView => view }.get
      val consumers = PreviewImageLoader.metrics.visibleConsumers
      (image.getImage != null) == expected && consumers == (if expected then 1 else 0)
    }

    try
      require(Runtime.version().feature() == 21, s"Java21 required, observed ${System.getProperty("java.version")}")
      val markedPath = artifact("NSFW-preview.png")
      val safePath = artifact("safe-preview.png")
      for path <- Vector(markedPath, safePath) do
        val image = new BufferedImage(32, 16, BufferedImage.TYPE_INT_RGB)
        for x <- 0 until 32; y <- 0 until 16 do image.setRGB(x, y, if x < 16 then 0xff236b8e else 0xffdc9944)
        require(ImageIO.write(image, "png", path.toFile), "PNG fixture writer unavailable")
        sourceHashes += ((path, digest(path)))
      val resolver = new ModelPreviewResolver(LocalPreviewFiles.scoped())
      val marked = resolver.validateExplicit(markedPath).fold(sys.error, identity)
      val safe = resolver.validateExplicit(safePath).fold(sys.error, identity)
      require(CardPreviewPolicy.matches(Some(marked.source)) && !CardPreviewPolicy.matches(Some(safe.source)),
        "Output directory name must not itself contain nsfw; otherwise a safe-source fixture is impossible")
      val id = CardId("fixture", artifact("model-identity-only.safetensors").toString)
      val markedRecord = ModelCardRecord(id, "Visibility fixture", None, Some(marked), None)
      val safeRecord = markedRecord.copy(preview = Some(safe))
      val missingRecord = markedRecord.copy(preview = None)
      val ready = new CountDownLatch(1)
      Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
      started = true
      require(ready.await(15, TimeUnit.SECONDS), "JavaFX startup timed out")
      fx {
        lease = PreviewImageLoader.acquire()
        card = new ModelPreviewCard(new CardPreviewSession("Blur"), lease.loader)
        card.node.setManaged(false)
        card.node.resize(480, 180)
        card.node.relocate(340, 120)
        parent = new Group(card.node)
        scene = new Scene(new Group(parent), 1000, 560)
        stage = new Stage()
        stage.setTitle("Native preview visibility regression fixture")
        stage.setScene(scene)
        stage.show()
        actualScale = ((stage.getOutputScaleX, stage.getOutputScaleY))
      }

      def prepare(clip: Rectangle, width: Double = 1000, height: Double = 560,
          record: ModelCardRecord = markedRecord, expectImage: Boolean = true): Unit =
        extent(width, height)
        fx {
          card.clear()
          parent.setClip(clip)
          card.node.resize(480, 180)
          card.node.relocate(340, 120)
          revision += 1
          card.bind(record, revision)
        }
        pulses()
        await("prepared visible/image ownership state", 15) { attached(expectImage) }
      def fullClip(): Rectangle = fx { new Rectangle(0, 0, 1000, 560) }
      def hiddenAfterMutation(before: Vector[Double]): Unit =
        pulses()
        stationary(before)
        require(attached(false), "Fully clipped/out-of-scene card retained image or visible-consumer ownership")
      def visibleAfterMutation(before: Vector[Double]): Unit =
        pulses()
        stationary(before)
        await("exposed stationary card acquires preview", 10) { attached(true) }

      phase("clip_width_shrink_hides_stationary_card") {
        val clip = fullClip(); prepare(clip)
        val before = transform()
        fx { clip.setWidth(100) }
        hiddenAfterMutation(before)
      }
      phase("clip_width_expand_exposes_stationary_card") {
        val clip = fx { new Rectangle(0, 0, 100, 560) }; prepare(clip, expectImage = false)
        val before = transform()
        fx { clip.setWidth(1000) }
        visibleAfterMutation(before)
      }
      phase("clip_height_shrink_hides_stationary_card") {
        val clip = fullClip(); prepare(clip)
        val before = transform()
        fx { clip.setHeight(40) }
        hiddenAfterMutation(before)
      }
      phase("clip_height_expand_exposes_stationary_card") {
        val clip = fx { new Rectangle(0, 0, 1000, 40) }; prepare(clip, expectImage = false)
        val before = transform()
        fx { clip.setHeight(560) }
        visibleAfterMutation(before)
      }
      phase("clip_translation_hides_stationary_card") {
        val clip = fullClip(); prepare(clip)
        val before = transform()
        fx { clip.setTranslateX(1200) }
        hiddenAfterMutation(before)
      }
      phase("clip_translation_exposes_stationary_card") {
        val clip = fullClip(); fx { clip.setTranslateY(800) }; prepare(clip, expectImage = false)
        val before = transform()
        fx { clip.setTranslateY(0) }
        visibleAfterMutation(before)
      }
      phase("clip_replacement_hides_stationary_card") {
        prepare(fullClip())
        val before = transform()
        fx { parent.setClip(new Rectangle(0, 0, 80, 80)) }
        hiddenAfterMutation(before)
      }
      phase("clip_replacement_exposes_stationary_card") {
        prepare(fx { new Rectangle(0, 0, 80, 80) }, expectImage = false)
        val before = transform()
        fx { parent.setClip(new Rectangle(0, 0, 1000, 560)) }
        visibleAfterMutation(before)
      }
      phase("scene_width_shrink_hides_stationary_card") {
        prepare(fullClip())
        val before = transform()
        extent(200, 560)
        hiddenAfterMutation(before)
      }
      phase("scene_width_expand_exposes_stationary_card") {
        prepare(fullClip(), width = 200, expectImage = false)
        val before = transform()
        extent(1000, 560)
        visibleAfterMutation(before)
      }
      phase("scene_height_shrink_hides_stationary_card") {
        prepare(fullClip())
        val before = transform()
        extent(1000, 70)
        hiddenAfterMutation(before)
      }
      phase("scene_height_expand_exposes_stationary_card") {
        prepare(fullClip(), height = 70, expectImage = false)
        val before = transform()
        extent(1000, 560)
        visibleAfterMutation(before)
      }
      phase("accessible_matching_source") {
        prepare(fullClip())
        val text = fx { card.node.getAccessibleText.toLowerCase(java.util.Locale.ROOT) }
        require(text.contains("nsfw") && text.contains("match") && !text.contains("does not match") && text.contains("blur"), text)
      }
      phase("accessible_safe_source") {
        prepare(fullClip(), record = safeRecord)
        val text = fx { card.node.getAccessibleText.toLowerCase(java.util.Locale.ROOT) }
        require(text.contains("does not match") || text.contains("no match") || text.contains("not matched"), text)
      }
      phase("accessible_missing_preview") {
        prepare(fullClip(), record = missingRecord, expectImage = false)
        val text = fx { card.node.getAccessibleText.toLowerCase(java.util.Locale.ROOT) }
        require(text.contains("no preview") || text.contains("missing preview"), text)
      }
      phase("dispose_then_clip_scene_and_visibility_mutations_do_not_reacquire") {
        val oldClip = fullClip(); prepare(oldClip)
        fx { card.dispose(); disposed = true }
        await("disposal releases actual consumer") { attached(false) }
        await("pre-disposal decode work actually exits") {
          val metrics = PreviewImageLoader.metrics
          metrics.activeWorkers == 0 && metrics.queued == 0
        }
        val completed = PreviewImageLoader.metrics.completed
        fx {
          oldClip.setWidth(0)
          oldClip.setTranslateX(1200)
          parent.setClip(new Rectangle(0, 0, 1000, 560))
          parent.setVisible(false)
          parent.setVisible(true)
        }
        extent(900, 500)
        pulses()
        require(attached(false), "Disposed card reacquired pixels")
        require(PreviewImageLoader.metrics.completed == completed, "Disposed listener activity started new decode work")
      }
      phase("sources_unchanged") {
        sourceHashes.foreach((path, hash) => require(digest(path) == hash, s"Fixture source modified: $path"))
      }
    catch case error: Throwable =>
      if failure == null then failure = error else if failure ne error then failure.addSuppressed(error)
    finally
      def cleanup(label: String)(body: => Unit): Unit =
        try body
        catch case error: Throwable =>
          cleanupErrors += s"$label: $error"
          throw error
      try NativeAcceptanceSafety.cleanup(failure)(
        () => cleanup("card/stage") { if started then fx {
          if card != null && !disposed then card.dispose()
          if stage != null then stage.close()
        } },
        () => cleanup("lease") { if lease != null then lease.close() },
        () => cleanup("process owner") { if lease != null then PreviewImageLoader.shutdownProcess() },
        () => cleanup("owner reservations") { if lease != null then
          await("all owner reservations released") {
            val metrics = PreviewImageLoader.metrics
            metrics.activeWorkers == 0 && metrics.admittedBuffers == 0 && metrics.ownedImages == 0 &&
              metrics.visibleConsumers == 0 && metrics.queued == 0 && metrics.mailboxItems == 0
          }
          val metrics = PreviewImageLoader.metrics
          finalOwner = ujson.Obj("activeWorkers" -> metrics.activeWorkers, "admittedBuffers" -> metrics.admittedBuffers,
            "sourceBytes" -> metrics.sourceBytes, "ownedImages" -> metrics.ownedImages,
            "visibleConsumers" -> metrics.visibleConsumers, "queued" -> metrics.queued, "mailboxItems" -> metrics.mailboxItems)
        },
        () => cleanup("JavaFX") { if started then Platform.exit() })
      catch case error: Throwable => if failure == null then failure = error

      val report = ujson.Obj(
        "scope" -> "native_preview_visibility", "status" -> (if failure == null && cleanupErrors.isEmpty then "passed" else "failed"),
        "javaVersion" -> System.getProperty("java.version"), "javaHome" -> System.getProperty("java.home"),
        "maxHeap" -> Runtime.getRuntime.maxMemory(), "actualOutputScale" -> ujson.Arr(actualScale._1, actualScale._2),
        "finalOwner" -> finalOwner,
        "output" -> realOutput.toString, "artifacts" -> artifacts.toString,
        "checks" -> ujson.Arr.from(phases), "screenshots" -> ujson.Arr.from(screenshots),
        "sources" -> ujson.Arr.from(sourceHashes.map((path, hash) => ujson.Obj("path" -> path.toString, "sha256" -> hash))),
        "cleanupErrors" -> ujson.Arr.from(cleanupErrors), "failure" -> Option(failure).map(_.toString).getOrElse(""),
        "limits" -> ujson.Arr("Not full Task7 acceptance", "Disposal checks observable inactivity; listener removal also requires source review",
          "No inference, memory stress, themes, scale matrix, ListView virtualization or native key-routing proof"))
      val reportPath = realOutput.resolve("visibility-report.json")
      require(reportPath.normalize().startsWith(realOutput), "Report path escaped isolated output")
      Files.writeString(reportPath, ujson.write(report, indent = 2))
    if failure != null then throw failure
