package sdfx.ui

import javafx.application.Platform
import javafx.stage.{Stage, Window}
import javafx.scene.control.{Button, ButtonType, ListView, TextField, TitledPane}
import javafx.scene.image.ImageView
import javafx.embed.swing.SwingFXUtils
import scala.jdk.CollectionConverters.*
import sdfx.plugins.prompt.StylePileCatalog
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit, LinkedBlockingQueue}
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.OutputImage
import sdfx.core.png.PngMetadata
import sdfx.core.infotext.Infotext
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.{StandardPlugins, StylePilePlugin}
import sdfx.ui.vm.{UiState, Mode}

/** Actual native Generate path and persisted PNG metadata, with a real model. */
object RealStylePileCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length >= 3, "home python checkpoint [custom]")
    val custom = args.drop(3).contains("custom")
    val full = args.drop(3).contains("full")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val checkpoint = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val customRoot = home.root.resolve("stylepile")
    if custom then
      Files.createDirectories(customRoot.resolve("Artists"))
      val fixture = new java.awt.image.BufferedImage(96, 96, java.awt.image.BufferedImage.TYPE_INT_RGB)
      val graphics = fixture.createGraphics()
      try
        graphics.setColor(java.awt.Color.MAGENTA)
        graphics.fillRect(0, 0, 96, 96)
      finally graphics.dispose()
      ImageIO.write(fixture, "png", customRoot.resolve("Artists/Synthetic Custom Artist.png").toFile)
    val services = new AppServices(home, Settings(
      pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString))), dryRun = false,
      builtins = StandardPlugins.withOwnAxes.map(plugin => if custom && plugin.id == "stylepile" then new StylePilePlugin(Some(customRoot)) else plugin), persistSettings = false, installShutdownHook = false)
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val cancelled = new LinkedBlockingQueue[Seq[OutputImage]]()
    val saved = new java.util.concurrent.atomic.AtomicInteger()
    services.queue.addListener {
      case event: JobEvent.JobFinished => finished.offer(Right(event.output.images)); ()
      case event: JobEvent.ImageSaved if !event.grid => saved.incrementAndGet(); ()
      case event: JobEvent.JobCancelled => cancelled.offer(event.partial.images); ()
      case event: JobEvent.JobFailed => finished.offer(Left(event.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    try
      services.start()
      services.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        val size = if checkpoint.getFileName.toString.toLowerCase.contains("sdxl") then 768 else 512
        window = new MainWindow(new Stage(), services, UiState(params = Map("txt2img" ->
          GenerationParams(prompt = "a ceramic teapot on a wooden table", seed = 7, steps = 8, width = size, height = size))), CliOptions(), _ => ())
        window.show()
        window.model.checkpoint.set(checkpoint.toString)
      }
      var gallery: Window = null
      val targetArtist = if custom then "Synthetic Custom Artist" else StylePileCatalog.choices("Artists").head
      fx {
        window.plugins(Mode.Txt2Img).selectedScript.set("stylepile")
        window.stage.getScene.getRoot.applyCss()
        window.stage.getScene.getRoot.lookupAll(".titled-pane").asScala.collect { case pane: TitledPane if pane.getText == "Artists and movements" => pane }.foreach(_.setExpanded(true))
        window.stage.getScene.getRoot.layout()
        val browse = window.stage.getScene.getRoot.lookupAll(".preview-choice-browse").asScala.collectFirst {
          case button: Button if button.getAccessibleText == "Browse Artist 1 examples" => button
        }.getOrElse(throw new IllegalStateException("Artist preview button missing"))
        browse.fire()
        gallery = Window.getWindows.asScala.find(w => w.isShowing && w != window.stage).get
        val search = gallery.getScene.lookup("#preview-choice-search").asInstanceOf[TextField]
        search.setText(targetArtist)
        val list = gallery.getScene.lookup("#preview-choice-list").asInstanceOf[ListView[String]]
        require(list.getItems.contains(targetArtist), "gallery search lost matching artist")
        list.getSelectionModel.select(targetArtist)
      }
      val imageDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
      def previewReady: Boolean = fx {
        val view = gallery.getScene.lookup("#preview-choice-image").asInstanceOf[ImageView]
        view.getImage != null && !view.getImage.isError
      }
      while !previewReady && System.nanoTime() < imageDeadline do Thread.sleep(50)
      require(previewReady, "native reference preview did not load")
      fx {
        val snapshot = gallery.getScene.snapshot(null)
        ImageIO.write(SwingFXUtils.fromFXImage(snapshot, null), "png", home.root.resolve("stylepile-gallery.png").toFile)
        val ok = gallery.getScene.getRoot.lookupAll(".button").asScala.collectFirst { case button: Button if button.isDefaultButton => button }.get
        ok.fire()
        require(window.plugins(Mode.Txt2Img).form("stylepile").get.props("artist1").get == targetArtist, "gallery confirmation did not update selection")
        window.plugins(Mode.Txt2Img).form("stylepile").get.props("artist1").set("Not set")
        window.plugins(Mode.Txt2Img).form("stylepile").get.props("tool_text").set("red scarf")
        window.vms(Mode.Txt2Img).prompt.set("teapot ")
        window.stage.getScene.lookup("#stylepile-action-weight").asInstanceOf[Button].fire()
        require(window.vms(Mode.Txt2Img).prompt.get == "teapot (red scarf:1.3)", "native helper did not append to prompt")
      }
      val outputs = Seq(false, true).map { enabled =>
        fx {
          window.plugins(Mode.Txt2Img).selectedScript.set(if enabled then "stylepile" else "")
          window.plugins(Mode.Txt2Img).form("stylepile").get.props("sequential").set("a ceramic teapot on a wooden table")
          window.vms(Mode.Txt2Img).prompt.set(if enabled then "[X]" else "a ceramic teapot on a wooden table")
          window.vms(Mode.Txt2Img).negativePrompt.set(if enabled then "" else "watermark,signature")
          window.statusBar.generateBtn.fire()
        }
        val result = finished.poll(120, TimeUnit.SECONDS)
        require(result != null, "native generation timed out")
        val image = result.fold(message => throw new IllegalStateException(message), identity).head
        val fields = Infotext.parse(PngMetadata.readTexts(image.path)(PngMetadata.ParametersKey)).fields
        if enabled then require(fields.contains("StylePile"), "saved StylePile settings missing")
        image.path
      }
      val images = outputs.map(path => ImageIO.read(path.toFile))
      val arrays = images.map(image => image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth))
      require(java.util.Arrays.equals(arrays.head, arrays(1)), "StylePile expansion differs from manual request")
      if full then
        def resultImages(): Seq[OutputImage] =
          val result = finished.poll(120, TimeUnit.SECONDS)
          require(result != null, "native job timed out")
          result.fold(message => throw new IllegalStateException(message), identity)
        def samePixels(paths: Seq[java.nio.file.Path]): Unit =
          val values = paths.map(path => ImageIO.read(path.toFile)).map(image => image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth))
          require(java.util.Arrays.equals(values.head, values(1)), "configured StylePile differs from manual request")
        val painted = Seq(false, true).map { manual =>
          fx {
            val form = window.plugins(Mode.Txt2Img).form("stylepile").get
            form.props("image_type").set("Painting")
            form.props("artist1").set("Pablo Picasso")
            form.props("movement1").set("Impressionism")
            form.props("sequential").set("")
            window.plugins(Mode.Txt2Img).selectedScript.set(if manual then "" else "stylepile")
            window.vms(Mode.Txt2Img).prompt.set(if manual then "(Painting:1.3) of ceramic teapot,(by Artist Pablo Picasso:1.3),(Impressionism:1.3)" else "ceramic teapot")
            window.vms(Mode.Txt2Img).negativePrompt.set(if manual then "blur,Low quality,Bad composition,Faded,(Photo:1.5),(Frame:1.3),watermark,signature" else "blur")
            window.statusBar.generateBtn.fire()
          }
          resultImages().head.path
        }
        samePixels(painted)
        fx {
          val form = window.plugins(Mode.Txt2Img).form("stylepile").get
          Seq("image_type", "artist1", "movement1").foreach(key => form.props(key).set("Not set"))
          form.props("sequential").set("teapot\nvase")
          form.props("subsequential").set("red\nblue")
          window.plugins(Mode.Txt2Img).selectedScript.set("stylepile")
          window.vms(Mode.Txt2Img).prompt.set("[Y] [X] on a wooden table")
          window.vms(Mode.Txt2Img).negativePrompt.set("")
          window.statusBar.generateBtn.fire()
        }
        val sequence = resultImages()
        require(sequence.size == 4, s"wrong sequence count: ${sequence.size}")
        val prompts = sequence.map(image => Infotext.parse(PngMetadata.readTexts(image.path)(PngMetadata.ParametersKey)).prompt)
        require(prompts == Seq("red teapot on a wooden table", "blue teapot on a wooden table", "red vase on a wooden table", "blue vase on a wooden table"), s"wrong sequence order: $prompts")
        require(sequence.map(_.seed) == Seq(7L, 8L, 7L, 8L), "wrong sequence seeds")
        val imgOutputs = Seq(false, true).map { enabled =>
          fx {
            window.select(Mode.Img2Img)
            val vm = window.vms(Mode.Img2Img)
            vm.initImage.set(outputs.last.toString)
            vm.width.set(images.head.getWidth); vm.height.set(images.head.getHeight)
            vm.steps.set(8); vm.seed.set(7); vm.denoise.set(.7)
            vm.prompt.set(if enabled then "[X]" else "a ceramic teapot on a wooden table")
            vm.negativePrompt.set(if enabled then "" else "watermark,signature")
            window.plugins(Mode.Img2Img).form("stylepile").get.props("sequential").set("a ceramic teapot on a wooden table")
            window.plugins(Mode.Img2Img).selectedScript.set(if enabled then "stylepile" else "")
            window.statusBar.generateBtn.fire()
          }
          resultImages().head.path
        }
        samePixels(imgOutputs)
        val before = saved.get
        fx {
          window.select(Mode.Txt2Img)
          val form = window.plugins(Mode.Txt2Img).form("stylepile").get
          form.props("sequential").set((1 to 20).map(i => s"teapot $i").mkString("\n"))
          form.props("subsequential").set("")
          window.vms(Mode.Txt2Img).prompt.set("[X]")
          window.vms(Mode.Txt2Img).steps.set(40)
          window.statusBar.generateBtn.fire()
        }
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(120)
        while saved.get <= before && System.nanoTime() < deadline do Thread.sleep(50)
        require(saved.get > before, "no sequence output before interrupt")
        fx { window.statusBar.generateBtn.fire() }
        val partial = cancelled.poll(30, TimeUnit.SECONDS)
        require(partial != null && partial.nonEmpty && partial.size < 20, "sequence cancel did not preserve partial images")
        val idleDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
        while !services.queue.isIdle && System.nanoTime() < idleDeadline do Thread.sleep(20)
        require(services.queue.isIdle)
        fx {
          window.plugins(Mode.Txt2Img).selectedScript.set("")
          window.vms(Mode.Txt2Img).steps.set(4)
          window.statusBar.generateBtn.fire()
        }
        require(resultImages().nonEmpty, "native recovery image missing")
        println(s"PASS StylePile configured style/manual RGB, four X/Y images and seeds, img2img RGB, interrupt partial=${partial.size}, recovery; styled=${painted.mkString(",")}")
      println(s"PASS native StylePile: saved settings, exact manually expanded RGB; ${outputs.mkString(", ")}")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown()
      Platform.exit()
