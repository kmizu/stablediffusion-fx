package sdfx.ui.view

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{ComboBox, ScrollPane, TextArea}
import javafx.stage.Stage
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.plugins.api.{ChoiceSource, Values}
import sdfx.engine.AppServices
import sdfx.plugins.Img2ImgHiresFixPlugin
import sdfx.ui.vm.*
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

/** Real generic JavaFX form binding and settings disk roundtrip; no backend/model launched. */
object RealImg2ImgHiresControlsCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 1, "output directory")
    val output = Paths.get(args(0)).toAbsolutePath
    Files.createDirectories(output)
    val services = new AppServices(AppHome(Files.createTempDirectory(output, "home-")), Settings(),
      dryRun = true, builtins = Nil, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    val task = new FutureTask[Unit](new Callable[Unit]:
      def call(): Unit =
        val stage = new Stage()
        try
          val catalog = new CatalogVM
          catalog.choices(ChoiceSource.Upscalers).setAll("Lanczos", "Nearest", "R-ESRGAN 4x+")
          catalog.choices(ChoiceSource.Samplers).setAll("Euler", "DPM++ 2M")
          catalog.choices(ChoiceSource.Schedulers).setAll("Automatic", "Karras")
          val ctx = new UiContext(services, catalog, stage)
          val plugin = new Img2ImgHiresFixPlugin
          val form = new PluginValuesVM(plugin)
          val root = PluginForm.render(ctx, form).asInstanceOf[Parent]
          val scroll = new ScrollPane(root)
          scroll.setFitToWidth(true)
          stage.setScene(new Scene(scroll, 800, 1000))
          stage.show()
          root.applyCss(); root.layout()
          def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
            case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
            case _ => Nil)
          val combos = nodes(root).collect { case c: ComboBox[?] => c.asInstanceOf[ComboBox[String]] }
          require(combos.size == 3)
          combos.find(_.getItems.contains("Lanczos")).get.setValue("Lanczos")
          combos.find(_.getItems.contains("Euler")).get.setValue("Euler")
          combos.find(_.getItems.contains("Karras")).get.setValue("Karras")
          val texts = nodes(root).collect { case t: TextArea => t }
          require(texts.size == 2)
          texts.head.setText("a blue-haired woman")
          texts.last.setText("blurry")
          require(form.values.str("upscaler") == "Lanczos" && form.values.str("sampler") == "Euler")
          require(form.values.str("prompt") == "a blue-haired woman" && form.values.str("negative_prompt") == "blurry")
          form.load(Values(Map("enabled" -> "true", "width" -> "640", "height" -> "768", "steps" -> "12", "cfg" -> "3", "denoise" -> "0.4")))
          val state = UiState(pluginValues = Map("txt2img" -> Map(plugin.id -> form.values.map)))
          val path = output.resolve("ui-state.json")
          Files.writeString(path, UiState.toJson(state))
          val restored = UiState.fromJson(Files.readString(path))
          require(restored.pluginValues("txt2img")(plugin.id) == form.values.map)
          val fields = plugin.process(sdfx.core.model.GenerationParams(), form.values).extra
          form.reset()
          form.load(Img2ImgHiresFixPlugin.replay(fields).get)
          require(combos.find(_.getItems.contains("Karras")).get.getValue == "Karras")
          require(texts.head.getText == "a blue-haired woman")
          root.applyCss(); root.layout()
          ImageIO.write(SwingFXUtils.fromFXImage(scroll.snapshot(null, null), null), "png", output.resolve("controls.png").toFile)
          println("img2img Hires native controls and disk restore passed")
        finally stage.close()
    )
    try
      Platform.runLater(task)
      task.get(60, TimeUnit.SECONDS)
    finally
      Platform.exit()
      services.shutdown()
