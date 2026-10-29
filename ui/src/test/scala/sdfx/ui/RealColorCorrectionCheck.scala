package sdfx.ui

import javafx.application.Platform
import javafx.scene.{Scene, Parent, Node}
import javafx.scene.control.{Button, CheckBox, ComboBox}
import javafx.stage.Stage
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.view.{ExtrasTab, UiContext}
import sdfx.ui.vm.CatalogVM

/** Actual native button -> plugin -> Python engine -> persisted results. No diffusion. */
object RealColorCorrectionCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def descendants(node: Node): Seq[Node] = node match
    case parent: Parent => node +: parent.getChildrenUnmodifiable.asScala.toSeq.flatMap(descendants)
    case _ => Seq(node)

  def main(args: Array[String]): Unit =
    require(args.length == 3 || args.length == 4, "home python-executable input-image [reference-image]")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    Files.createDirectories(home.pythonDir)
    val settings = Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString))
    val services = new AppServices(home, settings, dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null
    var context: UiContext = null
    try
      services.supervisor.start()
      val (tab, button) = fx {
        stage = new Stage()
        context = new UiContext(services, new CatalogVM, stage)
        val tab = new ExtrasTab(context, "", 2)
        stage.setScene(new Scene(tab.node, 800, 900))
        stage.show()
        tab.image.set(Paths.get(args(2)).toAbsolutePath.toString)
        if args.length == 4 then tab.correctionReference.set(Paths.get(args(3)).toAbsolutePath.toString)
        val nodes = descendants(tab.node)
        nodes.collectFirst { case check: CheckBox if check.getText == "Also output other methods" => check }.get.setSelected(true)
        val button = nodes.collectFirst { case b: Button if b.getText == "Apply color correction" => b }.get
        button.fire()
        require(button.isDisabled, "submission disables button")
        (tab, button)
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(120)
      while fx(button.isDisabled) && System.nanoTime() < deadline do Thread.sleep(50)
      fx {
        require(!button.isDisabled, "processing timed out")
        val messages = descendants(context.toaster.layer).collect { case label: javafx.scene.control.Label => label.getText }.mkString(" | ")
        require(tab.comparisonPair.nonEmpty, s"native result missing: $messages")
        val selector = descendants(tab.node).collectFirst {
          case c: ComboBox[?] if c.getPromptText == "Select correction output" => c
        }.get
        require(selector.getItems.size == 3, "all method outputs available")
        val choices = selector.asInstanceOf[ComboBox[String]]
        require(choices.getItems.asScala.toSet == Set("A1111", "AdaIN", "Wavelet"), "method labels come from saved metadata")
        val selectedPaths = choices.getItems.asScala.toSeq.map { method =>
          choices.setValue(method)
          val output = tab.comparisonPair.get._2
          val metadata = sdfx.core.png.PngMetadata.readTexts(output)
          val parsed = sdfx.core.infotext.Infotext.parse(metadata(sdfx.core.png.PngMetadata.ParametersKey))
          require(parsed.fields("Color Correction") == method, "selection shows the matching saved method")
          output
        }
        require(selectedPaths.distinct.size == 3, "comparison outputs must be separate files")
        tab.comparisonPair.foreach { (_, output) => require(Files.isRegularFile(output)) }
        tab.image.set("")
        require(tab.comparisonPair.isEmpty && selector.getItems.isEmpty, "input reset clears results")
      }
      fx {
        tab.image.set(Paths.get(args(2)).toAbsolutePath.toString)
        tab.correctionReference.set(home.root.resolve("missing-reference.png").toString)
        button.fire()
      }
      def waitForButton(): Unit =
        val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(120)
        while fx(button.isDisabled) && System.nanoTime() < end do Thread.sleep(50)
        require(fx(!button.isDisabled), "button must recover")
      waitForButton()
      fx {
        require(tab.comparisonPair.isEmpty, "failure must not publish a result")
        val errors = descendants(context.toaster.layer).collect { case label: javafx.scene.control.Label => label.getText }
        require(errors.exists(_.contains("Color correction failed:")), "failure must be visible")
        tab.correctionReference.set(if args.length == 4 then Paths.get(args(3)).toAbsolutePath.toString else "")
        button.fire()
      }
      waitForButton()
      fx { require(tab.comparisonPair.nonEmpty, "retry must persist a result") }
      fx {
        button.fire()
        val original = tab.image.get
        tab.image.set("")
        tab.image.set(original)
      }
      waitForButton()
      fx { require(tab.comparisonPair.isEmpty, "old completion must not publish after input changes away and back") }
      println("PASS native color correction: persisted result, three methods, reset")
    finally
      fx { if stage != null then stage.close() }
      services.shutdown()
      Platform.exit()
