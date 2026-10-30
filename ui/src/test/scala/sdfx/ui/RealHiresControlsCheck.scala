package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{Button, ComboBox, ListView, RadioButton, TitledPane}
import javafx.embed.swing.SwingFXUtils
import javax.imageio.ImageIO
import javafx.stage.Stage
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.model.PromptStyle
import sdfx.engine.AppServices
import sdfx.plugins.HiresFixTweaksPlugin
import sdfx.ui.vm.*
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

/** Actual JavaFX Hires style controls; no backend or model is launched. */
object RealHiresControlsCheck:
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
          val ctx = new UiContext(services, new CatalogVM, stage)
          ctx.styles.addAll(PromptStyle("A", "A({prompt})", "a({prompt})"), PromptStyle("B", "B({prompt})", "b({prompt})"))
          val vm = new GenerationVM(Mode.Txt2Img)
          val form = new PluginValuesVM(new HiresFixTweaksPlugin)
          val root = HiresTweaksControls.render(ctx, vm, form).asInstanceOf[Parent]
          stage.setScene(new Scene(root, 720, 900))
          stage.show()
          def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
            case t: TitledPane => nodes(t.getContent)
            case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
            case _ => Nil)
          val all = nodes(root)
          all.collect { case t: TitledPane => t }.foreach(_.setExpanded(true))
          root.applyCss(); root.layout()
          val list = all.find(_.getId == "hires-tweaks-styles").get.asInstanceOf[ListView[String]]
          val applyButton = all.collectFirst { case b: Button if b.getText == "Apply styles to Hires prompts" => b }.get
          list.getSelectionModel.select("B")
          list.getSelectionModel.select("A")
          require(form.props("styles").get == "[\"B\",\"A\"]", "style selection reordered to catalog order")
          ctx.styles.add(PromptStyle("C", "C({prompt})", "c({prompt})"))
          require(form.props("styles").get == "[\"B\",\"A\"]", "catalog refresh changed selection order")
          list.getSelectionModel.clearSelection(1)
          list.getSelectionModel.select("B")
          require(form.props("styles").get == "[\"A\",\"B\"]", "reselected style must append")
          for enabled <- Seq(false, true) do
            form.props("styles_enabled").set(enabled.toString)
            form.props("styles").set("[\"B\",\"A\"]")
            vm.hiresPrompt.set("cat"); vm.hiresNegative.set("bad")
            applyButton.fire()
            require(vm.hiresPrompt.get == "A(B(cat))" && vm.hiresNegative.get == "a(b(bad))", "Apply order incorrect")
            require(form.props("styles").get == "[]", "Apply did not clear styles")
            require(form.props("styles_enabled").get == enabled.toString, "Apply changed independent-style enable state")
          def node(id: String): Node = all.find(_.getId == id).get
          val radio = node("hires-tweaks-prompt_mode-radio")
          val dropdown = node("hires-tweaks-prompt_mode-dropdown").asInstanceOf[ComboBox[String]]
          require(radio.isVisible && !dropdown.isVisible)
          nodes(radio).collectFirst { case b: RadioButton if b.getText == "Prepend" => b }.get.fire()
          require(form.props("prompt_mode").get == "Prepend")
          ctx.hiresPreferences.set(ctx.hiresPreferences.get.copy(promptModeRadio = false, showBatchSeed = false, showStyles = false))
          require(!radio.isVisible && dropdown.isVisible && dropdown.getValue == "Prepend")
          require(!node("hires-tweaks-batch_seed").isManaged && !node("hires-tweaks-hr_styles").isVisible)
          dropdown.setValue("Append")
          ctx.hiresPreferences.set(ctx.hiresPreferences.get.copy(promptModeRadio = true, showBatchSeed = true, showStyles = true))
          require(nodes(radio).collectFirst { case b: RadioButton if b.getText == "Append" => b.isSelected }.contains(true))
          ctx.hiresSeedHistory = new HiresSeedHistory(Some(123), Some(456))
          all.collectFirst { case b: Button if b.getText == "Reuse last Hires seed" => b }.get.fire()
          all.collectFirst { case b: Button if b.getText == "Reuse last variation seed" => b }.get.fire()
          require(form.props("seed").get == "123" && form.props("subseed").get == "456")
          require(form.props("seed_enabled").get == "true" && form.props("extras").get == "true")
          form.props("styles").set("[\"B\",\"A\"]")
          require(node("hires-tweaks-style-order").asInstanceOf[javafx.scene.control.Label].getText == "Apply order: B → A")
          root.applyCss(); root.layout()
          ImageIO.write(SwingFXUtils.fromFXImage(node("hires-tweaks-hr_styles").snapshot(null, null), null), "png",
            output.resolve("styles.png").toFile)
          val entries = Seq(sdfx.core.protocol.ModelEntry("a/model", "D:/a/model.safetensors", "sd15"),
            sdfx.core.protocol.ModelEntry("b/model", "D:/b/model.safetensors", "sdxl"))
          ctx.catalog.snapshot.set(sdfx.engine.CatalogSnapshot(checkpoints = entries))
          val model = new ModelVM
          model.checkpoint.set(entries.head.name)
          ctx.checkpointName.set(entries.head.name)
          var selected = ""
          val top = new TopBar(ctx, model, n => { selected = n; model.checkpoint.set(n); ctx.checkpointName.set(n) },
            _ => (), () => (), () => (), () => ())
          val hires = ExtraSections.hires(ctx, vm, true, Some(form))
          val combined = new javafx.scene.layout.VBox(top.node, hires)
          stage.setScene(new Scene(combined, 1100, 900))
          combined.applyCss(); combined.layout()
          val cfg = nodes(hires).find(_.getId == "hires-tweaks-cfg").get
          val slider = nodes(cfg).collectFirst { case s: javafx.scene.control.Slider => s }.get
          slider.setValue(9.5)
          require(vm.hiresCfg.get == 9.5, "native Hires CFG did not update model")
          ctx.hiresPreferences.set(ctx.hiresPreferences.get.copy(showCfg = false))
          require(!cfg.isVisible && !cfg.isManaged && vm.hiresCfg.get == 9.5)
          val picker = nodes(top.node).collectFirst { case b: Button if b.getStyleClass.contains("picker") => b }.get
          for mode <- CheckpointLabels.modes do
            ctx.hiresPreferences.set(ctx.hiresPreferences.get.copy(checkpointLabels = mode))
            val current = entries.find(_.name == model.checkpoint.get).get
            val label = nodes(hires).find(_.getId == "hires-checkpoint-label").get.asInstanceOf[javafx.scene.control.Label]
            require(label.getText == CheckpointLabels.label(current, entries, mode, true))
            picker.fire()
            val popup = javafx.stage.Window.getWindows.asScala.collectFirst {
              case p: javafx.stage.Popup if p.isShowing => p }.get
            val choices = popup.getContent.asScala.toSeq.flatMap(nodes).collectFirst {
              case l: ListView[?] => l.asInstanceOf[ListView[sdfx.core.protocol.ModelEntry]] }.get
            require(choices.getItems.asScala.map(_.name).toSeq == entries.map(_.name))
            choices.getSelectionModel.select(1)
            choices.fireEvent(new javafx.scene.input.KeyEvent(javafx.scene.input.KeyEvent.KEY_PRESSED,
              "", "", javafx.scene.input.KeyCode.ENTER, false, false, false, false))
            require(selected == "b/model" && model.checkpoint.get == "b/model", "display label changed selected identity")
          val prefs = ctx.hiresPreferences.get.copy(outputDir = "hires-saved", saveTemplate = false, restoreTemplate = false)
          val settingsFile = output.resolve("settings-roundtrip.json")
          Settings.save(settingsFile, services.settings.copy(hiresFixTweaks = prefs))
          require(Settings.load(settingsFile).hiresFixTweaks == prefs, "Hires preferences did not persist")
          Files.writeString(output.resolve("report.json"), "{\"status\":\"passed\",\"selection_order\":true,\"apply\":true,\"display_preferences\":true,\"seed_reuse\":true,\"cfg_control\":true,\"checkpoint_picker_identity\":true,\"settings_roundtrip\":true}")
          println("HIRES CONTROLS VERIFIED " + output)
        finally stage.close())
    try
      Platform.runLater(task)
      task.get(60, TimeUnit.SECONDS)
    finally
      services.shutdown()
      Platform.exit()
