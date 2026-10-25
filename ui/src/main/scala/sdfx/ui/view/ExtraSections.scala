package sdfx.ui.view

import javafx.beans.property.SimpleStringProperty
import javafx.collections.FXCollections
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.image.ImageView
import javafx.scene.layout.VBox
import sdfx.core.plugins.api.ChoiceSource
import sdfx.ui.Fx
import sdfx.ui.vm.{ControlNetUnitVM, GenerationVM}

import java.nio.file.Paths

/** Hires fix and ControlNet sections of the generation panel. */
object ExtraSections:
  def hires(ctx: UiContext, vm: GenerationVM, expanded: Boolean): TitledPane =
    val target = W.label("", "hint")
    def updateTarget(): Unit =
      val (w, h) = vm.hiresTarget
      target.setText(s"${vm.width.get}×${vm.height.get}  →  $w×$h")
    Seq(vm.width, vm.height, vm.hiresScale, vm.hiresResizeX, vm.hiresResizeY).foreach(_.addListener((_, _, _) => updateTarget()))
    updateTarget()

    val samplers = withSame(ctx.catalog.samplers, "Use same sampler")
    val schedulers = withSame(ctx.catalog.schedulers, "Use same scheduler")
    val samplerProp = sameProp(vm.hiresSampler, "Use same sampler")
    val schedProp = sameProp(vm.hiresScheduler, "Use same scheduler")

    val hrPrompt = new TextArea()
    hrPrompt.textProperty.bindBidirectional(vm.hiresPrompt)
    hrPrompt.setPromptText("Hires prompt (empty = same as first pass)")
    hrPrompt.setPrefRowCount(2)
    hrPrompt.setWrapText(true)
    val hrNeg = new TextArea()
    hrNeg.textProperty.bindBidirectional(vm.hiresNegative)
    hrNeg.setPromptText("Hires negative prompt (empty = same)")
    hrNeg.setPrefRowCount(1)
    hrNeg.setWrapText(true)

    val advanced = W.form()
      .add("Sampler", W.combo(samplers, samplerProp), "Sampler for the second pass")
      .add("Schedule", W.combo(schedulers, schedProp), "Schedule type for the second pass")
      .add("CFG", W.doubleSlider(vm.hiresCfg, 0, 30, 0.5), "CFG scale for the second pass (0 = same as first pass)")
      .full(hrPrompt)
      .full(hrNeg)
    val adv = W.section("Advanced", advanced.grid, false)
    adv.getStyleClass.add("subsection")

    val f = W.form()
      .add("Upscaler", W.combo(ctx.catalog.hiresUpscalers, vm.hiresUpscaler), "Latent upscalers stay in latent space; ESRGAN-type models upscale the decoded image")
      .add("Upscale by", W.doubleSlider(vm.hiresScale, 1, 4, 0.05), "Scale factor of the second pass (ignored when Resize width/height are set)")
      .add("Hires steps", W.intSlider(vm.hiresSteps, 0, 150), "Second pass steps (0 = same as sampling steps)")
      .add("Denoising", W.doubleSlider(vm.hiresDenoise, 0, 1, 0.01), "How much the second pass may change the image")
      .add("Resize W", W.intSlider(vm.hiresResizeX, 0, 4096, 8), "Explicit target width (0 = use Upscale by)")
      .add("Resize H", W.intSlider(vm.hiresResizeY, 0, 4096, 8), "Explicit target height (0 = use Upscale by)")
      .add("Target", target, "Final image size after the hires pass")
      .full(adv)
    val pane = W.section("Hires. fix", f.grid, expanded, "Generate at the base size, then upscale and refine (A1111 Hires. fix)")
    val cb = new CheckBox()
    cb.selectedProperty.bindBidirectional(vm.hiresEnabled)
    W.tip(cb, "Enable Hires. fix")
    pane.setGraphic(cb)
    def mark(on: Boolean): Unit = if on then pane.getStyleClass.add("section-active") else pane.getStyleClass.remove("section-active")
    vm.hiresEnabled.addListener((_, _, on) => mark(on))
    mark(vm.hiresEnabled.get)
    pane

  /** A combo list with a leading "use same" entry that maps to "" in the VM. */
  private def withSame(src: javafx.collections.ObservableList[String], same: String) =
    val out = FXCollections.observableArrayList[String](same)
    def refresh(): Unit = out.setAll((same +: src.toArray(Array.empty[String]).toSeq)*)
    refresh()
    src.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) => refresh())
    out

  private def sameProp(p: javafx.beans.property.StringProperty, same: String) =
    val sp = new SimpleStringProperty(if p.get.isEmpty then same else p.get)
    sp.addListener((_, _, v) => p.set(if v == null || v == same then "" else v))
    p.addListener((_, _, v) => sp.set(if v == null || v.isEmpty then same else v))
    sp

  // ---- ControlNet --------------------------------------------------------------------------------------

  def controlNet(ctx: UiContext, vm: GenerationVM, expanded: Boolean): TitledPane =
    val tabs = new TabPane()
    tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
    tabs.getStyleClass.add("unit-tabs")
    vm.controlNet.zipWithIndex.foreach { (u, i) =>
      val t = new Tab(s"Unit ${i + 1}", unit(ctx, vm, u))
      def mark(): Unit = t.setText(s"Unit ${i + 1}${if u.enabled.get then " ●" else ""}")
      u.enabled.addListener((_, _, _) => mark())
      mark()
      tabs.getTabs.add(t)
    }
    val count = W.label("", "hint")
    def updateCount(): Unit =
      val n = vm.controlNet.count(_.enabled.get)
      count.setText(if n == 0 then "" else s"$n active")
    vm.controlNet.foreach(_.enabled.addListener((_, _, _) => updateCount()))
    updateCount()
    val pane = W.section("ControlNet", tabs, expanded, "Guide the composition with control images (canny, depth, pose…)")
    pane.setGraphic(count)
    pane.setContentDisplay(ContentDisplay.RIGHT)
    pane

  private def unit(ctx: UiContext, vm: GenerationVM, u: ControlNetUnitVM): Node =
    val models = FXCollections.observableArrayList[String]("None")
    val src = ctx.catalog.choices(ChoiceSource.ControlNets)
    def refresh(): Unit = models.setAll(("None" +: src.toArray(Array.empty[String]).toSeq)*)
    refresh()
    src.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) => refresh())
    val pre = FXCollections.observableArrayList[String]("none")
    val preSrc = ctx.catalog.choices(ChoiceSource.Preprocessors)
    def refreshPre(): Unit = pre.setAll(("none" +: preSrc.toArray(Array.empty[String]).toSeq.filterNot(_ == "none"))*)
    refreshPre()
    preSrc.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) => refreshPre())

    val hint = if vm.mode.usesInitImage then "Control image (empty = use the init image)" else "Control image"
    val drop = new ImageDropZone(ctx, u.image, hint, 170)
    val previewView = new ImageView()
    previewView.setPreserveRatio(true)
    previewView.setFitHeight(160)
    val previewBox = new VBox(4)
    val previewBtn = W.button("◉ Preview preprocessor", "Run the preprocessor on the control image and show the result")(())
    previewBtn.setOnAction(_ =>
      val img = Option(u.image.get).filter(_.nonEmpty).orElse(Option(vm.initImage.get).filter(_.nonEmpty))
      img match
        case None => ctx.info("Add a control image first")
        case Some(p) =>
          previewBtn.setDisable(true)
          Fx.async(ctx.services.generator.preprocess(Paths.get(p), u.preprocessor.get)) { out =>
            previewBtn.setDisable(false)
            ImageFiles.load(out.toString).foreach(previewView.setImage)
            previewBox.getChildren.setAll(W.label("Preprocessor result", "hint"), previewView)
          } { e =>
            previewBtn.setDisable(false)
            ctx.error(s"Preprocess failed: ${Fx.message(e)}")
          }
    )
    val f = W.form()
      .full(W.check("Enable", u.enabled, "Use this ControlNet unit"))
      .add("Model", W.combo(models, u.model), "ControlNet model (from models/controlnet)")
      .add("Preprocessor", W.combo(pre, u.preprocessor), "Turns the control image into a control map (none = already a map)")
      .add("Weight", W.doubleSlider(u.weight, 0, 2, 0.05), "Control strength")
      .add("Start", W.doubleSlider(u.guidanceStart, 0, 1, 0.01), "Fraction of the steps where control starts")
      .add("End", W.doubleSlider(u.guidanceEnd, 0, 1, 0.01), "Fraction of the steps where control ends")
    val v = new VBox(8, drop.node, f.grid, previewBtn, previewBox)
    v.getStyleClass.add("unit-body")
    v
