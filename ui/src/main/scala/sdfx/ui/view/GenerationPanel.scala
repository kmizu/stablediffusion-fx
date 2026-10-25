package sdfx.ui.view

import javafx.geometry.Insets
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.layout.{GridPane, Priority, VBox}
import sdfx.ui.vm.{GenerationVM, Mode, PluginsVM}

/** The scrollable parameter form of one generation tab. */
final class GenerationPanel(
    ctx: UiContext,
    val vm: GenerationVM,
    val plugins: PluginsVM,
    expanded: Map[String, Boolean],
    lastSeed: () => Long
):
  val mode: Mode = vm.mode
  /** Titled panes by persistence key ("<mode>.<section>"). */
  val sections = scala.collection.mutable.LinkedHashMap.empty[String, TitledPane]
  private def isExpanded(key: String, default: Boolean) = expanded.getOrElse(s"${mode.key}.$key", default)
  private def track(key: String, p: TitledPane): TitledPane =
    sections(s"${mode.key}.$key") = p
    p

  val prompts = new PromptEditor(ctx, vm.prompt, vm.negativePrompt)

  // ---- validation banner -------------------------------------------------------------------------------
  private val problems = new VBox(2)
  problems.getStyleClass.add("problems")
  private def refreshProblems(): Unit =
    problems.getChildren.setAll(vm.problems.toArray(Array.empty[String]).toSeq.map(p => W.label(s"⚠  $p", "problem"))*)
    problems.setVisible(!vm.problems.isEmpty)
    problems.setManaged(!vm.problems.isEmpty)
  vm.problems.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) => refreshProblems())
  refreshProblems()

  // ---- img2img / inpaint inputs ------------------------------------------------------------------------
  val initDrop: Option[ImageDropZone] = Option.when(mode == Mode.Img2Img)(new ImageDropZone(ctx, vm.initImage, "Init image", 280))
  val mask: Option[MaskCanvas] = Option.when(mode == Mode.Inpaint)(new MaskCanvas(ctx, vm.initImage, vm.hasMask))

  private def denoiseRow(f: W.Form) =
    f.add("Denoising", W.doubleSlider(vm.denoise, 0, 1, 0.01), "How much the image may change: 0 = keep, 1 = ignore the init image")

  private def imageSection: Option[Node] = mode match
    case Mode.Txt2Img => None
    case Mode.Img2Img =>
      val f = W.form()
        .add("Resize mode", W.indexCombo(GenerationVM.ResizeModes, vm.resizeMode), "How the init image is fitted to the target size")
      denoiseRow(f)
      Some(W.card("Init image", initDrop.get.node, f.grid))
    case Mode.Inpaint =>
      val f = W.form()
        .add("Mask blur", W.intSlider(vm.maskBlur, 0, 64), "Feathering of the mask edge in pixels")
        .add("Mask mode", W.indexCombo(Seq("Inpaint masked", "Inpaint not masked"), boolIndex(vm.maskInvert)), "Regenerate the painted area or everything else")
        .add("Masked content", W.indexCombo(GenerationVM.MaskedContent, vm.inpaintingFill), "What the masked area starts from: original keeps the colors, latent noise invents new content")
        .add("Inpaint area", W.indexCombo(Seq("Whole picture", "Only masked"), boolIndex(vm.inpaintFullRes)), "Only masked: crop around the mask, generate at full resolution and paste back")
        .add("Padding", W.intSlider(vm.inpaintPadding, 0, 256, 4), "Context pixels around the mask for 'Only masked'")
        .add("Resize mode", W.indexCombo(GenerationVM.ResizeModes, vm.resizeMode), "How the image is fitted to the target size")
      denoiseRow(f)
      Some(W.card("Inpaint", mask.get.node, f.grid))

  private def boolIndex(b: javafx.beans.property.BooleanProperty) =
    val ip = new javafx.beans.property.SimpleIntegerProperty(if b.get then 1 else 0)
    ip.addListener((_, _, v) => b.set(v.intValue == 1))
    b.addListener((_, _, v) => ip.set(if v then 1 else 0))
    ip

  // ---- sampling ----------------------------------------------------------------------------------------
  private def sampling: Node =
    val sampler = W.combo(ctx.catalog.samplers, vm.sampler)
    val sched = W.combo(ctx.catalog.schedulers, vm.scheduler)
    W.tip(sampler, "Sampling method (only samplers the engine supports are listed)")
    W.tip(sched, "Noise schedule (Automatic = the sampler's default)")
    val row = new GridPane()
    row.setHgap(8)
    val c = new javafx.scene.layout.ColumnConstraints()
    c.setPercentWidth(55)
    val c2 = new javafx.scene.layout.ColumnConstraints()
    c2.setPercentWidth(45)
    row.getColumnConstraints.addAll(c, c2)
    row.add(sampler, 0, 0)
    row.add(sched, 1, 0)
    val f = W.form()
      .add("Sampler", row)
      .add("Steps", W.intSlider(vm.steps, 1, 150), "Number of denoising steps (20–30 is typical)")
      .add("CFG Scale", W.doubleSlider(vm.cfgScale, 1, 30, 0.5), "How strongly the prompt is followed (5–8 is typical)")
    W.card("Sampling", f.grid)

  private def size: Node =
    val swap = W.iconButton("⇅", "Swap width and height") {
      val w = vm.width.get
      vm.width.set(vm.height.get)
      vm.height.set(w)
    }
    val presets = new MenuButton("▦")
    presets.getStyleClass.add("icon-button")
    W.tip(presets, "Size presets (SD1.5: 512–768, SDXL: ~1 megapixel)")
    GenerationVM.SizePresets.foreach { (w, h) =>
      val mi = new MenuItem(s"$w × $h")
      mi.setOnAction(_ =>
        vm.width.set(w)
        vm.height.set(h)
      )
      presets.getItems.add(mi)
    }
    val f = W.form()
      .add("Width", W.hbox(6, W.hgrow(W.intSlider(vm.width, 64, 2048, 8, 4096)), swap), "Image width (multiple of 8)")
      .add("Height", W.hbox(6, W.hgrow(W.intSlider(vm.height, 64, 2048, 8, 4096)), presets), "Image height (multiple of 8)")
      .add(
        "Batch",
        W.hbox(
          8,
          W.label("count", "hint"),
          W.tip(W.intSpinner(vm.batchCount, 1, 100), "Batch count: how many batches to run one after another"),
          W.label("size", "hint"),
          W.tip(W.intSpinner(vm.batchSize, 1, 16), "Batch size: images generated in parallel (uses more VRAM)")
        )
      )
    W.card("Size", f.grid)

  private def seed: Node =
    val field = W.longField(vm.seed)
    W.tip(field, "Seed (-1 = random)")
    val dice = W.iconButton("⚄", "Random seed (-1)")(vm.seed.set(-1))
    val recycle = W.iconButton("♻", "Reuse the seed of the last image")(if lastSeed() >= 0 then vm.seed.set(lastSeed()) else ctx.info("No image generated yet"))
    val sub = W.longField(vm.subseed)
    val variation = W.form()
      .add("Var. seed", W.hbox(6, W.hgrow(sub), W.iconButton("⚄", "Random variation seed (-1)")(vm.subseed.set(-1))), "Variation seed mixed into the main seed")
      .add("Strength", W.doubleSlider(vm.subseedStrength, 0, 1, 0.01), "0 = only the main seed, 1 = only the variation seed")
    val varPane = track("variation", W.section("Variation seed", variation.grid, isExpanded("variation", false)))
    varPane.getStyleClass.add("subsection")
    val f = W.form().add("Seed", W.hbox(6, W.hgrow(field), dice, recycle))
    W.card("Seed", f.grid, varPane)

  // ---- assembly ----------------------------------------------------------------------------------------
  val content: VBox =
    val v = new VBox(10)
    v.getStyleClass.add("generation-panel")
    v.setPadding(new Insets(10, 12, 16, 12))
    v.getChildren.add(problems)
    v.getChildren.add(W.card("Prompt", prompts.view))
    imageSection.foreach(v.getChildren.add)
    v.getChildren.addAll(sampling, size, seed)
    if mode == Mode.Txt2Img then v.getChildren.add(track("hires", ExtraSections.hires(ctx, vm, isExpanded("hires", false))))
    v.getChildren.add(track("controlnet", ExtraSections.controlNet(ctx, vm, isExpanded("controlnet", false))))
    v.getChildren.add(W.card("Script", PluginForm.scriptSection(ctx, plugins)))
    plugins.alwaysOnForms.foreach { f =>
      val key = s"plugin.${f.plugin.id}"
      v.getChildren.add(track(key, PluginForm.alwaysOnSection(ctx, f, isExpanded(key, false))))
    }
    v

  val node: ScrollPane =
    val sp = new ScrollPane(content)
    sp.setFitToWidth(true)
    sp.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
    sp.getStyleClass.add("panel-scroll")
    VBox.setVgrow(sp, Priority.ALWAYS)
    sp
