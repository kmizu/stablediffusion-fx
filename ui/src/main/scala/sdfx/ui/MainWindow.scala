package sdfx.ui

import javafx.animation.{KeyFrame, Timeline}
import javafx.geometry.Orientation
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.input.{KeyCode, KeyEvent}
import javafx.scene.layout.{BorderPane, StackPane, VBox}
import javafx.stage.{Screen, Stage}
import javafx.util.Duration
import sdfx.core.gallery.GalleryItem
import sdfx.core.model.ExtendedStyle
import sdfx.core.modelcards.{CardId, LocalPreviewFiles}
import sdfx.core.infotext.HardwareInfo
import sdfx.core.plugins.api.Plugin
import sdfx.engine.{AppServices, EngineState, JobEvent, WdBatchSpec}
import sdfx.core.presets.{LoadedPresets, PresetError}
import sdfx.ui.presets.{NativePresetFields, NativePresetFileDialogs, NativePresetsController, NativePresetsSession}
import sdfx.ui.view.*
import sdfx.ui.vm.*

import java.nio.file.{Files, Path}
import scala.util.control.NonFatal

/** Builds the main window and wires views, view models and services together. */
final class MainWindow(val stage: Stage, val services: AppServices, initial: UiState, opts: CliOptions,
    payloadClipboardWriter: String => Unit = RequestPayloadPane.systemClipboard,
    presetFileDialogs: NativePresetFileDialogs = NativePresetFileDialogs.system):
  val catalogVM = new CatalogVM
  // A window may be opened after the services have already scanned/loaded a model.
  catalogVM.update(services.catalog.snapshot, services.catalog.choices)
  val ctx = new UiContext(services, catalogVM, stage)
  val model = new ModelVM
  val progress = new ProgressVM
  val galleryVM = new GalleryVM
  @volatile var lastSeed: Long = initial.lastSeed
  // overlapping model / VAE loads: the chip shows the latest one until all are done
  private var loads = Vector.empty[(Long, String)]
  private var loadSeq = 0L
  private val cardWindowsPaths = LocalPreviewFiles.systemPlatform.windows
  private def validateCardModel(id: CardId): scala.concurrent.Future[Either[String, Unit]] =
    scala.concurrent.Future {
      val path = Path.of(id.modelPath)
      val directory = Files.isDirectory(path)
      LocalPreviewFiles.scoped().validateModel(path, directory).map(_ => ())
    }(using Fx.bg)
  private lazy val cardGate = new CheckpointCardGate(
    () => CardGateSnapshot(services.supervisor.state, services.queue.isIdle, loads.isEmpty,
      ctx.extendedStyleApplying.get, services.catalog.snapshot),
    validateCardModel,
    run => Fx.run(run()),
    name => {
      val exact = services.catalog.snapshot.checkpoints.filter(_.name == name)
      if exact.size != 1 || !services.catalog.findCheckpoint(name).contains(exact.head) then
        throw new IllegalStateException("Checkpoint catalog resolution changed during validation")
      startCheckpointLoad(name)
    },
    cardWindowsPaths
  )
  def cardSelectionRevision: javafx.beans.property.ReadOnlyLongProperty = cardGate.revision
  def cardCheckpointAdmission(id: CardId): CheckpointCardAdmission = cardGate.state(id)
  def admitCardCheckpoint(id: CardId): scala.concurrent.Future[Either[String, CheckpointAdmission]] = cardGate.admit(id)
  lazy val modelCardActions = new ModelCardActions(ctx, cardCheckpointAdmission, admitCardCheckpoint, cardSelectionRevision,
    validateCardModel, run => Fx.run(run()), cardWindowsPaths,
    Some((id, name) => cardGate.admit(id, Some(name))))
  private var browser: ModelBrowser = null
  ctx.openModelBrowser = (kind, target) =>
    modelCardActions.setTarget(target)
    if browser != null then browser.dispose()
    modelCardActions.setTarget(target)
    browser = new ModelBrowser(ctx, modelCardActions, services.modelCards)
    browser.show(kind)
  private lazy val cardIdle = new CardIdleObserver(() => services.queue.isIdle, run => {
    val timer = new javafx.animation.PauseTransition(Duration.millis(50))
    timer.setOnFinished(_ => run())
    timer.play()
    () => timer.stop()
  },
    () => cardGate.stateChanged())
  private def loading: Option[String] = loads.lastOption.map(_._2)
  private def beginLoad(what: String): Long =
    loadSeq += 1
    loads = loads :+ (loadSeq -> what)
    cardGate.stateChanged()
    refreshChip()
    loadSeq
  private def endLoad(id: Long): Unit =
    loads = loads.filterNot(_._1 == id)
    cardGate.stateChanged()
    refreshChip()
  private val exportGate = new SingleFlight
  /** Native acceptance holds the real export admission gate without starting a job. */
  private[ui] def withPresetMaskExportBusyForCheck(body: => Unit): Unit =
    require(javafx.application.Platform.isFxApplicationThread,
      "Preset export gate check must run on FX")
    require(!exportGate.isBusy, "A real mask export is already in flight")
    var release: () => Unit = () => ()
    require(exportGate.run(finish => release = finish, e => throw e),
      "Could not acquire the real mask export gate")
    try body
    finally release()
  // MCP applies run strictly in order on one thread; only the latest request updates the UI
  private val mcpExec = scala.concurrent.ExecutionContext.fromExecutorService(
    java.util.concurrent.Executors.newSingleThreadExecutor(r => {
      val t = new Thread(r, "sdfx-ui-mcp")
      t.setDaemon(true)
      t
    })
  )
  private var mcpGeneration = 0L
  private val mcp = new McpController((port, token) => sdfx.mcp.McpHttp.start(services, port, token))

  val vms: Map[Mode, GenerationVM] = Mode.values.map(m => m -> new GenerationVM(m, model)).toMap
  private def pluginValues(mode: Mode)(p: Plugin) =
    val saved = initial.pluginValues.get(mode.key).flatMap(_.get(p.id)).getOrElse(Map.empty)
    sdfx.core.plugins.api.Values(services.registry.values(p).map ++ saved)
  val plugins: Map[Mode, PluginsVM] = Mode.values.map { m =>
    val pv = new PluginsVM(m, services.registry.scripts(m.pluginTab), services.registry.alwaysOn(m.pluginTab), pluginValues(m))
    initial.selectedScript.get(m.key).filter(id => pv.scriptForms.exists(_.plugin.id == id)).foreach(pv.selectedScript.set)
    m -> pv
  }.toMap

  // restore last form state (init images only when the file still exists)
  Mode.values.foreach { m =>
    val variationGate = initial.variationEnabled.getOrElse(m.key, true)
    vms(m).variationEnabled.set(variationGate)
    initial.params.get(m.key).foreach { p =>
      val fixed = p.copy(img2img = p.img2img.map(i => i.copy(
        initImage = if i.initImage.nonEmpty && Files.isRegularFile(Path.of(i.initImage)) then i.initImage else "",
        mask = None
      )))
      try vms(m).load(fixed, variationEnabledState = variationGate)
      catch case NonFatal(e) => System.err.println(s"[sdfx] cannot restore ${m.key} state: ${e.getMessage}")
    }
  }
  model.vae.set(services.settings.lastVae.filter(_.nonEmpty).getOrElse("Automatic"))
  model.checkpoint.set(services.settings.lastModel.getOrElse(""))

  val comparisonHistory = new ComparisonHistory

  // ---- views -------------------------------------------------------------------------------------------
  private val actions: ImageActions = new ImageActions:
    def sendTo(mode: Mode, path: Path, infotext: Option[String]): Unit = MainWindow.this.sendTo(mode, path, infotext)
    def restore(infotext: String): Unit = MainWindow.this.restore(infotext)
    override def compare(sideB: Boolean, path: Path): Unit =
      comparison.send(sideB, path)
      tabs.getSelectionModel.select(comparisonTab)
    override def compareCurrent(): Unit =
      comparison.compareSource(activeMode.key)
      tabs.getSelectionModel.select(comparisonTab)
    def delete(path: Path): Unit = MainWindow.this.delete(path)

  val panels: Map[Mode, GenerationPanel] =
    Mode.values.map(m => m -> new GenerationPanel(ctx, vms(m), plugins(m), initial.expanded,
      () => lastSeed, openPayload, openPresets, openGenerationStates)).toMap
  ctx.commitPresetEditors = m => panels(m).commitPresetEditors()
  val preview = new PreviewPane(ctx, actions)
  val gallery = new GalleryView(ctx, galleryVM, actions, it => preview.show(it.path, it.parameters))
  val pngInfo = new PngInfoTab(ctx, actions)
  val extras = new ExtrasTab(ctx, initial.extrasUpscaler, initial.extrasScale)
  private def comparisonSource(kind: String): Option[(Path, Path)] =
    kind match
      case "txt2img" => comparisonHistory.pair
      case "extras" => extras.comparisonPair
      case other => Mode.values.find(_.key == other).flatMap { m =>
        for a <- Option(vms(m).initImage.get).filter(_.nonEmpty); b <- comparisonHistory.latest(m) yield java.nio.file.Paths.get(a) -> b
      }
  val comparison = new ImageComparisonTab(ctx, comparisonSource, () => preview.currentPath)
  val tokenizer = new TokenizerTab(ctx, () => vms(activeMode).prompt.get)
  val temporalLab = new TemporalLabTab(ctx)
  val systemInfo = new SystemInfoTab(ctx)
  val stylesEditor = new StylesEditorTab(ctx)
  val vaeTools = new VaeToolsTab(ctx)
  private val interruptionControls = new InterruptionControls(ctx)
  val statusBar = new StatusBar(progress, () => generate(false), () => generate(true), () => interrupt(),
    () => clearQueue(), interruptionControls.node)
  val topBar = new TopBar(ctx, model, loadCheckpoint, setVae, () => refreshCatalog(), () => restartEngine(), () => openSettings())

  private val tabs = new TabPane()
  tabs.setId("main-tabs")
  tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  tabs.getStyleClass.add("main-tabs")
  private val modeTabs: Map[Mode, Tab] = Mode.values.map(m => m -> new Tab(m.label, panels(m).node)).toMap
  Mode.values.foreach(m => tabs.getTabs.add(modeTabs(m)))
  private val pngTab = new Tab("PNG Info", pngInfo.node)
  private val extrasTab = new Tab("Extras", extras.node)
  private val temporalLabTab = new Tab("Temporal Lab", temporalLab.node)
  temporalLabTab.setId("temporal-lab-tab")
  private val tokenizerTab = new Tab("Tokenizer", tokenizer.node)
  tokenizerTab.setId("tokenizer-tab")
  private val systemInfoTab = new Tab("System Info", systemInfo.node)
  systemInfoTab.setId("system-info-tab")
  private val vaeToolsTab = new Tab("VAE Tools", vaeTools.node)
  vaeToolsTab.setId("vae-tools-tab")
  private val stylesTab = new Tab("Styles", stylesEditor.view)
  stylesTab.setId("styles-editor-tab")
  private val comparisonTab = new Tab("Comparison", comparison.node)
  comparisonTab.setId("comparison-tab")
  tabs.getTabs.addAll(pngTab, extrasTab, tokenizerTab, temporalLabTab, systemInfoTab, stylesTab, comparisonTab, vaeToolsTab)
  private var comparisonDivider = initial.dividers.get("main").flatMap(_.headOption).getOrElse(.42)
  private var lastMode: Mode = Mode.Txt2Img
  tabs.getSelectionModel.selectedItemProperty.addListener((_, previous, t) =>
    modeOf(t).foreach(lastMode = _)
    systemInfo.setActive(t eq systemInfoTab)
    if (previous eq stylesTab) && (t ne stylesTab) then stylesEditor.leave()
    refreshComparisonOptions()
    if t eq comparisonTab then
      if mainSplit.getItems.contains(rightBox) then
        if mainSplit.getWidth > 0 then comparisonDivider = mainSplit.getDividerPositions.headOption.getOrElse(comparisonDivider)
        mainSplit.getItems.remove(rightBox)
    else if !mainSplit.getItems.contains(rightBox) then
      mainSplit.getItems.add(rightBox)
      mainSplit.setDividerPositions(comparisonDivider)
  )
  private def modeOf(t: Tab): Option[Mode] = modeTabs.collectFirst { case (m, tab) if tab eq t => m }
  def activeMode: Mode = Option(tabs.getSelectionModel.getSelectedItem).flatMap(modeOf).getOrElse(lastMode)
  def select(mode: Mode): Unit = tabs.getSelectionModel.select(modeTabs(mode))

  private val rightSplit = new SplitPane(preview.imageNode, preview.infoNode)
  rightSplit.setOrientation(Orientation.VERTICAL)
  rightSplit.getStyleClass.add("right-split")
  SplitPane.setResizableWithParent(preview.infoNode, false)
  preview.infoNode.setMinHeight(96)
  private val rightBox = new VBox(rightSplit, gallery.node)
  VBox.setVgrow(rightSplit, javafx.scene.layout.Priority.ALWAYS)
  private val mainSplit = new SplitPane(tabs, rightBox)
  mainSplit.getStyleClass.add("main-split")
  SplitPane.setResizableWithParent(tabs, false)
  tabs.setMinWidth(420)

  private val layout = new BorderPane()
  layout.setTop(topBar.node)
  layout.setCenter(mainSplit)
  layout.setBottom(statusBar.node)
  val root = new StackPane(layout, ctx.toaster.layer)
  val scene = new Scene(root, initial.window.width, initial.window.height)
  Theme.apply(scene, services.settings.theme)

  private var payloadStage: Stage = null
  private var payloadPane: RequestPayloadPane = null
  private var presetStage: Stage = null
  private var presetPane: NativePresetsPane = null
  private var presetSession: NativePresetsSession = null

  private var generationStatesStage: Stage = null
  private var generationStatesPane: NamedGenerationStatesPane = null
  private var generationStatesSession: NativePresetsSession = null

  def namedStatesStage: Option[Stage] = Option(generationStatesStage)

  def openGenerationStates(mode: Mode): Unit =
    if !disposed then
      if generationStatesPane == null then
        val fields = Mode.values.map { m =>
          val panel = panels(m)
          m -> new NativePresetFields(vms(m), plugins(m), catalogVM,
            panel.sections(s"${m.key}.variation").expandedProperty, namedState = true,
            batchFields = Option.when(m == Mode.Img2Img)((panel.wdBatchEnabled, panel.wdBatchInput, panel.wdBatchOutputName)))
        }.toMap
        val controllers = fields.map((m, f) => m -> new NativePresetsController(f, () => exportGate.isBusy))
        generationStatesSession = new NativePresetsSession(services.home.configDir.resolve("generation-states"), controllers,
          result => if generationStatesPane != null then generationStatesPane.acceptResult(result),
          thunk => Fx.run(thunk()), true)
        generationStatesPane = new NamedGenerationStatesPane(ctx, generationStatesSession, fields, controllers)
        generationStatesStage = new Stage()
        generationStatesStage.initOwner(stage)
        generationStatesStage.setTitle("Named generation states")
        generationStatesStage.setMinWidth(760)
        generationStatesStage.setMinHeight(600)
        val statesScene = new Scene(generationStatesPane.node, 960, 780)
        Theme.apply(statesScene, services.settings.theme)
        generationStatesStage.setScene(statesScene)
        generationStatesSession.reload()
      generationStatesPane.openMode(mode)
      generationStatesStage.show()
      generationStatesStage.toFront()

  def presetsInspectorStage: Option[Stage] = Option(presetStage)

  def openPresets(mode: Mode): Unit =
    if !disposed then
      if presetPane == null then
        val controllers = Mode.values.map { m =>
          val fields = new NativePresetFields(vms(m), plugins(m), catalogVM,
            panels(m).sections(s"${m.key}.variation").expandedProperty)
          m -> new NativePresetsController(fields, () => exportGate.isBusy)
        }.toMap
        val directory = opts.configPresetsDir.map(Path.of(_))
          .getOrElse(services.home.configDir.resolve("native-presets"))
        presetSession = new NativePresetsSession(directory, controllers,
          result => if presetPane != null then presetPane.acceptResult(result),
          thunk => Fx.run(thunk()), opts.configPresetsDir.isEmpty)
        presetPane = new NativePresetsPane(ctx, presetSession, controllers, presetFileDialogs)
        presetStage = new Stage()
        presetStage.initOwner(stage)
        presetStage.setTitle("Native presets")
        presetStage.setMinWidth(700)
        presetStage.setMinHeight(500)
        val presetScene = new Scene(presetPane.node, 1040, 740)
        Theme.apply(presetScene, services.settings.theme)
        presetStage.setScene(presetScene)
        presetPane.reloadTools()
      presetPane.openMode(mode)
      presetStage.show()
      presetStage.toFront()

  /** Stable native acceptance seam. Opening only reads the memory-only store and never starts inference. */
  def payloadInspectorStage: Option[Stage] = Option(payloadStage)
  def openPayload(mode: Mode): Unit =
    if !disposed then
      if payloadPane == null then
        payloadPane = new RequestPayloadPane(ctx, payloadClipboardWriter)
        payloadStage = new Stage()
        payloadStage.initOwner(stage)
        payloadStage.setTitle("API payload inspector")
        payloadStage.setMinWidth(720)
        payloadStage.setMinHeight(540)
        val inspectorScene = new Scene(payloadPane.node, 1040, 740)
        Theme.apply(inspectorScene, services.settings.theme)
        payloadStage.setScene(inspectorScene)
      payloadPane.openMode(mode.key)
      payloadStage.show()
      payloadStage.toFront()

  private def refreshComparisonOptions(): Unit =
    preview.comparisonButton(if activeMode == Mode.Txt2Img then services.settings.comparisonTxtButton else services.settings.comparisonSendButtons)
    extras.refreshComparisonOptions()

  // ---- wiring ------------------------------------------------------------------------------------------
  ctx.sendToTab = (m, p, t) => sendTo(m, p, t)
  ctx.showImage = p => preview.show(p)
  ctx.openStylesEditor = () => { stylesEditor.reload(); tabs.getSelectionModel.select(stylesTab) }
  ctx.currentGeneration = () => vms(activeMode).toParams
  ctx.applyPromptStyle = style =>
    val mode = activeMode
    val vm = vms(mode)
    val (positive, negative) = style.applyTo(vm.prompt.get, vm.negativePrompt.get)
    vm.prompt.set(positive)
    vm.negativePrompt.set(negative)
    select(mode)
  ctx.applyExtendedStyle = style =>
    val mode = activeMode
    val vm = vms(mode)
    val candidate = style.restore(vm.toParams)
    val issues = candidate.validate
    val targetVae = style.effectiveVae
    if ctx.extendedStyleApplying.get then ctx.info("An extended style is already being applied")
    else if issues.nonEmpty then ctx.error(issues.mkString("; "))
    else if style.checkpoint.nonEmpty && services.catalog.findCheckpoint(style.checkpoint).isEmpty then
      ctx.error(s"Extended style checkpoint not found: ${style.checkpoint}")
    else if targetVae != "Automatic" && targetVae != "None" && services.catalog.findVae(targetVae).isEmpty then
      ctx.error(s"Extended style VAE not found: $targetVae")
    else
      ctx.extendedStyleApplying.set(true)
      val id = beginLoad(s"Style ${style.name}")
      val currentCheckpoint = model.checkpoint.get
      val currentVae = model.vae.get
      Fx.async {
        if style.checkpoint.nonEmpty && currentCheckpoint != style.checkpoint then services.loadCheckpoint(style.checkpoint)
        if ExtendedStyle.normalizeVae(currentVae) != targetVae || currentCheckpoint != style.checkpoint then services.setVae(Some(targetVae))
      } { _ =>
        try
          endLoad(id)
          // Update only documented settings; unrelated settings and disabled drafts remain intact.
          vm.prompt.set(style.prompt)
          vm.negativePrompt.set(style.negativePrompt)
          vm.width.set(style.width)
          vm.height.set(style.height)
          if style.checkpoint.nonEmpty then model.checkpoint.set(style.checkpoint)
          model.vae.set(targetVae)
          select(mode)
          ctx.success(s"Applied extended style '${style.name}'")
        finally ctx.extendedStyleApplying.set(false)
      } { e =>
        try
          endLoad(id)
          services.supervisor.loadedModel.foreach(m => model.checkpoint.set(m.name))
          ctx.error(s"Cannot apply extended style: ${Fx.message(e)}")
        finally ctx.extendedStyleApplying.set(false)
      }
  ctx.comparePair = (a, b) => { comparison.compare(a, b); tabs.getSelectionModel.select(comparisonTab) }
  comparison.onOptionsChanged = () => refreshComparisonOptions()
  refreshComparisonOptions()
  scene.addEventFilter(javafx.scene.input.MouseEvent.MOUSE_CLICKED, (event: javafx.scene.input.MouseEvent) =>
    if services.settings.comparisonMouseSend && event.isControlDown && event.isAltDown &&
        (event.getButton == javafx.scene.input.MouseButton.PRIMARY || event.getButton == javafx.scene.input.MouseButton.SECONDARY) then
      var node = event.getTarget match
        case n: javafx.scene.Node => n
        case _ => null
      var image: javafx.scene.image.Image = null
      var path: Option[Path] = None
      while node != null && path.isEmpty do
        node match
          case iv: javafx.scene.image.ImageView if image == null => image = iv.getImage
          case z: ZoomImageView if image == null => image = z.image
          case _ => ()
        node.getUserData match
          case p: Path => path = Some(p)
          case _ => ()
        node = node.getParent
      if image != null && !image.isError then
        val sideB = event.getButton == javafx.scene.input.MouseButton.SECONDARY
        def send(p: Path): Unit =
          comparison.send(sideB, p)
          if services.settings.comparisonMouseAuto then tabs.getSelectionModel.select(comparisonTab)
        path match
          case Some(p) => send(p)
          case None => Fx.async(ImageFiles.saveFxImage(ctx, image, "comparison-"))(send)(e => ctx.error(Fx.message(e)))
        event.consume()
  )
  ctx.interruptRuntime = () => interrupt()
  ctx.loadStyles()

  services.catalog.addListener(snap => Fx.run { catalogVM.update(snap, services.catalog.choices); cardGate.stateChanged() })
  services.supervisor.addStateListener(_ => Fx.run { refreshChip(); cardGate.stateChanged() })
  ctx.extendedStyleApplying.addListener((_, _, _) => cardGate.stateChanged())
  services.addLogListener(line => Fx.run(statusBar.log(line)))
  services.queue.addListener(e => { Fx.run {
    cardGate.stateChanged()
    e match
      case _: JobEvent.JobFinished | _: JobEvent.JobCancelled | _: JobEvent.JobFailed => cardIdle.terminal()
      case _ => ()
  }; QueueEventFxDispatch.forward(e, r => Fx.run(r()),
    // File stat stays on the job thread; only its small result crosses to FX.
    saved => {
      val size = try Files.size(saved.image.path) catch case NonFatal(_) => 0L
      Fx.run(onImageSaved(saved, size))
    },
    onJobEvent) })
  progress.state.addListener((_, _, _) => refreshChip())

  private def refreshChip(): Unit =
    val st = services.supervisor.state match
      case EngineState.Busy if !progress.get.running && loading.isEmpty =>
        services.supervisor.hello.map(EngineState.Ready(_)).getOrElse(EngineState.Busy)
      case s => s
    topBar.setState(st, loading)

  private def onImageSaved(e: JobEvent.ImageSaved, size: Long): Unit =
    progress.accept(e)
    val img = e.image
    lastSeed = img.seed
    if !e.grid then
      val mode = if !img.params.isImg2Img then Mode.Txt2Img else if img.params.isInpaint then Mode.Inpaint else Mode.Img2Img
      comparisonHistory.recordResult(e.jobId, img.path, mode)
    galleryVM.addResult(e.jobId, GalleryItem(img.path, System.currentTimeMillis(), size, Some(img.infotext)), e.grid)
    // selecting shows it (one decode); a gallery filter may hide it, then show directly
    if !e.grid && !gallery.select(img.path) then preview.show(img.path, Some(img.infotext))

  private def onJobEvent(e: JobEvent): Unit =
    progress.accept(e)
    e match
      case JobEvent.Preview(_, b64, _) if progress.get.running => preview.showLive(b64)
      case JobEvent.JobStarted(job)   =>
        galleryVM.beginGeneration(job.id)
        persistAsync()
      case JobEvent.Log(_, msg)       => statusBar.log(msg)
      case f: JobEvent.JobFailed      =>
        preview.stopLive()
        releaseTemps(f.jobId)
        ctx.error(friendly(f.message))
      case JobEvent.JobFinished(id, _) =>
        preview.stopLive()
        releaseTemps(id)
      case JobEvent.JobCancelled(id, _) =>
        preview.stopLive()
        releaseTemps(id)
      case _                          => ()

  private def releaseTemps(jobId: String): Unit =
    if ctx.temps.pendingJobs.contains(jobId) then Fx.async(ctx.temps.releaseJob(jobId))(_ => ())(_ => ())

  private def friendly(msg: String): String =
    if msg.contains("no_model") || msg.toLowerCase.contains("no model") || msg.contains("checkpoint not found") then
      s"$msg\nPut .safetensors in models/checkpoints or add folders in Settings."
    else if msg.contains("[oom]") || msg.toLowerCase.contains("out of memory") then
      s"$msg\nTry a smaller size / batch, or VRAM mode medvram in Settings."
    else msg

  // ---- actions -----------------------------------------------------------------------------------------
  def generate(queueOnly: Boolean): Unit =
    val mode = activeMode
    if !queueOnly && progress.get.running then
      ctx.info("A job is running — use Queue (Ctrl+Shift+Enter) to add another")
      return
    if exportGate.isBusy then return
    val vm = vms(mode)
    // a value typed into a spinner but not committed yet (Ctrl+Enter is consumed before focus moves)
    Option(scene.getFocusOwner).foreach(W.commitEnclosingSpinner)
    if Option(tabs.getSelectionModel.getSelectedItem).flatMap(modeOf).isEmpty then select(mode)
    val panel = panels(mode)
    val wdEnabled = mode == Mode.Img2Img && panel.wdBatchEnabled.get
    val wdAvailable = services.supervisor.hello.exists(_.features.contains("batch_interrogate_wd_v3_partial"))
    val wdChoice = WdBatchUiState.validate(wdEnabled, wdAvailable, panel.wdBatchInput.get, panel.wdBatchOutputName.get) match
      case Left(message) =>
        ctx.error(message)
        return
      case Right(value) => value
    // never reuse a mask exported for an earlier job: it is re-exported from the canvas below
    vm.maskImage.set("")
    val problems = vm.validate(batchImages = wdChoice.nonEmpty)
    if problems.nonEmpty then
      ctx.error(problems.mkString("\n"))
      return
    services.supervisor.state match
      case EngineState.Crashed(msg) =>
        ctx.error(s"The engine is not running: $msg\nUse Restart in the top bar.")
        return
      case _ => ()
    if !services.dryRun && catalogVM.snapshot.get.checkpoints.isEmpty && services.supervisor.loadedModel.isEmpty
      && services.supervisor.state.isInstanceOf[EngineState.Ready] then
      ctx.error("No checkpoint available.\nPut .safetensors in models/checkpoints or add folders in Settings.")
      return
    def submit(tempFiles: Seq[Path]): Unit =
      val ordinary = JobBuilder.build(vm, plugins(mode), sdfx.engine.CaptureOptions(services.settings.apiPayloadIncludeImageContents))
      val job = wdChoice match
        case None => ordinary
        case Some((input, name)) =>
          val spec = try
            val source = Path.of(input).toAbsolutePath.normalize()
            if !Files.isDirectory(source) then throw new IllegalArgumentException("Input image folder does not exist")
            val output = services.home.outputs(services.settings).toAbsolutePath.normalize()
            Files.createDirectories(output)
            if Files.exists(output.resolve(name)) then throw new IllegalArgumentException("Output folder already exists")
            WdBatchSpec(source.toString, output.toString, name)
          catch
            case NonFatal(e) =>
              ctx.error(s"Cannot prepare WD batch: ${Fx.message(e)}")
              return
          ordinary.copy(wdBatch = Some(spec))
      ctx.temps.registerJob(job.id, tempFiles)
      services.queue.submit(job).failed.foreach(_ => ())(using scala.concurrent.ExecutionContext.parasitic)
      if queueOnly && progress.get.running then ctx.info(s"Queued (${progress.get.pendingCount + 1} waiting)")
    panels(mode).mask match
      case Some(canvas) =>
        // PNG-encode the mask in the background, submit when it is written
        // the gate is released on every path (success, no image, async error, synchronous throw)
        val maskError: Throwable => Unit = e => ctx.error(s"Cannot export mask: ${Fx.message(e)}")
        exportGate.run(
          finish =>
            canvas.exportMaskAsync(
              {
                case Some(p) =>
                  finish()
                  vm.maskImage.set(p.toString)
                  submit(Seq(p))
                case None =>
                  finish()
                  ctx.error("The inpaint image is still loading — try again in a moment")
              },
              e =>
                finish()
                maskError(e)
            ),
          maskError
        )
      case None => submit(Nil)

  def interrupt(): Unit =
    Fx.onFx(services.interruptRuntime()) { result =>
      interruptionControls.refresh()
      if result.blocked then
        val message = result.message.getOrElse("Interruptions are blocked")
        statusBar.log(message)
        ctx.info(message)
      else if result.interrupted then statusBar.log("[ui] interrupt requested")
      else ctx.info(result.message.getOrElse("Nothing is running"))
    }(e => ctx.error("Cannot request interruption: " + Fx.message(e)))

  def clearQueue(): Unit =
    val n = services.queue.clear()
    if n > 0 then ctx.info(s"Removed $n pending job${if n == 1 then "" else "s"}")

  def sendTo(mode: Mode, path: Path, infotext: Option[String]): Unit =
    val vm = vms(mode)
    vm.initImage.set(path.toString)
    infotext.foreach { t =>
      restorePlugins(mode, t, vm.restoreInfotext(t))
      noticeHardware(t)
    }
    select(mode)
    ctx.info(s"Sent to ${mode.label}")

  def restore(infotext: String): Unit =
    val parsed = sdfx.core.infotext.Infotext.parse(infotext)
    noticeHardware(infotext)
    val mode = if activeMode == Mode.Txt2Img && sdfx.plugins.LatentUpscalePlugin.replay(parsed.fields).nonEmpty then Mode.Img2Img else activeMode
    restorePlugins(mode, infotext, vms(mode).restoreInfotext(infotext))
    select(mode)
    val vaeNotice = parsed.get("VAE").filter(_.nonEmpty).map { requested =>
      val available = if requested == "Automatic" || requested == "None" then Some(requested)
        else services.catalog.findVae(requested).map(_.name)
      available match
        case Some(value) => model.vae.set(value); ""
        case None => s" · VAE '$requested' was not found; current selection retained"
    }.getOrElse("")
    val parsedModel = sdfx.core.infotext.Infotext.parse(infotext).get("Model").filter(_.nonEmpty)
    parsedModel.flatMap(services.catalog.findCheckpoint).filter(_.name != model.checkpoint.get) match
      case Some(e) =>
        loadCheckpoint(e.name)
        ctx.info(s"Parameters restored · switching checkpoint to ${e.name}$vaeNotice")
      case None =>
        val modelNotice = parsedModel.filter(name => services.catalog.findCheckpoint(name).isEmpty)
          .map(name => s" · Checkpoint '$name' was not found; current selection retained").getOrElse("")
        ctx.success(s"Parameters restored into ${mode.label}$vaeNotice$modelNotice")

  private val hardwareNotices = new HardwareInfo.PasteNotices
  private var disposed = false
  private def noticeHardware(infotext: String): Unit =
    val imported = sdfx.core.infotext.Infotext.parse(infotext).get("Hardware Info")
    if imported.nonEmpty then Fx.onFx(ctx.hardwareInfo.profile()) { local =>
      if !disposed then hardwareNotices.pasted(local, imported).foreach { difference =>
        ctx.info(s"Recorded device differs: image ${difference.imageDevice}; current ${difference.currentDevice}")
      }
    }(_ => ()) // Provenance lookup failure must not interfere with ordinary parameter restoration.

  private def restorePlugins(mode: Mode, infotext: String, params: sdfx.core.model.GenerationParams): Unit =
    plugins(mode).restoreInfotext(infotext)
    if mode != Mode.Txt2Img then
      plugins(mode).restoreLatentUpscale(sdfx.core.infotext.Infotext.parse(infotext).fields)
    restoreComposable(mode, params)

  /** Restore this tool's complete PNG settings into its native form as well as the request. */
  private def restoreComposable(mode: Mode, params: sdfx.core.model.GenerationParams): Unit =
    params.hooks.get("composable_lora").collect { case o: ujson.Obj => o }.foreach { hook =>
      plugins(mode).form("composable_lora").foreach { form =>
        form.prop("enabled").foreach(_.set("true"))
        hook.obj.foreach { (key, value) =>
          form.prop(key).foreach(_.set(value.bool.toString))
        }
      }
    }

  def delete(path: Path): Unit =
    val a = new Alert(Alert.AlertType.CONFIRMATION, s"Delete ${path.getFileName}?\nThe file is removed from disk.", ButtonType.CANCEL, ButtonType.OK)
    a.initOwner(stage)
    a.setHeaderText(null)
    a.setTitle("Delete image")
    a.getDialogPane.getStylesheets.setAll(scene.getStylesheets)
    val r = a.showAndWait()
    if r.isPresent && r.get == ButtonType.OK then
      val next = galleryVM.neighbor(path, 1).orElse(galleryVM.neighbor(path, -1))
      Fx.async(Files.deleteIfExists(path)) { _ =>
        galleryVM.remove(path)
        gallery.evict(path)
        if preview.currentPath.contains(path) then next.fold(preview.clear())(it => gallery.select(it.path))
      }(e => ctx.error(s"Cannot delete: ${Fx.message(e)}"))

  /** Starts / stops / restarts the MCP HTTP server to match the settings (socket work off the FX thread). */
  def applyMcp(s: sdfx.core.config.Settings): Unit =
    mcpGeneration += 1
    val gen = mcpGeneration
    Fx.onFx(scala.concurrent.Future(mcp.apply(s.mcpHttpEnabled, s.mcpHttpPort, s.mcpToken))(using mcpExec)) { changed =>
      topBar.setMcp(mcpUrl)
      if changed && gen == mcpGeneration then
        mcpUrl match
          case Some(u) => ctx.success(s"MCP server listening on $u")
          case None    => ctx.info("MCP server stopped")
    } { e =>
      topBar.setMcp(mcpUrl)
      if gen == mcpGeneration then ctx.error(s"MCP server could not start on port ${s.mcpHttpPort}: ${Fx.message(e)}")
    }

  private def mcpUrl: Option[String] =
    mcp.handle.map {
      case t: sdfx.mcp.HttpTransport => t.url
      case _                         => s"http://127.0.0.1:${mcp.port.getOrElse(0)}/mcp"
    }

  def loadCheckpoint(name: String): Unit =
    startCheckpointLoad(name)
    ()

  private def startCheckpointLoad(name: String): Long =
    model.checkpoint.set(name)
    val id = beginLoad(name)
    Fx.async(services.loadCheckpoint(name)) { m =>
      endLoad(id)
      ctx.success(f"Loaded ${m.name}%s (${m.kind}%s) in ${m.loadedMs / 1000.0}%.1f s")
    } { e =>
      endLoad(id)
      ctx.error(s"Cannot load checkpoint: ${friendly(Fx.message(e))}")
    }
    id

  def setVae(v: String): Unit =
    val id = beginLoad(s"VAE $v")
    Fx.async(services.setVae(Some(v))) { _ =>
      endLoad(id)
    } { e =>
      endLoad(id)
      ctx.error(s"Cannot set VAE: ${Fx.message(e)}")
    }

  def refreshCatalog(): Unit =
    Fx.onFx(services.catalog.refreshAsync()) { s =>
      ctx.info(s"${s.checkpoints.size} checkpoints · ${s.loras.size} LoRAs · ${s.vaes.size} VAEs")
      updateBanner()
    }(e => ctx.error(s"Refresh failed: ${Fx.message(e)}"))

  def updateBanner(): Unit =
    val noModels = !services.dryRun && catalogVM.snapshot.get.checkpoints.isEmpty && services.supervisor.state.isUsable
    preview.setBanner(Option.when(noModels)("No checkpoints found. Put .safetensors in models/checkpoints or add folders in Settings (⚙)."))

  def reloadGallery(): Unit =
    val since = System.currentTimeMillis()
    Fx.async(services.gallery.scan(1000)) { items =>
      galleryVM.merge(items, since)
      if preview.currentPath.isEmpty then items.headOption.foreach(it => gallery.select(it.path))
    }(e => statusBar.log(s"[ui] gallery scan failed: ${Fx.message(e)}"))

  def startEngine(onReady: () => Unit = () => ()): Unit =
    refreshChip()
    Fx.onFx(services.startAsync()) { hello =>
      services.supervisor.loadedModel.foreach(m =>
        model.checkpoint.set(services.catalog.findCheckpoint(m.path).map(_.name).getOrElse(m.name))
      )
      refreshChip()
      updateBanner()
      statusBar.log(s"[ui] engine ready: ${hello.device}${if hello.dryRun then " (dry run)" else ""}")
      onReady()
    } { e =>
      val msg = Fx.message(e)
      topBar.setState(EngineState.Crashed(msg), None)
      ctx.error(s"Engine failed to start: $msg")
    }

  def restartEngine(): Unit =
    statusBar.log("[ui] restarting engine")
    Fx.async {
      services.supervisor.restart()
      services.catalog.refresh()
    }(_ =>
      refreshChip()
      updateBanner()
      ctx.success("Engine restarted")
    )(e => ctx.error(s"Restart failed: ${Fx.message(e)}"))

  def openSettings(tab: Int = 0): Unit =
    val before = services.settings
    val wdReported = services.supervisor.hello.map(_.features.contains("batch_interrogate_wd_v3_partial"))
    new SettingsDialog(stage, before, tab, wdReported).showAndWait().foreach { edited =>
      services.updateSettingsChecked(cur => SettingsMerge(cur, edited)) match
        case Left(reason) => ctx.error(reason)
        case Right(s) =>
          if s.mcpHttpEnabled != before.mcpHttpEnabled || s.mcpHttpPort != before.mcpHttpPort || s.mcpToken != before.mcpToken then applyMcp(s)
          if s.theme != before.theme then Theme.apply(scene, s.theme)
          if s.modelDirs != before.modelDirs then refreshCatalog()
          if s.outputDir != before.outputDir then
            services.resetGallery()
            reloadGallery()
          val engineChanged = s.pythonExe != before.pythonExe || s.dtype != before.dtype || s.vramMode != before.vramMode ||
            s.vaeTiling != before.vaeTiling || s.hfOffline != before.hfOffline ||
            WdSettingsState.restartRequired(before, s)
          if engineChanged then
            val a = new Alert(Alert.AlertType.CONFIRMATION, "Engine settings changed. Restart the engine now?", ButtonType.NO, ButtonType.YES)
            a.initOwner(stage)
            a.setHeaderText(null)
            a.getDialogPane.getStylesheets.setAll(scene.getStylesheets)
            val r = a.showAndWait()
            if r.isPresent && r.get == ButtonType.YES then restartEngine()
          ctx.success("Settings saved")
    }

  // ---- keyboard ----------------------------------------------------------------------------------------
  private def typing: Boolean = scene.getFocusOwner.isInstanceOf[TextInputControl]

  scene.addEventFilter(KeyEvent.KEY_PRESSED, (e: KeyEvent) =>
    val k = e.getCode
    val promptEditor = panels(activeMode).prompts
    if k == KeyCode.ENTER && e.isShortcutDown && !promptEditor.isComposingInput then
      generate(queueOnly = e.isShiftDown)
      e.consume()
    else if k == KeyCode.ESCAPE && progress.get.running && !promptEditor.isComposingInput && !promptEditor.completionVisible then
      interrupt()
      e.consume()
    else if e.isShortcutDown && k == KeyCode.V && !typing && activeMode.usesInitImage && tabs.getSelectionModel.getSelectedItem == modeTabs(activeMode) then
      activeMode match
        case Mode.Inpaint => panels(Mode.Inpaint).mask.foreach(_.paste())
        case m            => panels(m).initDrop.foreach(_.paste())
      e.consume()
    else if e.isShortcutDown && (k == KeyCode.Z || k == KeyCode.Y) && !typing && tabs.getSelectionModel.getSelectedItem == modeTabs(Mode.Inpaint) then
      panels(Mode.Inpaint).mask.foreach(m => if k == KeyCode.Z && !e.isShiftDown then m.undo() else m.redo())
      e.consume()
    else if (k == KeyCode.LEFT || k == KeyCode.RIGHT) && (scene.getFocusOwner == null || scene.getFocusOwner.isInstanceOf[ZoomImageView]) then
      gallery.selectNeighbor(if k == KeyCode.LEFT then -1 else 1)
      e.consume()
  )
  preview.zoom.setFocusTraversable(true)
  preview.zoom.addEventHandler(javafx.scene.input.MouseEvent.MOUSE_PRESSED, (_: javafx.scene.input.MouseEvent) => preview.zoom.requestFocus())

  // ---- VRAM polling ------------------------------------------------------------------------------------
  @volatile private var polling = false
  private val vramPoll = new Timeline(new KeyFrame(Duration.seconds(5), _ =>
    if !polling && services.queue.isIdle && loading.isEmpty && services.supervisor.state.isInstanceOf[EngineState.Ready] then
      polling = true
      Fx.onFx(services.client.memory()) { m =>
        polling = false
        topBar.setMemory(Some(m))
      } { _ => polling = false }
  ))
  vramPoll.setCycleCount(javafx.animation.Animation.INDEFINITE)

  // ---- window state ------------------------------------------------------------------------------------
  def show(): Unit =
    stage.setTitle(s"Stable Diffusion NUI${if services.dryRun then " (dry run)" else ""}")
    stage.setScene(scene)
    stage.setMinWidth(1280)
    stage.setMinHeight(800)
    val w = initial.window
    val onScreen = for x <- w.x; y <- w.y if !Screen.getScreensForRectangle(x, y, 200, 120).isEmpty yield (x, y)
    onScreen.foreach { (x, y) =>
      stage.setX(x)
      stage.setY(y)
    }
    stage.setWidth(w.width.max(1280))
    stage.setHeight(w.height.max(800))
    stage.setMaximized(w.maximized)
    stage.show()
    // Divider positions are re-applied while the window settles (the first heights can be pre-layout ones),
    // then left alone so user drags stick.
    def applyDividers(): Unit =
      val divider = initial.dividers.get("main").flatMap(_.headOption).getOrElse(0.42)
      if mainSplit.getItems.contains(rightBox) then mainSplit.setDividerPositions(divider)
      else comparisonDivider = divider
      val hh = rightSplit.getHeight
      if hh > 200 then rightSplit.setDividerPositions(initial.dividers.get("right").flatMap(_.headOption).getOrElse((hh - 160) / hh))
    var settling = true
    val settle = new javafx.animation.PauseTransition(Duration.seconds(1.5))
    settle.setOnFinished(_ => settling = false)
    settle.play()
    rightSplit.heightProperty.addListener((_, _, _) => if settling then Fx.later(applyDividers()))
    Fx.later(applyDividers())
    initial.galleryQuery.headOption.foreach(_ => galleryVM.query.set(initial.galleryQuery))
    opts.tab.orElse(Some(initial.selectedTab)).foreach { t =>
      Mode.fromKey(t).foreach(select)
      if t == "pnginfo" then tabs.getSelectionModel.select(pngTab)
      if t == "extras" then tabs.getSelectionModel.select(extrasTab)
      if t == "tokenizer" then tabs.getSelectionModel.select(tokenizerTab)
      if t == "temporal_lab" then tabs.getSelectionModel.select(temporalLabTab)
      if t == "systeminfo" then tabs.getSelectionModel.select(systemInfoTab)
      if t == "styles" then tabs.getSelectionModel.select(stylesTab)
      if t == "comparison" then tabs.getSelectionModel.select(comparisonTab)
      if t == "vae_tools" then tabs.getSelectionModel.select(vaeToolsTab)
    }
    statusBar.logToggle.setSelected(initial.logVisible)
    reloadGallery()
    vramPoll.play()
    applyMcp(services.settings)
    sweepTemps()

  def state: UiState =
    val selected = Option(tabs.getSelectionModel.getSelectedItem) match
      case Some(t) if t eq pngTab    => "pnginfo"
      case Some(t) if t eq extrasTab => "extras"
      case Some(t) if t eq tokenizerTab => "tokenizer"
      case Some(t) if t eq temporalLabTab => "temporal_lab"
      case Some(t) if t eq systemInfoTab => "systeminfo"
      case Some(t) if t eq stylesTab => "styles"
      case Some(t) if t eq comparisonTab => "comparison"
      case Some(t) if t eq vaeToolsTab => "vae_tools"
      case Some(t)                   => modeOf(t).map(_.key).getOrElse("txt2img")
      case None                      => "txt2img"
    UiState(
      window = WindowState(
        Some(stage.getX).filterNot(_.isNaN),
        Some(stage.getY).filterNot(_.isNaN),
        stage.getWidth,
        stage.getHeight,
        stage.isMaximized
      ),
      dividers = Map("main" -> (if mainSplit.getItems.contains(rightBox) then mainSplit.getDividerPositions.toSeq else Seq(comparisonDivider)), "right" -> rightSplit.getDividerPositions.toSeq),
      selectedTab = selected,
      params = Mode.values.map(m => m.key -> vms(m).snapshot).toMap,
      variationEnabled = Mode.values.map(m => m.key -> vms(m).variationEnabled.get).toMap,
      pluginValues = Mode.values.map(m => m.key -> plugins(m).snapshot).toMap,
      selectedScript = Mode.values.map(m => m.key -> plugins(m).selectedScript.get).toMap,
      expanded = panels.values.flatMap(_.sections.map((k, p) => k -> p.isExpanded)).toMap,
      logVisible = statusBar.logToggle.isSelected,
      galleryQuery = galleryVM.query.get,
      lastSeed = lastSeed,
      extrasUpscaler = Option(extras.upscaler.get).getOrElse(""),
      extrasScale = extras.scale.get
    )

  /** Images the form state still points at (kept by temp sweeps). */
  private def referencedImages: Set[String] =
    (Mode.values.toSeq.flatMap(m => vms(m).initImage.get +: vms(m).controlNet.map(_.image.get)) ++
      Seq(pngInfo.image.get, extras.image.get)).filter(s => s != null && s.nonEmpty).toSet ++ comparison.sourcePaths

  /** Deletes stale UI temp files (masks of finished jobs, unreferenced pasted images). */
  def sweepTemps(): Unit =
    val keep = referencedImages
    Fx.async(ctx.temps.sweep(keep))(n => if n > 0 then statusBar.log(s"[ui] removed $n temp files"))(_ => ())

  /** Saves the UI state in the background (at job start, so a crash loses little). */
  def persistAsync(): Unit =
    val s = state
    Fx.async(UiState.save(UiState.file(services.home), s))(_ => ())(e => statusBar.log(s"[ui] cannot save ui state: ${Fx.message(e)}"))

  def persist(): Unit =
    try
      val s = state
      UiState.save(UiState.file(services.home), s)
      services.updateSettings(st => st.copy(pluginValues = st.pluginValues ++ plugins(Mode.Txt2Img).snapshot))
    catch case NonFatal(e) => System.err.println(s"[sdfx] cannot save ui state: ${e.getMessage}")

  private val cleanup = new UiCleanup
  def dispose(): Unit =
    disposed = true
    cleanup.run(Seq(
      "interruptions" -> (() => interruptionControls.dispose()),
      "card idle observer" -> (() => cardIdle.dispose()),
      "model browser" -> (() => if browser != null then { val old = browser; browser = null; old.dispose() }),
      "card actions" -> (() => modelCardActions.dispose()),
      "card gate" -> (() => cardGate.dispose()),
      "payload pane" -> (() => if payloadPane != null then payloadPane.dispose()),
      "payload stage" -> (() => if payloadStage != null then payloadStage.close()),
      "generation states pane" -> (() => if generationStatesPane != null then generationStatesPane.dispose()),
      "generation states session" -> (() => if generationStatesSession != null then generationStatesSession.close()),
      "generation states stage" -> (() => if generationStatesStage != null then generationStatesStage.close()),
      "preset pane" -> (() => if presetPane != null then presetPane.dispose()),
      "preset session" -> (() => if presetSession != null then presetSession.close()),
      "preset stage" -> (() => if presetStage != null then presetStage.close()),
      "system information" -> (() => systemInfo.dispose()),
      "temporal lab" -> (() => temporalLab.dispose()),
      "styles editor" -> (() => stylesEditor.dispose()),
      "VRAM timer" -> (() => vramPoll.stop()),
      "MCP controller" -> (() => mcpExec.execute(() => mcp.close())),
      "MCP executor" -> (() => mcpExec.shutdown()),
      "temporary files" -> (() => { ctx.temps.sweep(referencedImages); () })
    ))
