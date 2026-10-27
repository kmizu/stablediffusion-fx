package sdfx.ui.view

import javafx.beans.property.*
import javafx.collections.{FXCollections, ObservableList}
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{HBox, Priority, VBox}
import javafx.stage.{DirectoryChooser, FileChooser, Stage}
import sdfx.core.config.{ModelDirs, Settings}
import sdfx.ui.vm.{VramModes, WdSettingsState}

import scala.jdk.CollectionConverters.*

/** Settings editor. Returns the edited settings; the caller applies them. */
final class SettingsDialog(owner: Stage, initial: Settings, initialTab: Int = 0, wdWorkerReported: Option[Boolean] = None):
  private val kinds = Seq("Checkpoints", "VAE", "LoRA", "ControlNet", "Embeddings", "Upscalers", "Face restore", "SVD",
    "Zero123", "PhotoMaker")
  private val dirs: Map[String, ObservableList[String]] =
    val d = initial.modelDirs
    Seq(d.checkpoints, d.vae, d.lora, d.controlnet, d.embeddings, d.upscalers, d.faceRestore, d.svd, d.zero123, d.photomaker).zip(kinds).map((xs, k) => k -> FXCollections.observableArrayList(xs*)).toMap

  private val outputDir = new SimpleStringProperty(initial.outputDir)
  private val pythonExe = new SimpleStringProperty(initial.pythonExe.getOrElse(""))
  private val dtype = new SimpleStringProperty(initial.dtype)
  private val vramIndex = new SimpleIntegerProperty(VramModes.index(initial.vramMode).max(0))
  private val vaeTiling = new SimpleBooleanProperty(initial.vaeTiling)
  private val previewEvery = new SimpleIntegerProperty(initial.previewEvery)
  private val previewMethod = new SimpleStringProperty(initial.previewMethod)
  private val hfOffline = new SimpleBooleanProperty(initial.hfOffline)
  private val mcpEnabled = new SimpleBooleanProperty(initial.mcpHttpEnabled)
  private val mcpPort = new SimpleIntegerProperty(initial.mcpHttpPort)
  private val mcpToken = new SimpleStringProperty(initial.mcpToken.getOrElse(""))
  private val theme = new SimpleStringProperty(initial.theme)
  private val cardDefault = new SimpleStringProperty(initial.nsfwCardBlurDefault)
  private val allowCode = new SimpleBooleanProperty(initial.allowCustomCode)
  private val wdModel = new SimpleStringProperty(initial.wdV3Model.getOrElse(""))
  private val wdTags = new SimpleStringProperty(initial.wdV3Tags.getOrElse(""))

  // ---- model folders -----------------------------------------------------------------------------------
  private val kindList = new ListView[String](FXCollections.observableArrayList(kinds*))
  kindList.setPrefWidth(150)
  kindList.getStyleClass.add("kind-list")
  private val folderList = new ListView[String]()
  folderList.setEditable(true)
  folderList.setCellFactory(javafx.scene.control.cell.TextFieldListCell.forListView())
  W.tip(folderList, "Folders searched for this model type (relative paths are relative to the app folder). Double click to edit.")
  kindList.getSelectionModel.selectedItemProperty.addListener((_, _, k) => if k != null then folderList.setItems(dirs(k)))
  kindList.getSelectionModel.select(0)

  private def chooseDir(title: String): Option[String] =
    val dc = new DirectoryChooser()
    dc.setTitle(title)
    Option(dc.showDialog(owner)).map(_.getAbsolutePath)

  private def currentKind = Option(kindList.getSelectionModel.getSelectedItem).getOrElse(kinds.head)
  private val addBtn = W.button("＋ Add folder…", "Add a folder for the selected model type")(
    chooseDir("Add model folder").foreach(d => if !dirs(currentKind).contains(d) then dirs(currentKind).add(d))
  )
  private val removeBtn = W.button("－ Remove", "Remove the selected folder")(
    Option(folderList.getSelectionModel.getSelectedIndex).filter(_ >= 0).foreach(i => folderList.getItems.remove(i.intValue))
  )
  private val a1111Btn = W.button("Add A1111/reForge install…", "Use the model folders of an existing A1111 / reForge / Forge install (read-only)", Seq("accent-button"))(
    chooseDir("Select the A1111 / reForge folder").foreach { root =>
      val d = currentDirs.withA1111(root)
      Seq(d.checkpoints, d.vae, d.lora, d.controlnet, d.embeddings, d.upscalers, d.faceRestore, d.svd, d.zero123, d.photomaker).zip(kinds).foreach((xs, k) => dirs(k).setAll(xs*))
    }
  )

  private def currentDirs: ModelDirs =
    def l(k: String) = dirs(k).asScala.toSeq.map(_.trim).filter(_.nonEmpty)
    ModelDirs(l("Checkpoints"), l("VAE"), l("LoRA"), l("ControlNet"), l("Embeddings"), l("Upscalers"), l("Face restore"),
      l("SVD"), l("Zero123"), l("PhotoMaker"))

  private val modelsTab =
    val right = new VBox(8, folderList, W.hbox(6, addBtn, removeBtn))
    VBox.setVgrow(folderList, Priority.ALWAYS)
    val h = new HBox(10, kindList, right)
    HBox.setHgrow(right, Priority.ALWAYS)
    val v = new VBox(10, h, a1111Btn)
    VBox.setVgrow(h, Priority.ALWAYS)
    v.setPadding(new Insets(12))
    new Tab("Model folders", v)

  // ---- other tabs --------------------------------------------------------------------------------------
  private def browseRow(p: StringProperty, dir: Boolean, title: String) =
    val f = W.textField(p)
    val b = W.iconButton("…", title) {
      if dir then chooseDir(title).foreach(p.set)
      else
        val fc = new FileChooser()
        fc.setTitle(title)
        Option(fc.showOpenDialog(owner)).foreach(x => p.set(x.getAbsolutePath))
    }
    W.hbox(6, W.hgrow(f), b)

  private def page(f: W.Form): VBox =
    val v = new VBox(f.grid)
    v.setPadding(new Insets(14))
    v

  private val generalTab = new Tab(
    "General",
    page(
      W.form()
        .add("Output folder", browseRow(outputDir, true, "Choose the output folder"), "Where generated images are saved (relative = inside the app folder)")
        .add("Named NSFW previews default", W.combo(FXCollections.observableArrayList("Blur", "Hide", "Show"), cardDefault), "Saved default for a new app session; changing it does not change the current temporary mode.")
        .add("Theme", W.combo(FXCollections.observableArrayList("dark", "light"), theme), "Color theme")
        .add("Live preview", W.intSlider(previewEvery, 0, 20), "Send a preview image every N sampling steps (0 = off)")
        .add("Preview method", W.combo(FXCollections.observableArrayList("approx", "taesd"), previewMethod), "approx: instant, rough colors · taesd: better quality, small extra cost")
    )
  )

  private val engineTab = new Tab(
    "Engine",
    page(
      W.form()
        .add("Python", browseRow(pythonExe, false, "Choose python executable"), "Python interpreter for the engine (empty = app venv, then system Python)")
        .add("Precision", W.combo(FXCollections.observableArrayList("fp16", "bf16", "fp32"), dtype), "Model weight precision (fp16 for most GPUs)")
        .add("VRAM mode", W.indexCombo(VramModes.Modes.map(_._2), vramIndex), "GPU: everything stays on the GPU · Model offload: whole models move to the GPU when used · Sequential offload: layer by layer, slowest but lowest VRAM")
        .full(W.check("VAE tiling (decode large images in tiles to save VRAM)", vaeTiling))
        .full(W.check("Offline mode (never download from Hugging Face)", hfOffline))
        .full(W.check("Allow custom code (the Custom code script runs Python in the engine; like A1111 --allow-code)", allowCode))
        .full(W.label("Python, precision and VRAM changes apply after an engine restart.", "hint"))
    )
  )

  private val mcpTab = new Tab(
    "MCP",
    page(
      W.form()
        .full(W.check("Enable MCP HTTP server", mcpEnabled, "Let AI agents drive this app over MCP (HTTP)"))
        .add("Port", W.intSpinner(mcpPort, 1024, 65535, 1, 110), "Local port of the MCP HTTP server")
        .add("Token", W.textField(mcpToken, "optional bearer token"), "Clients must send this bearer token (recommended)")
        .full(W.label("Applies immediately when you press OK.", "hint"))
    )
  )

  private val wdTab =
    val status = W.label(WdSettingsState.status(wdModel.get, wdTags.get), "hint")
    status.setId("wd-v3-status")
    val update = () => status.setText(WdSettingsState.status(wdModel.get, wdTags.get))
    wdModel.addListener((_, _, _) => update())
    wdTags.addListener((_, _, _) => update())
    val note = W.label("Partial support: WD v3 only. No models are downloaded. The batch UI is not yet available. Restart the engine to check the selected files.", "hint")
    note.setWrapText(true)
    val worker = W.label(WdSettingsState.workerStatus(wdWorkerReported), "hint")
    worker.setId("wd-v3-worker-status")
    worker.setWrapText(true)
    val content = page(W.form()
      .full(note)
      .add("Model (.onnx)", browseRow(wdModel, false, "Choose local WD v3 model"), "Select the local WD v3 ONNX file")
      .add("Tags (.csv)", browseRow(wdTags, false, "Choose local WD v3 tags"), "Select the matching local selected_tags.csv")
      .full(status)
      .full(worker))
    val scroll = new ScrollPane(content)
    scroll.setFitToWidth(true)
    new Tab("WD v3 (partial)", scroll)

  private val tabs = new TabPane(modelsTab, generalTab, engineTab, mcpTab, wdTab)
  tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  tabs.setPrefSize(720, 440)
  tabs.getSelectionModel.select(initialTab.max(0).min(tabs.getTabs.size - 1))

  /** Shows the dialog; Some(updated) when saved. */
  def showAndWait(): Option[Settings] =
    val d = new Dialog[ButtonType]()
    d.initOwner(owner)
    d.setTitle("Settings")
    d.setHeaderText(null)
    d.getDialogPane.setContent(tabs)
    d.getDialogPane.getButtonTypes.addAll(ButtonType.CANCEL, ButtonType.OK)
    d.getDialogPane.getStylesheets.setAll(owner.getScene.getStylesheets)
    d.getDialogPane.getStyleClass.add("settings-dialog")
    d.setResizable(true)
    val r = d.showAndWait()
    if r.isPresent && r.get == ButtonType.OK then
      Some(
        initial.copy(
          modelDirs = currentDirs,
          outputDir = Option(outputDir.get).map(_.trim).filter(_.nonEmpty).getOrElse("outputs"),
          pythonExe = Option(pythonExe.get).map(_.trim).filter(_.nonEmpty),
          dtype = dtype.get,
          vramMode = VramModes.Modes(vramIndex.get.max(0).min(VramModes.Modes.size - 1))._1,
          vaeTiling = vaeTiling.get,
          previewEvery = previewEvery.get,
          previewMethod = previewMethod.get,
          hfOffline = hfOffline.get,
          mcpHttpEnabled = mcpEnabled.get,
          mcpHttpPort = mcpPort.get,
          mcpToken = Option(mcpToken.get).map(_.trim).filter(_.nonEmpty),
          theme = theme.get,
          nsfwCardBlurDefault = cardDefault.get,
          allowCustomCode = allowCode.get,
          wdV3Model = Option(wdModel.get).map(_.trim).filter(_.nonEmpty),
          wdV3Tags = Option(wdTags.get).map(_.trim).filter(_.nonEmpty)
        )
      )
    else None
