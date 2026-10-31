package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.image.{Image, ImageView}
import javafx.scene.input.{Clipboard, ClipboardContent}
import javafx.scene.layout.{FlowPane, Priority, VBox}
import javafx.stage.FileChooser
import java.nio.file.{Files, Path}
import java.util.concurrent.CancellationException
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal
import sdfx.core.modelpresets.*
import sdfx.ui.Fx
import sdfx.ui.presets.ModelPresetAsync
import sdfx.ui.vm.Mode

/** Model-bound infotext library, independent from generation and general settings presets. */
final class ModelPresetsTab(ctx: UiContext, currentModel: () => Option[Path],
    capture: () => Option[String], send: (Mode, String, Path) => Unit):
  private val service = ctx.services.modelPresets
  private val model = new ComboBox[String](ctx.catalog.choices(sdfx.core.plugins.api.ChoiceSource.Checkpoints))
  model.setId("model-presets-model"); model.setMaxWidth(Double.MaxValue)
  private val identity = W.label("Choose a checkpoint or use the currently loaded model.", "muted")
  identity.setId("model-presets-identity"); identity.setWrapText(true)
  private val names = new ComboBox[String]()
  names.setId("model-presets-list"); names.setMaxWidth(Double.MaxValue)
  private val name = new TextField()
  name.setId("model-presets-name"); name.setPromptText("Preset name")
  private val generationLabel = W.label("Generation data")
  private val editor = new TextArea()
  editor.setId("model-presets-editor"); editor.setPrefRowCount(8); editor.setWrapText(true)
  private val url = new TextField()
  url.setId("model-presets-url"); url.setPromptText("Model page URL (optional)")
  private val words = new TextArea()
  words.setId("model-presets-words"); words.setPrefRowCount(2); words.setPromptText("Trigger words, one per line")
  private val triggers = new FlowPane(8, 6)
  private val thumbnail = new ImageView()
  thumbnail.setFitWidth(200); thumbnail.setFitHeight(200); thumbnail.setPreserveRatio(true)
  private val shared = new TextArea()
  shared.setId("model-presets-sharing"); shared.setPrefRowCount(3); shared.setEditable(false)
  private val status = W.label("No model library loaded.", "muted")
  status.setId("model-presets-status"); status.setWrapText(true)
  private val buttons = scala.collection.mutable.ArrayBuffer.empty[(Button, Boolean)]
  private var state: Option[(PresetModelIdentity, LoadedModelPresets)] = None
  private val drafts = scala.collection.mutable.Map.empty[(String, String), (String, String)]
  private var working = false
  private var updating = false
  private def controls(): Unit =
    buttons.foreach((b, needsModel) => b.setDisable(working || (needsModel && state.isEmpty)))
    Seq(model, names, name, editor, url, words).foreach(_.setDisable(working))
  private val async = new ModelPresetAsync(task => { scala.concurrent.Future(task())(using Fx.bg); () },
    task => Fx.run(task()), value => { working = value; controls() }, error => status.setText(Fx.message(error)))
  private def safe(f: => Unit): Unit = try f catch case NonFatal(e) => status.setText(Fx.message(e))
  private def button(id: String, label: String, needsModel: Boolean = true)(action: => Unit): Button =
    val result = W.button(label)(safe(action))
    result.setId("model-presets-" + id); buttons += ((result, needsModel)); result
  private def details(document: ModelPresetDocument): ModelPresetDocument = document.copy(url = url.getText.trim,
    triggerWords = words.getText.split("\\r?\\n").toVector.map(_.trim).filter(_.nonEmpty)).validate()
  private def rememberDraft(selected: String): Unit =
    if selected != null then state.foreach { (id, loaded) =>
      if loaded.document.presets.contains(selected) then drafts((id.sha256, selected)) = (name.getText, editor.getText)
    }
  private def sharingDocument(id: PresetModelIdentity, loaded: LoadedModelPresets): ModelPresetDocument =
    val text = loaded.document.presets.map { (key, saved) => key -> drafts.get((id.sha256, key)).map(_._2).getOrElse(saved) }
    details(loaded.document.copy(presets = text))
  private def show(modelId: PresetModelIdentity, loaded: LoadedModelPresets, selected: Option[String] = None): Unit =
    state = Some((modelId, loaded)); updating = true
    try
      val doc = loaded.document
      identity.setText(s"${modelId.path.getFileName} · SHA256 ${modelId.sha256}")
      url.setText(doc.url); words.setText(doc.triggerWords.mkString("\n"))
      names.getItems.setAll(doc.presets.keys.toSeq*)
      val picked = selected.filter(doc.presets.contains).getOrElse(doc.defaultPreset)
      val draft = drafts.getOrElse((modelId.sha256, picked), (picked, doc.presets(picked)))
      names.setValue(picked); name.setText(draft._1); editor.setText(draft._2)
      generationLabel.setText(if picked == doc.defaultPreset then "Generation data (default preset)" else "Generation data")
      rebuildTriggers()
      val image = service.thumbnailPath(modelId)
      thumbnail.setImage(if Files.isRegularFile(image) then new Image(image.toUri.toString, true) else null)
      shared.clear(); controls()
    finally updating = false
  private def rebuildTriggers(): Unit =
    triggers.getChildren.clear()
    words.getText.split("\\r?\\n").toVector.map(_.trim).filter(_.nonEmpty).distinct.take(1024).foreach { word =>
      val check = new CheckBox(word)
      check.setSelected(ModelPresetDocument.containsTrigger(editor.getText, word))
      check.setOnAction(_ => editor.setText(ModelPresetDocument.toggleTrigger(editor.getText, word, check.isSelected)))
      triggers.getChildren.add(check)
    }
  names.valueProperty.addListener((_, previous, selected) => if !updating && selected != null then state.foreach { (id, loaded) =>
    rememberDraft(previous)
    updating = true
    try
      val draft = drafts.getOrElse((id.sha256, selected), (selected, loaded.document.presets(selected)))
      name.setText(draft._1); editor.setText(draft._2)
      generationLabel.setText(if selected == loaded.document.defaultPreset then "Generation data (default preset)" else "Generation data")
      rebuildTriggers(); shared.clear()
    finally updating = false
  })
  editor.textProperty.addListener((_, _, _) => if !updating then
    rememberDraft(names.getValue); shared.clear()
    triggers.getChildren.asScala.collect { case c: CheckBox => c }.foreach(c => c.setSelected(ModelPresetDocument.containsTrigger(editor.getText, c.getText))))
  name.textProperty.addListener((_, _, _) => if !updating then { rememberDraft(names.getValue); shared.clear() })
  words.textProperty.addListener((_, _, _) => if !updating then rebuildTriggers())
  private def saveDocument(document: ModelPresetDocument, selected: Option[String] = None, preserveDraft: Boolean = false, replaceDrafts: Boolean = false): Unit = state.foreach { (modelId, previous) =>
    val draft = (name.getText, editor.getText)
    val submitted = Option(names.getValue)
    status.setText("Saving model presets…")
    async.run(check => { check(); service.save(modelId, document, previous.revision) }) { loaded =>
      drafts.keys.filter(_._1 == modelId.sha256).toVector.foreach { key =>
        val entry = drafts(key)
        if replaceDrafts || !document.presets.contains(key._2) || (!preserveDraft &&
            (submitted.contains(key._2) || (entry._1 == key._2 && document.presets.get(key._2).contains(entry._2)))) then drafts.remove(key)
      }
      show(modelId, loaded, selected)
      if preserveDraft then { name.setText(draft._1); editor.setText(draft._2) }
      status.setText("Model presets saved.")
    }
  }
  def loadModel(path: Path, autoFetch: Boolean = true): Unit =
    state = None; controls(); status.setText("Hashing checkpoint identity and reading presets…")
    async.run(check => {
      val modelId = service.identify(path, check)
      (modelId, service.load(modelId))
    }) { (modelId, loaded) =>
      show(modelId, loaded); status.setText("Local model presets loaded.")
      if autoFetch && loaded.document.url.isEmpty then fetch(overwrite = false)
    }
  private def fetch(overwrite: Boolean): Unit = state.foreach { (modelId, previous) =>
    val local = details(previous.document)
    status.setText("Fetching Civitai information; cancellation leaves existing presets unchanged until publication…")
    async.run(check => {
      val fetched = service.civitai.fetch(modelId.sha256, local, check)
      val document = fetched.selectDocument(overwrite)
      // Library conflicts fail before any thumbnail work; downloaded thumbnails never replace existing ones.
      check()
      val saved = service.save(modelId, document, previous.revision)
      val imageMessage = try
        fetched.thumbnail.foreach(bytes => service.saveThumbnail(modelId, bytes, false, check))
        ""
      catch case NonFatal(e) => s" Presets saved; thumbnail not saved: ${Fx.message(e)}"
      (saved, imageMessage)
    }) { (loaded, notice) =>
      if overwrite then drafts.keys.filter(_._1 == modelId.sha256).toVector.foreach(drafts.remove)
      show(modelId, loaded); status.setText("Civitai information saved." + notice)
    }
  }
  private def choose(open: Boolean, description: String, patterns: Seq[String]): Option[Path] =
    val dialog = new FileChooser()
    dialog.setTitle(description)
    dialog.getExtensionFilters.add(new FileChooser.ExtensionFilter(description, patterns*))
    if !open then state.foreach((modelId, _) => dialog.setInitialFileName(modelId.sha256.take(10) + ".json"))
    Option(if open then dialog.showOpenDialog(ctx.stage) else dialog.showSaveDialog(ctx.stage)).map(_.toPath)
  private val load = button("load", "Retrieve model info", false) {
    val path = Option(model.getValue).flatMap(ctx.services.catalog.findCheckpoint).map(c => Path.of(c.path))
      .orElse(currentModel()).getOrElse(throw new IllegalArgumentException("Select a checkpoint"))
    loadModel(path)
  }
  private val current = button("current", "Use loaded model", false) {
    loadModel(currentModel().getOrElse(throw new IllegalArgumentException("No checkpoint loaded")))
  }
  private val refresh = button("fetch", "Download and overwrite info") { fetch(overwrite = true) }
  private val save = button("save", "Save preset") { state.foreach { (_, loaded) =>
    saveDocument(details(loaded.document).save(name.getText, editor.getText), Some(name.getText))
  }}
  private val rename = button("rename", "Rename preset") { state.foreach { (_, loaded) =>
    saveDocument(details(loaded.document).save(names.getValue, editor.getText).rename(names.getValue, name.getText), Some(name.getText))
  }}
  private val delete = button("delete", "Delete preset") { state.foreach { (_, loaded) => saveDocument(details(loaded.document).delete(names.getValue)) }}
  private val default = button("default", "Save as default") { state.foreach { (_, loaded) =>
    saveDocument(details(loaded.document).save(name.getText, editor.getText).setDefault(name.getText), Some(name.getText))
  }}
  private val template = button("template", "Append template") { editor.setText(ModelPresetDocument.appendTemplate(editor.getText)) }
  private val captureButton = button("capture", "Use generated image parameters") {
    editor.setText(capture().getOrElse(throw new IllegalArgumentException("Generate or select an image first")))
  }
  private val saveInfo = button("save-info", "Save URL / trigger words") { state.foreach((_, loaded) => saveDocument(details(loaded.document), Option(names.getValue), preserveDraft = true)) }
  private val open = button("open", "Open model page") {
    val target = java.net.URI.create(url.getText.trim)
    require(Set("http", "https")(target.getScheme) && target.getHost != null && target.getUserInfo == null, "Use an HTTP(S) model URL")
    java.awt.Desktop.getDesktop.browse(target)
  }
  private val reveal = button("reveal", "Reveal presets folder") { state.foreach { (id, _) =>
    require(Files.exists(service.store.path(id.sha256)), "Save a preset first")
    java.awt.Desktop.getDesktop.open(service.directory.toFile)
  }}
  private val importButton = button("import", "Import JSON / sharing text") {
    choose(true, "Preset JSON or text", Seq("*.json", "*.txt")).foreach { path =>
      async.run(_ => service.store.importDocument(path)) { doc => saveDocument(doc, replaceDrafts = true) }
    }
  }
  private val exportButton = button("export", "Export JSON") { state.foreach { (id, loaded) =>
    val doc = sharingDocument(id, loaded)
    choose(false, "Preset JSON", Seq("*.json")).foreach { path =>
      async.run(check => { check(); service.store.exportDocument(doc, path) })(_ => status.setText("Preset JSON exported without overwriting."))
    }
  }}
  private val shareButton = button("share", "Get sharing text") { state.foreach { (id, loaded) =>
    shared.setText(sharingDocument(id, loaded).share)
    status.setText("Sharing includes unsaved generation text under existing preset names; Save/Rename applies a name change.")
  }}
  private val copy = button("copy", "Copy sharing text") {
    state.foreach((id, loaded) => shared.setText(sharingDocument(id, loaded).share))
    val content = new ClipboardContent(); content.putString(shared.getText); Clipboard.getSystemClipboard.setContent(content)
  }
  private val picture = button("thumbnail", "Replace thumbnail…") {
    choose(true, "Image", Seq("*.png", "*.jpg", "*.jpeg", "*.webp")).foreach { path => state.foreach { (modelId, loaded) =>
      val secret = ctx.services.imageScrambleSession.snapshot
      async.run(check => {
        check()
        val display = ImageFiles.readForDisplay(path, secret)
        require(!display.scrambled || (secret.nonEmpty && !display.passwordVerified.contains(false)), "Unlock the image with its correct password first")
        val image = javafx.embed.swing.SwingFXUtils.fromFXImage(display.image, null)
        val encoded = new java.io.ByteArrayOutputStream()
        javax.imageio.ImageIO.write(image, "png", encoded)
        service.saveThumbnail(modelId, encoded.toByteArray, true, () => {
          check()
          if ctx.services.imageScrambleSession.snapshot != secret then throw new CancellationException("Image session changed")
        })
      }) { path => thumbnail.setImage(new Image(path.toUri.toString, true)); status.setText("Thumbnail replaced.") }
    }}
  }
  private val sendButtons = Mode.values.map { mode => button("send-" + mode.key, "Send to " + mode.label) {
    state.foreach((id, _) => send(mode, editor.getText, id.path))
  }}
  private val cancel = W.button("Cancel") { async.cancel() }
  cancel.setId("model-presets-cancel")
  private val help = W.label("Retrieve model info and Use loaded model contact Civitai when the saved URL is empty, sending the full checkpoint SHA256 hash (never model bytes). " +
    "Download and overwrite info explicitly imports remote sharing presets; automatic retrieval only updates URL/triggers and a missing thumbnail. " +
    "Unsaved generation text is retained per preset and included in sharing/export under existing names; Save/Rename applies the name field. " +
    "Downloaded thumbnails preserve an existing image. Manual replacement writes a readable 300px PNG beside the checkpoint, including when the selected image is protected. Infotext and sharing files are readable text.", "muted")
  help.setWrapText(true)
  private val content = new VBox(10, W.label("Model Preset Manager", "section-title"), help,
    W.hbox(8, W.hgrow(model), load, current), identity, W.hbox(8, W.hgrow(url), open, saveInfo),
    W.hbox(8, thumbnail, new VBox(8, refresh, picture, reveal)), W.label("Trigger words"), words, triggers,
    W.hbox(8, W.hgrow(names), W.hgrow(name)), generationLabel, editor, W.hbox(8, save, default, rename, delete),
    W.hbox(8, template, captureButton), W.hbox(8, sendButtons*), W.hbox(8, importButton, exportButton, shareButton, copy),
    shared, W.hbox(8, cancel, status))
  content.setPadding(new Insets(14)); content.setMinWidth(0); VBox.setVgrow(editor, Priority.ALWAYS)
  val node = new ScrollPane(content)
  node.setFitToWidth(true); node.setId("model-presets-pane")
  controls()
  def dispose(): Unit = async.dispose()
