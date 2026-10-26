package sdfx.ui.view

import javafx.animation.{KeyFrame, PauseTransition, Timeline}
import javafx.beans.property.{SimpleBooleanProperty, SimpleStringProperty}
import javafx.collections.FXCollections
import javafx.geometry.{Insets, Orientation}
import javafx.scene.control.*
import javafx.scene.control.cell.TextFieldTableCell
import javafx.scene.input.{Clipboard, ClipboardContent, KeyCode, KeyEvent}
import javafx.scene.layout.{BorderPane, FlowPane, Priority, VBox}
import javafx.stage.FileChooser
import javafx.util.Duration
import sdfx.core.model.*
import sdfx.ui.Fx
import java.nio.file.{Files, Path, StandardCopyOption}
import java.util.UUID
import scala.jdk.CollectionConverters.*
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.*

/** Native spreadsheet and extended preset browser. Disk writes precede live-list publication. */
final class StylesEditorTab(ctx: UiContext):
  val busy = new SimpleBooleanProperty(false)
  val scope = new ComboBox[String](FXCollections.observableArrayList("Prompt styles", "Extended presets"))
  scope.setValue("Prompt styles")
  scope.setId("styles-scope")
  val category = new ComboBox[String]()
  category.setValue("Master styles")
  category.setId("styles-category")
  val search = new TextField()
  search.setPromptText("Search every column…")
  search.setId("styles-search")
  val filterMode = new ComboBox[String](FXCollections.observableArrayList("Exact match", "Case insensitive", "regex"))
  filterMode.setValue("Case insensitive")
  val target = new CheckBox("Current model + VAE")
  target.setId("styles-target")
  val autosort = new CheckBox("Autosort")
  private val status = W.label("Loading styles…", "muted")
  status.setId("styles-status")
  status.setWrapText(true)
  private val rows = FXCollections.observableArrayList[StyleEntry]()
  val table = new TableView[StyleEntry](rows)
  table.setId("styles-table")
  table.setEditable(true)
  table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN)
  table.getSelectionModel.setSelectionMode(SelectionMode.MULTIPLE)
  table.setPlaceholder(W.label("No matching styles. Create a style or clear the filter.", "muted"))
  private var current: Option[StyleDocument] = None
  def document: Option[StyleDocument] = current
  private var updating = false
  private var selectedId: Option[String] = None
  private var writeInFlight: Option[Future[StyleDocument]] = None
  private val recoveryBlocked = new SimpleBooleanProperty(false)
  private var recoveredDraft: Option[Path] = None
  private var pendingDraft: Option[StyleDocument] = None
  def lastRecoveryPath: Option[Path] = recoveredDraft
  private class RecoveredDraft(val path: Path, cause: Throwable)
      extends RuntimeException(s"${Fx.message(cause)}; draft retained at $path", cause)
  private var disposed = false
  private var lastSaved = Map.empty[String, String]
  val name = new TextField()
  val positive = new TextArea()
  val negative = new TextArea()
  val notes = new TextArea()
  val order = new TextField()
  val checkpoint = new TextField()
  val vae = new TextField()
  val width = new TextField()
  val height = new TextField()
  Seq("styles-name" -> name, "styles-positive" -> positive, "styles-negative" -> negative, "styles-notes" -> notes,
    "styles-sort" -> order, "styles-checkpoint" -> checkpoint, "styles-vae" -> vae, "styles-width" -> width, "styles-height" -> height)
    .foreach((id, node) => node.setId(id))
  Seq(positive, negative, notes).foreach { area => area.setWrapText(true); area.setPrefRowCount(3) }
  notes.setPrefRowCount(2)
  private val editDelay = new PauseTransition(Duration.millis(500))
  private val encryption = new CheckBox("Encrypt future backups (.aes)")
  encryption.setId("styles-backup-encrypt")
  private val password = new PasswordField()
  password.setPromptText("Backup password — never saved")
  password.setId("styles-backup-password")
  private val backupChoice = new ComboBox[Path]()
  backupChoice.setId("styles-backups")
  backupChoice.setPrefWidth(260)
  backupChoice.setMaxWidth(320)
  private def backupCell(): ListCell[Path] = new ListCell[Path]:
    override def updateItem(path: Path, empty: Boolean): Unit =
      super.updateItem(path, empty)
      setText(if empty || path == null then "Select a backup" else path.getFileName.toString)
  backupChoice.setButtonCell(backupCell())
  backupChoice.setCellFactory(_ => backupCell())
  private def isExtended = scope.getValue == "Extended presets"
  private def selectedIds = table.getSelectionModel.getSelectedItems.asScala.map(_.id).toVector.distinct
  private def categoryName = Option(category.getValue).filter(_ != "Master styles")
  private def keyPassword: Option[String] =
    if encryption.isSelected then
      require(password.getText.nonEmpty, "Enter an encryption password")
      Some(password.getText)
    else None
  private def button(label: String, id: String)(body: => Unit): Button =
    val b = W.button(label) { if !busy.get && (!recoveryBlocked.get || id == "styles-save-changes") then try body catch case e: Exception => failure(e) }
    b.setId(id)
    b.disableProperty.bind(if id == "styles-save-changes" then busy else busy.or(recoveryBlocked))
    b
  private def failure(e: Throwable): Unit =
    status.setText(s"Not saved: ${Fx.message(e)}")
    status.getStyleClass.add("problem")
    ctx.error(Fx.message(e))
  private def publish(doc: StyleDocument, prefer: Option[String] = selectedId): Unit =
    updating = true
    current = Some(doc)
    refreshRows()
    prefer.flatMap(id => rows.asScala.find(_.id == id)).orElse(rows.asScala.headOption).foreach(table.getSelectionModel.select)
    updating = false
    showSelection()
    status.getStyleClass.remove("problem")
    status.setText(s"${doc.promptStyles.size} styles · ${doc.path.toAbsolutePath} · Autosaved")
    ctx.loadStyles()
  private def refreshRows(): Unit = current.foreach { doc =>
    doc.filtered(search.getText, filterMode.getValue) match
      case Left(error) => status.setText(s"Invalid regex: $error"); search.setStyle("-fx-border-color: #e66;")
      case Right(found) =>
        search.setStyle("")
        val p = ctx.currentGeneration()
        val visible = if isExtended && target.isSelected then found.filter(e =>
          doc.value(e, "sd_model_checkpoint") == p.checkpoint.getOrElse("") &&
            ExtendedStyle.normalizeVae(doc.value(e, "sd_vae")) == ExtendedStyle.normalizeVae(p.vae.getOrElse("Automatic"))) else found
        rows.setAll(visible*)
        if status.getText.startsWith("Invalid regex:") then status.setText(s"${visible.size} matching styles")
  }
  def reload(): Unit =
    if busy.get then return
    if recoveryBlocked.get then
      failure(new IllegalStateException("The unsaved draft could not be preserved. Retry Save before leaving this file."))
      return
    editDelay.stop()
    busy.set(true)
    val extended = isExtended
    val cat = categoryName
    val previousDoc = current
    current = None
    selectedId = None
    updating = true
    rows.clear()
    updating = false
    showSelection()
    Fx.async {
      val library = ctx.styleLibrary
      val nextPath = if extended then library.extendedFile else cat.fold(library.master)(library.categoryFile)
      previousDoc.filter(d => d.path.toAbsolutePath.normalize() != nextPath.toAbsolutePath.normalize() && isCategoryDocument(d)).foreach { previous =>
        library.saveCategory(previous.path.getFileName.toString.stripSuffix(".csv"), previous)
        library.pruneEmptyCategories()
      }
      val doc = if extended then library.loadExtended() else cat.fold(library.load())(library.loadCategory)
      (doc, library.categories(), library.backups())
    } { (doc, categories, backups) =>
      busy.set(false)
      updating = true
      category.getItems.setAll(("Master styles" +: categories)*)
      category.setValue(cat.getOrElse("Master styles"))
      backupChoice.getItems.setAll(backups*)
      updating = false
      publish(doc)
    } { e => busy.set(false); failure(e) }
  private def save(changed: StyleDocument, prefer: Option[String] = selectedId): Unit =
    if busy.get then return
    pendingDraft = Some(changed)
    val extended = isExtended
    val cat = categoryName
    busy.set(true)
    val future = Future {
      try
        if extended || cat.isEmpty then ctx.styleLibrary.save(changed) else ctx.styleLibrary.saveCategory(cat.get, changed)
      catch case original: Exception =>
        val recovery = try ctx.styleLibrary.preserveDraft(changed)
        catch case preservation: Exception =>
          original.addSuppressed(preservation)
          throw original
        throw new RecoveredDraft(recovery, original)
    }(using Fx.bg)
    writeInFlight = Some(future)
    Fx.onFx(future) { saved =>
      writeInFlight = None
      pendingDraft = None
      recoveryBlocked.set(false)
      busy.set(false)
      if !disposed then publish(saved, prefer)
    } { e =>
      writeInFlight = None
      e match
        case recovered: RecoveredDraft =>
          recoveredDraft = Some(recovered.path)
          pendingDraft = None
          recoveryBlocked.set(false)
          // Navigation may now replace the widgets: their exact proposed document is safely on disk.
          lastSaved = detailValues
        case _ => recoveryBlocked.set(true)
      busy.set(false)
      if !disposed then failure(e)
    }
  private def isCategoryDocument(doc: StyleDocument): Boolean =
    doc.path.toAbsolutePath.normalize().getParent == ctx.styleLibrary.categoryFile("category").getParent
  private def edit(id: String, fields: Map[String, String]): Unit = current.foreach { doc =>
    if isExtended then
      Seq("width", "height").foreach(k => fields.get(k).foreach { value =>
        val n = value.toIntOption.getOrElse(throw new IllegalArgumentException(s"$k must be an integer"))
        require(n >= 64 && n <= 4096 && n % 8 == 0, s"$k must be a multiple of 8 in 64..4096")
      })
    val selected = doc.entries.find(_.id == id).getOrElse(throw new IllegalStateException("Selected style changed; reload"))
    val changes = fields.filter((key, value) => doc.value(selected, key) != value)
    val changed = doc.edit(id, changes, uniqueNames = !isExtended)
    save(if autosort.isSelected then changed.numericSort else changed, Some(id))
  }
  private def showSelection(): Unit =
    if recoveryBlocked.get then
      updating = true
      selectedId.flatMap(id => rows.asScala.find(_.id == id)).foreach(table.getSelectionModel.select)
      updating = false
      return
    val row = Option(table.getSelectionModel.getSelectedItem)
    val id = row.map(_.id)
    editDelay.stop()
    selectedId = id
    updating = true
    def value(key: String) = row.flatMap(r => current.map(_.value(r, key))).getOrElse("")
    name.setText(value("name")); positive.setText(value("prompt")); negative.setText(value("negative_prompt"))
    notes.setText(value("notes")); order.setText(value("sort")); checkpoint.setText(value("sd_model_checkpoint"))
    vae.setText(value("sd_vae")); width.setText(value("width")); height.setText(value("height"))
    lastSaved = detailValues
    updating = false
  private def detailValues: Map[String, String] =
    val common = Map("name" -> name.getText, "prompt" -> positive.getText, "negative_prompt" -> negative.getText, "notes" -> notes.getText)
    if isExtended then common ++ Map("sd_model_checkpoint" -> checkpoint.getText, "sd_vae" -> vae.getText, "width" -> width.getText, "height" -> height.getText)
    else common + ("sort" -> order.getText)
  private def saveDetails(force: Boolean = false): Unit =
    if !updating && !busy.get && detailValues != lastSaved then selectedId.foreach(id =>
      try edit(id, detailValues) catch case e: Exception => failure(e))
    else if force && !busy.get && pendingDraft.nonEmpty then save(pendingDraft.get)
    else if force && !busy.get then current.foreach(save(_))
  editDelay.setOnFinished(_ => saveDetails())
  Seq(name, order, checkpoint, vae, width, height).foreach(_.textProperty.addListener((_, _, _) => if !updating then editDelay.playFromStart()))
  Seq(positive, negative, notes).foreach(_.textProperty.addListener((_, _, _) => if !updating then editDelay.playFromStart()))
  // Commit when leaving a detail field, before a row/scope switch could replace its draft.
  Seq(name, order, checkpoint, vae, width, height).foreach(_.focusedProperty.addListener((_, _, focused) => if !focused then saveDetails()))
  Seq(positive, negative, notes).foreach(_.focusedProperty.addListener((_, _, focused) => if !focused then saveDetails()))
  table.getSelectionModel.selectedItemProperty.addListener((_, _, _) => if !updating then showSelection())
  table.disableProperty.bind(busy.or(recoveryBlocked))
  private def addColumn(title: String, field: String, size: Double): Unit =
    val column = new TableColumn[StyleEntry, String](title)
    column.setCellValueFactory(c => new SimpleStringProperty(current.map(_.value(c.getValue, field)).getOrElse("")))
    column.setCellFactory(TextFieldTableCell.forTableColumn[StyleEntry]())
    column.setOnEditCommit(e => try edit(e.getRowValue.id, Map(field -> e.getNewValue)) catch case error: Exception => failure(error))
    column.setPrefWidth(size)
    column.setMinWidth(65)
    column.setSortable(false)
    column.setUserData(field)
    column.setEditable(field != "id")
    table.getColumns.add(column)
  addColumn("Sort", "sort", 65); addColumn("Name", "name", 140); addColumn("Prompt", "prompt", 220)
  addColumn("Negative", "negative_prompt", 180); addColumn("Notes", "notes", 150)
  addColumn("Model", "sd_model_checkpoint", 160); addColumn("VAE", "sd_vae", 130)
  addColumn("Width", "width", 75); addColumn("Height", "height", 75); addColumn("UUID", "id", 180)
  private def confirm(title: String, text: String): Boolean =
    val dialog = new Alert(Alert.AlertType.CONFIRMATION, text, ButtonType.CANCEL, ButtonType.OK)
    dialog.initOwner(ctx.stage)
    dialog.setHeaderText(title)
    dialog.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.OK
  private def ask(title: String, initial: String = ""): Option[String] =
    val dialog = new TextInputDialog(initial)
    dialog.initOwner(ctx.stage); dialog.setTitle(title); dialog.setHeaderText(title)
    val result = dialog.showAndWait()
    if result.isPresent then Option(result.get.trim).filter(_.nonEmpty) else None
  private def create(fromPrompts: Boolean): Unit = ask(if fromPrompts then "Save current prompts" else "New style", "New style").foreach { label =>
    if isExtended then
      val params = ctx.currentGeneration()
      busy.set(true)
      Fx.async(ctx.styleLibrary.createExtended(label, params)) { _ => busy.set(false); reload() } { e => busy.set(false); failure(e) }
    else current.foreach { doc =>
      val p = ctx.currentGeneration()
      val changed = doc.append(Map("name" -> label, "prompt" -> (if fromPrompts then p.prompt else ""),
        "negative_prompt" -> (if fromPrompts then p.negativePrompt else "")))
      save(changed, changed.entries.lastOption.map(_.id))
    }
  }
  private def duplicate(): Unit = current.foreach { doc =>
    var next = doc
    selectedIds.foreach { id => doc.entries.find(_.id == id).foreach { row =>
      val fields = doc.fields(row)
      next = next.append(if isExtended then fields.updated("id", UUID.randomUUID().toString) else fields, uniqueNames = !isExtended)
    }}
    save(next)
  }
  private def delete(): Unit = current.foreach { doc =>
    if selectedIds.nonEmpty && confirm("Delete selected styles?", selectedIds.flatMap(id => doc.entries.find(_.id == id).map(doc.value(_, "name"))).mkString("\n")) then
      save(doc.delete(selectedIds), None)
  }
  private def move(delta: Int): Unit = current.foreach { doc =>
    val ids = doc.entries.map(_.id).toBuffer
    val selected = selectedIds.toSet
    val indices = if delta < 0 then ids.indices else ids.indices.reverse
    indices.foreach { i =>
      val j = i + delta
      if j >= 0 && j < ids.size && selected(ids(i)) && !selected(ids(j)) then
        val swap = ids(i); ids(i) = ids(j); ids(j) = swap
    }
    save(doc.reorder(ids.toSeq))
  }
  private def moveCategory(): Unit = if !isExtended then ask("Move selected styles to category (prefix)").foreach { dest =>
    current.foreach { doc =>
      val ids = selectedIds
      if ids.nonEmpty then
        busy.set(true)
        Fx.async(ctx.styleLibrary.moveToCategory(doc, ids, dest)) { _ => busy.set(false); reload() } { e => busy.set(false); failure(e) }
    }
  }
  private def applySelected(): Unit = current.foreach { doc =>
    Option(table.getSelectionModel.getSelectedItem).foreach { row =>
      if isExtended && ctx.extendedStyleApplying.get then ctx.info("An extended style is already being applied")
      else if isExtended then
        val id = doc.value(row, "id")
        busy.set(true)
        Fx.async(ctx.styleLibrary.extended().find(_.id == id).getOrElse(throw new IllegalStateException("Preset changed; refresh before applying"))) {
          preset => busy.set(false); ctx.applyExtendedStyle(preset)
        } { e => busy.set(false); failure(e) }
      else ctx.applyPromptStyle(PromptStyle(doc.value(row, "name"), doc.value(row, "prompt"), doc.value(row, "negative_prompt")))
    }
  }
  private val findText = new TextField()
  private val replacement = new TextField()
  findText.setPromptText("Find literal text"); replacement.setPromptText("Replace with")
  findText.setId("styles-find"); replacement.setId("styles-replacement")
  private val replaceButton = button("Replace all", "styles-replace") {
    current.foreach(doc => if findText.getText.nonEmpty && confirm("Replace across this style file?", "All columns and hidden rows are included.") then
      save(doc.replace(findText.getText, replacement.getText, uniqueNames = !isExtended, protectedFields = if isExtended then Set("id") else Set.empty)))
  }
  private def chooser(title: String): FileChooser =
    val c = new FileChooser(); c.setTitle(title)
    c.getExtensionFilters.add(new FileChooser.ExtensionFilter("Style CSV / AES Crypt backup", "*.csv", "*.aes")); c
  private def restore(path: Path): Unit =
    if confirm("Restore prompt styles and categories?", s"Replace styles.csv and its category mirrors with ${path.getFileName}? The current prompt-style master is backed up first. Extended presets are separate.") then
      val key = Option(password.getText).filter(_.nonEmpty)
      busy.set(true)
      Fx.async(ctx.styleLibrary.restore(path, key)) { _ =>
        busy.set(false)
        current = None
        selectedId = None
        updating = true
        scope.setValue("Prompt styles"); category.setValue("Master styles")
        updating = false
        reload()
      } { e => busy.set(false); failure(e) }
  private val backupButton = button("Back up now", "styles-backup-now") {
    val key = keyPassword
    busy.set(true)
    Fx.async(ctx.styleLibrary.backupNow(key)) { path => busy.set(false); status.setText(s"Backup saved: ${path.getFileName}"); reload() } { e => busy.set(false); failure(e) }
  }
  private val restoreButton = button("Restore", "styles-restore-backup") { Option(backupChoice.getValue).foreach(restore) }
  private val importButton = button("Import / restore…", "styles-import") {
    Option(chooser("Restore styles from CSV or .aes").showOpenDialog(ctx.stage)).foreach(f => restore(f.toPath))
  }
  private val exportButton = button("Export…", "styles-export") {
    val source = Option(backupChoice.getValue).getOrElse(ctx.styleLibrary.master)
    val c = chooser("Export selected backup, or active master")
    c.setInitialFileName(source.getFileName.toString)
    Option(c.showSaveDialog(ctx.stage)).foreach { file =>
      Fx.async(Files.copy(source, file.toPath, StandardCopyOption.REPLACE_EXISTING))(_ => status.setText("Exported"))(failure)
    }
  }
  private val addCategory = button("New category…", "styles-new-category") {
    ask("New category").foreach { label =>
      busy.set(true)
      Fx.async(ctx.styleLibrary.createCategory(label)) { _ =>
        busy.set(false)
        updating = true
        if !category.getItems.contains(label) then category.getItems.add(label)
        category.setValue(label)
        updating = false
        reload()
      } { e => busy.set(false); failure(e) }
    }
  }
  private val applyButton = button("Apply", "styles-apply")(applySelected())
  applyButton.disableProperty.unbind()
  applyButton.disableProperty.bind(busy.or(ctx.extendedStyleApplying).or(recoveryBlocked))
  private val toolbar = new FlowPane(7, 7, scope, category,
    button("New", "styles-new")(create(false)), button("From prompts…", "styles-from-prompts")(create(true)),
    button("Duplicate", "styles-duplicate")(duplicate()), button("Delete", "styles-delete")(delete()),
    button("↑", "styles-up")(move(-1)), button("↓", "styles-down")(move(1)),
    button("Move to…", "styles-move-category")(moveCategory()), addCategory,
    applyButton, button("Reload", "styles-reload") { target.setSelected(false); reload() })
  private val filters = new FlowPane(7, 7, search, filterMode, target, autosort, findText, replacement, replaceButton)
  filters.disableProperty.bind(recoveryBlocked)
  private val details = W.form().add("Name", name).add("Sort", order)
    .add("Positive prompt", positive).add("Negative prompt", negative).add("Notes", notes)
    .add("Checkpoint", checkpoint).add("VAE", vae).add("Width", width).add("Height", height)
  // Keep the exact pending document intact if neither the active file nor recovery copy can be written.
  // The retry button is outside this grid, so the user can retry once storage is available again.
  details.grid.disableProperty.bind(recoveryBlocked)
  private val detailBox = new VBox(7, W.label("Selected style", "card-title"), details.grid,
    button("Save / merge changes", "styles-save-changes")(saveDetails(force = true)))
  detailBox.setPadding(new Insets(10))
  detailBox.disableProperty.bind(busy)
  private val detailScroll = new ScrollPane(detailBox)
  detailScroll.setFitToWidth(true)
  private val split = new SplitPane(table, detailScroll)
  split.setDividerPositions(.62)
  split.setMinHeight(250)
  private val backupHelp = W.label("Changed prompt styles are backed up every ten minutes; the latest 24 are kept. Encryption applies to future backups. Active CSVs stay readable. Extended presets are stored separately.", "muted")
  backupHelp.setWrapText(true)
  private val backups = new TitledPane("Backups · prompt styles and categories", new VBox(7, backupHelp, new FlowPane(7, 7,
    encryption, password, backupChoice, backupButton, restoreButton, importButton, exportButton)))
  backups.setExpanded(false)
  backups.setAnimated(false)
  backups.setId("styles-backup-panel")
  private val hint = W.label("Edits autosave after a short pause or on leaving a field. Double-click cells to edit; Ctrl/Cmd-X/C/V cuts, copies or pastes a cell. Ctrl/Cmd-click rows to select several. D duplicates; Delete removes rows; M moves categories. Duplicate prompt-style names gain x; extended presets keep separate UUIDs and restore prompts, size, model and VAE only.", "muted")
  hint.setWrapText(true)
  val node = new VBox(9, W.label("Styles", "page-title"), hint, toolbar, filters, split, backups, status)
  node.setPadding(new Insets(12))
  node.setId("styles-editor")
  VBox.setVgrow(split, Priority.ALWAYS)
  val view = new ScrollPane(node)
  view.setFitToWidth(true)
  view.setFitToHeight(true)
  view.setId("styles-editor-scroll")
  node.widthProperty.addListener((_, _, w) => split.setOrientation(if w.doubleValue < 820 then Orientation.VERTICAL else Orientation.HORIZONTAL))
  search.textProperty.addListener((_, _, _) => refreshRows())
  filterMode.valueProperty.addListener((_, _, _) => refreshRows())
  target.selectedProperty.addListener((_, _, _) => refreshRows())
  autosort.selectedProperty.addListener((_, _, on) => if on && !busy.get then current.foreach(d => save(d.numericSort)))
  scope.valueProperty.addListener((_, _, _) =>
    updateScope()
    if !updating then { selectedId = None; reload() })
  category.valueProperty.addListener((_, _, _) => if !updating then { selectedId = None; reload() })
  table.addEventFilter(KeyEvent.KEY_PRESSED, (event: KeyEvent) =>
    if table.getEditingCell == null && !busy.get then
      if event.isShortcutDown && Set(KeyCode.C, KeyCode.X, KeyCode.V).contains(event.getCode) then
        val position = table.getFocusModel.getFocusedCell
        val row = Option(table.getSelectionModel.getSelectedItem)
        val column = Option(position.getTableColumn).filter(_.isVisible).getOrElse(table.getColumns.get(1))
        val field = column.getUserData.toString
        row.foreach { r => current.foreach { doc =>
          if event.getCode == KeyCode.V && field != "id" then
            Option(Clipboard.getSystemClipboard.getString).foreach(text => edit(r.id, Map(field -> text)))
          else if event.getCode != KeyCode.V then
            val content = new ClipboardContent()
            content.putString(doc.value(r, field))
            Clipboard.getSystemClipboard.setContent(content)
            if event.getCode == KeyCode.X && field != "id" then edit(r.id, Map(field -> ""))
        }}
        event.consume()
      else if event.getCode == KeyCode.DELETE || event.getCode == KeyCode.BACK_SPACE then { delete(); event.consume() }
      else if event.getCode == KeyCode.D && !event.isShortcutDown then { duplicate(); event.consume() }
      else if event.getCode == KeyCode.M && !event.isShortcutDown then { moveCategory(); event.consume() }
  )
  private val timer = new Timeline(new KeyFrame(Duration.seconds(30), _ =>
    if !busy.get then
      try
        val key = keyPassword
        Fx.async(ctx.styleLibrary.backupIfDue(key))(_ => ())(e => status.setText(s"Backup failed: ${Fx.message(e)}"))
      catch case e: Exception => status.setText(s"Backup pending: ${e.getMessage}")
  ))
  timer.setCycleCount(javafx.animation.Animation.INDEFINITE)
  timer.play()
  scope.disableProperty.bind(busy.or(recoveryBlocked))
  category.disableProperty.bind(busy.or(recoveryBlocked).or(scope.valueProperty.isEqualTo("Extended presets")))
  target.disableProperty.bind(busy.or(scope.valueProperty.isNotEqualTo("Extended presets")))
  addCategory.disableProperty.unbind()
  addCategory.disableProperty.bind(busy.or(recoveryBlocked).or(scope.valueProperty.isEqualTo("Extended presets")))
  private def updateScope(): Unit =
    val extendedFields = Set("sd_model_checkpoint", "sd_vae", "width", "height", "id")
    table.getColumns.asScala.foreach { c =>
      val field = c.getUserData.toString
      c.setVisible(if isExtended then field == "name" || extendedFields(field) else !extendedFields(field))
    }
    details.grid.getChildren.asScala.foreach { n =>
      val row = Option(javafx.scene.layout.GridPane.getRowIndex(n)).map(_.intValue).getOrElse(0)
      val visible = if row >= 5 then isExtended else if row == 1 then !isExtended else true
      n.setVisible(visible); n.setManaged(visible)
    }
  updateScope()
  reload()
  def dispose(): Unit =
    if disposed then return
    disposed = true
    timer.stop(); editDelay.stop(); password.clear()
    // Closing must not drop the last half-second of typing. I/O still runs on the worker pool; shutdown waits
    // for this small write before the JVM exits. A conflict preserves a separate recovery CSV, never overwrites.
    val detailDraft = for doc <- current; id <- selectedId if detailValues != lastSaved yield
      val row = doc.entries.find(_.id == id).get
      doc.edit(id, detailValues.filter((k, v) => doc.value(row, k) != v), uniqueNames = !isExtended)
    val draft = detailDraft.orElse(pendingDraft)
    try
      writeInFlight match
        case Some(future) => Await.result(future, 10.seconds)
        case None => draft.orElse(current.filter(isCategoryDocument)).foreach { doc =>
          val extended = isExtended
          val cat = categoryName
          Await.result(Future {
            if extended || cat.isEmpty then ctx.styleLibrary.save(doc) else ctx.styleLibrary.saveCategory(cat.get, doc)
          }(using Fx.bg), 10.seconds)
        }
    catch case e: Exception =>
      System.err.println(s"[sdfx] styles could not be saved on close: ${Fx.message(e)}")
      draft.foreach { doc =>
        try
          val path = Await.result(Future(ctx.styleLibrary.preserveDraft(doc))(using Fx.bg), 10.seconds)
          System.err.println(s"[sdfx] styles could not be saved (${Fx.message(e)}); draft retained at $path")
        catch case failure: Exception => System.err.println(s"[sdfx] styles draft could not be preserved: ${Fx.message(failure)}")
      }
  def leave(): Unit =
    if !busy.get then
      val categoryDoc = current.filter(isCategoryDocument)
      Fx.async {
        categoryDoc.foreach(d => ctx.styleLibrary.saveCategory(d.path.getFileName.toString.stripSuffix(".csv"), d))
        ctx.styleLibrary.pruneEmptyCategories()
      }(_ => ctx.loadStyles())(e => { status.setText(Fx.message(e)); ctx.error(Fx.message(e)) })
