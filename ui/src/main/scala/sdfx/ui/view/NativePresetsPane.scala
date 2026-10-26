package sdfx.ui.view

import javafx.beans.property.SimpleStringProperty
import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.{Node, Parent}
import javafx.scene.control.*
import javafx.scene.layout.{FlowPane, Priority, VBox}
import javafx.stage.Window

import java.nio.file.Path
import java.nio.charset.StandardCharsets.UTF_8
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

import sdfx.core.presets.*
import sdfx.ui.Fx
import sdfx.ui.presets.{NativePresetFileDialogs, NativePresetsController, NativePresetsSession}
import sdfx.ui.vm.Mode

/** The one owned native editor for Config Presets and Preset Utilities. All form
  * reads/writes run on FX; only the session owns authoritative file operations.
  */
final class NativePresetsPane private[ui] (
    ctx: UiContext,
    session: NativePresetsSession,
    controllers: Map[Mode, NativePresetsController],
    dialogs: NativePresetFileDialogs
):
  def this(ctx: UiContext, session: NativePresetsSession,
           controllers: Map[Mode, NativePresetsController]) =
    this(ctx, session, controllers, NativePresetFileDialogs.system)

  private val limits = PresetLimits()
  private var loaded: Option[LoadedPresets] = None
  private var everLoaded = false
  private var seedAttempted = false
  private var readOnly = false
  private var awaitingReload = false
  private var closed = false
  private var quiet = false
  private var mode = Mode.Txt2Img
  private var selectedName: Option[String] = None
  private var pendingDeleteName: Option[String] = None
  private var checkedFields = Set.empty[FieldKey]
  private var importRevision: Option[PresetRevision] = None
  private var importTicket = 0L
  private final case class ExportReview(
    libraryRevision: PresetRevision,
    targetRevision: PresetRevision,
    path: Path,
    format: String,
    mode: Option[PresetMode],
    collection: PresetCollection,
    library: PresetLibrary,
    profileFormat: String,
    profileMode: Option[PresetMode],
    catalogs: Map[PresetMode, PresetSchema]
  )
  private var exportReview: Option[ExportReview] = None
  private var exportTicket = 0L
  private var importedAdditional = Vector.empty[(String, Option[String])]
  private var importedTracked = Vector.empty[String]
  private final case class AdditionalRow(label: String, componentId: Option[String],
                                         target: Option[FieldKey], note: String)
  private final case class AdditionalReview(rows: Vector[AdditionalRow],
                                            bindings: Vector[SourceBinding],
                                            trackable: Set[FieldKey],
                                            sourceMetadata: RawJson, ownershipMetadata: RawJson)
  private var additionalReview: Option[AdditionalReview] = None
  private var sourceCatalog = Map.empty[PresetMode, PresetSchema]
  private var invalidDraft: Option[String] = None

  /** Display-only preview: never changes the complete atom retained in the library. */
  private def previewValue(raw: String, limit: Int): String =
    val text = Option(raw).getOrElse("")
    val out = new StringBuilder
    var offset = 0
    var truncated = false
    while offset < text.length && out.length < limit && !truncated do
      val cp = text.codePointAt(offset)
      val part = cp match
        case 10 => "↵"
        case 13 => "↵"
        case 9 => "⇥"
        case _ => new String(Character.toChars(cp))
      if out.length + part.length <= limit then
        out.append(part)
        offset += Character.charCount(cp)
      else truncated = true
    if truncated || offset < text.length then out.append(" [truncated]")
    out.toString

  private def atomPreview(value: Option[PresetAtom], limit: Int): String = value match
    case None => "unavailable"
    case Some(PresetAtom.Text(v)) => previewValue(v, limit)
    case Some(PresetAtom.Integer(v)) => previewValue(v.toString, limit)
    case Some(PresetAtom.Decimal(v)) => previewValue(v.toString, limit)
    case Some(PresetAtom.Flag(v)) => v.toString

  private def aliasesPreview(ids: Vector[String], limit: Int = 160): String =
    val shown = ids.iterator.take(5).map(id => previewValue(id, 64)).mkString(", ")
    val base = previewValue(shown, limit)
    if ids.size > 5 then s"$base [truncated; ${ids.size} aliases total]" else base

  private def id[A <: Node](node: A, name: String): A = { node.setId(name); node }
  private def label(text: String, name: String): Label = id(new Label(text), name)
  private def button(text: String, name: String)(action: => Unit): Button =
    val b = id(new Button(text), name)
    b.setOnAction(_ => action)
    b
  private def boxed(nodes: Node*): FlowPane =
    val box = new FlowPane(8, 6)
    box.getChildren.addAll(nodes*)
    box
  private def message(e: PresetError): String =
    e.issues.take(8).map(i => s"${i.code} ${i.path}: ${i.message}").mkString("; ").take(4096)
  private def fail(text: String): Unit =
    status.setText(text.take(4096))
  private def fail(e: PresetError): Unit = fail(message(e))
  private def current: Option[PresetCollection] =
    loaded.flatMap(_.library.collections.find(_.id == collectionId))
  private def collectionId: String =
    if profile.getValue == "Utilities" then "utilities"
    else if mode == Mode.Txt2Img then "config-txt2img" else "config-img2img"
  private def sourceMode: Option[PresetMode] =
    if profile.getValue == "Utilities" then None
    else Some(if mode == Mode.Txt2Img then PresetMode.Txt2Img else PresetMode.Img2Img)
  private def sourceFormat: String =
    if profile.getValue == "Utilities" then "preset-utilities" else "config-presets"
  private def owner: Window = Option(node.getScene).map(_.getWindow).getOrElse(ctx.stage)
  private def selected: Option[PresetRecord] =
    selectedName.flatMap(name => current.flatMap(_.presets.find(_.name == name)))
  private def controller = controllers(mode)
  private def inspect(): Either[PresetError, (PresetSchema, FieldSnapshot)] = controller.inspect()

  private val profile = id(new ComboBox[String](FXCollections.observableArrayList("Config Presets", "Utilities")), "presets-profile")
  profile.setValue("Config Presets")
  private val destination = label("txt2img", "presets-destination")
  private val modeChoice = new ComboBox[Mode](FXCollections.observableArrayList(Mode.values*))
  modeChoice.setValue(mode)
  private val collection = label("config-txt2img", "presets-collection")
  private val status = label("Loading preset library…", "presets-status")
  status.setWrapText(true)
  private val invalid = label("", "presets-invalid-draft")
  invalid.setWrapText(true)

  private val configSelect = id(new ComboBox[String](), "presets-select")
  private val quickSelect = id(new ComboBox[String](), "presets-utilities-quick-select")
  private val detailedSelect = id(new ComboBox[String](), "presets-utilities-detailed-select")
  private val quickStack = id(new CheckBox("Stackable over current values"), "presets-utilities-quick-stackable")
  private val detailedStack = id(new CheckBox("Stackable over current values"), "presets-utilities-detailed-stackable")
  quickStack.setSelected(true)
  detailedStack.setSelected(true)
  private val quickName = id(new TextField(), "presets-quick-name")
  quickName.setPromptText("New Utilities preset name")
  private val detailedName = id(new TextField(), "presets-name")
  detailedName.setPromptText("Preset name")
  private val quickSavePreview = id(new TextArea(), "presets-quick-save-preview")
  quickSavePreview.setEditable(false)
  quickSavePreview.setWrapText(true)
  quickSavePreview.setPrefRowCount(3)
  private val savePreview = id(new TextArea(), "presets-save-preview")
  savePreview.setEditable(false)
  savePreview.setWrapText(true)
  savePreview.setPrefRowCount(3)
  private val applyButton = button("Apply", "presets-apply") { applySelected(detailed = false) }
  private val reapplyButton = button("Reapply", "presets-reapply") { applySelected(detailed = activeDetailed) }
  private val quickSaveButton = button("Save all tracked", "presets-quick-save") { savePreset(quickName.getText, quick = true) }
  private val saveButton = button("Save selected", "presets-save") { savePreset(detailedName.getText, quick = false) }
  private val editButton = button("Edit", "presets-edit") { setEditorOpen(true) }
  private val cancelButton = button("Cancel", "presets-cancel") { setEditorOpen(false) }
  private val savePrevious = button("Previous save review page", "presets-save-previous") { previousSavePage(quick = false) }
  private val saveNext = button("Next save review page", "presets-save-next") { nextSavePage(quick = false) }
  private val savePageLabel = label("No save review", "presets-save-page")
  private val quickSavePrevious = button("Previous Quick save review page", "presets-quick-save-previous") { previousSavePage(quick = true) }
  private val quickSaveNext = button("Next Quick save review page", "presets-quick-save-next") { nextSavePage(quick = true) }
  private val quickSavePageLabel = label("No Quick save review", "presets-quick-save-page")
  private val quickSaveNav = boxed(quickSavePrevious, quickSaveNext, quickSavePageLabel)
  private val deleteButton = button("Delete", "presets-delete") { deleteSelected() }
  private val refreshButton = button("Refresh", "presets-refresh") { refresh() }
  private val reloadButton = button("Reload preset tools", "presets-reload-tools") { reloadTools() }
  private val restoreButton = button("Restore backup", "presets-restore-backup") { restoreBackup() }

  private val importFormat = id(new ComboBox[String](FXCollections.observableArrayList(
    "Config Presets JSON", "Preset Utilities JSON", "Tracked IDs text", "Additional components JSON")), "presets-import-format")
  importFormat.setValue("Config Presets JSON")
  private val importPreview = id(new TextArea(), "presets-import-preview")
  importPreview.setEditable(false)
  importPreview.setWrapText(true)
  importPreview.setPrefRowCount(6)
  private val collision = id(new ComboBox[String](FXCollections.observableArrayList("Skip", "Overwrite")), "presets-import-collision")
  collision.setValue("Skip")
  private val addTracked = id(new CheckBox("Add recognized fields to tracked list"), "presets-import-add-tracked")
  private val importButton = button("Import…", "presets-import") { startImport() }
  private val importConfirm = button("Confirm import", "presets-import-confirm") { confirmImport() }
  importConfirm.setDisable(true)
  private val importPrevious = button("Previous review page", "presets-import-previous") { previousImportPage() }
  private val importNext = button("Next review page", "presets-import-next") { nextImportPage() }
  private val importPageLabel = label("No import review", "presets-import-page")
  importPrevious.setDisable(true)
  importNext.setDisable(true)

  private val exportFormat = id(new ComboBox[String](FXCollections.observableArrayList(
    "Native library JSON", "Config Presets JSON", "Preset Utilities JSON",
    "Tracked IDs text", "Additional components JSON")), "presets-export-format")
  exportFormat.setValue("Native library JSON")
  private val exportPreview = id(new TextArea(), "presets-export-preview")
  exportPreview.setEditable(false)
  exportPreview.setWrapText(true)
  exportPreview.setPrefRowCount(5)
  private val supportedOnly = id(new CheckBox("Supported fields only; omit listed native-only fields"),
    "presets-export-supported-only")
  private val exportButton = button("Export…", "presets-export") { startExport() }
  private val exportConfirm = button("Confirm export", "presets-export-confirm") { confirmExport() }
  exportConfirm.setDisable(true)
  private val exportPrevious = button("Previous omission page", "presets-export-previous") { previousExportPage() }
  private val exportNext = button("Next omission page", "presets-export-next") { nextExportPage() }
  private val exportPageLabel = label("No omissions", "presets-export-page")
  exportPrevious.setDisable(true)
  exportNext.setDisable(true)

  private val applyPreview = id(new TextArea(), "presets-apply-preview")
  applyPreview.setEditable(false)
  applyPreview.setWrapText(true)
  applyPreview.setPrefRowCount(3)
  private val applyPrevious = button("Previous warning page", "presets-apply-previous") { previousApplyPage() }
  private val applyNext = button("Next warning page", "presets-apply-next") { nextApplyPage() }
  private val applyPageLabel = label("No apply warnings", "presets-apply-page")
  applyPrevious.setDisable(true)
  applyNext.setDisable(true)

  private val inspectPreview = id(new TextArea(), "presets-inspect-preview")
  inspectPreview.setEditable(false)
  inspectPreview.setWrapText(true)
  private val inspectSearch = id(new TextField(), "presets-inspect-search")
  inspectSearch.setPromptText("Search native labels, IDs and source aliases")
  private val inspectButton = button("Gather / inspect", "presets-inspect") { gather() }
  private val openBacking = button("Open backing file", "presets-open-backing") { openExternal(session.directoryPath.resolve("library-v1.json")) }
  private val openFields = button("Open fields file", "presets-open-fields") { openExternal(session.directoryPath.resolve("library-v1.json")) }

  private val fields = id(new TableView[FieldRow](), "presets-fields")
  private final case class FieldRow(spec: PresetFieldSpec, description: String)
  private val fieldColumn = new TableColumn[FieldRow, String]("Tracked fields, current / default, source and availability")
  fieldColumn.setCellValueFactory(f => new SimpleStringProperty(f.getValue.spec.key.render))
  fieldColumn.setCellFactory(_ => new TableCell[FieldRow, String]:
    override def updateItem(value: String, empty: Boolean): Unit =
      super.updateItem(value, empty)
      if empty || value == null || getIndex < 0 || getIndex >= getTableView.getItems.size then
        setGraphic(null)
      else
        val row = getTableView.getItems.get(getIndex)
        val check = new CheckBox()
        check.setId("presets-field-check-" + row.spec.key.render)
        check.setSelected(checkedFields(row.spec.key))
        check.setOnAction(_ =>
          if check.isSelected then checkedFields += row.spec.key else checkedFields -= row.spec.key
          resetSavePages())
        val description = new Label(row.description)
        description.setWrapText(true)
        description.setMaxWidth(600)
        description.setMaxHeight(90)
        setGraphic(boxed(check, description))
        setText(null)
  )
  fields.getColumns.add(fieldColumn)
  fields.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY)
  fields.setFixedCellSize(104)
  fields.setPrefHeight(300)
  private val fieldChecks = id(new VBox(fields), "presets-field-checks")
  VBox.setVgrow(fields, Priority.ALWAYS)

  private val quick = new VBox(9, boxed(configSelect, quickSelect, quickStack, applyButton),
    boxed(quickName, quickSaveButton), quickSavePreview,
    quickSaveNav,
    new Label("Choosing a preset applies it. Reapply runs the current selection again."),
    applyPreview, boxed(applyPrevious, applyNext, applyPageLabel))
  private val editor = id(new VBox(8, fieldChecks, boxed(detailedName, saveButton, cancelButton),
    savePreview, boxed(savePrevious, saveNext, savePageLabel)), "presets-editor")
  VBox.setVgrow(fieldChecks, Priority.ALWAYS)
  private val detailed = new VBox(9, boxed(detailedSelect, detailedStack, editButton, deleteButton), editor,
    boxed(importFormat, importButton, collision, addTracked), importPreview,
    boxed(importPrevious, importNext, importPageLabel, importConfirm),
    boxed(exportFormat, exportButton, supportedOnly), exportPreview,
    boxed(exportPrevious, exportNext, exportPageLabel, exportConfirm))
  VBox.setVgrow(editor, Priority.ALWAYS)
  private val detailedScroll = new ScrollPane(detailed)
  detailedScroll.setFitToWidth(true)
  detailedScroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  private val inspectTab = new VBox(9, boxed(inspectButton, openBacking, openFields),
    new Label("Open fields file opens the fixed native library. Each collection's tracked and bindings sections hold the field list and declarative source aliases."),
    inspectSearch, inspectPreview)
  VBox.setVgrow(inspectPreview, Priority.ALWAYS)
  private val about = new VBox(8,
    new Label("Native Config Presets and Preset Utilities"),
    new Label("Save and apply typed form values. Unsupported source fields are retained and reported."),
    new Label("Reload preset tools refreshes the editor and mappings without changing generation values."),
    new Label("Config Presets source: Zyin055/Config-Presets e252e60288b1104a45ea12106d92a8d145364fa1 (MIT)."),
    new Label("Preset Utilities behaviour reference: Gerschel/sd_web_ui_preset_utils e301734567f37ac53c6225aec3a5c9858b357e7e (archived; no tracked licence)."),
    new Label("This native implementation uses independent code and its own library format. Model-bearing quicksettings, unavailable extension controls and unsupported source fields are not applied."))
  about.getChildren.forEach(n => n match
    case l: Label => l.setWrapText(true)
    case _ => ())
  private val tabs = id(new TabPane(new Tab("Quick", quick), new Tab("Detailed", detailedScroll),
    new Tab("Inspect", inspectTab), new Tab("About", about)), "presets-tabs")
  tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  private def activeDetailed: Boolean = tabs.getSelectionModel.getSelectedIndex == 1
  private val top = boxed(new Label("Profile"), profile, new Label("Destination"), modeChoice,
    destination, collection, refreshButton, reloadButton, restoreButton)
  private val root = new VBox(9, top, boxed(reapplyButton), tabs, invalid, status)
  root.setPadding(new Insets(10))
  val node: Parent = root
  VBox.setVgrow(tabs, Priority.ALWAYS)

  profile.valueProperty.addListener((_, _, _) => if !quiet then contextChanged())
  modeChoice.valueProperty.addListener((_, _, next) =>
    if !quiet && next != null then { mode = next; contextChanged() })
  configSelect.valueProperty.addListener((_, _, next) => choosePreset(next, detailed = false))
  quickSelect.valueProperty.addListener((_, _, next) => choosePreset(next, detailed = false))
  detailedSelect.valueProperty.addListener((_, _, next) => choosePreset(next, detailed = true))
  collision.valueProperty.addListener((_, _, _) => { resetImportPage(); updateImportPreview() })
  addTracked.selectedProperty.addListener((_, _, _) => { resetImportPage(); updateImportPreview() })
  inspectSearch.textProperty.addListener((_, _, _) => gather())
  quickName.textProperty.addListener((_, _, _) => resetSavePages())
  detailedName.textProperty.addListener((_, _, _) => resetSavePages())

  private var savePageStarts = Vector(0)
  private var quickSavePageStarts = Vector(0)
  private var savePageShown = 0
  private var quickSavePageShown = 0

  private def setEditorOpen(open: Boolean): Unit =
    editor.setVisible(open)
    editor.setManaged(open)

  private def saveReviewRows(quick: Boolean): Vector[String] =
    val name = if quick then quickName.getText else detailedName.getText
    val available = fields.getItems.asScala.iterator.map(_.spec.key).toSet
    val c = current
    val chosen = c.map(collection =>
      if quick then collection.tracked.intersect(available) else checkedFields).getOrElse(Set.empty)
    val existing = c.flatMap(_.presets.find(_.name == name))
    val removed = existing.toVector.flatMap(_.fields.keys)
      .filter(key => c.exists(_.tracked(key)) && available(key) && !chosen(key))
      .sortBy(_.render)
    val scope = if quick then "all available tracked fields" else "selected checked fields"
    val verb = if existing.nonEmpty then "overwrite exact same-name preset" else "create new preset"
    val summary = s"${previewValue(name, 256)}: $verb in collection $collectionId; " +
      s"scope=$scope (${chosen.size} fields); remove ${removed.size} previously saved known fields; " +
      s"opaque source fields retained=${existing.map(_.opaque.size).getOrElse(0)}; " +
      "unavailable/untracked saved fields retained"
    Vector(summary) ++ removed.map(key => s"remove ${key.render}")

  private def updateSavePreview(quick: Boolean): Unit =
    val rows = saveReviewRows(quick)
    val starts = if quick then quickSavePageStarts else savePageStarts
    val start = starts.last
    val (text, shown) = boundedImportPreview(rows.size, start, rows.iterator.drop(start))
    val view = if quick then quickSavePreview else savePreview
    val previous = if quick then quickSavePrevious else savePrevious
    val next = if quick then quickSaveNext else saveNext
    val page = if quick then quickSavePageLabel else savePageLabel
    view.setText(text)
    page.setText(s"Rows ${start + 1}-${start + shown} of ${rows.size}")
    previous.setDisable(starts.size == 1)
    next.setDisable(start + shown >= rows.size || shown == 0)
    if quick then quickSavePageShown = shown else savePageShown = shown

  private def updateSavePreviews(): Unit =
    updateSavePreview(quick = false)
    updateSavePreview(quick = true)

  private def resetSavePages(): Unit =
    savePageStarts = Vector(0)
    quickSavePageStarts = Vector(0)
    updateSavePreviews()

  private def previousSavePage(quick: Boolean): Unit =
    if quick && quickSavePageStarts.size > 1 then
      quickSavePageStarts = quickSavePageStarts.dropRight(1)
    else if !quick && savePageStarts.size > 1 then
      savePageStarts = savePageStarts.dropRight(1)
    updateSavePreview(quick)

  private def nextSavePage(quick: Boolean): Unit =
    val rows = saveReviewRows(quick)
    val starts = if quick then quickSavePageStarts else savePageStarts
    val shown = if quick then quickSavePageShown else savePageShown
    val next = starts.last + shown
    if shown > 0 && next < rows.size then
      if quick then quickSavePageStarts = starts :+ next else savePageStarts = starts :+ next
      updateSavePreview(quick)

  private def pm(m: Mode): PresetMode = m match
    case Mode.Txt2Img => PresetMode.Txt2Img
    case Mode.Img2Img => PresetMode.Img2Img
    case Mode.Inpaint => PresetMode.Inpaint

  private def schemas(): Either[PresetError, Map[PresetMode, PresetSchema]] =
    Mode.values.foldLeft[Either[PresetError, Map[PresetMode, PresetSchema]]](Right(Map.empty)) {
      (soFar, m) => for
        known <- soFar
        pair <- controllers(m).inspect()
      yield known.updated(pm(m), pair._1)
    }

  private def rows(): Unit =
    inspect() match
      case Left(e) => fail(e)
      case Right((schema, snapshot)) =>
        val tracked = current.map(_.tracked).getOrElse(Set.empty)
        val bindings = current.map(_.bindings).getOrElse(Vector.empty)
        val list = schema.fields.map { spec =>
          val resolved = PresetCompatibility.sourceIdsForField(spec.key, sourceFormat,
            sourceMode, bindings)
          val source = resolved.fold(i => s"unavailable: ${previewValue(i.message, 120)}",
            ids => aliasesPreview(ids))
          val description = s"${previewValue(spec.label, 80)}  [${spec.key.render}]  " +
            s"current=${atomPreview(snapshot.values.get(spec.key), 150)}  " +
            s"default=${atomPreview(Some(spec.defaultValue), 90)}  source=$source  " +
            (if tracked(spec.key) then "tracked" else "available; select to track")
          FieldRow(spec, previewValue(description, 430))
        }
        fields.setItems(FXCollections.observableArrayList(list*))
        val unresolved = selected.toVector.flatMap(_.fields.keys).filterNot(k => schema.fields.exists(_.key == k))
        if unresolved.nonEmpty then invalid.setText("Unavailable saved fields: " + unresolved.map(_.render).sorted.take(200).mkString(", "))

  private def syncSelection(): Unit =
    quiet = true
    try
      val names = current.map(_.presets.map(_.name)).getOrElse(Vector.empty)
      configSelect.setItems(FXCollections.observableArrayList(names*))
      quickSelect.setItems(FXCollections.observableArrayList(names*))
      detailedSelect.setItems(FXCollections.observableArrayList(names*))
      selectedName = selectedName.filter(names.contains)
      selectedName.foreach { name =>
        configSelect.setValue(name)
        quickSelect.setValue(name)
        detailedSelect.setValue(name)
      }
      applyButton.setDisable(selectedName.isEmpty)
      reapplyButton.setDisable(selectedName.isEmpty)
      deleteButton.setDisable(selectedName.isEmpty)
      configSelect.setVisible(profile.getValue != "Utilities")
      configSelect.setManaged(profile.getValue != "Utilities")
      quickSelect.setVisible(profile.getValue == "Utilities")
      quickSelect.setManaged(profile.getValue == "Utilities")
      quickStack.setVisible(profile.getValue == "Utilities")
      quickStack.setManaged(profile.getValue == "Utilities")
      quickName.setVisible(profile.getValue == "Utilities")
      quickName.setManaged(profile.getValue == "Utilities")
      quickSaveButton.setVisible(profile.getValue == "Utilities")
      quickSaveButton.setManaged(profile.getValue == "Utilities")
      quickSavePreview.setVisible(profile.getValue == "Utilities")
      quickSavePreview.setManaged(profile.getValue == "Utilities")
      quickSaveNav.setVisible(profile.getValue == "Utilities")
      quickSaveNav.setManaged(profile.getValue == "Utilities")
      detailedSelect.setVisible(profile.getValue == "Utilities")
      detailedSelect.setManaged(profile.getValue == "Utilities")
      detailedStack.setVisible(profile.getValue == "Utilities")
      detailedStack.setManaged(profile.getValue == "Utilities")
    finally quiet = false
    rows()
    resetSavePages()

  private def contextChanged(): Unit =
    selectedName = None
    checkedFields = Set.empty
    importRevision = None
    importTicket += 1
    exportReview = None
    exportTicket += 1
    exportConfirm.setDisable(true)
    clearExportDiagnostics()
    sourceCandidates = Vector.empty
    importedTracked = Vector.empty
    importedAdditional = Vector.empty
    additionalReview = None
    resetImportPage()
    importConfirm.setDisable(true)
    sourceCatalog = Map.empty
    invalidateInspect()
    clearApplyWarnings()
    destination.setText(mode.key)
    collection.setText(collectionId)
    syncSelection()

  private def choosePreset(name: String, detailed: Boolean): Unit =
    if !quiet && name != null then
      selectedName = Some(name)
      selected.foreach { record =>
        val available = inspect().toOption.map(_._1.fields.map(_.key).toSet).getOrElse(Set.empty)
        checkedFields = record.fields.keySet.intersect(available)
        detailedName.setText(record.name)
      }
      syncSelection()
      applySelected(detailed)

  private def preflight(): Boolean = ctx.commitPresetEditors(mode) match
    case Left(problem) =>
      invalidDraft = Some(problem)
      invalid.setText(problem)
      fail(problem)
      false
    case Right(_) =>
      invalidDraft = None
      invalid.setText("")
      true

  private def applySelected(detailed: Boolean): Unit =
    controller.admitApply() match
      case Left(e) => { fail(e); return }
      case Right(_) => ()
    if !preflight() then return
    for record <- selected; c <- current do
      val base = if profile.getValue == "Utilities" &&
          !(if detailed then detailedStack else quickStack).isSelected then ApplyBase.Defaults
        else ApplyBase.Current
      controller.apply(record, c.tracked, base) match
        case Left(e) => fail(e)
        case Right(plan) =>
          rows()
          setApplyWarnings(plan.issues)
          val warnings = plan.issues.take(5).map(_.message).mkString("; ")
          fail(if warnings.isEmpty then s"Applied ${record.name}"
            else s"Applied ${record.name}; ${plan.issues.size} compatibility warnings: $warnings")

  private def mutate(next: PresetLibrary): Unit =
    if session.isMutationPending then fail("Busy: another preset operation is pending")
    else if readOnly then fail("Read-only after preset reload error; reload or restore backup")
    else loaded match
      case None => fail("Preset library has not loaded")
      case Some(base) =>
        fail("Saving preset library…")
        session.save(base.revision, next)

  private def updatedCollection(change: PresetCollection => PresetCollection): Option[PresetLibrary] =
    loaded.flatMap { base =>
      val index = base.library.collections.indexWhere(_.id == collectionId)
      Option.when(index >= 0)(base.library.copy(collections =
        base.library.collections.updated(index, change(base.library.collections(index)))))
    }

  private def savePreset(name: String, quick: Boolean): Unit =
    if session.isMutationPending then { fail("Busy: another preset operation is pending"); return }
    if !preflight() then return
    current match
      case None => fail("Preset collection has not loaded")
      case Some(c) =>
        inspect() match
          case Left(e) => fail(e)
          case Right((schema, _)) =>
            val available = schema.fields.map(_.key).toSet
            val chosen = if quick then c.tracked.intersect(available) else checkedFields
            val expanded = if quick then c else c.copy(tracked = c.tracked ++ chosen)
            controller.captureSelected(name, expanded, chosen) match
              case Left(e) =>
                invalidDraft = Some(message(e))
                invalid.setText(invalidDraft.get)
                fail(e)
              case Right(record) =>
                selectedName = Some(record.name)
                updatedCollection { old =>
                  old.copy(tracked = old.tracked ++ chosen,
                    presets = old.presets.filterNot(_.id == record.id) :+ record)
                }.foreach(mutate)

  private def deleteSelected(): Unit =
    if session.isMutationPending then { fail("Busy: another preset operation is pending"); return }
    selected.foreach { record =>
      updatedCollection(c => c.copy(presets = c.presets.filterNot(_.id == record.id)))
        .foreach(mutate)
      if session.isMutationPending then pendingDeleteName = Some(record.name)
    }

  def openMode(next: Mode): Unit =
    if !closed && next != mode then
      quiet = true
      try modeChoice.setValue(next)
      finally quiet = false
      mode = next
      contextChanged()

  def acceptResult(result: Either[PresetError, LoadedPresets]): Unit =
    if closed then return
    val wasReload = awaitingReload
    awaitingReload = false
    result match
      case Left(e) =>
        pendingDeleteName = None
        if wasReload then readOnly = true
        fail(e)
      case Right(value) =>
        readOnly = false
        if !everLoaded && value.revision.sha256.isEmpty && !seedAttempted then
          seedAttempted = true
          schemas() match
            case Left(e) => fail(e)
            case Right(catalogs) =>
              loaded = Some(value)
              fail("Seeding built-in Config defaults…")
              try session.save(value.revision, PresetDefaults.library(catalogs))
              catch case NonFatal(e) => fail(s"Cannot construct built-in presets: ${e.getMessage}")
        else
          everLoaded = true
          loaded = Some(value)
          invalidateInspect()
          pendingDeleteName.foreach { name =>
            if current.exists(c => !c.presets.exists(_.name == name)) then selectedName = None
          }
          pendingDeleteName = None
          syncSelection()
          if invalidDraft.nonEmpty then fail(invalidDraft.get)
          else fail(s"Loaded ${value.library.collections.map(_.presets.size).sum} presets")

  private def refresh(): Unit =
    if session.isMutationPending then fail("Busy: another preset operation is pending")
    else { awaitingReload = true; fail("Loading preset library…"); session.reload() }

  def reloadTools(): Unit =
    if session.isMutationPending then fail("Busy: another preset operation is pending")
    else
      sourceCatalog = Map.empty
      rows()
      refresh()

  private def restoreBackup(): Unit =
    if session.isMutationPending then fail("Busy: another preset operation is pending")
    else
      fail("Checking current backing revision before explicit restore…")
      session.readLibraryRevision(result => result match
        case Left(e) => fail(e)
        case Right(actualRevision) => session.restoreBackup(actualRevision))

  private def openExternal(path: Path): Unit =
    try dialogs.openExternal(path)
    catch case NonFatal(e) => fail(s"Open ${path}: ${e.getMessage}")

  private var applyIssues = Vector.empty[PresetIssue]
  private var applyPageStarts = Vector(0)
  private var applyPageShown = 0
  private var exportIssues = Vector.empty[PresetIssue]
  private var exportPageStarts = Vector(0)
  private var exportPageShown = 0
  private var exportReadyText = ""
  private var exportDiagnosticKind = "omissions"

  private def diagnosticRow(prefix: String, issue: PresetIssue): String =
    s"$prefix ${issue.path}: ${previewValue(issue.message, 800)}"

  private def clearApplyWarnings(): Unit =
    applyIssues = Vector.empty
    applyPageStarts = Vector(0)
    updateApplyPreview()

  private def setApplyWarnings(issues: Vector[PresetIssue]): Unit =
    applyIssues = issues
    applyPageStarts = Vector(0)
    updateApplyPreview()

  private def updateApplyPreview(): Unit =
    if applyIssues.isEmpty then
      applyPreview.setText("No compatibility warnings from the last Apply")
      applyPageLabel.setText("No apply warnings")
      applyPageShown = 0
    else
      val start = applyPageStarts.last
      val (shown, count) = boundedImportPreview(applyIssues.size, start,
        applyIssues.iterator.drop(start).map(i => diagnosticRow("Warning", i)))
      applyPreview.setText(shown)
      applyPageShown = count
      applyPageLabel.setText(s"Rows ${start + 1}-${start + count} of ${applyIssues.size} warnings")
    applyPrevious.setDisable(applyPageStarts.size == 1)
    applyNext.setDisable(applyPageShown == 0 || applyPageStarts.last + applyPageShown >= applyIssues.size)

  private def previousApplyPage(): Unit =
    if applyPageStarts.size > 1 then applyPageStarts = applyPageStarts.dropRight(1)
    updateApplyPreview()

  private def nextApplyPage(): Unit =
    val next = applyPageStarts.last + applyPageShown
    if applyPageShown > 0 && next < applyIssues.size then
      applyPageStarts = applyPageStarts :+ next
      updateApplyPreview()

  private def clearExportDiagnostics(): Unit =
    exportIssues = Vector.empty
    exportPageStarts = Vector(0)
    exportPageShown = 0
    exportReadyText = ""
    exportDiagnosticKind = "omissions"
    exportPrevious.setDisable(true)
    exportNext.setDisable(true)
    exportPageLabel.setText("No omissions")

  private def setExportDiagnostics(issues: Vector[PresetIssue], readyText: String,
      kind: String = "omissions"): Unit =
    exportIssues = issues
    exportReadyText = readyText
    exportDiagnosticKind = kind
    exportPageStarts = Vector(0)
    updateExportPreview()

  private def updateExportPreview(): Unit =
    if exportIssues.isEmpty then
      exportPreview.setText(exportReadyText)
      exportPageShown = 0
      exportPageLabel.setText("No omissions")
    else
      val start = exportPageStarts.last
      val (shown, count) = boundedImportPreview(exportIssues.size, start,
        exportIssues.iterator.drop(start).map(i => diagnosticRow(
          if exportDiagnosticKind == "omissions" then "Omission" else "Export error", i)))
      exportPreview.setText(shown)
      exportPageShown = count
      exportPageLabel.setText(s"Rows ${start + 1}-${start + count} of ${exportIssues.size} $exportDiagnosticKind")
    exportPrevious.setDisable(exportPageStarts.size == 1)
    exportNext.setDisable(exportPageShown == 0 || exportPageStarts.last + exportPageShown >= exportIssues.size)

  private def previousExportPage(): Unit =
    if exportPageStarts.size > 1 then exportPageStarts = exportPageStarts.dropRight(1)
    updateExportPreview()

  private def nextExportPage(): Unit =
    val next = exportPageStarts.last + exportPageShown
    if exportPageShown > 0 && next < exportIssues.size then
      exportPageStarts = exportPageStarts :+ next
      updateExportPreview()

  private def invalidateInspect(): Unit =
    inspectPreview.setText("Gather / inspect to show the current profile, destination and library")

  private def gather(): Unit =
    inspect() match
      case Left(e) => fail(e)
      case Right((schema, snapshot)) =>
        val c = current
        val tracked = c.map(_.tracked).getOrElse(Set.empty)
        val grouped = schema.fields.groupBy(_.key)
        val specs = grouped.collect { case (key, candidates) if candidates.size == 1 =>
          key -> candidates.head
        }.toMap
        val saved = selected.map(_.fields.keySet).getOrElse(Set.empty[FieldKey])
        val ordered = schema.fields.map(_.key).distinct ++
          saved.diff(grouped.keySet).toVector.sortBy(_.render)
        val query = inspectSearch.getText.trim.toLowerCase
        def sourceIds(key: FieldKey): Vector[String] = c.toVector.flatMap(collection =>
          PresetCompatibility.sourceIdsForField(key, sourceFormat, sourceMode,
            collection.bindings).toOption.getOrElse(Vector.empty))
        val matching = ordered.filter { key =>
          query.isEmpty || key.render.toLowerCase.contains(query) ||
            specs.get(key).exists(_.label.toLowerCase.contains(query)) ||
            sourceIds(key).exists(_.toLowerCase.contains(query))
        }
        val diagnostics = schema.issues.map(i => diagnosticRow(i.code, i)) ++
          selected.toVector.flatMap(_.opaque.map((id, _) =>
            s"Unresolved source ID ${previewValue(id, 512)}"))
        val matchingDiagnostics = diagnostics.filter(line =>
          query.isEmpty || line.toLowerCase.contains(query))
        val total = matching.size + matchingDiagnostics.size
        val header = s"Collection $collectionId; mode ${mode.key}; $total matching of " +
          s"${ordered.size + diagnostics.size} field and diagnostic rows"
        val fieldRows = matching.iterator.map { key =>
          val spec = specs.get(key)
          val available = spec.nonEmpty && snapshot.values.contains(key)
          val allSource = sourceIds(key)
          val shownSource = if query.isEmpty then allSource else
            val hits = allSource.filter(_.toLowerCase.contains(query))
            if hits.nonEmpty then hits else allSource
          val source = aliasesPreview(shownSource) +
            (if shownSource.size != allSource.size then s" (${allSource.size} aliases total)" else "")
          val choices = spec.map(s =>
            aliasesPreview(s.choices.iterator.take(5).map(v => atomPreview(Some(v), 30)).toVector) +
              s" (total ${s.choices.size})").getOrElse("unavailable")
          val default = spec.map(s => atomPreview(Some(s.defaultValue), 110)).getOrElse("unavailable")
          val nativeInitial = if key.render == "generation.variationEnabled" then
            "  Native initial value=true" else ""
          s"${key.render}  ${spec.map(s => previewValue(s.label, 80)).getOrElse("saved canonical field")}  " +
            s"tracked=${tracked(key)} checked=${checkedFields(key)} saved=${saved(key)} " +
            s"available=$available  current=${atomPreview(snapshot.values.get(key), 150)}  " +
            s"Source preset default=$default$nativeInitial  " +
            s"kind=${spec.map(_.kind.toString).getOrElse("unavailable")}  " +
            s"choices=$choices  source=$source"
        }
        val rows = Iterator.single(header) ++ fieldRows ++
          matchingDiagnostics.iterator.map(previewValue(_, 900))
        val (shown, _) = boundedImportPreview(total + 1, 0, rows)
        inspectPreview.setText(shown)
        fail(s"Gathered $total field and diagnostic rows")

  private def emptyError(code: String, path: String, message: String): PresetError =
    PresetError(Vector(PresetIssue(code, path, message)))

  private val additionalMeta = "sdfx.native-presets.additional-components.v1"
  private val additionalOwnershipMeta = "sdfx.native-presets.additional-owned-bindings.v1"
  private val trackedMeta = "sdfx.native-presets.tracked-ids.v1"

  private def componentField(id: String, library: PresetLibrary,
      catalogs: Map[PresetMode, PresetSchema]): Either[PresetIssue, FieldKey] =
    val attempts = Vector(
      PresetMode.Txt2Img -> "config-txt2img",
      PresetMode.Img2Img -> "config-img2img").map { (sourceMode, collection) =>
      val bindings = library.collections.find(_.id == collection)
        .map(_.bindings).getOrElse(Vector.empty)
      PresetCompatibility.resolveSourceId(id, "config-presets", Some(sourceMode),
        bindings, catalogs)
    }
    val failures = attempts.collect {
      case Left(issue) if issue.code != "mode" && issue.code != "unsupported" => issue
    }
    val resolved = attempts.flatMap(_.toOption.toVector.flatten).distinct
    if failures.nonEmpty then Left(failures.head)
    else if resolved.size > 1 then Left(PresetIssue("alias-conflict", id,
      "component ID resolves to different or compound native fields across Config modes"))
    else resolved.headOption match
      case Some(key) => Right(key)
      case None =>
        val registered = catalogs.valuesIterator.flatMap(_.fields.iterator.map(_.key)).toSet
        FieldKey.parse(id).toOption.filter(registered).toRight(
          PresetIssue("unsupported", id, "no verified Config component or registered native field"))

  private def buildAdditionalReview(entries: Vector[(String, Option[String])],
      library: PresetLibrary, collection: PresetCollection,
      catalogs: Map[PresetMode, PresetSchema]): Either[PresetError, AdditionalReview] =
    metadataEntries(collection, additionalOwnershipMeta).flatMap { previous =>
      val owned = previous.foldLeft[Either[PresetError, Set[(String, FieldKey)]]](Right(Set.empty)) {
        case (soFar, (label, target)) => soFar.flatMap { known =>
          target match
            case None => Right(known)
            case Some(rendered) => FieldKey.parse(rendered).map(field => known + (label -> field))
        }
      }
      owned.flatMap { ownedBindings =>
        var bindings = collection.bindings.filterNot(b =>
          b.format == "preset-utilities" && ownedBindings(b.sourceKey -> b.target))
        var ownedNow = Vector.empty[(String, Option[String])]
        var rows = Vector.empty[AdditionalRow]
        var trackable = Set.empty[FieldKey]
        var failure: Option[PresetError] = None
        val registered = catalogs.valuesIterator.flatMap(_.fields.iterator.map(_.key)).toSet
        entries.foreach { (label, componentId) =>
          if failure.isEmpty then componentId match
            case None =>
              rows :+= AdditionalRow(label, None, None,
                "null retained; any matching Additional-owned binding removed")
            case Some(id) => componentField(id, library, catalogs) match
              case Left(issue) =>
                rows :+= AdditionalRow(label, Some(id), None,
                  s"${issue.code}: ${issue.message}; retained without binding")
              case Right(target) =>
                val existing = bindings.find(b =>
                  b.format == "preset-utilities" && b.sourceKey == label)
                val route = PresetCompatibility.resolveSourceId(label, "preset-utilities",
                  None, bindings, catalogs)
                existing match
                  case Some(manual) if manual.target != target =>
                    rows :+= AdditionalRow(label, Some(id), Some(target),
                      s"conflict: existing binding ${manual.target.render} retained; incoming mapping not applied")
                  case Some(_) if route != Right(Vector(target)) =>
                    rows :+= AdditionalRow(label, Some(id), Some(target),
                      "conflict: existing binding disagrees with a standard source route; incoming mapping not applied")
                  case Some(_) =>
                    rows :+= AdditionalRow(label, Some(id), Some(target),
                      "existing manual binding retained; incoming mapping agrees")
                    if registered(target) then trackable += target
                  case None => route match
                    case Right(keys) if keys == Vector(target) =>
                      rows :+= AdditionalRow(label, Some(id), Some(target),
                        "verified standard source route; no new binding needed")
                      if registered(target) then trackable += target
                    case Right(keys) =>
                      rows :+= AdditionalRow(label, Some(id), Some(target),
                        s"conflict: standard route ${keys.map(_.render).mkString(", ")} retained; incoming mapping not applied")
                    case Left(issue) if issue.code == "unsupported" =>
                      if bindings.size >= limits.maxBindings then
                        failure = Some(emptyError("limit", label, "too many source bindings"))
                      else
                        bindings :+= SourceBinding("preset-utilities", label, target)
                        ownedNow :+= (label -> Some(target.render))
                        rows :+= AdditionalRow(label, Some(id), Some(target),
                          if registered(target) then "binding configured; available"
                          else "binding configured; native field currently unavailable")
                        if registered(target) then trackable += target
                    case Left(issue) =>
                      rows :+= AdditionalRow(label, Some(id), Some(target),
                        s"conflict: ${issue.message}; incoming mapping not applied")
        }
        failure match
          case Some(e) => Left(e)
          case None =>
            for
              source <- metadataRaw(entries)
              ownership <- metadataRaw(ownedNow)
            yield AdditionalReview(rows, bindings, trackable, source, ownership)
      }
    }

  private def metadataRaw(entries: Vector[(String, Option[String])]): Either[PresetError, RawJson] =
    PresetSourceFiles.writeAdditionalComponents(entries, limits)
      .flatMap(bytes => PresetJson.readSource(bytes, limits))

  private def metadataEntries(c: PresetCollection, key: String): Either[PresetError, Vector[(String, Option[String])]] =
    c.extra.find(_._1 == key) match
      case None => Right(Vector.empty)
      case Some((_, raw)) => PresetSourceFiles.readAdditionalComponents(raw.text.getBytes(UTF_8), limits)

  private def putMetadata(c: PresetCollection, key: String, raw: RawJson): PresetCollection =
    c.copy(extra = c.extra.filterNot(_._1 == key) :+ (key -> raw))

  private def startImport(): Unit =
    if session.isMutationPending then { fail("Busy: another preset operation is pending"); return }
    if readOnly then { fail("Read-only after failed reload"); return }
    if (importFormat.getValue.startsWith("Config") && profile.getValue == "Utilities") ||
        (importFormat.getValue.startsWith("Preset Utilities") && profile.getValue != "Utilities") then
      { fail("Choose the matching preset profile before importing this source format"); return }
    val chosen = dialogs.chooseOpen(owner)
    for path <- chosen do
      importTicket += 1
      val ticket = importTicket
      importRevision = loaded.map(_.revision)
      sourceCandidates = Vector.empty
      importedTracked = Vector.empty
      importedAdditional = Vector.empty
      additionalReview = None
      resetImportPage()
      importConfirm.setDisable(true)
      importPreview.setText("Reading source…")
      importFormat.getValue match
        case format if format.startsWith("Config") || format.startsWith("Preset Utilities") =>
          val kind = if format.startsWith("Config") then "config-presets" else "preset-utilities"
          val m = if kind == "config-presets" then Some(if mode == Mode.Txt2Img then PresetMode.Txt2Img else PresetMode.Img2Img) else None
          session.readSource(path, kind, m, result => result match
            case Left(e) if ticket == importTicket => fail(e)
            case Right(document) if ticket == importTicket => reviewSource(document)
            case _ => ())
        case format if format.startsWith("Tracked") =>
          session.readTrackedIds(path, result => result match
            case Left(e) if ticket == importTicket => fail(e)
            case Right(ids) if ticket == importTicket => reviewTracked(ids)
            case _ => ())
        case _ =>
          session.readAdditionalComponents(path, result => result match
            case Left(e) if ticket == importTicket => fail(e)
            case Right(entries) if ticket == importTicket => reviewAdditional(entries)
            case _ => ())

  private final case class SourceCandidate(result: CompatibilityResult, sourceIds: Vector[String])
  private var sourceCandidates = Vector.empty[SourceCandidate]
  private var importPageStarts = Vector(0)
  private var importPageShown = 0
  private var importRowTotal = 0

  private def resetImportPage(): Unit =
    importPageStarts = Vector(0)
    importPageShown = 0
    importRowTotal = 0
    importPrevious.setDisable(true)
    importNext.setDisable(true)
    importPageLabel.setText("No import review")

  private def previousImportPage(): Unit =
    if importPageStarts.size > 1 then
      importPageStarts = importPageStarts.dropRight(1)
      updateImportPreview()

  private def nextImportPage(): Unit =
    val next = importPageStarts.last + importPageShown
    if importPageShown > 0 && next < importRowTotal then
      importPageStarts = importPageStarts :+ next
      updateImportPreview()

  private def reviewSource(document: SourceDocument): Unit =
    (schemas(), current) match
      case (Left(e), _) => fail(e)
      case (_, None) => fail("Preset collection has not loaded")
      case (Right(catalogs), Some(c)) =>
        val ticket = importTicket
        importPreview.setText("Resolving source IDs and compatibility…")
        Fx.async(PresetCompatibility.importRecords(document, catalogs, c.bindings).flatMap { results =>
          results.foldLeft[Either[PresetError, Vector[SourceCandidate]]](Right(Vector.empty)) {
            (soFar, item) => for
              known <- soFar
              ids <- PresetCompatibility.sourceFieldIds(item.record)
            yield known :+ SourceCandidate(item, ids)
          }
        }) { result =>
          if ticket == importTicket then result match
            case Left(e) => fail(e)
            case Right(results) =>
              sourceCatalog = catalogs
              sourceCandidates = results
              importedTracked = Vector.empty
              importedAdditional = Vector.empty
              additionalReview = None
              resetImportPage()
              updateImportPreview()
              importConfirm.setDisable(results.isEmpty)
        }(e => if ticket == importTicket then fail(s"Import review failed: ${e.getMessage}"))

  private def reviewTracked(ids: Vector[String]): Unit =
    sourceCatalog = schemas().toOption.getOrElse(Map.empty)
    sourceCandidates = Vector.empty
    importedTracked = ids
    importedAdditional = Vector.empty
    additionalReview = None
    resetImportPage()
    updateImportPreview()
    importConfirm.setDisable(ids.isEmpty)

  private def reviewAdditional(entries: Vector[(String, Option[String])]): Unit =
    (schemas(), loaded, current) match
      case (Left(e), _, _) => fail(e)
      case (_, None, _) => fail("Preset collection has not loaded")
      case (_, _, None) => fail("Preset collection has not loaded")
      case (Right(catalogs), Some(base), Some(c)) =>
        val ticket = importTicket
        importPreview.setText("Resolving Additional Components and current bindings…")
        Fx.async(buildAdditionalReview(entries, base.library, c, catalogs)) { result =>
          if ticket == importTicket then result match
            case Left(e) => fail(e)
            case Right(review) =>
              sourceCatalog = catalogs
              sourceCandidates = Vector.empty
              importedTracked = Vector.empty
              importedAdditional = entries
              additionalReview = Some(review)
              resetImportPage()
              updateImportPreview()
              importConfirm.setDisable(entries.isEmpty)
        }(e => if ticket == importTicket then fail(s"Additional review failed: ${e.getMessage}"))

  private def sourceResolution(id: String): Either[PresetIssue, Vector[FieldKey]] =
    if sourceCatalog.isEmpty then sourceCatalog = schemas().toOption.getOrElse(Map.empty)
    PresetCompatibility.resolveSourceId(id, sourceFormat, sourceMode,
      current.map(_.bindings).getOrElse(Vector.empty), sourceCatalog)

  private def sourceResolutionText(id: String): String = sourceResolution(id) match
    case Left(issue) => s"$id unavailable: ${issue.message} (retained)"
    case Right(keys) =>
      val relevant = if profile.getValue == "Utilities" then PresetMode.values.toVector
        else if mode == Mode.Txt2Img then Vector(PresetMode.Txt2Img)
        else Vector(PresetMode.Img2Img, PresetMode.Inpaint)
      val available = relevant.flatMap(sourceCatalog.get).flatMap(_.fields.map(_.key)).toSet
      s"$id -> ${keys.map(k => k.render + (if available(k) then " (available)" else " (unavailable; retained)")).mkString(", ")}"

  private def resolvedTracked(ids: Vector[String]): Set[FieldKey] =
    ids.flatMap(id => sourceResolution(id).toOption.toVector.flatten).toSet

  private def boundedImportPreview(total: Int, start: Int, rows: Iterator[String]): (String, Int) =
    // Keep an entire next row for the next page when it does not fit here.
    // Reserve the count footer before rendering any row or issue text.
    val reserved = s"\nShowing $total-$total of $total rows; total $total".length
    val bodyLimit = 4096 - reserved
    val text = new StringBuilder
    var shown = 0
    var space = true
    while shown < 200 && space && rows.hasNext && text.length < bodyLimit do
      val row = rows.next()
      val separator = if shown == 0 then 0 else 1
      val room = bodyLimit - text.length - separator
      if room >= row.length then
        if separator != 0 then text.append('\n')
        text.append(row)
        shown += 1
      else if shown == 0 && room > 0 then
        text.append(row.take(room))
        shown = 1
      else space = false
    val range = if shown == 0 then "0" else s"${start + 1}-${start + shown}"
    text.append(s"\nShowing $range of $total rows; total $total")
    (text.toString, shown)

  private def sourceReviewRow(item: CompatibilityResult, id: String,
      bindings: Vector[SourceBinding], available: Set[FieldKey]): String =
    item.record.source match
      case None => s"$id -> missing source provenance (retained)"
      case Some(source) =>
        PresetCompatibility.resolveSourceId(id, source.format, source.mode,
          bindings, sourceCatalog) match
          case Left(issue) => s"$id -> ${issue.code}: ${issue.message} (retained)"
          case Right(keys) =>
            val targets = keys.iterator.take(5).map { key =>
              val state = if !available(key) then "unavailable native field; retained"
                else if item.record.fields.contains(key) then "compatible"
                else "source value unavailable; retained"
              s"${key.render} ($state)"
            }.mkString(", ")
            s"$id -> $targets${if keys.size > 5 then s" (total ${keys.size} targets)" else ""}"

  private def updateImportPreview(): Unit =
    val c = current
    val start = importPageStarts.last
    val (total, rows) =
      if sourceCandidates.nonEmpty then
        val bindings = c.map(_.bindings).getOrElse(Vector.empty)
        val available = sourceCatalog.get(pm(mode))
          .map(_.fields.iterator.map(_.key).toSet).getOrElse(Set.empty[FieldKey])
        val count = sourceCandidates.iterator.map { candidate =>
          val item = candidate.result
          1 + candidate.sourceIds.size + item.issues.size + item.record.opaque.size
        }.sum
        var skip = start
        val rendered = sourceCandidates.iterator.flatMap { candidate =>
          val item = candidate.result
          val groupSize = 1 + candidate.sourceIds.size + item.issues.size + item.record.opaque.size
          if skip >= groupSize then
            skip -= groupSize
            Iterator.empty
          else
            val within = skip
            skip = 0
            val header = if within == 0 then Iterator.single {
              val conflict = c.exists(_.presets.exists(_.name == item.record.name))
              s"${item.record.name}: ${item.record.fields.size} native fields" +
                (if conflict then s"; conflict (${collision.getValue})" else "")
            } else Iterator.empty
            header ++
              candidate.sourceIds.iterator.drop(math.max(0, within - 1))
                .map(id => sourceReviewRow(item, id, bindings, available)) ++
              item.issues.iterator.drop(math.max(0, within - 1 - candidate.sourceIds.size))
                .map(i => s"${i.code} ${i.path}: ${i.message}") ++
              item.record.opaque.iterator.drop(math.max(0,
                within - 1 - candidate.sourceIds.size - item.issues.size))
                .map((id, _) => s"unavailable source ID $id retained")
        }
        (count, rendered)
      else if importedTracked.nonEmpty then
        (importedTracked.size, importedTracked.iterator.drop(start).map(sourceResolutionText))
      else
        val reviewed = additionalReview.map(_.rows).getOrElse(Vector.empty)
        (reviewed.size, reviewed.iterator.drop(start).map { row =>
          s"${row.label} -> ${row.componentId.getOrElse("null")} -> " +
            row.target.fold("unresolved")(_.render) + s" (${row.note})"
        })
    val (preview, shown) = boundedImportPreview(total, start, rows)
    importRowTotal = total
    importPageShown = shown
    importPreview.setText(preview)
    importPrevious.setDisable(importPageStarts.size == 1)
    importNext.setDisable(shown == 0 || start + shown >= total)
    importPageLabel.setText(if total == 0 then "No import rows"
      else s"Rows ${start + 1}-${start + shown} of $total")
    fail(s"Review $total import rows; choose ${collision.getValue} before confirming")

  private def confirmImport(): Unit =
    if session.isMutationPending then { fail("Busy: another preset operation is pending"); return }
    if importConfirm.isDisabled then { fail("Wait for import review to finish"); return }
    if importRevision != loaded.map(_.revision) then
      { fail("Import review is stale; read the source again"); return }
    val imported = sourceCandidates
    val tracked = importedTracked
    val additional = importedAdditional
    if imported.isEmpty && tracked.isEmpty && additional.isEmpty then { fail("No import review to confirm"); return }
    if imported.nonEmpty then
      updatedCollection { c =>
        val incoming = imported.map(_.result.record)
        val names = incoming.map(_.name).toSet
        val kept = if collision.getValue == "Overwrite" then c.presets.filterNot(p => names(p.name)) else c.presets
        val accepted = if collision.getValue == "Overwrite" then incoming
          else incoming.filterNot(p => c.presets.exists(_.name == p.name))
        val stable = accepted.map { record =>
          c.presets.find(_.name == record.name).fold(record) { old =>
            val oldOpaque = old.opaque.filterNot(x => record.opaque.exists(_._1 == x._1))
            record.copy(id = old.id, opaque = oldOpaque ++ record.opaque, extra = old.extra,
              fields = old.fields.filterNot((key, _) => record.fields.contains(key)) ++ record.fields)
          }
        }
        c.copy(presets = kept ++ stable,
          tracked = if addTracked.isSelected then c.tracked ++ accepted.flatMap(_.fields.keys) else c.tracked)
      }.foreach(mutate)
    else if tracked.nonEmpty then
      val entries = tracked.zipWithIndex.map((name, index) => index.toString -> Some(name))
      metadataRaw(entries) match
        case Left(e) => fail(e)
        case Right(raw) => updatedCollection { c =>
          val marked = putMetadata(c, trackedMeta, raw)
          if addTracked.isSelected then marked.copy(tracked = marked.tracked ++ resolvedTracked(tracked))
          else marked
        }.foreach(mutate)
    else
      additionalReview match
        case None => fail("Additional Components review is incomplete")
        case Some(review) => updatedCollection { c =>
          val marked = putMetadata(putMetadata(c, additionalMeta, review.sourceMetadata),
            additionalOwnershipMeta, review.ownershipMetadata)
          marked.copy(bindings = review.bindings,
            tracked = if addTracked.isSelected then marked.tracked ++ review.trackable else marked.tracked)
        }.foreach(mutate)
    importRevision = None
    importConfirm.setDisable(true)

  private def startExport(): Unit =
    if session.isMutationPending then { fail("Busy: another preset operation is pending"); return }
    if readOnly then { fail("Read-only after failed reload"); return }
    if (exportFormat.getValue.startsWith("Config") && profile.getValue == "Utilities") ||
        (exportFormat.getValue.startsWith("Preset Utilities") && profile.getValue != "Utilities") then
      { fail("Choose the matching preset profile before exporting this source format"); return }
    val catalogs = schemas() match
      case Left(e) => { fail(e); return }
      case Right(value) => value
    for path <- dialogs.chooseSave(owner); c <- current; base <- loaded do
      val kind = exportFormat.getValue
      val format = if kind.startsWith("Native") then "native"
        else if kind.startsWith("Tracked") then "tracked"
        else if kind.startsWith("Additional") then "additional"
        else if kind.startsWith("Config") then "config-presets" else "preset-utilities"
      val exportMode = if format == "config-presets" then
        Some(if mode == Mode.Txt2Img then PresetMode.Txt2Img else PresetMode.Img2Img) else None
      exportTicket += 1
      val ticket = exportTicket
      val profileFormat = sourceFormat
      val profileMode = sourceMode
      exportReview = None
      exportConfirm.setDisable(true)
      clearExportDiagnostics()
      exportPreview.setText("Checking destination revision…")
      fail("Checking export destination revision…")
      session.readExportRevision(path, result =>
        if ticket == exportTicket then result match
          case Left(e) => fail(e)
          case Right(targetRevision) =>
            val review = ExportReview(base.revision, targetRevision, path, format,
              exportMode, c, base.library, profileFormat, profileMode, catalogs)
            exportReview = Some(review)
            supportedOnly.setSelected(false)
            exportPreview.setText("Checking format and omissions…")
            Fx.async {
              val assessment = if format == "config-presets" || format == "preset-utilities" then
                PresetCompatibility.exportRecordIssues(c.presets, format, exportMode, c.bindings)
              else Right(Vector.empty[PresetIssue])
              val strict = exportBytes(review, false)
              val supported = assessment.toOption.filter(_.nonEmpty)
                .map(_ => exportBytes(review, true))
              (assessment, strict, supported)
            } { result =>
              val (assessment, strict, supported) = result
              if ticket == exportTicket then
                assessment match
                  case Left(hard) =>
                    setExportDiagnostics(hard.issues, "", kind = "errors")
                    exportConfirm.setDisable(true)
                    fail("Export has a non-omittable format or binding error")
                  case Right(omissions) =>
                    supported.flatMap(_.left.toOption) match
                      case Some(hard) =>
                        setExportDiagnostics(hard.issues, "", kind = "errors")
                        exportConfirm.setDisable(true)
                        fail("Supported-only export also has a non-omittable error")
                      case None if omissions.nonEmpty =>
                        setExportDiagnostics(omissions, "")
                        exportConfirm.setDisable(false)
                        fail(s"Review all ${omissions.size} omissions; confirm supported-only explicitly")
                      case None => strict match
                        case Right(bytes) =>
                          setExportDiagnostics(Vector.empty,
                            s"${c.presets.size} records; ${bytes.length} bytes. Destination revision ${targetRevision.sha256.getOrElse("missing")}. Ready to export $format.")
                          exportConfirm.setDisable(false)
                          fail("Review export format and destination; confirm explicitly")
                        case Left(e) =>
                          val omittable = e.issues.nonEmpty && e.issues.forall(_.code == "unsupported")
                          setExportDiagnostics(e.issues, "",
                            kind = if omittable then "omissions" else "errors")
                          exportConfirm.setDisable(!omittable)
                          fail(if omittable then "Review omissions; confirm supported-only explicitly"
                            else "Export has a non-omittable error")
            }(e => if ticket == exportTicket then fail(s"Export review failed: ${e.getMessage}")))

  private def exportBytes(review: ExportReview, onlySupported: Boolean): Either[PresetError, Array[Byte]] =
    val c = review.collection
    if review.format == "native" then PresetJson.writeNative(review.library, limits)
    else if review.format == "tracked" then
      for
        represented <- PresetCompatibility.exportTrackedIds(c.tracked, review.profileFormat,
          review.profileMode, c.bindings, onlySupported)
        original <- metadataEntries(c, trackedMeta)
        resolve = (id: String) => PresetCompatibility.resolveSourceId(id,
          review.profileFormat, review.profileMode, c.bindings, review.catalogs)
        retained = original.flatMap(_._2).filter { id =>
          resolve(id) match
            case Left(_) => true // retain unavailable original IDs as inert text
            case Right(keys) => keys.forall(c.tracked) &&
              PresetCompatibility.exportTrackedIds(keys.toSet, review.profileFormat,
                review.profileMode, c.bindings, false).toOption.exists(_.nonEmpty)
        }
        covered = retained.flatMap(id => resolve(id).toOption.toVector.flatten).toSet
        remaining = represented.filter(id => resolve(id).toOption.forall(keys =>
          !keys.forall(covered)))
        bytes <- PresetSourceFiles.writeTrackedIds((retained ++ remaining).distinct, limits)
      yield bytes
    else if review.format == "additional" then
      metadataEntries(c, additionalMeta).flatMap(PresetSourceFiles.writeAdditionalComponents(_, limits))
    else PresetCompatibility.exportRecords(c.presets, review.format, review.mode, c.bindings, onlySupported)

  private def confirmExport(): Unit =
    if session.isMutationPending then { fail("Busy: another preset operation is pending"); return }
    if exportConfirm.isDisabled then { fail("Wait for export review to finish"); return }
    exportReview match
      case None => fail("No export review to confirm")
      case Some(review) if loaded.exists(_.revision != review.libraryRevision) =>
        fail("Export review is stale; review the current library again")
      case Some(review) =>
        val ticket = exportTicket
        val onlySupported = supportedOnly.isSelected
        fail(s"Preparing ${review.format} export…")
        Fx.async(exportBytes(review, onlySupported)) { encoded =>
          if ticket == exportTicket then encoded match
            case Left(e) => fail(e)
            case Right(bytes) if loaded.forall(_.revision != review.libraryRevision) =>
              fail("Export review is stale; review the current library again")
            case Right(bytes) =>
              fail(s"Writing ${review.format} export…")
              session.exportFile(review.path, bytes, review.targetRevision, result => result match
                case Left(e) => fail(e)
                case Right(_) => fail(s"Exported ${review.format} to ${review.path}"))
        }(e => if ticket == exportTicket then fail(s"Export encoding failed: ${e.getMessage}"))

  def dispose(): Unit =
    closed = true
    importTicket += 1
    exportTicket += 1
    importRevision = None
    exportReview = None
