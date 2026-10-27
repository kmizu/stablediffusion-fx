package sdfx.ui.view

import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{HBox, Priority, VBox}
import sdfx.core.presets.*
import sdfx.ui.presets.{NamedGenerationStates, NativePresetFields, NativePresetsController, NativePresetsSession}
import sdfx.ui.vm.Mode

/** Separate full-state library; selected-field presets are never read or mutated here. */
final class NamedGenerationStatesPane(
    ctx: UiContext,
    session: NativePresetsSession,
    fields: Map[Mode, NativePresetFields],
    controllers: Map[Mode, NativePresetsController]
):
  private var mode = Mode.Txt2Img
  private var loaded: Option[LoadedPresets] = None
  private var pending = false
  private var readOnly = true
  private var closed = false
  private val heading = W.label("Named generation states", "section-title")
  private val scope = W.label(NamedGenerationStates.scope, "hint")
  scope.setWrapText(true)
  private val list = new ListView[String]()
  list.setId("generation-states-list")
  list.setPrefHeight(170)
  private val name = new TextField()
  name.setId("generation-states-name")
  name.setPromptText("Name this complete settings snapshot")
  private val detail = new TextArea()
  detail.setId("generation-states-detail")
  detail.setEditable(false)
  detail.setWrapText(true)
  private val status = new TextArea()
  status.setId("generation-states-status")
  status.setEditable(false)
  status.setWrapText(true)
  status.setPrefRowCount(3)
  private def selected: Option[PresetRecord] = for
    state <- loaded
    selectedName <- Option(list.getSelectionModel.getSelectedItem)
    record <- NamedGenerationStates.collection(state.library, mode).presets.find(_.name == selectedName)
  yield record

  private def confirm(title: String, message: String): Boolean =
    val alert = new Alert(Alert.AlertType.CONFIRMATION, message, ButtonType.OK, ButtonType.CANCEL)
    alert.initOwner(ctx.stage)
    alert.setTitle(title)
    alert.setHeaderText(title)
    alert.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.OK

  private def failure(error: PresetError): Unit = status.setText(NamedGenerationStates.describe(error))
  private def refreshButtons(): Unit =
    save.setDisable(pending || readOnly || loaded.isEmpty)
    restore.setDisable(pending || selected.isEmpty)
    delete.setDisable(pending || readOnly || selected.isEmpty)
    reload.setDisable(pending)
    backup.setDisable(pending)

  private def showSelected(): Unit =
    selected match
      case None => detail.clear()
      case Some(record) =>
        name.setText(record.name)
        val warnings = NamedGenerationStates.preflight(record, fields(mode)).left.toOption
          .map(e => "Restore blocked; current settings remain unchanged:\n" + NamedGenerationStates.describe(e) + "\n\n").getOrElse("")
        detail.setText(warnings + record.fields.toVector.sortBy(_._1.render)
          .map((key, value) => s"${key.render}: $value").mkString("\n"))
    refreshButtons()

  private val save = W.button("Save named state", "Capture all supported controls for this mode") {
    val targetMode = mode
    ctx.commitPresetEditors(targetMode) match
      case Left(error) => status.setText(error)
      case Right(_) => NamedGenerationStates.capture(name.getText, fields(targetMode)) match
        case Left(error) => failure(error)
        case Right(record) => loaded.foreach { state =>
          val exists = NamedGenerationStates.collection(state.library, targetMode).presets.exists(_.name == record.name)
          if !exists || confirm("Replace named state?", s"Replace '${record.name}' with the current ${targetMode.key} settings?") then
            pending = true
            refreshButtons()
            session.save(state.revision, NamedGenerationStates.put(state.library, targetMode, record))
        }
  }
  save.setId("generation-states-save")
  private val restore = W.button("Restore state", "Restore every saved setting after checking all resources") {
    val targetMode = mode
    selected.foreach { record =>
      controllers(targetMode).admitApply() match
        case Left(error) => failure(error)
        case Right(_) => NamedGenerationStates.preflight(record, fields(targetMode)) match
          case Left(error) => failure(error)
          case Right(_) =>
            if confirm("Restore named state?", s"Replace current ${targetMode.key} settings with '${record.name}'?\nShared checkpoint/VAE/clip skip also change. Current init image and painted mask stay in place.") then
              // Confirmations run a nested event loop: repeat admission and preflight afterward.
              val result = for
                _ <- controllers(targetMode).admitApply()
                _ <- NamedGenerationStates.preflight(record, fields(targetMode))
                _ <- ctx.commitPresetEditors(targetMode).left.map(message =>
                  PresetError(Vector(PresetIssue("editor", targetMode.key, message))))
                plan <- controllers(targetMode).apply(record, record.fields.keySet, ApplyBase.Current)
              yield plan
              result match
                case Left(error) => failure(error)
                case Right(_) => status.setText(s"Restored '${record.name}'. Init image and painted mask were kept; no generation was started.")
    }
  }
  restore.setId("generation-states-restore")
  private val delete = W.button("Delete", "Delete the selected named state") {
    val targetMode = mode
    selected.foreach { record => loaded.foreach { state =>
      if confirm("Delete named state?", s"Delete '${record.name}' from ${targetMode.key} named states?") then
        pending = true
        refreshButtons()
        session.save(state.revision, NamedGenerationStates.remove(state.library, targetMode, record.name))
    }}
  }
  delete.setId("generation-states-delete")
  private val reload = W.button("Reload", "Reload the library after an external change or error") {
    pending = true
    refreshButtons()
    session.reload()
  }
  private val backup = W.button("Restore backup", "Recover the previous saved library") {
    if confirm("Restore library backup?", "Replace the entire named-state library with its previous saved version?") then
      pending = true
      refreshButtons()
      session.readLibraryRevision {
        case Left(error) =>
          pending = false
          failure(error)
          refreshButtons()
        case Right(revision) => session.restoreBackup(revision)
      }
  }

  list.getSelectionModel.selectedItemProperty.addListener((_, _, _) => showSelected())
  private val actions = new HBox(8, save, restore, delete, reload, backup)
  val node = new VBox(10, heading, scope,
    W.label("Unlike Settings presets, a named state always captures all supported settings for one mode.", "hint"),
    list, name, actions, detail, status)
  node.setPadding(new Insets(16))
  VBox.setVgrow(detail, Priority.ALWAYS)
  refreshButtons()

  def openMode(value: Mode): Unit =
    mode = value
    heading.setText(s"Named generation states — ${mode.key}")
    refreshList()

  private def refreshList(): Unit =
    val old = Option(list.getSelectionModel.getSelectedItem)
    val names = loaded.toVector.flatMap(state => NamedGenerationStates.collection(state.library, mode).presets.map(_.name)).sorted
    list.getItems.setAll(names*)
    old.filter(names.contains).foreach(list.getSelectionModel.select)
    showSelected()

  def acceptResult(result: Either[PresetError, LoadedPresets]): Unit =
    if !closed then
      pending = false
      result match
        case Left(error) =>
          readOnly = true
          failure(error)
        case Right(value) =>
          loaded = Some(value)
          readOnly = false
          status.setText(s"Library ready: ${session.directoryPath}. Input files are referenced, not archived.")
      refreshList()

  def dispose(): Unit = closed = true
