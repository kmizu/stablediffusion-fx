package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{Priority, VBox}
import sdfx.core.protocol.{EngineEvent, MetadataUtilsRequest}
import sdfx.engine.MetadataUtilsHandle
import sdfx.ui.Fx
import scala.util.control.NonFatal

/** A separate copy-only workflow; selecting a model never changes the loaded checkpoint. */
final class MetadataUtilsTab(ctx: UiContext):
  private val kind = new ComboBox[String](FXCollections.observableArrayList("Checkpoint", "LoRA"))
  kind.setId("metadata-kind")
  kind.getSelectionModel.selectFirst()
  private val files = new ComboBox[String]()
  files.setId("metadata-file")
  files.setPromptText("Select a .safetensors file")
  files.setMaxWidth(Double.MaxValue)
  files.setMinWidth(0)
  private val editor = new TextArea("{}")
  editor.setId("metadata-json")
  editor.setAccessibleText("Metadata as JSON")
  editor.setPrefRowCount(18)
  private val name = new TextField()
  name.setId("metadata-name")
  name.setPromptText("Optional new filename stem; blank appends _md")
  private val status = W.label("Select a checkpoint or LoRA, then load its metadata.", "muted")
  status.setId("metadata-status")
  status.setWrapText(true)
  private var active: Option[MetadataUtilsHandle] = None
  private var loadedPath: Option[String] = None
  private var loadedToken = ""
  private var working = false
  private var updating = false
  private var disposed = false
  private val refresh = W.button("Refresh", "Rescan configured checkpoint and LoRA folders")(refreshFiles())
  refresh.setId("metadata-refresh")
  private val load = W.button("Load metadata", "Read metadata without loading model weights")(run(false))
  load.setId("metadata-load")
  private val save = W.button("Save copy", "Write a new file; existing files are never overwritten", Seq("primary"))(run(true))
  save.setId("metadata-save")
  private val cancel = W.button("Cancel", "Cancel only this metadata operation") {
    active.foreach(h => Fx.onFx(h.interrupt()) { decision =>
      if !disposed then status.setText(if decision.tooLate then "Copy already committed; cancellation is too late."
        else if decision.blocked then "Cancellation blocked by interruption protection."
        else if decision.affected then "Cancelling metadata operation…" else "No active metadata operation to cancel.")
    }(e => if !disposed then status.setText(Fx.message(e))))
  }
  cancel.setId("metadata-cancel")
  private val help = W.label("Editing metadata changes the model file hash. Tensor bytes stay unchanged. " +
    "Save copy writes beside the source; an existing name is always rejected. JSON values are stored as strings, " +
    "with objects and other values encoded as JSON. An empty object clears metadata in the copy.", "muted")
  help.setWrapText(true)
  val node = new VBox(10, W.label("Metadata Utils", "section-title"), help,
    W.hbox(8, kind, refresh), files, load, editor, name, W.hbox(8, save, cancel), status)
  node.setPadding(new Insets(14))
  node.setMinWidth(0)
  VBox.setVgrow(editor, Priority.ALWAYS)

  private def resetLoaded(): Unit =
    loadedPath = None
    loadedToken = ""
    editor.setText("{}")
    name.clear()
    controls()
  private def controls(): Unit =
    kind.setDisable(working)
    files.setDisable(working)
    refresh.setDisable(working)
    load.setDisable(working || files.getValue == null)
    editor.setDisable(working || loadedPath.isEmpty)
    name.setDisable(working || loadedPath.isEmpty)
    save.setDisable(working || loadedPath.isEmpty || loadedToken.isEmpty)
    cancel.setDisable(active.isEmpty)
  private def populate(): Unit =
    val selected = Option(files.getValue)
    val snapshot = ctx.catalog.snapshot.get
    val paths = (if kind.getValue == "LoRA" then snapshot.loras.map(_.path) else snapshot.checkpoints.map(_.path))
      .filter(_.toLowerCase(java.util.Locale.ROOT).endsWith(".safetensors"))
    updating = true
    try
      files.getItems.setAll(paths*)
      selected.filter(paths.contains) match
        case Some(path) => files.setValue(path)
        case None => files.getSelectionModel.clearSelection(); files.setValue(null)
    finally updating = false
    if loadedPath != Option(files.getValue) then resetLoaded()
    controls()
  private def refreshFiles(): Unit =
    working = true; controls()
    status.setText("Refreshing model files…")
    Fx.onFx(ctx.services.catalog.refreshAsync()) { _ =>
      if !disposed then
        working = false; populate()
        status.setText("Model file list refreshed.")
    } { e => if !disposed then { working = false; controls(); status.setText(Fx.message(e)) } }
  kind.valueProperty.addListener((_, _, _) => if !updating then { resetLoaded(); populate() })
  files.valueProperty.addListener((_, _, _) => if !updating then resetLoaded())
  private val catalogListener: javafx.beans.InvalidationListener = _ => if !disposed && !working then populate()
  ctx.catalog.snapshot.addListener(catalogListener)
  populate()

  private def run(write: Boolean): Unit =
    val source = if write then loadedPath else Option(files.getValue)
    source.foreach { path =>
      try
        working = true
        if !write then loadedToken = ""
        status.setText(if write then "Copy queued; original file remains unchanged." else "Reading metadata and fingerprinting source…")
        val request = MetadataUtilsRequest(if write then "write" else "read", path,
          if write then editor.getText else "", if write then name.getText else "", if write then loadedToken else "")
        val handle = ctx.services.client.metadataUtilsOwned(request, {
          case p: EngineEvent.Progress => Fx.run {
            if !disposed && working then
              val action = if p.pass == "metadata_utils_read" then "Reading source fingerprint" else "Copying tensor bytes"
              status.setText(s"$action: ${p.step} / ${p.total} MiB")
          }
          case _ => ()
        })
        active = Some(handle); controls()
        Fx.onFx(handle.result) { result =>
          if !disposed then
            active = None; working = false
            if write then
              status.setText(s"Saved copy: ${result.path}. Tensor bytes: ${result.tensorBytes}.")
              Fx.onFx(ctx.services.catalog.refreshAsync())(_ => ())(
                e => if !disposed then status.setText(s"Copy saved: ${result.path}; catalog refresh failed: ${Fx.message(e)}"))
            else
              loadedPath = Some(path)
              loadedToken = result.sourceToken
              editor.setText(result.editable)
              status.setText(s"Metadata loaded. Tensor bytes: ${result.tensorBytes}.")
            controls()
        } { e => if !disposed then
          active = None; working = false
          val message = Fx.message(e)
          if message.contains("reload metadata") then loadedToken = ""
          controls(); status.setText(message)
        }
      catch case NonFatal(e) =>
        active = None; working = false; controls(); status.setText(Fx.message(e))
    }

  def dispose(): Unit =
    disposed = true
    ctx.catalog.snapshot.removeListener(catalogListener)
    active.foreach(_.interrupt())
    active = None
