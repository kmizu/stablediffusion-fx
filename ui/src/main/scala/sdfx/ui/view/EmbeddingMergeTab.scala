package sdfx.ui.view

import javafx.beans.property.SimpleStringProperty
import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{Priority, VBox}
import sdfx.core.protocol.*
import sdfx.engine.EmbeddingMergeHandle
import sdfx.ui.Fx
import scala.util.control.NonFatal

/** Raw CLIP vector algebra; the worker serializes model access and publication. */
final class EmbeddingMergeTab(ctx: UiContext):
  private val input = new TextArea()
  input.setId("embedding-merge-input")
  input.setPrefRowCount(4)
  input.setWrapText(true)
  input.setPromptText("Plain prompt, 'red hair' * 0.7 + 'blue hair' * 0.3, or pasted PNG infotext")
  private val grouping = new ComboBox[String](FXCollections.observableArrayList("none", "comma", "parts", "words", "tokens", "vectors"))
  grouping.setId("embedding-merge-grouping")
  grouping.setValue("parts")
  private val name = new TextField()
  name.setId("embedding-merge-name")
  name.setPromptText("New embedding filename (never overwrites)")
  private val directory = new TextField(ctx.services.home.resolve(
    ctx.services.settings.modelDirs.embeddings.headOption.getOrElse("embeddings")).resolve("embedding_merge").toString)
  directory.setId("embedding-merge-output-dir")
  directory.setAccessibleText("Embedding output directory")
  private val safetensors = new CheckBox("Safetensors + SD1/SDXL conversion")
  safetensors.setId("embedding-merge-safetensors")
  safetensors.setSelected(true)
  private val results = new TabPane()
  results.setId("embedding-merge-results")
  results.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  private val status = W.label("Load a checkpoint, then inspect its raw CLIP vectors.", "muted")
  status.setId("embedding-merge-status")
  status.setWrapText(true)
  private var active: Option[EmbeddingMergeHandle] = None
  private var disposed = false
  private val inspect = W.button("Inspect / restore", "Inspect vectors or restore pasted PNG expressions", Seq("primary"))(run(false))
  inspect.setId("embedding-merge-inspect")
  private val save = W.button("Save new embedding", "Save and activate the primary embedding; existing files are rejected")(run(true))
  save.setId("embedding-merge-save")
  private val cancel = W.button("Cancel", "Cancel only this merge operation") {
    active.foreach(handle => Fx.onFx(handle.interrupt()) { decision =>
      if !disposed then status.setText(if decision.tooLate then "Save already committed; cancellation is too late."
        else if decision.blocked then "Cancellation blocked by interruption protection."
        else if decision.affected then "Cancelling…" else "No active merge operation.")
    }(error => if !disposed then status.setText(Fx.message(error))))
  }
  cancel.setId("embedding-merge-cancel")
  private val fromPrompt = W.button("Use current prompt", "Copy the current positive prompt") {
    input.setText(ctx.currentGeneration().prompt)
  }
  private val toPrompt = W.button("Use in prompt", "Insert the current expression into the active positive prompt") {
    val text = input.getText.trim
    ctx.editPositivePrompt(if text.startsWith("'") then s"<$text>" else text, false)
  }
  private val help = W.label("Quote text or TI names. + / − combine rows; * / / scale; :N crops or pads; :+N / :-N rotates. " +
    "Prefix a modifier with = for the accumulated result; suffix L/G for SDXL. Commas append text or token IDs (00 pad, 000 end, 0000 start). " +
    "Use <'expression'> or {'expression'} in generation prompts. PT is used when Safetensors is off. " +
    "Converted copies live in sd1/ or sdxl/; only the current model's primary file is activated.", "muted")
  help.setWrapText(true)
  val node = new VBox(9, W.label("Embedding Merge", "section-title"), help, input,
    W.hbox(8, grouping, inspect, fromPrompt, toPrompt), name, directory, safetensors,
    W.hbox(8, save, cancel), status, results)
  node.setPadding(new Insets(14))
  node.setMinWidth(0)
  VBox.setVgrow(results, Priority.ALWAYS)
  input.textProperty.addListener((_, _, _) => controls())
  name.textProperty.addListener((_, _, _) => controls())
  directory.textProperty.addListener((_, _, _) => controls())
  controls()

  private def controls(): Unit =
    val busy = active.nonEmpty
    input.setDisable(busy); grouping.setDisable(busy); name.setDisable(busy)
    directory.setDisable(busy); safetensors.setDisable(busy); fromPrompt.setDisable(busy)
    inspect.setDisable(busy || input.getText.trim.isEmpty)
    save.setDisable(busy || input.getText.trim.isEmpty || name.getText.trim.isEmpty || directory.getText.trim.isEmpty)
    toPrompt.setDisable(busy || input.getText.trim.isEmpty)
    cancel.setDisable(!busy)

  private def number(value: Double): String = f"$value%.6g"
  private def show(stream: EmbeddingMergeStream): VBox =
    val table = new TableView[EmbeddingMergeRow](FXCollections.observableArrayList(stream.rows*))
    table.setId(s"embedding-merge-${stream.name}-table")
    def column(title: String, width: Double)(value: EmbeddingMergeRow => String): Unit =
      val col = new TableColumn[EmbeddingMergeRow, String](title)
      col.setCellValueFactory(cell => new SimpleStringProperty(value(cell.getValue)))
      col.setPrefWidth(width)
      table.getColumns.add(col)
    column("Rows", 70)(r => if r.start == r.end then r.start.toString else s"${r.start}–${r.end}")
    column("Count", 60)(_.count.toString)
    column("Text", 190)(_.text)
    column("Tokens", 160)(_.tokens.map(v => v.strOpt.getOrElse(v.toString)).mkString(", "))
    column("Min", 90)(r => number(r.statistics.min))
    column("Max", 90)(r => number(r.statistics.max))
    column("Sum", 90)(r => number(r.statistics.sum))
    column("Abs sum", 90)(r => number(r.statistics.absSum))
    column("Norm", 90)(r => number(r.statistics.norm))
    column("Std", 90)(r => r.statistics.std.map(number).getOrElse("—"))
    val stats = stream.total
    val total = W.label(s"${stream.count} vectors · min ${number(stats.min)} · max ${number(stats.max)} · sum ${number(stats.sum)} · " +
      s"abs ${number(stats.absSum)} · matrix spectral norm ${number(stats.norm)} · std ${stats.std.map(number).getOrElse("—")}")
    total.setWrapText(true)
    val box = new VBox(8, total, table)
    VBox.setVgrow(table, Priority.ALWAYS)
    box

  private def run(write: Boolean): Unit =
    if active.nonEmpty || disposed then return
    try
      val request = EmbeddingMergeRequest(input.getText, if write then "save" else "inspect", grouping.getValue,
        name.getText.trim, directory.getText.trim, ctx.catalog.snapshot.get.embeddings.map(_.path), safetensors.isSelected)
      val handle = ctx.services.client.embeddingMergeOwned(request)
      active = Some(handle); controls()
      status.setText(if write then "Save queued…" else "Inspection queued…")
      Fx.onFx(handle.result) { result =>
        if !disposed then
          active = None
          input.setText(result.text)
          results.getTabs.setAll(result.streams.map(s => new Tab(s.name, show(s)))*)
          status.setText(if result.warning.nonEmpty then result.warning
            else if result.paths.nonEmpty then s"Saved ${result.paths.mkString(", ")}. ${if result.active then "Primary embedding active." else "Not active."}"
            else if result.streams.isEmpty then "Prompt restored."
            else s"Inspected ${result.streams.map(s => s"${s.name}: ${s.count} vectors").mkString(" · ")}")
          controls()
          if result.paths.nonEmpty then Fx.onFx(ctx.services.catalog.refreshAsync())(_ => ())(
            error => if !disposed then status.setText(s"Saved ${result.paths.mkString(", ")}; catalog refresh failed: ${Fx.message(error)}"))
      } { error => if !disposed then
        active = None; controls(); status.setText(Fx.message(error))
      }
    catch case NonFatal(error) =>
      active = None; controls(); status.setText(Fx.message(error))

  def dispose(): Unit =
    disposed = true
    active.foreach(_.interrupt())
    active = None
