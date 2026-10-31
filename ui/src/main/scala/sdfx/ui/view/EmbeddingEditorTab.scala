package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{HBox, Pane, Priority, Region, VBox}
import javafx.scene.paint.Color
import javafx.scene.shape.Rectangle
import sdfx.ui.Fx
import sdfx.ui.vm.{EmbeddingEditorDraft, EmbeddingEditorRequestGate}
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

/** Edits an owned, loaded embedding. Preview is submitted through the generation queue. */
final class EmbeddingEditorTab(ctx: UiContext, preview: ujson.Value => Unit,
    sendOverride: Option[ujson.Obj => scala.concurrent.Future[ujson.Value]] = None):
  private def send(params: ujson.Obj): scala.concurrent.Future[ujson.Value] =
    sendOverride.fold(ctx.services.client.raw("embedding_editor", params))(request => request(params))
  private val name = new TextField()
  name.setId("embedding-editor-name"); name.setPromptText("Loaded embedding name")
  HBox.setHgrow(name, Priority.ALWAYS)
  private val catalog = new ComboBox[String]()
  catalog.setId("embedding-editor-catalog"); catalog.setPromptText("Choose an embedding file")
  catalog.setMaxWidth(Double.MaxValue); HBox.setHgrow(catalog, Priority.ALWAYS)
  private def populateCatalog(): Unit =
    val selected = Option(catalog.getValue)
    val paths = ctx.services.catalog.embeddings.map(_.path).distinct.sorted
    catalog.setItems(FXCollections.observableArrayList(paths.asJava))
    selected.filter(paths.contains).foreach(catalog.setValue)
  private val refreshCatalog: Button = W.button("Refresh list", "Find embeddings in configured folders") {
    requestGate.begin().foreach { token =>
      status.setText("Refreshing embedding list…")
      try
        Fx.onFx(ctx.services.catalog.refreshAsync()) { _ =>
          requestGate.settle(token) { populateCatalog(); status.setText("Embedding list refreshed; unsaved edits are unchanged.") }
        } { e => requestGate.settle(token) { status.setText(s"Could not refresh embedding list: ${Fx.message(e)}") } }
      catch case NonFatal(e) => requestGate.settle(token) { status.setText(s"Could not refresh embedding list: ${Fx.message(e)}") }
    }
  }
  refreshCatalog.setId("embedding-editor-refresh-list")
  private val guidance = new TextField()
  guidance.setId("embedding-editor-guidance")
  guidance.setPromptText("Guidance: word:#RRGGBB, word:#RRGGBB")
  HBox.setHgrow(guidance, Priority.ALWAYS)
  private val encoder = new ComboBox[String]()
  encoder.setId("embedding-editor-encoder")
  private val vector = new ComboBox[Integer]()
  vector.setId("embedding-editor-vector")
  private val page = new ComboBox[Integer]()
  page.setId("embedding-editor-page")
  private val coordinates = new VBox(8)
  private val status: Label = W.label("Load a checkpoint and open an embedding.", "muted")
  status.setId("embedding-editor-status"); status.setWrapText(true)
  private val editing = W.label("No embedding open.", "muted")
  editing.setWrapText(true)
  private val overwrite = new CheckBox("Overwrite the loaded embedding file")
  overwrite.setId("embedding-editor-overwrite")
  private var source: Option[ujson.Value] = None
  private var draft: Option[EmbeddingEditorDraft] = None
  private var updating = false
  private var coordinateError = false
  private val pageSize = 24

  private def confirmDiscard(): Boolean =
    if !draft.exists(_.dirty) then true
    else
      val dialog = new Alert(Alert.AlertType.CONFIRMATION, "Discard unsaved embedding edits?", ButtonType.CANCEL, ButtonType.OK)
      dialog.initOwner(ctx.stage)
      dialog.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.OK

  private val open = W.button("Open / refresh", "Read the loaded embedding and guidance", Seq("primary")) {
    if commitPendingCoordinate() && confirmDiscard() then requestOpen()
  }
  open.setId("embedding-editor-open")
  private val resetVector = W.button("Reset vector", "Discard edits to this vector") {
    for d <- draft; e <- Option(encoder.getValue); i <- Option(vector.getValue) do
      d.resetVector(e, i.intValue); coordinateError = false; render()
  }
  private val resetAll = W.button("Reset all", "Discard all unsaved edits") {
    draft.foreach(_.resetAll()); coordinateError = false; render()
  }
  private val save: Button = W.button("Save", "Back up and overwrite the loaded embedding", Seq("primary")) {
    if !commitPendingCoordinate() then ()
    else if !overwrite.isSelected then status.setText("Select overwrite consent before saving.")
    else if draft.isEmpty then status.setText("Open an embedding before saving.")
    else for payload <- currentPayload(); token <- requestGate.begin() do
      status.setText("Queued after any running engine operation…")
      try
        Fx.onFx(send(ujson.Obj("operation" -> "save", "name" -> payload("name"),
          "draft" -> payload, "overwrite" -> true))) { result =>
          requestGate.settle(token) {
            status.setText(s"Saved and reloaded. Backup: ${result("backup").str}. Reopen to continue editing.")
            source = None; draft = None; coordinates.getChildren.clear(); overwrite.setSelected(false)
            editing.setText("No embedding open.")
            ctx.services.catalog.refreshAsync()
          }
        } { e => requestGate.settle(token) { status.setText(Fx.message(e)) } }
      catch case NonFatal(e) => requestGate.settle(token) { status.setText(Fx.message(e)) }
  }
  private val previewButton = W.button("Preview", "Queue a preview using the current generation settings and unsaved edits") {
    if commitPendingCoordinate() then
      if draft.isEmpty then status.setText("Open an embedding before previewing.")
      else currentPayload().foreach(preview)
  }
  private val updateGuidance = W.button("Update guidance", "Update reference markers without discarding edits") {
    if !commitPendingCoordinate() then ()
    else if draft.isEmpty then status.setText("Open an embedding before updating guidance.")
    else currentPayload().foreach { payload =>
      requestGate.begin().foreach { token =>
        status.setText("Updating guidance…")
        try
          Fx.onFx(send(ujson.Obj("operation" -> "guidance", "name" -> payload("name"),
            "draft" -> payload, "guidance" -> guidance.getText))) { result =>
            requestGate.settle(token) {
              source.foreach { current =>
                current("ranges") = result("ranges")
                current("guidance") = result("guidance")
              }
              render()
              status.setText(if draft.exists(_.dirty) then "Guidance updated; edits are unsaved." else "Guidance updated.")
            }
          } { e => requestGate.settle(token) { status.setText(Fx.message(e)) } }
        catch case NonFatal(e) => requestGate.settle(token) { status.setText(Fx.message(e)) }
      }
    }
  }
  updateGuidance.setId("embedding-editor-update-guidance")
  previewButton.setId("embedding-editor-preview")
  save.setId("embedding-editor-save")
  private val warning = W.label("Select a configured embedding file or enter a loaded embedding name. Opening an unloaded file activates it in the current model. " +
    "Edits stay in memory until saved. Saving makes a backup and reloads the embedding. " +
    "Guidance uses the first token of each word. Preview uses the current generation settings and output protection.", "muted")
  warning.setWrapText(true)
  private val body: VBox = new VBox(10, W.label("Embedding Editor", "section-title"), warning,
    W.hbox(8, catalog, refreshCatalog), W.hbox(8, name, open), editing, W.hbox(8, guidance, updateGuidance),
    W.hbox(8, W.label("Encoder"), encoder, W.label("Vector"), vector, W.label("Page"), page),
    W.hbox(8, resetVector, resetAll, previewButton), W.hbox(8, overwrite, save), status, coordinates)
  body.setPadding(new Insets(14)); body.setMinWidth(0); body.setMinHeight(Region.USE_PREF_SIZE)
  private lazy val requestGate = new EmbeddingEditorRequestGate(body.setDisable)
  private val scroll = new ScrollPane(body)
  scroll.setFitToWidth(true); scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  val node = new VBox(scroll)
  populateCatalog()
  catalog.valueProperty.addListener((_, _, selected) => Option(selected).foreach { path =>
    val file = java.nio.file.Path.of(path).getFileName.toString
    val extension = file.lastIndexOf('.')
    name.setText(if extension > 0 then file.substring(0, extension) else file)
  })
  encoder.valueProperty.addListener((_, _, _) => if !updating then selectEncoder())
  vector.valueProperty.addListener((_, _, _) => if !updating then render())
  page.valueProperty.addListener((_, _, _) => if !updating then render())

  private def commitPendingCoordinate(): Boolean =
    Option(ctx.stage.getScene).flatMap(scene => Option(scene.getFocusOwner)) match
      case Some(field: TextField) if Option(field.getId).exists(_.startsWith("embedding-editor-coordinate-value-")) =>
        val valid = scala.util.Try(field.getText.trim.toDouble).toOption.exists(_.isFinite)
        field.fireEvent(new javafx.event.ActionEvent())
        valid && !coordinateError
      case _ => !coordinateError

  private def requestOpen(): Unit =
    requestGate.begin().foreach { token =>
      status.setText("Queued after any running engine operation…")
      try
        val request = ujson.Obj("operation" -> "open", "name" -> name.getText, "guidance" -> guidance.getText)
        Option(catalog.getValue).foreach { path =>
          val file = java.nio.file.Path.of(path).getFileName.toString
          val extension = file.lastIndexOf('.')
          val stem = if extension > 0 then file.substring(0, extension) else file
          if stem.equalsIgnoreCase(name.getText.trim) then request("path") = ujson.Str(path)
        }
        Fx.onFx(send(request)) { result =>
          requestGate.settle(token) {
            try
              val values = result("vectors").obj.toMap.map { (key, rows) =>
                key -> rows.arr.toVector.map(_.arr.toVector.map(_.num))
              }
              val next = EmbeddingEditorDraft(values)
              source = Some(result); draft = Some(next)
              editing.setText(s"Editing ${result("name").str}: ${result("path").str}")
              updating = true
              encoder.setItems(FXCollections.observableArrayList(values.keys.toSeq.sorted.asJava))
              encoder.getSelectionModel.selectFirst()
              updating = false
              selectEncoder()
              status.setText(s"Opened ${result("name").str}; edits are unsaved.")
            catch case NonFatal(e) => status.setText(s"Cannot open editor data: ${Fx.message(e)}")
          }
        } { e => requestGate.settle(token) { status.setText(Fx.message(e)) } }
      catch case NonFatal(e) => requestGate.settle(token) { status.setText(Fx.message(e)) }
    }

  private def selectEncoder(): Unit =
    for d <- draft; e <- Option(encoder.getValue) do
      updating = true
      val rows = d.values(e)
      vector.setItems(FXCollections.observableArrayList(rows.indices.map(i => Integer.valueOf(i)).asJava))
      vector.getSelectionModel.selectFirst()
      page.setItems(FXCollections.observableArrayList((0 until ((rows.head.size + pageSize - 1) / pageSize)).map(i => Integer.valueOf(i)).asJava))
      page.getSelectionModel.selectFirst()
      updating = false
      render()

  private def currentPayload(): Option[ujson.Value] =
    for s <- source; d <- draft yield
      val payload = ujson.Obj("name" -> s("name"), "model_token" -> s("model_token"),
        "path" -> s("path"), "file_identity" -> s("file_identity"))
      payload("vectors") = ujson.Obj.from(d.values.toSeq.map { (key, rows) =>
        key -> ujson.Arr.from(rows.map(row => ujson.Arr.from(row.map(ujson.Num(_)))))
      })
      payload

  private def render(): Unit =
    coordinates.getChildren.clear()
    for d <- draft; s <- source; e <- Option(encoder.getValue); v <- Option(vector.getValue); p <- Option(page.getValue) do
      val row = d.vector(e, v.intValue)
      val limits = s("ranges")(e).arr
      val refs = s("guidance").arr
      val start = p.intValue * pageSize
      (start until math.min(start + pageSize, row.size)).foreach { i =>
        val lo = math.min(limits(i)(0).num, row(i)); val hi = math.max(limits(i)(1).num, row(i))
        val slider = new Slider(lo, if hi == lo then lo + 1e-9 else hi, row(i))
        slider.setId(s"embedding-editor-coordinate-$i")
        slider.setBlockIncrement((slider.getMax - slider.getMin) / 100.0)
        HBox.setHgrow(slider, Priority.ALWAYS)
        val value = new TextField(java.lang.Double.toString(row(i)))
        value.setId(s"embedding-editor-coordinate-value-$i")
        value.setPrefColumnCount(11); value.setMaxWidth(150)
        def commitValue(): Unit =
          scala.util.Try(value.getText.trim.toDouble).toOption.filter(_.isFinite) match
            case Some(number) =>
              coordinateError = false
              if number < slider.getMin then slider.setMin(number)
              if number > slider.getMax then slider.setMax(number)
              slider.setValue(number)
              d.setCoordinate(e, v.intValue, i, number)
              value.setText(java.lang.Double.toString(number))
              status.setText("Unsaved edits")
            case None =>
              coordinateError = true
              value.setText(java.lang.Double.toString(d.vector(e, v.intValue)(i)))
              status.setText("Enter a finite numeric coordinate.")
        value.setOnAction(_ => commitValue())
        value.focusedProperty.addListener((_, wasFocused, focused) =>
          if wasFocused && !focused then commitValue())
        slider.valueProperty.addListener((_, _, n) => {
          d.setCoordinate(e, v.intValue, i, n.doubleValue)
          value.setText(java.lang.Double.toString(n.doubleValue))
          status.setText("Unsaved edits")
        })
        val marks = refs.flatMap { ref =>
          val position = ref("markers")(e)(i)
          if position == ujson.Null then None else
            val mark = W.label(s"${ref("text").str}: ${f"${position.num * 100}%.1f"}%")
            mark.setStyle(s"-fx-text-fill: ${ref("color").str};")
            Some(mark)
        }
        val bar = new Pane()
        bar.setMinHeight(6); bar.setPrefHeight(6); bar.setMaxHeight(6)
        bar.setId(s"embedding-editor-guidance-bar-$i")
        refs.foreach { ref =>
          val position = ref("markers")(e)(i)
          if position != ujson.Null then
            val lower = limits(i)(0).num; val upper = limits(i)(1).num
            val reference = lower + position.num * (upper - lower)
            val marker = new Rectangle(3, 6, Color.web(ref("color").str))
            marker.setMouseTransparent(true)
            val place = () =>
              val fraction = (reference - slider.getMin) / (slider.getMax - slider.getMin)
              marker.setLayoutX(math.max(0.0, math.min(bar.getWidth - 3.0, fraction * bar.getWidth - 1.5)))
            bar.widthProperty.addListener((_, _, _) => place())
            slider.minProperty.addListener((_, _, _) => place())
            slider.maxProperty.addListener((_, _, _) => place())
            bar.getChildren.add(marker)
        }
        val sliderColumn = new VBox(1, slider, bar)
        HBox.setHgrow(sliderColumn, Priority.ALWAYS)
        slider.prefWidthProperty.bind(sliderColumn.widthProperty)
        bar.prefWidthProperty.bind(sliderColumn.widthProperty)
        val line = W.hbox(8, W.label(i.toString), sliderColumn, value)
        coordinates.getChildren.add(new VBox(2, line, W.hbox(8, marks.toSeq*)))
      }
