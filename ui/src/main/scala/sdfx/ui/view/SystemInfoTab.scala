package sdfx.ui.view

import javafx.animation.{KeyFrame, Timeline}
import javafx.beans.property.SimpleStringProperty
import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.Node
import javafx.scene.chart.{LineChart, NumberAxis, XYChart}
import javafx.scene.control.*
import javafx.scene.layout.{FlowPane, Priority, VBox}
import javafx.stage.FileChooser
import javafx.util.Duration
import sdfx.engine.SystemInfoService
import sdfx.ui.Fx
import java.nio.file.Files

/** Read-only diagnostics, explicit benchmark workload, and preview-before-send legacy submission. */
final class SystemInfoTab(ctx: UiContext):
  private val service = new SystemInfoService(ctx.services)
  private val hardwareMetadata = new HardwareMetadataPane(ctx)
  private var current: ujson.Value = ujson.Obj()
  private var active = false
  private var initialized = false
  private var statusBusy = false
  private var fullBusy = false
  private var benchmarking = false
  private var pendingToken: Option[String] = None
  private var previewGeneration = 0L
  private var samples = 0
  private var history: Seq[ujson.Value] = Nil
  private val status = W.label("Open this tab to inspect the engine. No report is sent automatically.", "muted")
  status.setId("system-info-status")
  status.setWrapText(true)
  private val overviewRows = FXCollections.observableArrayList[(String, String)]()
  private val overview = table(overviewRows)
  overview.setId("system-info-overview")
  private val networks = new TextArea()
  networks.setId("system-info-networks")
  networks.setEditable(false)
  networks.setWrapText(true)
  private val json = new TextArea()
  json.setId("system-info-json")
  json.setEditable(false)
  json.setStyle("-fx-font-family: monospace;")
  private val refresh = W.button("Refresh data", "Refresh runtime, libraries and network inventory", Seq("primary"))(refreshFull())
  refresh.setId("system-info-refresh")
  private val hash = W.button("Hash inventory", "Read model files to calculate SHA256 prefixes; this can take time")(refreshFull(true))
  hash.setId("system-info-hash")
  private val exportButton = W.button("Export JSON…", "Save the report you can inspect in Info JSON")(exportJson(current, "system-info.json"))
  exportButton.setId("system-info-export")
  private val interrupt = W.button("Send interrupt", "Interrupt the active engine request, including a benchmark") {
    ctx.interruptRuntime()
  }
  interrupt.setId("system-info-interrupt")
  private val xAxis = new NumberAxis()
  private val chart = new LineChart[Number, Number](xAxis, new NumberAxis(0, 100, 25))
  chart.setId("system-info-chart")
  chart.setTitle("GPU utilization / GPU memory (%) · latest 120 visible samples")
  chart.setAnimated(false)
  chart.setCreateSymbols(false)
  chart.setMinHeight(180)
  chart.setPrefHeight(220)
  private val utilization = new XYChart.Series[Number, Number]()
  utilization.setName("Utilization")
  private val gpuMemory = new XYChart.Series[Number, Number]()
  gpuMemory.setName("Memory")
  chart.getData.addAll(utilization, gpuMemory)
  private val overviewBox = new VBox(8, overview, chart)
  VBox.setVgrow(overview, Priority.ALWAYS)

  private val metadata = new FlowPane(8, 6)
  private val metadataChecks: Seq[CheckBox] = SystemInfoService.metadataFields.map { field =>
    val check = new CheckBox(field)
    check.setSelected(ctx.services.settings.systemInfoMetadataFields.contains(field))
    check.setOnAction(_ => ctx.services.updateSettings(_.copy(systemInfoMetadataFields = metadataChecks.filter(_.isSelected).map(_.getText))))
    metadata.getChildren.add(check)
    check
  }
  metadata.setId("system-info-metadata-fields")
  private val metadataHelp = W.label("Optional image metadata: selected fields are written into future generated PNGs. All fields start off; this setting persists locally.", "muted")
  metadataHelp.setWrapText(true)
  private val metadataPane = new TitledPane("Image metadata (opt-in)", new VBox(8, metadataHelp, metadata))
  metadataPane.setId("system-info-metadata")
  metadataPane.setExpanded(false)

  private val level = combo("system-benchmark-level", Seq("quick", "normal", "extensive", "ludicrous"), "normal")
  private val steps = combo("system-benchmark-steps", Seq("turbo", "default", "long"), "default")
  private val width = new Spinner[Integer](64, 8192, 512, 8)
  private val height = new Spinner[Integer](64, 8192, 512, 8)
  width.setId("system-benchmark-width")
  height.setId("system-benchmark-height")
  private val warmup = new CheckBox("Warm up first (excluded from scores)")
  warmup.setSelected(true)
  private val note = new TextField()
  note.setId("system-benchmark-note")
  note.setPromptText("Optional local note")
  private val benchmarkStatus = W.label("10 / 20 / 50 steps. Increasing batches can exhaust VRAM; the run stops after the first failed batch.", "muted")
  benchmarkStatus.setId("system-benchmark-status")
  benchmarkStatus.setWrapText(true)
  private val benchmarkRows = FXCollections.observableArrayList[(String, String)]()
  private val benchmarkTable = table(benchmarkRows)
  benchmarkTable.setId("system-benchmark-results")
  benchmarkTable.setPrefHeight(190)
  private val historyChoice = new ComboBox[String]()
  historyChoice.setId("system-benchmark-history")
  historyChoice.setMaxWidth(Double.MaxValue)
  private val runBenchmark = W.button("Run benchmark", "Runs real generation using the fixed source benchmark scene", Seq("primary"))(benchmark())
  runBenchmark.setId("system-benchmark-run")
  private val exportBenchmark = W.button("Export benchmark JSON…", "Save the selected record without sending it") {
    selectedRecord.foreach(record => exportJson(record, "benchmark.json"))
  }
  private val username = new TextField()
  username.setId("system-benchmark-username")
  username.setPromptText("Submission username (not auto-filled)")
  private val hostname = new CheckBox("Include this computer's hostname")
  hostname.setId("system-benchmark-hostname")
  private val preview = new TextArea()
  preview.setId("system-benchmark-preview")
  preview.setEditable(false)
  preview.setWrapText(true)
  preview.setPrefRowCount(6)
  private val previewButton = W.button("Preview submission", "Construct the exact legacy UDP message without sending it")(previewSubmission())
  previewButton.setId("system-benchmark-preview-button")
  private val submitButton = W.button("Submit previewed payload", "Send only the exact previewed packet to the shown legacy endpoint")(submit())
  submitButton.setId("system-benchmark-submit")
  submitButton.setDisable(true)
  private val destination = W.label("Legacy destination: logs3.papertrailapp.com:32554 (UDP). Receipt cannot be confirmed. Nothing is sent until Submit is pressed.", "muted")
  destination.setWrapText(true)
  private val leaderboard = new Hyperlink("Open published benchmark results")
  leaderboard.setOnAction(_ => java.awt.Desktop.getDesktop.browse(java.net.URI.create("https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html")))
  private val submission = new TitledPane("Optional legacy cloud submission", new VBox(8, destination, username, hostname,
    W.hbox(6, previewButton, submitButton), preview, leaderboard))
  submission.setExpanded(false)
  private val benchmarkBox = new VBox(10, W.label("Fixed scene benchmark", "section-title"), benchmarkStatus,
    W.hbox(6, W.label("Level"), level, W.label("Steps"), steps),
    W.hbox(6, W.label("Width"), width, W.label("Height"), height), warmup, note,
    W.hbox(6, runBenchmark, exportBenchmark), W.label("Local history (same system fingerprint is replaced)"),
    historyChoice, benchmarkTable, submission)
  benchmarkBox.setPadding(new Insets(10))
  private val tabs = new TabPane()
  tabs.setId("system-info-sections")
  tabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  tabs.getTabs.addAll(new Tab("Overview", overviewBox), new Tab("Models & Networks", networks),
    new Tab("Benchmark", scroll(benchmarkBox)), new Tab("Info JSON", json))
  private val content = new VBox(10, W.hbox(6, refresh, hash, exportButton, interrupt), status, hardwareMetadata.node, tabs, metadataPane)
  content.setPadding(new Insets(12))
  VBox.setVgrow(tabs, Priority.ALWAYS)
  tabs.setMinHeight(280)
  val node = new ScrollPane(content)
  node.setFitToWidth(true)
  node.setFitToHeight(true)
  node.setId("system-info-tool")
  private val timer = new Timeline(new KeyFrame(Duration.seconds(5), _ => refreshStatus()))
  timer.setCycleCount(javafx.animation.Animation.INDEFINITE)
  historyChoice.valueProperty.addListener((_, _, _) => { invalidatePreview(); renderBenchmark() })
  username.textProperty.addListener((_, _, _) => invalidatePreview())
  hostname.selectedProperty.addListener((_, _, _) => invalidatePreview())
  Fx.async(service.history())(records => { history = records; refreshHistory() })(e => benchmarkStatus.setText(s"History could not be read: ${Fx.message(e)}"))

  def setActive(value: Boolean): Unit =
    active = value
    if value then
      if !initialized then refreshFull()
      refreshStatus()
      timer.play()
    else timer.stop()

  def dispose(): Unit =
    timer.stop()
    hardwareMetadata.dispose()

  private def refreshStatus(): Unit =
    if active && !statusBusy then
      statusBusy = true
      Fx.onFx(service.status()) { data =>
        statusBusy = false
        data.obj.foreach((key, value) => current(key) = value)
        render()
        appendChart(data)
      } { error => statusBusy = false; status.setText(Fx.message(error)) }

  private def refreshFull(hashes: Boolean = false): Unit =
    if !fullBusy then
      fullBusy = true
      refresh.setDisable(true)
      hash.setDisable(true)
      status.setText(if hashes then "Reading inventory hashes… queued after any active generation" else "Refreshing full data…")
      Fx.onFx(service.full(hashes)) { data =>
        current = data
        initialized = true
        fullBusy = false
        refresh.setDisable(false)
        hash.setDisable(false)
        render()
      } { error =>
        fullBusy = false
        refresh.setDisable(false)
        hash.setDisable(false)
        status.setText(Fx.message(error))
      }

  private def render(): Unit =
    json.setText(current.render(indent = 2))
    overviewRows.setAll(SystemInfoTab.flatten(current).filterNot((key, _) => key.startsWith("networks") || key.startsWith("native.plugins"))* )
    networks.setText(current.obj.get("networks").map(_.render(indent = 2)).getOrElse("Refresh data to inspect inventory"))
    status.setText("Updated " + current.obj.get("timestamp").map(_.str).getOrElse("") + " · automatic updates every 5 seconds while visible")

  private def appendChart(data: ujson.Value): Unit =
    samples += 1
    data.obj.get("memory").flatMap(_.obj.get("gpu")).foreach { gpu =>
      def add(series: XYChart.Series[Number, Number], value: Double): Unit =
        series.getData.add(new XYChart.Data[Number, Number](Int.box(samples), Double.box(value)))
        if series.getData.size > 120 then series.getData.remove(0)
      gpu.obj.get("utilization_percent").foreach(v => add(utilization, v.num))
      for total <- gpu.obj.get("total_mb"); used <- gpu.obj.get("used_mb") if total.num > 0 do add(gpuMemory, used.num / total.num * 100)
    }

  private def benchmark(): Unit =
    if !benchmarking then
      benchmarking = true
      runBenchmark.setDisable(true)
      invalidatePreview()
      benchmarkStatus.setText("Benchmark queued. Use Send interrupt to stop; generated images are temporary and are not added to the gallery.")
      val params = ujson.Obj("level" -> level.getValue, "steps" -> steps.getValue, "width" -> width.getValue.intValue,
        "height" -> height.getValue.intValue, "warmup" -> warmup.isSelected, "note" -> note.getText)
      Fx.onFx(ctx.services.client.systemBenchmark(params, {
        case sdfx.core.protocol.EngineEvent.Log(_, message) => Fx.run(benchmarkStatus.setText(message))
        case _ => ()
      })) { record =>
        benchmarking = false
        runBenchmark.setDisable(false)
        Fx.async(service.remember(record)) { records =>
          history = records
          refreshHistory()
          benchmarkStatus.setText("Benchmark finished. End-to-end iterations/s = steps × batch ÷ wall time. Data saved locally; no submission sent.")
        } { error =>
          history = history :+ record
          refreshHistory()
          benchmarkStatus.setText("Result available; local history save failed: " + Fx.message(error))
        }
      } { error => benchmarking = false; runBenchmark.setDisable(false); benchmarkStatus.setText(Fx.message(error)) }

  private def refreshHistory(): Unit =
    historyChoice.getItems.setAll(history.map(r => r("timestamp").str + " · " + r("fingerprint").str)* )
    if history.nonEmpty then historyChoice.getSelectionModel.selectLast()

  private def selectedRecord: Option[ujson.Value] = history.lift(historyChoice.getSelectionModel.getSelectedIndex)
  private def renderBenchmark(): Unit =
    benchmarkRows.clear()
    selectedRecord.foreach(r => benchmarkRows.setAll(SystemInfoTab.benchmarkRows(r)*))

  private def invalidatePreview(): Unit =
    previewGeneration += 1
    pendingToken = None
    submitButton.setDisable(true)
    preview.clear()

  private def previewSubmission(): Unit = selectedRecord.foreach { record =>
    invalidatePreview()
    val generation = previewGeneration
    Fx.onFx(ctx.services.client.raw("system_submission_preview", ujson.Obj("record" -> record,
      "username" -> username.getText, "include_hostname" -> hostname.isSelected))) { data =>
      if generation == previewGeneration then
        pendingToken = Some(data("token").str)
        preview.setText(data("destination").str + "\n\n" + data("packets").arr.map(_.str.replace("\u0000", "[NUL]")).mkString("\n"))
        submitButton.setDisable(false)
    }(e => benchmarkStatus.setText(Fx.message(e)))
  }

  private def submit(): Unit = pendingToken.foreach { token =>
    submitButton.setDisable(true)
    pendingToken = None
    Fx.onFx(ctx.services.client.raw("system_submit", ujson.Obj("token" -> token))) { data =>
      benchmarkStatus.setText(data("message").str)
    }(e => benchmarkStatus.setText(Fx.message(e)))
  }

  private def exportJson(data: ujson.Value, name: String): Unit =
    val chooser = new FileChooser()
    chooser.setTitle("Export inspected diagnostic data")
    chooser.setInitialFileName(name)
    chooser.getExtensionFilters.add(new FileChooser.ExtensionFilter("JSON", "*.json"))
    Option(chooser.showSaveDialog(ctx.stage)).foreach { file =>
      val text = data.render(indent = 2)
      Fx.async(Files.writeString(file.toPath, text))(_ => ctx.info("JSON saved locally"))(e => ctx.error(Fx.message(e)))
    }

  private def combo(id: String, values: Seq[String], default: String): ComboBox[String] =
    val box = new ComboBox[String](FXCollections.observableArrayList(values*))
    box.setId(id)
    box.setValue(default)
    box

  private def table(rows: javafx.collections.ObservableList[(String, String)]): TableView[(String, String)] =
    val table = new TableView[(String, String)](rows)
    val key = new TableColumn[(String, String), String]("Field")
    key.setCellValueFactory(v => new SimpleStringProperty(v.getValue._1))
    key.setPrefWidth(210)
    val value = new TableColumn[(String, String), String]("Value")
    value.setCellValueFactory(v => new SimpleStringProperty(v.getValue._2))
    value.setPrefWidth(370)
    table.getColumns.addAll(key, value)
    table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN)
    table.setPlaceholder(W.label("No data collected yet"))
    table

  private def scroll(node: Node): ScrollPane =
    val pane = new ScrollPane(node)
    pane.setFitToWidth(true)
    pane

object SystemInfoTab:
  def benchmarkRows(record: ujson.Value): Seq[(String, String)] =
    val scores = record.obj.get("results").toSeq.flatMap(_.arr.toSeq).map { row =>
      val key = "Batch " + row("batch").num.toInt
      val value = if row("status").str == "complete" then
        f"${row("iterations_per_second").num}%.2f it/s · ${row("seconds").num}%.2f s · ${row("megapixels_per_second").num}%.2f MP/s"
      else row("status").str + ": " + row.obj.get("error").map(_.str).getOrElse("unavailable")
      key -> value
    }
    scores ++ flatten(record).filterNot((key, _) => key.startsWith("system") || key.startsWith("results["))

  def flatten(value: ujson.Value, prefix: String = ""): Seq[(String, String)] = value match
    case obj: ujson.Obj => obj.value.toSeq.flatMap((k, v) => flatten(v, if prefix.isEmpty then k else s"$prefix.$k"))
    case array: ujson.Arr => array.value.zipWithIndex.flatMap((v, i) => flatten(v, s"$prefix[$i]")).toSeq
    case ujson.Str(s) => Seq(prefix -> s)
    case ujson.Null => Seq(prefix -> "unavailable")
    case other => Seq(prefix -> other.render())
