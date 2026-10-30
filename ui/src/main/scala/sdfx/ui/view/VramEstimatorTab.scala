package sdfx.ui.view

import javafx.geometry.Insets
import javafx.scene.chart.{LineChart, NumberAxis, XYChart}
import javafx.scene.control.*
import javafx.scene.layout.{FlowPane, GridPane, VBox}
import javafx.stage.FileChooser
import sdfx.ui.Fx

/** Explicit calibration only: opening the view never starts a GPU workload. */
final class VramEstimatorTab(ctx: UiContext):
  private val service = ctx.vramEstimator
  private var closed = false
  private var records = Vector.empty[ujson.Value]
  private val history = new ComboBox[String]()
  history.setId("vram-profile-history")
  private val status = new Label("Calibrate the loaded CUDA model. Results apply to the recorded runtime only.")
  status.setWrapText(true)
  private val maximum = new Spinner[Integer](256, 2048, 960, 64)
  maximum.setId("vram-maximum-size")
  private val batches = new Spinner[Integer](1, 16, 8)
  batches.setId("vram-maximum-batch")
  private val steps = new Spinner[Integer](1, 100, 1)
  private val modes = Vector("txt2img", "txt2img_hr", "txt2img_hr_latent", "img2img")
  private val charts = modes.map { mode =>
    val x = new NumberAxis(); x.setLabel("Pixels")
    val y = new NumberAxis(); y.setLabel("Additional reserved VRAM (MiB)")
    val chart = new LineChart[Number, Number](x, y)
    chart.setTitle(mode); chart.setAnimated(false); chart.setPrefSize(490, 300)
    chart.setId(s"vram-chart-$mode")
    chart
  }
  private def selected: Option[ujson.Value] = records.lift(history.getSelectionModel.getSelectedIndex)
  private def reportError(e: Throwable): Unit = if !closed then status.setText(Fx.message(e))
  private def render(): Unit = selected.foreach { record =>
    charts.zip(modes).foreach { (chart, mode) =>
      chart.getData.clear()
      record("rows").arr.filter(r => r("mode").str == mode && r("status").str == "complete")
        .groupBy(_("batch_size").num.toInt).toSeq.sortBy(_._1).foreach { (batch, rows) =>
          val series = new XYChart.Series[Number, Number]()
          series.setName(s"Batch $batch")
          rows.sortBy(_("pixels").num).foreach(r => series.getData.add(new XYChart.Data[Number, Number](r("pixels").num, r("reserved_delta_mb").num)))
          chart.getData.add(series)
        }
    }
    status.setText(s"${record("status").str}: ${record("rows").arr.size} attempted cases · ${record("fingerprint").str}")
  }
  private def reload(): Unit = Fx.async(service.history().reverse.toVector) { loaded =>
    if !closed then
      records = loaded
      history.getItems.setAll(records.map(r => s"${r("status").str} · ${r("fingerprint").str} · ${r("profile_path").str}")*)
      if records.nonEmpty then history.getSelectionModel.select(0)
  }(reportError)
  history.valueProperty.addListener((_, _, _) => render())
  private val run = W.button("Run calibration", "Runs four vanilla workloads over every configured size and batch") {
    runCalibration()
  }
  run.setId("vram-calibrate")
  private val stop = W.button("Stop calibration", "Stops this calibration; completed measurements remain in history") {
    Fx.onFx(service.interrupt())(decision => if !closed then status.setText(if decision.accepted then "Stop requested" else "Interruptions are protected"))(reportError)
  }
  stop.setId("vram-calibration-stop")
  private def runCalibration(): Unit =
    run.setDisable(true)
    status.setText("Calibrating… Models remain loaded; larger cases can run out of memory.")
    Fx.onFx(service.benchmark(ujson.Obj("max_size" -> maximum.getValue.intValue, "max_batch" -> batches.getValue.intValue, "steps" -> steps.getValue.intValue), {
      case progress: sdfx.core.protocol.EngineEvent.Progress if progress.pass == "vram_estimator" =>
        Fx.run { if !closed && service.isBenchmarking then status.setText(s"Calibrating: ${progress.step} / ${progress.total} planned cases (failed size groups skip larger cases)") }
      case _ => ()
    })) { _ =>
      if !closed then { run.setDisable(false); reload() }
    } { e => if !closed then { run.setDisable(false); reload(); reportError(e) } }
  private val refresh = W.button("Reload history", "Read saved complete and partial calibrations")(reload())
  private val exportButton = W.button("Export profile…", "Export the selected measurements and runtime identity") {
    selected.foreach { record =>
      val chooser = new FileChooser(); chooser.setInitialFileName("vram-profile.json")
      Option(chooser.showSaveDialog(ctx.stage)).foreach(file => Fx.async {
        service.exportProfile(record, file.toPath)
      }(_ => if !closed then status.setText("Profile exported"))(reportError))
    }
  }
  private val importButton = W.button("Import profile…", "Validate a saved profile before adding it to local history") {
    val chooser = new FileChooser()
    Option(chooser.showOpenDialog(ctx.stage)).foreach { file =>
      Fx.onFx(service.importProfile(file.toPath))(_ => if !closed then reload())(reportError)
    }
  }
  private val grid = new GridPane()
  charts.zipWithIndex.foreach((chart, index) => grid.add(chart, index % 2, index / 2))
  private val controls = new FlowPane(8, 8, new Label("Maximum side"), maximum, new Label("Maximum batch"), batches,
    new Label("Steps"), steps, run, stop, refresh, importButton, exportButton)
  private val content = new VBox(10, controls, history, status,
    new Label("Vanilla workloads only. Extension costs are excluded. A finite sweep is not a device capacity guarantee."), grid)
  content.setPadding(new Insets(12))
  val node = new ScrollPane(content)
  node.setFitToWidth(true)
  def dispose(): Unit = closed = true
  reload()
