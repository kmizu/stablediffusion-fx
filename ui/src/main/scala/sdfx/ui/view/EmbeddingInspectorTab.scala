package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.chart.{LineChart, NumberAxis, XYChart}
import javafx.scene.control.*
import javafx.scene.layout.{FlowPane, HBox, Priority, Region, VBox}
import javafx.stage.FileChooser
import sdfx.ui.Fx
import scala.util.control.NonFatal

/** Inspection and editing of the loaded model's token and textual-inversion vectors. */
final class EmbeddingInspectorTab(ctx: UiContext):
  private val input = new TextField()
  input.setId("embedding-inspector-input")
  input.setMinWidth(100); HBox.setHgrow(input, Priority.ALWAYS)
  input.setPromptText("Token, #token ID, or embedding name")
  private val tokenizer = new TextField()
  tokenizer.setPromptText("Short prompt for the mini tokenizer")
  tokenizer.setId("embedding-inspector-tokenizer")
  tokenizer.setMinWidth(100); HBox.setHgrow(tokenizer, Priority.ALWAYS)
  private val rows = new TextArea()
  rows.setPromptText("One token or embedding per line; append |weight, e.g. cat|0.5")
  rows.setPrefRowCount(5)
  rows.setId("embedding-inspector-rows")
  private val concat = new CheckBox("Concatenate vectors")
  concat.setId("embedding-inspector-concat")
  private val combine = new CheckBox("Combine as one vector")
  combine.setId("embedding-inspector-combine")
  private val overwrite = new CheckBox("Allow overwrite")
  overwrite.setId("embedding-inspector-overwrite")
  private val allPresets = new CheckBox("Save all eval presets")
  private val output = new TextField()
  output.setPromptText("Output embedding .safetensors or .pt path")
  output.setId("embedding-inspector-output")
  output.setMinWidth(100); HBox.setHgrow(output, Priority.ALWAYS)
  private val step = new TextField()
  step.setPromptText("Optional training step")
  step.setPrefColumnCount(12)
  private val expression = new TextField()
  expression.setPromptText("Optional Python eval expression")
  expression.setId("embedding-inspector-eval")
  private val presets = new ComboBox[String](FXCollections.observableArrayList(
    "None", "Boost", "Digitize", "Binary", "Randomize", "Sine", "Comb", "Crop_high", "Crop_low"))
  presets.getSelectionModel.selectFirst()
  private val presetExpressions = Map("None" -> "", "Boost" -> "=v*8", "Digitize" -> "=math.ceil(v*8)/8",
    "Binary" -> "=(1*(v>=0)-1*(v<0))/50", "Randomize" -> "=v*random.random()",
    "Sine" -> "=v*math.sin(i/maxi*math.pi)", "Comb" -> "=v*((i%2)==0)",
    "Crop_high" -> "=v*(i<maxi//2)", "Crop_low" -> "=v*(i>=maxi//2)")
  presets.valueProperty.addListener((_, _, value) => expression.setText(presetExpressions.getOrElse(value, "")))
  private val status = W.label("Load a checkpoint before inspecting embeddings.", "muted")
  status.setWrapText(true)
  status.setId("embedding-inspector-status")
  private val report = new TextArea()
  report.setEditable(false)
  report.setPrefRowCount(12)
  report.setId("embedding-inspector-report")
  private val graph = new LineChart[Number, Number](new NumberAxis(), new NumberAxis())
  graph.setTitle("Vectors (selected encoder)")
  graph.setAnimated(false); graph.setCreateSymbols(false); graph.setLegendVisible(false)
  graph.setPrefHeight(220)
  private val encoder = new ComboBox[String](FXCollections.observableArrayList("clip_l", "clip_g"))
  encoder.getSelectionModel.selectFirst()
  private var lastResult: Option[ujson.Value] = None
  encoder.valueProperty.addListener((_, _, _) => updateGraph())
  private val choose = W.button("Output file…", "Choose a safetensors or .pt embedding file") {
    val fc = new FileChooser()
    fc.setTitle("Save embedding")
    fc.getExtensionFilters.addAll(new FileChooser.ExtensionFilter("Safe tensors", "*.safetensors"),
      new FileChooser.ExtensionFilter("PyTorch embedding", "*.pt"))
    Option(fc.showSaveDialog(ctx.stage)).foreach(f => output.setText(f.getAbsolutePath))
  }
  private val inspect = W.button("Inspect", "Inspect token vector and nearest vocabulary entries", Seq("primary")) {
    request(ujson.Obj("operation" -> "inspect", "text" -> input.getText))
  }
  inspect.setId("embedding-inspector-inspect")
  private val list = W.button("List embeddings", "List available embeddings") {
    request(ujson.Obj("operation" -> "list"))
  }
  list.setId("embedding-inspector-list")
  private val exportVector = W.button("Export vector…", "Write vector values to a UTF-8 text file") {
    val fc = new FileChooser()
    fc.setTitle("Export vector as text")
    fc.getExtensionFilters.add(new FileChooser.ExtensionFilter("Text files", "*.txt"))
    Option(fc.showSaveDialog(ctx.stage)).foreach { file =>
      request(ujson.Obj("operation" -> "export", "text" -> input.getText, "output" -> file.getAbsolutePath,
        "overwrite" -> overwrite.isSelected))
    }
  }
  private val tokenize = W.button("Tokenize", "Tokenize without special tokens") {
    request(ujson.Obj("operation" -> "tokenize", "text" -> tokenizer.getText))
  }
  private val send = W.button("Send token IDs to mixer", "Tokenize and use the chosen encoder's token IDs") {
    request(ujson.Obj("operation" -> "tokenize", "text" -> tokenizer.getText), sendToMixer = true)
  }
  private val reset = W.button("Reset mixer", "Clear mixer rows and options") {
    rows.clear(); concat.setSelected(false); combine.setSelected(false); expression.clear()
  }
  private val save = W.button("Save mixed", "Save a textual-inversion embedding", Seq("primary")) { saveMixed() }
  save.setId("embedding-inspector-save")
  private val warning = W.label("Eval executes arbitrary Python with your user permissions and requires Settings → Allow custom code, followed by an engine restart. " +
    "For SDXL, vectors are preserved separately for both text encoders; mixer IDs use #clip_l/#clip_g. " +
    "Embedding and vector exports remain readable when image scrambling is enabled. Saved embeddings are loaded into the active model and the model catalog is refreshed; files outside configured embedding folders remain active only for this session.", "muted")
  warning.setWrapText(true)
  private val body = new VBox(10, W.label("Embedding Inspector", "section-title"), warning,
    W.hbox(8, input, inspect, list, exportVector), W.hbox(8, tokenizer, tokenize, send),
    W.label("Mixer: name or #ID | weight"), rows, new FlowPane(8, 8, concat, combine, reset),
    W.hbox(8, W.label("Eval preset"), presets), expression,
    W.hbox(8, choose, output), new FlowPane(8, 8, step, overwrite, allPresets, save), status,
    W.hbox(8, W.label("Graph encoder"), encoder), graph, report)
  body.setPadding(new Insets(14)); body.setMinHeight(Region.USE_PREF_SIZE); body.setMinWidth(0)
  private val scroll = new ScrollPane(body)
  scroll.setFitToWidth(true); scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  val node: VBox = new VBox(scroll)
  node.setId("embedding-inspector-tool")

  private def request(params: ujson.Obj, sendToMixer: Boolean = false): Unit =
    params("embedding_paths") = ujson.Arr.from(ctx.services.catalog.embeddings.map(e => ujson.Str(e.path)))
    status.setText("Queued after any running engine operation…")
    Fx.onFx(ctx.services.client.raw("embedding_inspector", params)) { value =>
      lastResult = Some(value)
      report.setText(formatResult(value))
      if sendToMixer then
        val streams = value("encoders").obj
        val left = streams("clip_l").arr
        streams.get("clip_g") match
          case Some(right) if right.arr.size == left.size =>
            rows.setText(left.zip(right.arr).map((l, r) => s"#${l("id").num.toInt}/#${r("id").num.toInt}|1").mkString("\n"))
            concat.setSelected(true); combine.setSelected(false)
          case Some(_) => status.setText("The two SDXL tokenizers produced different token counts; enter names manually.")
          case None =>
            rows.setText(left.map(v => s"#${v("id").num.toInt}|1").mkString("\n"))
            concat.setSelected(true); combine.setSelected(false)
      updateGraph()
      if !sendToMixer || !value("encoders").obj.get("clip_g").exists(_.arr.size != value("encoders")("clip_l").arr.size) then
        val count = value.obj.get("warnings").collect { case a: ujson.Arr => a.value.size }.getOrElse(0)
        val outcome = if count > 0 then s"Completed with $count warning(s); see the result details below."
          else "Completed. Inspect the results below."
        if params("operation").str == "save" then
          status.setText("Saved and reloaded; refreshing the model catalog…")
          Fx.onFx(ctx.services.catalog.refreshAsync()) { _ => status.setText(outcome) } { error =>
            status.setText(s"Saved and reloaded, but catalog refresh failed: ${Fx.message(error)}")
          }
        else status.setText(outcome)
    } { error => status.setText(Fx.message(error)) }

  private def formatResult(value: ujson.Value): String =
    val lines = scala.collection.mutable.ArrayBuffer.empty[String]
    value.obj.get("loaded").collect { case a: ujson.Arr => a }.foreach { loaded =>
      lines += s"Loaded embeddings (${loaded.value.size}):"
      loaded.value.foreach { item =>
        val shapes = item("encoders").obj.toSeq.map { (key, shape) => s"$key ${shape.render()}" }.mkString(", ")
        lines += s"${item("name").str}: $shapes"
        item.obj.get("file_metadata").collect { case m: ujson.Obj => m }.foreach { metadata =>
          metadata.value.foreach((key, v) => lines += s"  file $key: ${v.str}")
        }
        item.obj.get("file_metadata_status").collect { case ujson.Str(status) => status }
          .foreach(status => lines += s"  File metadata: $status")
      }
    }
    value.obj.get("available").collect { case a: ujson.Arr => a }.foreach { available =>
      lines += s"\nAvailable in catalog, not loaded (${available.value.size}):"
      available.value.foreach(item => lines += item.str)
    }
    value.obj.get("metadata").collect { case m: ujson.Obj => m }.foreach { metadata =>
      metadata.value.foreach((key, item) => lines += s"$key: ${item.str}")
    }
    value.obj.get("paths").collect { case a: ujson.Arr => a }.foreach { paths =>
      paths.value.foreach(path => lines += s"Saved: ${path.str}")
    }
    value.obj.get("path").collect { case ujson.Str(path) => path }.foreach { path =>
      if !value.obj.contains("paths") then lines += s"Saved: $path"
    }
    value.obj.get("warnings").collect { case a: ujson.Arr => a }.foreach { warnings =>
      warnings.value.foreach(warning => lines += s"Warning: ${warning.str}")
    }
    value.obj.get("encoders").collect { case o: ujson.Obj => o }.foreach { encoders =>
      encoders.value.foreach { (name, value) =>
        lines += s"\n$name"
        value match
          case data: ujson.Obj =>
            data.value.get("token").collect { case ujson.Str(token) => token }.foreach(token => lines += s"Token: $token")
            data.value.get("shape").collect { case a: ujson.Arr => a }.foreach(shape => lines += s"Shape: ${shape.render()}")
            data.value.get("per_vector").collect { case a: ujson.Arr => a }.foreach { vectors =>
              vectors.value.zipWithIndex.foreach { (vector, index) =>
                lines += f"Vector $index: magnitude ${vector("magnitude").num}%.6f, min ${vector("min").num}%.6f, max ${vector("max").num}%.6f"
                vector("nearest").arr.foreach { entry =>
                  lines += f"  ${entry("token").str} (#${entry("id").num.toInt}): ${entry("score").num}%.5f"
                }
              }
            }
          case a: ujson.Arr => a.value.foreach(item => lines += item.toString)
          case _ => ()
      }
    }
    lines.mkString("\n")

  private def updateGraph(): Unit =
    graph.getData.clear()
    lastResult.foreach { value =>
      for
        streams <- value.obj.get("encoders")
        entry <- streams.obj.get(encoder.getValue).collect { case o: ujson.Obj => o }
        vectors <- entry.obj.get("vectors")
      do
        vectors.arr.take(16).foreach { vector =>
          val series = new XYChart.Series[Number, Number]()
          vector.arr.zipWithIndex.foreach((v, i) => series.getData.add(new XYChart.Data[Number, Number](i, v.num)))
          graph.getData.add(series)
        }
    }

  private def saveMixed(): Unit =
    try
      val entries = rows.getText.linesIterator.map(_.trim).filter(_.nonEmpty).map { line =>
        val parts = line.split("\\|", 2)
        val weight = if parts.length == 1 then 1.0 else parts(1).trim.toDouble
        val name = parts(0).trim
        val pair = name.split("/", 2)
        if pair.length == 2 && pair.forall(_.startsWith("#")) then
          ujson.Obj("name" -> name, "weight" -> weight,
            "ids" -> ujson.Obj("clip_l" -> pair(0).drop(1).toInt, "clip_g" -> pair(1).drop(1).toInt))
        else ujson.Obj("name" -> name, "weight" -> weight)
      }.toVector
      require(entries.nonEmpty, "Add at least one mixer row")
      val p = ujson.Obj("operation" -> "save", "rows" -> ujson.Arr.from(entries), "output" -> output.getText,
        "concat" -> concat.isSelected, "combine" -> combine.isSelected, "eval" -> expression.getText,
        "overwrite" -> overwrite.isSelected, "batch_presets" -> allPresets.isSelected)
      if step.getText.trim.nonEmpty then p("step") = step.getText.trim.toInt
      request(p)
    catch case NonFatal(e) => status.setText(Fx.message(e))
