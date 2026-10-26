package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.image.{Image, ImageView}
import javafx.scene.layout.{FlowPane, Priority, Region, VBox}
import javafx.stage.{DirectoryChooser, FileChooser}
import sdfx.engine.VaeToolsService
import sdfx.ui.Fx
import java.nio.file.{Path, Paths}
import scala.jdk.CollectionConverters.*

/** Separate file workflow: never re-save a selected latent through an RGB image picker. */
final class VaeToolsTab(ctx: UiContext):
  private val service = new VaeToolsService(ctx.services)
  private def combo(id: String, values: String*) =
    val c = new ComboBox[String](FXCollections.observableArrayList(values*))
    c.setId(id)
    c.setAccessibleText(id.stripPrefix("vae-tools-").replace('-', ' '))
    c.getSelectionModel.selectFirst()
    c
  private val operation = combo("vae-tools-operation", "encode", "decode", "encode+decode")
  private val representation = combo("vae-tools-representation", "L", "RGBA", "CMYK")
  private val approximation = combo("vae-tools-approximation", "Full", "Approx NN", "Approx cheap", "TAESD")
  private val format = combo("vae-tools-cmyk-format", "TIFF", "JPEG100")
  private val input = new TextArea()
  input.setId("vae-tools-input")
  input.setAccessibleText("Input file paths, one per line")
  input.setPromptText("Input files, one path per line. Decode accepts unchanged L/RGBA PNG, CMYK TIFF/JPEG, or tensor.safetensors.")
  input.setPrefRowCount(3)
  private val output = new TextField(ctx.services.home.outputs(ctx.services.settings).toAbsolutePath.toString)
  output.setId("vae-tools-output")
  output.setAccessibleText("Output directory")
  private val seed = new TextField("0")
  seed.setId("vae-tools-seed")
  seed.setAccessibleText("Encoding seed")
  seed.setPrefColumnCount(11)
  private val tiled = new CheckBox("Tiled VAE")
  tiled.setId("vae-tools-tiled")
  W.tip(tiled, "Use native diffusers VAE tiling for this operation; previous VAE state is restored afterwards")
  private val download = new CheckBox("Allow optional model downloads")
  download.setId("vae-tools-download")
  private val approxWeights = new TextField()
  approxWeights.setId("vae-tools-approx-weights")
  approxWeights.setPromptText("Optional local Approx NN model.pt / vaeapprox-sdxl.pt")
  approxWeights.setAccessibleText("Local Approx NN weights file")
  private val status = W.label("Uses the checkpoint and VAE selected in the top bar. Load a checkpoint first.", "muted")
  status.setId("vae-tools-status")
  status.setWrapText(true)
  private val report = new TextArea()
  report.setId("vae-tools-report")
  report.setEditable(false)
  report.setStyle("-fx-font-family: monospace;")
  report.setPrefRowCount(10)
  private val choices = new ComboBox[String]()
  choices.setId("vae-tools-results")
  choices.setMaxWidth(Double.MaxValue)
  private val preview = new ImageView()
  preview.setId("vae-tools-preview")
  preview.setPreserveRatio(true)
  preview.setFitHeight(300)
  preview.setFitWidth(600)
  private var results = Vector.empty[ujson.Value]
  private def showResult(): Unit =
    val index = choices.getSelectionModel.getSelectedIndex
    if index >= 0 && index < results.size then
      val item = results(index)
      item.obj.get("preview_path") match
        case Some(ujson.Str(path)) => preview.setImage(new Image(Paths.get(path).toUri.toString, true))
        case _ => preview.setImage(null)
  choices.valueProperty.addListener((_, _, _) => showResult())
  private val selectFiles = W.button("Choose files…", "Select one or more image or latent files without converting their modes") {
    val chooser = new FileChooser()
    chooser.setTitle("VAE Tools input files — original bytes retained")
    chooser.getExtensionFilters.addAll(
      new FileChooser.ExtensionFilter("Images and latent tensors", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp", "*.safetensors"),
      new FileChooser.ExtensionFilter("All files", "*.*"))
    Option(chooser.showOpenMultipleDialog(ctx.stage)).foreach(files => input.setText(files.asScala.map(_.getAbsolutePath).mkString("\n")))
  }
  selectFiles.setId("vae-tools-choose")
  private val outputButton = W.button("Output folder…", "Choose where to save encoded data, previews and reports") {
    val chooser = new DirectoryChooser()
    Option(chooser.showDialog(ctx.stage)).foreach(file => output.setText(file.getAbsolutePath))
  }
  private val runButton = W.button("Run VAE Tools", "Run the selected operation on every input file", Seq("primary"))(run())
  runButton.setId("vae-tools-run")
  private val interrupt = W.button("Interrupt", "Stop at the next safe model/batch boundary")(ctx.services.client.interrupt())
  interrupt.setId("vae-tools-interrupt")
  private val help = W.label("Explore VAE compression, latent editing and cross-VAE decoding. L is an exact tensor container in a grayscale PNG. " +
    "RGBA/CMYK quantize 4 latent channels to 8-bit over [-10,10]; out-of-range values clip. Sixteen channels use a 2×2 grid. " +
    "CMYK TIFF retains all channels; JPEG 100 still adds loss. Do not resize, flatten alpha or use lossy formats for L/RGBA. " +
    "Encoding/decoding itself is lossy. The separate CMYK preview is converted to RGB for display only.", "muted")
  help.setWrapText(true)
  private val approximationHelp = W.label("Full uses the selected VAE. TAESD uses madebyollin/taesd or taesdxl (MIT); first use needs cached weights or explicit download permission. " +
    "Approx NN uses official A1111 release weights (local path or explicit download), and decodes at 2× latent size; Approx cheap at latent size. " +
    "Only Full/TAESD can encode; other choices encode with Full. Tiling uses native diffusers, not the external Web UI Tiled VAE script. " +
    "Generation decode errors automatically retain recovery PNG/safetensors and mark the approximate result Broken.", "muted")
  approximationHelp.setWrapText(true)
  private val body = new VBox(10, W.label("VAE Tools", "section-title"), help,
    new FlowPane(8, 8, W.hbox(6, W.label("Operation"), operation), W.hbox(6, W.label("Representation"), representation)),
    new FlowPane(8, 8, W.hbox(6, W.label("Approximation"), approximation), tiled), approximationHelp,
    new FlowPane(8, 8, selectFiles, W.hbox(6, W.label("Encoding seed"), seed)), input,
    W.hbox(8, W.label("CMYK save format"), format), approxWeights, download,
    W.hbox(8, outputButton, output), W.hbox(8, runButton, interrupt), status, choices, preview,
    new TitledPane("Result metadata and saved paths", report))
  body.setPadding(new Insets(14))
  body.setMinHeight(Region.USE_PREF_SIZE)
  body.setMinWidth(0)
  private val scroll = new ScrollPane(body)
  scroll.setFitToWidth(true)
  scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  scroll.setId("vae-tools-scroll")
  preview.fitWidthProperty.bind(javafx.beans.binding.Bindings.max(1.0, scroll.widthProperty.subtract(48)))
  VBox.setVgrow(scroll, Priority.ALWAYS)
  val node: VBox = new VBox(scroll)
  node.setId("vae-tools-tool")
  private val controls = Seq(input, output, seed, operation, representation, approximation, format, tiled, download, approxWeights, selectFiles, outputButton, runButton)
  private def busy(value: Boolean): Unit = controls.foreach(_.setDisable(value))
  private def run(): Unit =
    try
      val paths = input.getText.linesIterator.map(_.trim).filter(_.nonEmpty).map(Paths.get(_)).toSeq
      val number = seed.getText.trim.toLongOption.getOrElse(throw new IllegalArgumentException("Seed must be an integer"))
      busy(true)
      status.setText("VAE operation queued; original input files will be retained.")
      val future = service.run(paths, operation.getValue, representation.getValue, approximation.getValue,
        Paths.get(output.getText), number, tiled.isSelected, format.getValue, download.isSelected, approxWeights.getText)
      Fx.onFx(future) { value =>
        busy(false)
        report.setText(value.render(indent = 2))
        results = value("results").arr.toVector
        choices.getItems.setAll(results.zipWithIndex.map((item, i) => s"${i + 1}. ${Paths.get(item("input_path").str).getFileName} · ${item("status").str}")*)
        choices.getSelectionModel.selectFirst()
        showResult() // Repeated same-file runs can keep the same selected display label.
        val completed = results.count(_("status").str == "complete")
        status.setText(s"$completed / ${results.size} completed. Saved report: ${value("report_path").str}. " + value("warnings").arr.map(_.str).mkString(" "))
      } { error => busy(false); status.setText(Fx.message(error)) }
    catch
      case scala.util.control.NonFatal(error) =>
        busy(false)
        status.setText(Fx.message(error))
