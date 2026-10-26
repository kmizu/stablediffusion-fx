package sdfx.ui.view

import javafx.beans.property.SimpleStringProperty
import javafx.collections.FXCollections
import javafx.embed.swing.SwingFXUtils
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{FlowPane, HBox, Priority, VBox}
import javafx.scene.text.{Text, TextFlow}
import javafx.stage.FileChooser
import sdfx.core.comparison.ImageComparison
import sdfx.core.comparison.ImageComparison.*
import sdfx.core.png.PngMetadata
import sdfx.ui.Fx
import java.awt.image.BufferedImage
import java.nio.file.{Path, Paths}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*

/** Native read-only image comparison. Decode/metadata/export are off the JavaFX thread. */
final class ImageComparisonTab(ctx: UiContext, source: String => Option[(Path, Path)], selected: () => Option[Path]):
  val inputA = new SimpleStringProperty("")
  val inputB = new SimpleStringProperty("")
  val canvas = new ComparisonCanvas
  private val status = W.label("Choose A and B, then Compare upload. Original files stay unchanged.", "muted")
  status.setId("comparison-status")
  status.setWrapText(true)
  private case class Loaded(path: Path, pixels: BufferedImage, metadata: Option[String])
  private var images: Option[(Loaded, Loaded)] = None
  private var revision = 0L
  private var writingInputs = false
  private var pendingLoads = 0
  private val loading = new javafx.beans.property.SimpleBooleanProperty(false)
  def isLoading: Boolean = loading.get
  private def writeInputs(a: Path, b: Path): Unit =
    writingInputs = true
    try
      inputA.set(a.toString)
      inputB.set(b.toString)
    finally writingInputs = false
  Seq(inputA, inputB).foreach(_.addListener((_, _, _) =>
    if !writingInputs then
      revision += 1
      status.setText("Inputs changed; Compare upload to update the displayed pair")
  ))
  private var syncing = false
  var onOptionsChanged: () => Unit = () => ()
  def sourcePaths: Set[String] =
    Set(inputA.get, inputB.get).filter(s => s != null && s.nonEmpty) ++
      images.toSeq.flatMap((a, b) => Seq(a.path.toString, b.path.toString))
  def loadedPaths: Option[(Path, Path)] = images.map((a, b) => (a.path, b.path))
  private def load(path: Path): Loaded =
    val im = ImageIO.read(path.toFile)
    require(im != null, "Image format is unreadable: " + path.getFileName)
    Loaded(path, im, PngMetadata.readParameters(path))
  def compare(a: Path, b: Path): Unit =
    revision += 1
    val ticket = revision
    status.setText("Loading comparison…")
    pendingLoads += 1
    loading.set(true)
    Fx.async((load(a), load(b))) { pair =>
      pendingLoads -= 1
      loading.set(pendingLoads > 0)
      if ticket == revision then
        images = Some(pair)
        writeInputs(a, b)
        showPair()
    } { e =>
      pendingLoads -= 1
      loading.set(pendingLoads > 0)
      if ticket == revision then status.setText("Cannot compare: " + Fx.message(e))
    }
  def send(sideB: Boolean, path: Path): Unit =
    revision += 1 // invalidate any earlier pair request, even if only one new input is present
    (if sideB then inputB else inputA).set(path.toString)
    val a = Option(inputA.get).filter(_.nonEmpty)
    val b = Option(inputB.get).filter(_.nonEmpty)
    if a.nonEmpty && b.nonEmpty then compare(Paths.get(a.get), Paths.get(b.get))
    else status.setText(s"Image ${if sideB then "B" else "A"} ready; choose the other image")
  def compareSource(kind: String): Unit = source(kind) match
    case Some((a, b)) => compare(a, b)
    case None => ctx.info("This source has no input/output pair yet")
  private val firstInfo = new VBox(5)
  private val secondInfo = new VBox(5)
  firstInfo.setMinWidth(0)
  secondInfo.setMinWidth(0)
  firstInfo.setId("comparison-info-a")
  secondInfo.setId("comparison-info-b")
  private def showPair(reset: Boolean = true): Unit = images.foreach { (a, b) =>
    val previousView = canvas.view
    canvas.setImages(SwingFXUtils.toFXImage(a.pixels, null), SwingFXUtils.toFXImage(b.pixels, null))
    if !reset then canvas.update(previousView)
    status.setText(s"A: ${a.path.getFileName} · ${a.pixels.getWidth}×${a.pixels.getHeight}    |    B: ${b.path.getFileName} · ${b.pixels.getWidth}×${b.pixels.getHeight}")
    val diff = differences(a.metadata, b.metadata)
    def info(box: VBox, image: Loaded, positive: Seq[Term], negative: Seq[Term], sideB: Boolean): Unit =
      box.getChildren.clear()
      def terms(label: String, values: Seq[Term]): Unit =
        box.getChildren.add(W.label(label, "pane-title"))
        val flow = new TextFlow()
        values.zipWithIndex.foreach { (term, index) =>
          if index > 0 then flow.getChildren.add(new Text(", "))
          val text = new Text(term.text)
          text.getStyleClass.add(if term.changed then "comparison-diff" else "comparison-term")
          text.setAccessibleText((if term.changed then "Changed: " else "Unchanged: ") + term.text)
          flow.getChildren.add(text)
        }
        box.getChildren.add(flow)
      if image.metadata.isEmpty then box.getChildren.add(W.label("No embedded generation parameters", "muted"))
      else
        terms("Positive", positive)
        terms("Negative", negative)
        val params = new VBox(3)
        diff.parameters.foreach { p =>
          val value = (if sideB then p.b else p.a).getOrElse("— missing —")
          val label = W.label((if p.changed then "Changed · " else "") + p.key + ": " + value)
          label.setWrapText(true)
          if p.changed then label.getStyleClass.add("comparison-diff")
          params.getChildren.add(label)
        }
        box.getChildren.add(params)
      val raw = new TextArea(image.metadata.getOrElse(""))
      raw.setEditable(false)
      raw.setWrapText(true)
      raw.setPrefRowCount(4)
      val pane = new TitledPane("Full embedded text", raw)
      pane.setExpanded(false)
      box.getChildren.add(pane)
    info(firstInfo, a, diff.positiveA, diff.negativeA, false)
    info(secondInfo, b, diff.positiveB, diff.negativeB, true)
  }
  private val direction = new CheckBox("Horizontal slider (left / right)")
  direction.setSelected(true)
  direction.setId("comparison-horizontal")
  private val edge = new Slider(0, 1, .5)
  edge.setId("comparison-reveal")
  edge.setAccessibleText("Reveal edge position")
  private val opacity = new Slider(0, 1, 1)
  opacity.setId("comparison-opacity")
  opacity.setAccessibleText("Image B opacity")
  private val zoom = new Slider(.5, 10, 1)
  zoom.setId("comparison-zoom")
  zoom.setAccessibleText("Shared image zoom")
  private val zoomLabel = W.label("100%", "hint")
  canvas.onViewChanged = v =>
    syncing = true
    edge.setValue(v.split)
    opacity.setValue(v.opacity)
    zoom.setValue(v.zoom)
    zoomLabel.setText(f"${v.zoom * 100}%.0f%%")
    syncing = false
  direction.setOnAction(_ => canvas.update(canvas.view.copy(horizontal = direction.isSelected, split = .5, opacity = 1)))
  edge.valueProperty.addListener((_, _, v) => if !syncing then canvas.update(canvas.view.copy(split = v.doubleValue)))
  opacity.valueProperty.addListener((_, _, v) => if !syncing then canvas.update(canvas.view.copy(opacity = v.doubleValue)))
  zoom.valueProperty.addListener((_, _, v) => if !syncing then canvas.update(canvas.view.copy(zoom = v.doubleValue)))
  def swap(): Unit = images.foreach { (a, b) =>
    revision += 1
    images = Some((b, a))
    writeInputs(b.path, a.path)
    showPair(reset = false)
  }
  private def exportDialog(): Unit = images.foreach { (a, b) =>
    val chooser = new FileChooser()
    chooser.setTitle("Export visible comparison (originals are preserved)")
    chooser.setInitialFileName("comparison.png")
    chooser.getExtensionFilters.add(new FileChooser.ExtensionFilter("PNG", "*.png"))
    Option(chooser.showSaveDialog(ctx.stage)).foreach { file =>
      val view = canvas.view
      val w = canvas.getWidth.toInt.max(1)
      val h = canvas.getHeight.toInt.max(1)
      Fx.async(ImageComparison.exportPng(render(a.pixels, b.pixels, w, h, view), file.toPath, Seq(a.path, b.path))) {
        _ => ctx.success("Comparison exported; originals unchanged")
      }(e => ctx.error(Fx.message(e)))
    }
  }
  /** Manual QA/export integration without a file chooser; exact same rendering and overwrite guard. */
  def exportTo(path: Path): Unit =
    val (a, b) = images.getOrElse(throw new IllegalStateException("Load two images first"))
    ImageComparison.exportPng(render(a.pixels, b.pixels, canvas.getWidth.toInt.max(1), canvas.getHeight.toInt.max(1), canvas.view),
      path, Seq(a.path, b.path))

  private def identified[A <: javafx.scene.Node](id: String, value: A): A = { value.setId(id); value }
  private val compareUpload = identified("comparison-upload", W.button("Compare upload", "Load the two chosen files", Seq("primary")) {
    if inputA.get.nonEmpty && inputB.get.nonEmpty then compare(Paths.get(inputA.get), Paths.get(inputB.get))
    else ctx.info("Choose both image files first")
  })
  compareUpload.disableProperty.bind(loading)
  private val dropA = new ImageDropZone(ctx, inputA, "A · first image", 110)
  private val dropB = new ImageDropZone(ctx, inputB, "B · second image", 110)
  dropA.node.setId("comparison-input-a")
  dropB.node.setId("comparison-input-b")
  Seq(dropA, dropB).foreach { drop =>
    drop.node.setFocusTraversable(true)
    drop.node.addEventHandler(javafx.scene.input.MouseEvent.MOUSE_PRESSED,
      (_: javafx.scene.input.MouseEvent) => drop.node.requestFocus())
    drop.node.setOnKeyPressed(e =>
      if e.isShortcutDown && e.getCode == javafx.scene.input.KeyCode.V then
        drop.paste()
        e.consume()
    )
  }
  private val inputs = new HBox(10, dropA.node, dropB.node)
  HBox.setHgrow(dropA.node, Priority.ALWAYS)
  HBox.setHgrow(dropB.node, Priority.ALWAYS)
  private val sources = new FlowPane(6, 6, compareUpload,
    identified("comparison-img2img", W.button("Compare img2img", "Input versus shown img2img result")(compareSource("img2img"))),
    identified("comparison-inpaint", W.button("Compare Inpaint", "Input versus shown Inpaint result")(compareSource("inpaint"))),
    identified("comparison-extras", W.button("Compare Extras", "Upscale input versus output")(compareSource("extras"))),
    identified("comparison-previous", W.button("Compare previous", "Current txt2img result versus previous generation")(compareSource("txt2img"))),
    W.button("Selected → A", "Send the selected gallery/preview image to A")(selected().foreach(p => send(false, p))),
    W.button("Selected → B", "Send the selected gallery/preview image to B")(selected().foreach(p => send(true, p))))
  private val controls = new FlowPane(8, 6,
    identified("comparison-swap", W.button("Swap A / B", "Exchange images and metadata")(swap())), direction,
    identified("comparison-in", W.button("+", "Zoom in by100%")(canvas.update(canvas.view.zoomBy(1)))),
    identified("comparison-reset", W.button("Reset view", "Reset zoom and position; keep reveal and opacity")(canvas.resetView())),
    identified("comparison-out", W.button("−", "Zoom out by100%")(canvas.update(canvas.view.zoomBy(-1)))),
    identified("comparison-export", W.button("Export PNG…", "Flatten the current comparison to a new file")(exportDialog())))
  private val sliders = new FlowPane(10, 6, W.label("Reveal"), edge, W.label("B opacity"), opacity,
    W.label("Zoom"), zoom, zoomLabel)
  private val pans = new FlowPane(6, 6, W.label("Pan"),
    W.button("←", "Pan left")(canvas.update(canvas.view.panBy(100, 0))),
    W.button("→", "Pan right")(canvas.update(canvas.view.panBy(-100, 0))),
    W.button("↑", "Pan up")(canvas.update(canvas.view.panBy(0, 100))),
    W.button("↓", "Pan down")(canvas.update(canvas.view.panBy(0, -100))))
  private val help = W.label("Drag the edge · Shift+drag: pan · Shift+wheel or +/−: zoom · arrows: pan · Backspace or Shift+0: reset. Keyboard sliders offer the same controls. Different sizes are fitted without stretching.", "muted")
  help.setWrapText(true)
  help.setMinHeight(javafx.scene.layout.Region.USE_PREF_SIZE)
  private val infoColumns = new HBox(14, firstInfo, secondInfo)
  HBox.setHgrow(firstInfo, Priority.ALWAYS)
  HBox.setHgrow(secondInfo, Priority.ALWAYS)
  firstInfo.prefWidthProperty.bind(infoColumns.widthProperty.divide(2).subtract(7))
  secondInfo.prefWidthProperty.bind(firstInfo.prefWidthProperty)
  private val infoPane = new TitledPane("Infotext · changed tags and fields are marked in red", infoColumns)
  infoPane.setId("comparison-infotext")
  infoPane.setExpanded(false)
  private val options = new VBox(8)
  private def setting(label: String, current: String)(save: String => Unit): ComboBox[String] =
    val c = new ComboBox[String](FXCollections.observableArrayList("Off", "Text", "Icon"))
    c.setValue(current)
    c.setOnAction(_ => { save(c.getValue); onOptionsChanged() })
    options.getChildren.add(W.hbox(8, W.label(label), c))
    c
  setting("img2img / Inpaint / Extras send buttons", ctx.services.settings.comparisonSendButtons) { v =>
    ctx.services.updateSettings(_.copy(comparisonSendButtons = v))
  }
  setting("txt2img previous-result send button", ctx.services.settings.comparisonTxtButton) { v =>
    ctx.services.updateSettings(_.copy(comparisonTxtButton = v))
  }
  private def check(text: String, current: Boolean)(save: Boolean => Unit): Unit =
    val c = new CheckBox(text)
    c.setSelected(current)
    c.setOnAction(_ => save(c.isSelected))
    options.getChildren.add(c)
  check("Ctrl+Alt+left/right click an image to send to A/B", ctx.services.settings.comparisonMouseSend)(v =>
    ctx.services.updateSettings(_.copy(comparisonMouseSend = v)))
  check("Switch to Comparison after image shortcut", ctx.services.settings.comparisonMouseAuto)(v =>
    ctx.services.updateSettings(_.copy(comparisonMouseAuto = v)))
  private val optionPane = new TitledPane("Comparison options", options)
  optionPane.setExpanded(false)
  private val content = new VBox(10, W.label("Image comparison", "section-title"), status, controls, sliders, pans,
    canvas, help, inputs, sources, infoPane, optionPane)
  content.setPadding(new Insets(12))
  canvas.prefHeightProperty.bind(javafx.beans.binding.Bindings.min(650, javafx.beans.binding.Bindings.max(340, content.widthProperty.multiply(.55))))
  canvas.setMinHeight(javafx.scene.layout.Region.USE_PREF_SIZE)
  val node = new ScrollPane(content)
  node.setId("comparison-tool")
  node.setFitToWidth(true)
  node.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
