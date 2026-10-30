package sdfx.ui.view

import javafx.animation.PauseTransition
import javafx.geometry.Pos
import javafx.scene.Node
import javafx.scene.control.{CheckBox, Label}
import javafx.scene.image.{Image, ImageView}
import javafx.scene.layout.{HBox, VBox}
import javafx.util.Duration
import sdfx.engine.InpaintDifferencePreview
import sdfx.ui.Fx
import sdfx.ui.vm.{GenerationVM, InpaintDifferenceInputs, LatestPreview, PluginValuesVM}

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import scala.jdk.CollectionConverters.*
import scala.concurrent.ExecutionContext
import scala.util.{Failure, Success, Try}
import scala.util.control.NonFatal

/** Native controls and owned asynchronous preview for the two-image inpaint mode. */
final class InpaintDifferenceControls(ctx: UiContext, vm: GenerationVM, form: PluginValuesVM,
                                      inputs: InpaintDifferenceInputs):
  private case class Request(base: String, altered: String, options: ujson.Obj)
  private given ExecutionContext = ExecutionContext.parasitic
  private val rootDir = ctx.tmpDir.resolve("inpaint-difference-previews").toAbsolutePath.normalize()
  private val image = new ImageView()
  image.setPreserveRatio(true); image.setSmooth(true); image.setFitHeight(280); image.setFitWidth(520)
  private val status = new Label("Choose a base and an altered image")
  status.setWrapText(true); status.setId("difference-preview-status")
  private var displayed: Option[InpaintDifferencePreview.Artifact] = None
  private var closed = false
  private val shuttingDown = new AtomicBoolean(false)
  private val pendingDelivery = ConcurrentHashMap.newKeySet[InpaintDifferencePreview.Artifact]()
  private def remove(a: InpaintDifferencePreview.Artifact): Unit =
    try a.cleanup() catch case NonFatal(_) => ()
  private def clear(): Unit =
    image.setImage(null)
    displayed.foreach(remove)
    displayed = None

  private val work: LatestPreview[Request, InpaintDifferencePreview.Artifact] = new LatestPreview[Request, InpaintDifferencePreview.Artifact](
    r => InpaintDifferencePreview.run(ctx.services, r.base, r.altered, rootDir, r.options),
    (id, outcome) =>
      outcome.foreach(pendingDelivery.add)
      if shuttingDown.get then outcome.foreach(a => if pendingDelivery.remove(a) then remove(a))
      else Fx.run {
      val owned = outcome match
        case Success(a) => pendingDelivery.remove(a)
        case Failure(_) => true
      if owned && (closed || !work.isCurrent(id)) then outcome.foreach(remove)
      else if owned then outcome match
        case Success(artifact) =>
          clear()
          displayed = Some(artifact)
          val loaded = new Image(artifact.preview.toUri.toString, true)
          image.setImage(loaded)
          status.setText("Loading preview…")
          ImageFiles.whenLoaded(loaded) {
            if !closed && work.isCurrent(id) && displayed.contains(artifact) then
              if loaded.isError then
                clear(); status.setText("Could not open the generated preview")
              else status.setText(s"Difference mask · ${artifact.width} × ${artifact.height} · before resize/crop")
          }
        case Failure(error) =>
          clear(); status.setText(s"Preview unavailable: ${Fx.message(error)}")
    },
    remove)
  private val debounce = new PauseTransition(Duration.millis(300))

  private def active = form.isActive
  private def file(path: String): Boolean =
    Option(path).exists(p => p.nonEmpty && Try(Files.isRegularFile(Paths.get(p))).getOrElse(false))
  private def refresh(): Unit =
    debounce.stop()
    work.invalidate()
    clear()
    if !active then status.setText("Difference mode is off")
    else if !file(form.props("base_path").get) || !file(inputs.altered.get) then
      status.setText("Choose a base and an altered image")
    else
      status.setText("Waiting to update preview…")
      debounce.setOnFinished(_ =>
        try
          val values = form.values
          val opts = form.plugin.asInstanceOf[sdfx.plugins.InpaintDifferencePlugin].hookFields(values)
          opts("blur") = vm.maskBlur.get
          opts("color") = values.str("color", "#ffffff")
          opts("show_under") = values.bool("show_under", true)
          opts("invert") = vm.maskInvert.get
          opts("preview_mode") = "effective"
          status.setText("Calculating preview…")
          work.request(Request(values.str("base_path"), inputs.altered.get, opts))
        catch case NonFatal(error) => status.setText(s"Preview unavailable: ${Fx.message(error)}")
      )
      debounce.playFromStart()

  private val toggle = new CheckBox("Create mask from image difference")
  toggle.setId("difference-enabled")
  toggle.setSelected(active)
  toggle.selectedProperty.addListener((_, _, value) => form.props("enabled").set(value.toString))
  form.props("enabled").addListener((_, _, _) => if toggle.isSelected != active then toggle.setSelected(active))
  private val base = new ImageDropZone(ctx, form.props("base_path"), "Base image", 210)
  private val altered = new ImageDropZone(ctx, inputs.altered, "Altered image to inpaint", 210)
  base.node.setId("difference-base"); altered.node.setId("difference-altered")
  private val swap = W.button("⇄ Swap images", "Swap base and altered images") {
    val old = form.props("base_path").get
    form.props("base_path").set(inputs.altered.get)
    inputs.altered.set(old)
  }
  swap.setId("difference-swap")
  private val baseColumn = new VBox(4, W.label("Base image", "muted"), base.node)
  private val alteredColumn = new VBox(4, W.label("Altered image · used for inpainting", "muted"), altered.node)
  private val images = new HBox(8, baseColumn, alteredColumn)
  images.setAlignment(Pos.CENTER)
  javafx.scene.layout.HBox.setHgrow(baseColumn, javafx.scene.layout.Priority.ALWAYS)
  javafx.scene.layout.HBox.setHgrow(alteredColumn, javafx.scene.layout.Priority.ALWAYS)
  private val controls = PluginForm.render(ctx, form, form.plugin.controls.filterNot(c =>
    Set("enabled", "base_path", "painted_path", "altered_path")(c.key)))
  private val detail = new VBox(8, images, swap, controls, status, image)
  detail.visibleProperty.bind(toggle.selectedProperty); detail.managedProperty.bind(toggle.selectedProperty)
  val node: Node = new VBox(8, toggle, detail)
  image.setId("difference-preview")
  form.props.values.foreach(_.addListener((_, _, _) => refresh()))
  inputs.altered.addListener((_, _, _) => refresh())
  vm.maskBlur.addListener((_, _, _) => refresh())
  vm.maskInvert.addListener((_, _, _) => refresh())
  refresh()

  def enabled: Boolean = active
  def basePath: String = form.props("base_path").get
  def referencedImages: Seq[String] = Seq(basePath, inputs.painted.get, inputs.altered.get).filter(s => s != null && s.nonEmpty)
  def dispose(): Unit =
    if !closed then
      closed = true
      shuttingDown.set(true)
      debounce.stop(); work.close(); clear()
      pendingDelivery.asScala.toSeq.foreach(a => if pendingDelivery.remove(a) then remove(a))
