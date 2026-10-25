package sdfx.ui.view

import javafx.beans.property.{SimpleBooleanProperty, SimpleDoubleProperty, StringProperty}
import javafx.geometry.Pos
import javafx.scene.canvas.Canvas
import javafx.scene.control.{Label, ToggleButton, ToggleGroup}
import javafx.scene.image.{Image, ImageView, PixelFormat, WritableImage}
import javafx.scene.input.{MouseButton, MouseEvent, ScrollEvent}
import javafx.scene.layout.{StackPane, VBox}
import javafx.scene.paint.Color
import sdfx.ui.Fx
import sdfx.ui.vm.{MaskGeometry, MaskModel}

import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO
import scala.util.control.NonFatal

/** Inpaint mask painter: init image + mask overlay at image resolution, scaled to fit.
  * Left drag paints, right drag (or eraser) erases, Ctrl+wheel changes the brush, Ctrl+Z / Ctrl+Y undo / redo.
  */
final class MaskCanvas(ctx: UiContext, imagePath: StringProperty, hasMask: javafx.beans.property.BooleanProperty, height: Double = 420):
  val brushSize = new SimpleDoubleProperty(40)
  val eraser = new SimpleBooleanProperty(false)
  val overlayOpacity = new SimpleDoubleProperty(0.6)

  private var model: Option[MaskModel] = None
  private var overlay: Option[WritableImage] = None
  private val OverlayArgb = 0xff7c5cff

  private val base = new ImageView()
  base.setPreserveRatio(true)
  base.setSmooth(true)
  private val over = new ImageView()
  over.setPreserveRatio(true)
  over.setSmooth(false)
  over.opacityProperty.bind(overlayOpacity)
  private val cursor = new Canvas()
  cursor.setMouseTransparent(true)
  private val placeholder = new Label("◐\nOpen, drop or paste the image to inpaint\nthen paint over the area to change")
  placeholder.getStyleClass.add("drop-hint")

  private val stack = new StackPane(placeholder, base, over, cursor)
  stack.getStyleClass.addAll("drop-zone", "mask-canvas")
  stack.setMinHeight(height)
  stack.setMinWidth(0)
  stack.setPrefHeight(height)
  Seq(base, over).foreach { v =>
    v.fitWidthProperty.bind(stack.widthProperty.subtract(16))
    v.fitHeightProperty.bind(stack.heightProperty.subtract(16))
  }
  cursor.widthProperty.bind(stack.widthProperty)
  cursor.heightProperty.bind(stack.heightProperty)
  StackPane.setAlignment(cursor, Pos.TOP_LEFT)
  cursor.setManaged(false)

  /** The drop zone logic is reused for drag & drop / paste / open. */
  private val input = new ImageInput(ctx, imagePath)
  input.install(stack)

  def open(): Unit = input.open()
  def paste(): Boolean = input.paste()

  // ---- painting ----------------------------------------------------------------------------------------

  private var last: Option[(Double, Double)] = None
  private var strokeErase = false

  /** Image pixel per displayed pixel. */
  private def scale: Double = model.fold(1.0)(m => over.getBoundsInLocal.getWidth / m.width)

  private def toImage(e: MouseEvent): (Double, Double) =
    val p = over.sceneToLocal(e.getSceneX, e.getSceneY)
    val s = scale
    (p.getX / s, p.getY / s)

  over.setOnMousePressed(e =>
    model.foreach { m =>
      strokeErase = eraser.get || e.getButton == MouseButton.SECONDARY
      m.checkpoint()
      val pt = toImage(e)
      paint(m, pt, pt)
      last = Some(pt)
    }
  )
  over.setOnMouseDragged(e =>
    model.foreach { m =>
      val pt = toImage(e)
      paint(m, last.getOrElse(pt), pt)
      last = Some(pt)
    }
    drawCursor(e)
  )
  over.setOnMouseReleased(_ =>
    last = None
    changed()
  )
  over.setOnMouseMoved(e => drawCursor(e))
  over.setOnMouseExited(_ => cursor.getGraphicsContext2D.clearRect(0, 0, cursor.getWidth, cursor.getHeight))
  stack.addEventFilter(ScrollEvent.SCROLL, (e: ScrollEvent) =>
    if e.isShortcutDown then
      val f = if e.getDeltaY > 0 then 1.15 else 1 / 1.15
      brushSize.set((brushSize.get * f).max(2).min(400))
      e.consume()
  )

  private def paint(m: MaskModel, from: (Double, Double), to: (Double, Double)): Unit =
    val r = MaskGeometry.brushRadius(brushSize.get, scale)
    m.stroke(from._1, from._2, to._1, to._2, r, strokeErase)
    val (x, y, w, h) = MaskGeometry.segmentBounds(from._1, from._2, to._1, to._2, r, m.width, m.height)
    updateOverlay(x, y, w, h)

  private def drawCursor(e: MouseEvent): Unit =
    val g = cursor.getGraphicsContext2D
    g.clearRect(0, 0, cursor.getWidth, cursor.getHeight)
    val p = cursor.sceneToLocal(e.getSceneX, e.getSceneY)
    val r = brushSize.get / 2
    g.setStroke(Color.WHITE)
    g.setLineWidth(1.5)
    g.strokeOval(p.getX - r, p.getY - r, r * 2, r * 2)
    g.setStroke(Color.rgb(0, 0, 0, 0.6))
    g.setLineWidth(1)
    g.strokeOval(p.getX - r - 1.5, p.getY - r - 1.5, r * 2 + 3, r * 2 + 3)

  private def updateOverlay(x: Int, y: Int, w: Int, h: Int): Unit =
    for m <- model; img <- overlay if w > 0 && h > 0 do
      val buf = new Array[Int](w * h)
      var j = 0
      while j < h do
        var i = 0
        while i < w do
          buf(j * w + i) = if m.valueAt(x + i, y + j) != 0 then OverlayArgb else 0
          i += 1
        j += 1
      img.getPixelWriter.setPixels(x, y, w, h, PixelFormat.getIntArgbInstance, buf, 0, w)

  private def redrawAll(): Unit = model.foreach(m => updateOverlay(0, 0, m.width, m.height))

  private def changed(): Unit = hasMask.set(model.exists(!_.isEmpty))

  /** Dev helper (--demo-mask): paints a sample stroke. */
  def paintDemo(): Unit = model.foreach { m =>
    m.checkpoint()
    m.stroke(m.width * 0.3, m.height * 0.45, m.width * 0.65, m.height * 0.55, m.width * 0.09, erase = false)
    redrawAll()
    changed()
  }

  def undo(): Unit =model.foreach(m => if m.undo() then { redrawAll(); changed() })
  def redo(): Unit = model.foreach(m => if m.redo() then { redrawAll(); changed() })
  def clear(): Unit = model.foreach { m => m.checkpoint(); m.clear(); redrawAll(); changed() }
  def invert(): Unit = model.foreach { m => m.checkpoint(); m.invert(); redrawAll(); changed() }

  /** Writes the mask (white = inpaint) into the temp dir; None when there is no image. */
  def exportMask(): Option[Path] =
    model.map(m => m.writePng(ctx.tmpFile("mask-", ".png")))

  /** Copies the mask on the FX thread and PNG-encodes it in the background; `done(None)` when there is no image. */
  def exportMaskAsync(done: Option[Path] => Unit, failed: Throwable => Unit): Unit =
    model match
      case None => done(None)
      case Some(m) =>
        // a synchronous failure (copy) is reported like an async one, so callers always get a callback
        val copy =
          try Some(m.copyImage())
          catch
            case scala.util.control.NonFatal(e) =>
              failed(e)
              None
        copy.foreach(c => Fx.async(MaskModel.writePng(c, ctx.tmpFile("mask-", ".png")))(p => done(Some(p)))(failed))

  // ---- image loading -----------------------------------------------------------------------------------

  private def setImage(img: Option[Image]): Unit =
    base.setImage(img.orNull)
    placeholder.setVisible(img.isEmpty)
    model = None
    overlay = None
    over.setImage(null)
    changed()
    img.foreach { i =>
      // only the image that is still shown may create the mask grid (A loading late must not win over B)
      def ready(): Unit =
        if (base.getImage eq i) && !i.isError && i.getWidth > 0 then
          val m = new MaskModel(i.getWidth.toInt, i.getHeight.toInt)
          val w = new WritableImage(m.width, m.height)
          model = Some(m)
          overlay = Some(w)
          over.setImage(w)
          changed()
      ImageFiles.whenLoaded(i)(ready())
    }

  /** Loads an existing mask file into the current mask (e.g. restored state). */
  def loadMask(file: String): Unit =
    for m <- model if file.nonEmpty do
      Fx.async(Option(ImageIO.read(Paths.get(file).toFile))) { src =>
        if model.exists(_ eq m) then
          src.foreach { s =>
            m.checkpoint()
            m.loadFrom(s)
            redrawAll()
            changed()
          }
      }(e => ctx.error(s"Cannot load mask: ${Fx.message(e)}"))

  imagePath.addListener((_, _, p) => setImage(ImageFiles.load(p)))
  setImage(ImageFiles.load(imagePath.get))

  // ---- tools -------------------------------------------------------------------------------------------

  private val group = new ToggleGroup()
  private val brushBtn = new ToggleButton("✎ Brush")
  private val eraserBtn = new ToggleButton("⌫ Eraser")
  brushBtn.setToggleGroup(group)
  eraserBtn.setToggleGroup(group)
  brushBtn.setSelected(true)
  W.tip(brushBtn, "Paint the area to regenerate (left drag)")
  W.tip(eraserBtn, "Erase painted mask (or drag with the right mouse button)")
  eraserBtn.selectedProperty.bindBidirectional(eraser)
  group.selectedToggleProperty.addListener((_, old, n) => if n == null then group.selectToggle(old))

  val tools = W.hbox(
    6,
    brushBtn,
    eraserBtn,
    W.iconButton("↶", "Undo (Ctrl+Z)")(undo()),
    W.iconButton("↷", "Redo (Ctrl+Y)")(redo()),
    W.iconButton("◑", "Invert the mask")(invert()),
    W.button("Clear", "Clear the whole mask")(clear()),
    W.spacer(),
    W.iconButton("…", "Open an image")(open())
  )

  val sliders: javafx.scene.Node =
    W.form()
      .add("Brush size", W.doubleSlider(brushSize, 2, 400, 1), "Brush diameter in screen pixels (Ctrl+mouse wheel over the image)")
      .add("Mask opacity", W.doubleSlider(overlayOpacity, 0.1, 1, 0.05), "How strongly the painted mask is shown (display only)")
      .grid

  val node: VBox = new VBox(8, tools, stack, sliders)
