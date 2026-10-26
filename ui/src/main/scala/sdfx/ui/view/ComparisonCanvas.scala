package sdfx.ui.view

import javafx.scene.canvas.Canvas
import javafx.scene.image.Image
import javafx.scene.input.{KeyCode, MouseButton}
import javafx.scene.layout.Region
import javafx.scene.paint.Color
import sdfx.core.comparison.ImageComparison.*

/** Both images share a transform and reveal plane. Input pixels never enter a mutable editor. */
final class ComparisonCanvas extends Region:
  private val canvas = new Canvas()
  getChildren.add(canvas)
  setId("comparison-canvas")
  setFocusTraversable(true)
  setAccessibleText("Image comparison. Drag to reveal B; Shift drag to pan; plus/minus zoom; arrows pan; Backspace resets view.")
  setMinSize(240, 280)
  setPrefSize(700, 500)
  private var a: Image = null
  private var b: Image = null
  private var state = View()
  var onViewChanged: View => Unit = _ => ()
  def view: View = state
  def update(v: View): Unit =
    state = v.copy(split = clamp(v.split), opacity = clamp(v.opacity), zoom = v.zoom.max(.5).min(10))
    redraw()
    onViewChanged(state)
  def setImages(first: Image, second: Image): Unit =
    a = first
    b = second
    update(state.forNewPair)
  def resetView(): Unit = update(state.reset)
  override protected def layoutChildren(): Unit =
    canvas.setWidth(getWidth)
    canvas.setHeight(getHeight)
    redraw()
  private def redraw(): Unit =
    val w = canvas.getWidth
    val h = canvas.getHeight
    if w <= 0 || h <= 0 then return
    val g = canvas.getGraphicsContext2D
    g.setGlobalAlpha(1)
    g.setFill(Color.web("#17212b"))
    g.fillRect(0, 0, w, h)
    def draw(im: Image): Unit =
      if im != null && !im.isError && im.getWidth > 0 then
        val r = fit(im.getWidth, im.getHeight, plane(w, h, state))
        g.drawImage(im, r.x, r.y, r.width, r.height)
    draw(a)
    val r = reveal(w, h, state)
    g.save()
    g.beginPath()
    g.rect(r.x, r.y, r.width, r.height)
    g.clip()
    g.setGlobalAlpha(state.opacity)
    draw(b)
    g.restore()
    if a != null && b != null then
      g.setStroke(Color.WHITE)
      g.setLineWidth(2)
      if state.horizontal then g.strokeLine(r.x, 0, r.x, h) else g.strokeLine(0, r.y, w, r.y)
      g.setFill(Color.web("#17212b", .85))
      g.fillRoundRect(10, 10, 28, 24, 6, 6)
      g.fillRoundRect(w - 38, h - 34, 28, 24, 6, 6)
      g.setFill(Color.WHITE)
      g.fillText("A", 19, 27)
      g.fillText("B", w - 29, h - 17)
    else
      g.setFill(Color.web("#aebdcc"))
      g.fillText("Load two images to begin", 24, 40)

  private var dragX = 0d
  private var dragY = 0d
  private def moveEdge(x: Double, y: Double): Unit =
    if getWidth > 0 && getHeight > 0 then update(state.copy(split = revealAt(x, y, getWidth, getHeight, state)))
  setOnMousePressed(e =>
    if e.getButton == MouseButton.PRIMARY then
      requestFocus()
      dragX = e.getX
      dragY = e.getY
      if !e.isShiftDown then moveEdge(e.getX, e.getY)
      e.consume()
  )
  setOnMouseDragged(e =>
    if e.isPrimaryButtonDown then
      if e.isShiftDown then update(state.panBy(e.getX - dragX, e.getY - dragY))
      else moveEdge(e.getX, e.getY)
      dragX = e.getX
      dragY = e.getY
      e.consume()
  )
  setOnTouchPressed(e => { requestFocus(); moveEdge(e.getTouchPoint.getX, e.getTouchPoint.getY); e.consume() })
  setOnTouchMoved(e => { moveEdge(e.getTouchPoint.getX, e.getTouchPoint.getY); e.consume() })
  setOnScroll(e =>
    if e.isShiftDown then
      update(state.zoomBy(if e.getDeltaY > 0 then .5 else if e.getDeltaY < 0 then -.5 else 0))
      e.consume()
  )
  setOnKeyPressed(e =>
    val next = e.getCode match
      case KeyCode.ADD | KeyCode.PLUS | KeyCode.EQUALS => Some(state.zoomBy(.5))
      case KeyCode.SUBTRACT | KeyCode.MINUS => Some(state.zoomBy(-.5))
      case KeyCode.LEFT => Some(state.panBy(100, 0))
      case KeyCode.RIGHT => Some(state.panBy(-100, 0))
      case KeyCode.UP => Some(state.panBy(0, 100))
      case KeyCode.DOWN => Some(state.panBy(0, -100))
      case KeyCode.BACK_SPACE | KeyCode.INSERT => Some(state.reset)
      case KeyCode.DIGIT0 if e.isShiftDown => Some(state.reset)
      case _ => None
    next.foreach(v => { update(v); e.consume() })
  )
