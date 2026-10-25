package sdfx.ui.view

import javafx.scene.image.{Image, ImageView}
import javafx.scene.input.{MouseButton, ScrollEvent}
import javafx.scene.layout.Region
import javafx.scene.shape.Rectangle
import sdfx.ui.vm.MaskGeometry

/** Image viewer: fit by default, Ctrl+wheel zoom around the cursor, drag to pan, double click = fit / 100%. */
final class ZoomImageView extends Region:
  private val view = new ImageView()
  view.setSmooth(true)
  getChildren.add(view)
  getStyleClass.add("zoom-view")
  private val clip = new Rectangle()
  clip.widthProperty.bind(widthProperty)
  clip.heightProperty.bind(heightProperty)
  setClip(clip)

  private var zoom: Option[Double] = None // None = fit
  private var ox = 0.0
  private var oy = 0.0
  private var dragFrom: Option[(Double, Double, Double, Double)] = None

  def image: Image = view.getImage

  private var lastSize: Option[(Double, Double)] = None

  /** Shows an image. Until a background image is loaded the view is hidden (no stretched stale fit);
    * the zoom is kept only when the loaded image has the same size as the previous one.
    */
  def setImage(img: Image): Unit =
    view.setImage(img)
    if img == null then
      zoom = None
      lastSize = None
      requestLayout()
    else
      view.setVisible(false)
      ImageFiles.whenLoaded(img) {
        if view.getImage eq img then
          val size = (img.getWidth, img.getHeight)
          if !lastSize.contains(size) then zoom = None
          lastSize = Some(size)
          view.setVisible(!img.isError)
          requestLayout()
      }

  def fit(): Unit =
    zoom = None
    requestLayout()

  private def imgSize: (Double, Double) =
    val i = view.getImage
    if i == null then (0.0, 0.0) else (i.getWidth, i.getHeight)

  private def currentScale: Double =
    val (iw, ih) = imgSize
    zoom.getOrElse(MaskGeometry.fit(iw, ih, getWidth, getHeight).scale)

  override protected def layoutChildren(): Unit =
    val (iw, ih) = imgSize
    if iw > 0 && ih > 0 then
      zoom match
        case None =>
          val f = MaskGeometry.fit(iw, ih, getWidth, getHeight)
          ox = f.offsetX
          oy = f.offsetY
          view.setFitWidth(f.width)
          view.setFitHeight(f.height)
        case Some(z) =>
          view.setFitWidth(iw * z)
          view.setFitHeight(ih * z)
          // keep the zoomed image reachable after resizes: center a dimension that fits, else clamp the pan
          ox = clampOffset(ox, iw * z, getWidth)
          oy = clampOffset(oy, ih * z, getHeight)
      view.relocate(ox, oy)

  private def clampOffset(o: Double, size: Double, box: Double): Double =
    if size <= box then (box - size) / 2 else o.min(0).max(box - size)

  private def zoomAt(x: Double, y: Double, z: Double): Unit =
    val s = currentScale
    val nz = z.max(0.05).min(16)
    val ix = (x - ox) / s
    val iy = (y - oy) / s
    ox = x - ix * nz
    oy = y - iy * nz
    zoom = Some(nz)
    requestLayout()

  addEventFilter(ScrollEvent.SCROLL, (e: ScrollEvent) =>
    if e.isShortcutDown && view.getImage != null then
      val f = if e.getDeltaY > 0 then 1.2 else 1 / 1.2
      zoomAt(e.getX, e.getY, currentScale * f)
      e.consume()
  )
  setOnMousePressed(e =>
    if e.getButton == MouseButton.PRIMARY && zoom.isDefined then
      dragFrom = Some((e.getX, e.getY, ox, oy))
      setCursor(javafx.scene.Cursor.CLOSED_HAND)
  )
  setOnMouseDragged(e =>
    dragFrom.foreach { (sx, sy, sox, soy) =>
      ox = sox + e.getX - sx
      oy = soy + e.getY - sy
      requestLayout()
    }
  )
  setOnMouseReleased(_ =>
    dragFrom = None
    setCursor(javafx.scene.Cursor.DEFAULT)
  )
  setOnMouseClicked(e =>
    if e.getButton == MouseButton.PRIMARY && e.getClickCount == 2 && view.getImage != null then
      if zoom.isDefined then fit() else zoomAt(e.getX, e.getY, 1.0)
  )
