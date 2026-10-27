package sdfx.ui.view

import javafx.animation.{Interpolator, KeyFrame, KeyValue, Timeline}
import javafx.application.Platform
import javafx.beans.{InvalidationListener, Observable}
import javafx.beans.value.ChangeListener
import javafx.geometry.{Insets, Pos}
import javafx.scene.{Node, Scene}
import javafx.scene.control.{Button, Label, OverrunStyle, Tooltip}
import javafx.scene.effect.GaussianBlur
import javafx.scene.image.{Image, ImageView}
import javafx.scene.input.KeyEvent
import javafx.scene.layout.{HBox, Priority, Region, StackPane, VBox}
import javafx.stage.Window
import javafx.util.Duration
import sdfx.core.modelcards.*
import sdfx.engine.CardPreviewSession

private[ui] final case class PreviewBlurTransition(durationMillis: Double,
    spline: (Double, Double, Double, Double))
private[ui] final case class PreviewCardVisual(record: Option[ModelCardRecord], image: Option[Image],
    presentation: PreviewPresentation, targetBlurRadius: Double, transition: Option[PreviewBlurTransition],
    previewVisible: Boolean, previewManaged: Boolean, revealControlVisible: Boolean, status: Option[String])

/** The native card delegates all policy, attachment and ticket lifetime decisions to this controller. */
private[ui] final class ModelPreviewCardState(session: CardPreviewSession,
    request: (PreviewReference, Int, PreviewImageResult => Unit) => PreviewTicket,
    render: PreviewCardVisual => Unit,
    sessionDispatch: (() => Unit) => Unit = task => task()):
  private var record: Option[ModelCardRecord] = None
  private var revision = 0L
  private var token = 0L
  private var ticket: Option[PreviewTicket] = None
  private var image: Option[Image] = None
  private var status: Option[String] = None
  private var hovered = false
  private var revealFocused = false
  private var windowActive = true
  private var held = Set.empty[String]
  private var blocked = Set.empty[String]
  private var disposed = false
  private var visibleCapacityRefused = false
  private var previousRadius = 20.0
  private val unsubscribe = session.addListener(_ => sessionDispatch(() => modeChanged()))

  private def source: Option[String] = record.flatMap(_.preview).map(_.source)
  private def matched: Boolean = CardPreviewPolicy.matches(source)
  private def revealed: Boolean = windowActive && (hovered || (revealFocused && held.nonEmpty))
  private def presentation: PreviewPresentation = CardPreviewPolicy.presentation(source, session.current, revealed)
  private def publish(animate: Boolean = false): Unit =
    if !disposed then
      val display = presentation
      val radius = if display == PreviewPresentation.Blurred then 20.0 else 0.0
      val hidden = display == PreviewPresentation.Hidden
      val transition =
        if animate && image.nonEmpty && !hidden && radius != previousRadius then
          Some(PreviewBlurTransition(150.0, (0.79, 0.14, 0.15, 0.86)))
        else None
      previousRadius = radius
      render(PreviewCardVisual(record, if hidden then None else image, display, radius, transition,
        !hidden && image.nonEmpty, !hidden, matched && session.current == CardPreviewMode.Blur,
        status.orElse(record.flatMap(_.problem))))

  private def resetReveal(): Unit =
    hovered = false
    // Recycling must not turn an old key's auto-repeat into a new deliberate hold.
    blocked ++= held
    held = Set.empty
    revealFocused = false

  /** Detach native pixels synchronously before releasing their loader reference. */
  private def dropTicket(): Unit =
    token += 1
    image = None
    visibleCapacityRefused = false
    publish()
    val old = ticket
    ticket = None
    old.foreach(_.cancel())

  def bind(next: ModelCardRecord, nextRevision: Long): Unit =
    if !disposed then
      resetReveal()
      dropTicket()
      record = Some(next)
      revision = nextRevision
      status = if next.preview.isEmpty then Some("No preview") else Some("Loading preview…")
      publish()
      begin()

  private def begin(): Unit =
    if !disposed && ticket.isEmpty && presentation != PreviewPresentation.Hidden then
      record.flatMap(_.preview).foreach { reference =>
        val expectedToken = token
        val expectedRevision = revision
        val expectedId = record.get.id
        val next = request(reference, 112, result =>
          val currentReference = record.flatMap(_.preview)
          val valid = !disposed && expectedToken == token && expectedRevision == revision &&
            record.exists(_.id == expectedId) && currentReference.exists(ref =>
              ref.source == reference.source && ref.identity == reference.identity &&
                ref.localPath == reference.localPath && ref.allowedRealParent == reference.allowedRealParent)
          if valid && presentation != PreviewPresentation.Hidden then
            result match
              case PreviewImageResult.Ready(pixels) =>
                image = Some(pixels)
                visibleCapacityRefused = false
                status = if reference.localPath.toString.toLowerCase(java.util.Locale.ROOT).endsWith(".gif") then
                  Some("GIF · first frame only") else None
              case PreviewImageResult.Unavailable(reason) =>
                image = None
                status = Some(reason)
                visibleCapacityRefused = reason == PreviewImageLoader.visibleCapacityReason
            publish() // Never animate from unconcealed pixels at an asynchronous attachment.
        )
        if !disposed && expectedToken == token && presentation != PreviewPresentation.Hidden then ticket = Some(next)
        else next.cancel()
      }

  private def modeChanged(): Unit =
    if !disposed then
      hovered = false
      blocked ++= held
      held = Set.empty
      if presentation == PreviewPresentation.Hidden then
        status = Some("Preview hidden by file-name policy")
        dropTicket()
      else
        publish()
        begin()

  def setHover(value: Boolean): Unit =
    if !disposed then
      hovered = value && windowActive && session.current == CardPreviewMode.Blur
      publish(animate = true)

  def setRevealFocused(value: Boolean): Unit =
    if !disposed then
      revealFocused = value
      if !value then dismissReveal()
      else publish()

  def setWindowActive(value: Boolean): Unit =
    if !disposed then
      windowActive = value
      if !value then dismissReveal()
      else publish()

  def revealKey(key: String, pressed: Boolean): Unit =
    if !disposed && (key == "SPACE" || key == "ENTER") then
      if !pressed then
        held -= key
        blocked -= key
      else if revealFocused && windowActive && matched && session.current == CardPreviewMode.Blur && !blocked(key) then
        held += key
      publish(animate = true)

  private def dismissReveal(): Unit =
    blocked ++= held
    held = Set.empty
    hovered = false
    publish(animate = true)

  def escape(): Unit = if !disposed then dismissReveal()

  /** Called only by the browser's shared capacity-release subscription, never by a timer. */
  def retryVisibleCapacity(): Unit =
    if !disposed && visibleCapacityRefused && presentation != PreviewPresentation.Hidden then
      dropTicket()
      status = Some("Loading preview…")
      publish()
      begin()

  def clear(): Unit =
    if !disposed then
      resetReveal()
      record = None
      status = None
      dropTicket()

  def dispose(): Unit =
    if !disposed then
      clear()
      disposed = true
      unsubscribe()

/** Preview-only effects; selection and insertion remain explicit browser-owned actions. */
final class ModelPreviewCard(session: CardPreviewSession, loader: PreviewImageLoader):
  private val pixels = new ImageView()
  pixels.setFitWidth(112)
  pixels.setFitHeight(112)
  pixels.setPreserveRatio(true)
  pixels.setSmooth(true)
  private val blur = new GaussianBlur(20)
  private var animation: Option[Timeline] = None
  private val placeholder = new Label("No preview")
  placeholder.setWrapText(true)
  placeholder.setMaxWidth(108)
  private val preview = new StackPane(placeholder, pixels)
  preview.setMinSize(112, 112)
  preview.setPrefSize(112, 112)
  preview.setMaxSize(112, 112)
  private val name = new Label()
  name.setWrapText(true)
  name.getStyleClass.add("model-preview-name")
  private val kind = new Label()
  private val path = new Label()
  path.setTextOverrun(OverrunStyle.CENTER_ELLIPSIS)
  path.setMaxWidth(Double.MaxValue)
  private val sourcePath = new Label()
  sourcePath.setTextOverrun(OverrunStyle.CENTER_ELLIPSIS)
  sourcePath.setMaxWidth(Double.MaxValue)
  private val detail = new Label()
  detail.setWrapText(true)
  private val reveal = new Button("Hold to reveal preview")
  reveal.setFocusTraversable(true)
  reveal.setTooltip(new Tooltip("Hold Space or Enter to reveal. Release, Escape or loss of focus conceals the preview."))
  private val text = new VBox(4, name, kind, path, sourcePath, detail, reveal)
  text.setMinWidth(0)
  HBox.setHgrow(text, Priority.ALWAYS)
  private val root = new HBox(12, preview, text)
  root.setAlignment(Pos.CENTER_LEFT)
  root.setPadding(new Insets(10))
  root.setMinHeight(132)
  root.getStyleClass.add("model-preview-card")
  val node: Region = root
  private var disposed = false
  private var scene: Scene = null
  private var window: Window = null
  private var modeQueued = false
  private var pendingMode: Option[() => Unit] = None
  private var binding: Option[(ModelCardRecord, Long)] = None
  private var activeBinding = false
  private var observedVisibilityInputs = Vector.empty[Observable]
  private var visibilityQueued = false
  private val state = new ModelPreviewCardState(session, loader.request, render, dispatchMode)

  // The session is engine-owned and may notify from a non-FX caller. Coalesce only its latest state.
  private def dispatchMode(action: () => Unit): Unit =
    if Platform.isFxApplicationThread then action()
    else synchronized {
      if !disposed then
        pendingMode = Some(action)
        if !modeQueued then
          modeQueued = true
          Platform.runLater(() =>
            val next = synchronized {
              modeQueued = false
              val value = pendingMode
              pendingMode = None
              value
            }
            if !disposed then next.foreach(_())
          )
    }

  private def render(visual: PreviewCardVisual): Unit =
    animation.foreach(_.stop())
    animation = None
    val hidden = visual.presentation == PreviewPresentation.Hidden
    // Establish visibility/effect before installing pixels, including cached or late completions.
    pixels.setVisible(visual.previewVisible)
    pixels.setManaged(visual.previewManaged)
    if hidden then
      pixels.setImage(null)
      blur.setRadius(20)
      pixels.setEffect(blur)
    else
      pixels.setEffect(if visual.revealControlVisible || visual.targetBlurRadius > 0 then blur else null)
      visual.transition match
        case Some(change) =>
          val (x1, y1, x2, y2) = change.spline
          val timeline = new Timeline(new KeyFrame(Duration.millis(change.durationMillis),
            new KeyValue(blur.radiusProperty(), Double.box(visual.targetBlurRadius), Interpolator.SPLINE(x1, y1, x2, y2))))
          animation = Some(timeline)
          timeline.play()
        case None => blur.setRadius(visual.targetBlurRadius)
      pixels.setImage(visual.image.orNull)
    placeholder.setVisible(visual.image.isEmpty)
    placeholder.setText(if hidden then "Preview hidden" else visual.status.getOrElse("No preview"))
    name.setText(visual.record.map(_.name).getOrElse(""))
    kind.setText(visual.record.map(r => r.id.kind + r.family.map(f => s" · $f").getOrElse("")).getOrElse(""))
    val modelPath = visual.record.map(_.id.modelPath).getOrElse("")
    path.setText(modelPath)
    path.setTooltip(if modelPath.isEmpty then null else new Tooltip(modelPath))
    val selectedSource = visual.record.flatMap(_.preview).map(_.source)
    sourcePath.setText(selectedSource.map(s => s"Preview: $s").getOrElse(""))
    sourcePath.setTooltip(selectedSource.map(s => new Tooltip(s)).orNull)
    sourcePath.setVisible(selectedSource.nonEmpty)
    sourcePath.setManaged(selectedSource.nonEmpty)
    pixels.setUserData(selectedSource.orNull)
    pixels.setAccessibleText(selectedSource.map(s => s"Local preview $s").getOrElse("No local preview"))
    detail.setText(visual.status.getOrElse(""))
    detail.setManaged(visual.status.nonEmpty)
    detail.setVisible(visual.status.nonEmpty)
    reveal.setVisible(visual.revealControlVisible)
    reveal.setManaged(visual.revealControlVisible)
    val concealed = if hidden then "hidden" else if visual.targetBlurRadius > 0 then "concealed" else "revealed"
    val sourcePolicy =
      if selectedSource.isEmpty then "No preview file reference."
      else if CardPreviewPolicy.matches(selectedSource) then "Matches nsfw in the preview file reference."
      else "Does not match nsfw in the preview file reference."
    root.setAccessibleText(s"${name.getText}. ${kind.getText}. Preview $concealed. Mode ${session.current}. $sourcePolicy")
    reveal.setAccessibleText(s"Hold Space or Enter to reveal preview. Preview $concealed. Focus alone does not reveal.")

  private val hoverListener: ChangeListener[java.lang.Boolean] = (_, _, value) => state.setHover(value.booleanValue())
  private val revealFocusListener: ChangeListener[java.lang.Boolean] = (_, _, value) => state.setRevealFocused(value.booleanValue())
  private val windowFocusListener: ChangeListener[java.lang.Boolean] = (_, _, value) => state.setWindowActive(value.booleanValue())
  private val escapeHandler: javafx.event.EventHandler[KeyEvent] = event =>
    if event.getCode == javafx.scene.input.KeyCode.ESCAPE then state.escape()
  private val windowListener: ChangeListener[Window] = (_, _, next) => attachWindow(next)
  private val sceneListener: ChangeListener[Scene] = (_, _, next) => attachScene(next)
  private val visibilityListener: InvalidationListener = _ => queueVisibility()

  root.hoverProperty().addListener(hoverListener)
  reveal.focusedProperty().addListener(revealFocusListener)
  root.sceneProperty().addListener(sceneListener)
  root.visibleProperty().addListener(visibilityListener)
  root.parentProperty().addListener(visibilityListener)
  root.boundsInLocalProperty().addListener(visibilityListener)
  root.localToSceneTransformProperty().addListener(visibilityListener)
  reveal.addEventFilter(KeyEvent.KEY_PRESSED, event =>
    val key = event.getCode.toString
    if key == "SPACE" || key == "ENTER" then
      state.setRevealFocused(reveal.isFocused)
      state.revealKey(key, true)
      event.consume()
  )
  reveal.addEventFilter(KeyEvent.KEY_RELEASED, event =>
    val key = event.getCode.toString
    if key == "SPACE" || key == "ENTER" then
      state.revealKey(key, false)
      event.consume()
  )
  // No onAction: clicking or holding this control cannot select a model or insert prompt text.
  private def attachWindow(next: Window): Unit =
    if window != null then
      window.focusedProperty().removeListener(windowFocusListener)
      window.showingProperty().removeListener(visibilityListener)
    window = next
    if window != null then
      window.focusedProperty().addListener(windowFocusListener)
      window.showingProperty().addListener(visibilityListener)
      state.setWindowActive(window.isFocused)
    else state.setWindowActive(false)
    refreshVisibility()

  private def attachScene(next: Scene): Unit =
    if scene != null then
      scene.windowProperty().removeListener(windowListener)
      scene.widthProperty().removeListener(visibilityListener)
      scene.heightProperty().removeListener(visibilityListener)
      scene.removeEventFilter(KeyEvent.KEY_PRESSED, escapeHandler)
    attachWindow(null)
    scene = next
    if scene != null then
      scene.windowProperty().addListener(windowListener)
      scene.widthProperty().addListener(visibilityListener)
      scene.heightProperty().addListener(visibilityListener)
      scene.addEventFilter(KeyEvent.KEY_PRESSED, escapeHandler)
      attachWindow(scene.getWindow)

  private def observeVisibilityInputs(ancestors: Vector[Node]): Unit =
    val inputs = ancestors.flatMap { ancestor =>
      val properties = Vector[Observable](ancestor.visibleProperty(), ancestor.parentProperty(), ancestor.clipProperty())
      // boundsInParent includes both clip geometry and its local transforms. A stationary
      // card's own transform need not invalidate when either of those inputs changes.
      properties ++ Option(ancestor.getClip).map(_.boundsInParentProperty())
    }
    if inputs != observedVisibilityInputs then
      observedVisibilityInputs.foreach(_.removeListener(visibilityListener))
      observedVisibilityInputs = inputs
      observedVisibilityInputs.foreach(_.addListener(visibilityListener))

  /** ListView may bind an overscan cell before it is drawable. Such cells own no request/pixels. */
  private def queueVisibility(): Unit =
    if !disposed && !visibilityQueued then
      visibilityQueued = true
      Platform.runLater(() =>
        visibilityQueued = false
        refreshVisibility()
      )

  private def refreshVisibility(): Unit =
    if !disposed then
      val ancestors = Iterator.iterate(root.getParent: Node)(_.getParent).takeWhile(_ != null).toVector
      observeVisibilityInputs(ancestors)
      val bounds = root.localToScene(root.getBoundsInLocal)
      val drawable = scene != null && window != null && window.isShowing && root.isVisible &&
        bounds != null && bounds.getWidth > 0 && bounds.getHeight > 0 &&
        bounds.intersects(0, 0, scene.getWidth, scene.getHeight) && ancestors.forall { ancestor =>
          val clip = ancestor.getClip
          ancestor.isVisible && (clip == null || bounds.intersects(ancestor.localToScene(clip.getBoundsInParent)))
        }
      if drawable && binding.nonEmpty && !activeBinding then
        activeBinding = true // Set before layout changes caused by rendering labels.
        val (record, revision) = binding.get
        state.bind(record, revision)
      else if !drawable && activeBinding then
        activeBinding = false
        state.clear()

  def bind(record: ModelCardRecord, revision: Long): Unit =
    binding = Some((record, revision))
    activeBinding = false
    state.clear()
    refreshVisibility()
  def clear(): Unit =
    binding = None
    activeBinding = false
    state.clear()
  private[ui] def retryVisibleCapacity(): Unit = if activeBinding then state.retryVisibleCapacity()
  def dispose(): Unit =
    if !disposed then
      synchronized { pendingMode = None; disposed = true }
      binding = None
      activeBinding = false
      state.dispose()
      attachScene(null)
      root.hoverProperty().removeListener(hoverListener)
      reveal.focusedProperty().removeListener(revealFocusListener)
      root.sceneProperty().removeListener(sceneListener)
      root.visibleProperty().removeListener(visibilityListener)
      root.parentProperty().removeListener(visibilityListener)
      root.boundsInLocalProperty().removeListener(visibilityListener)
      root.localToSceneTransformProperty().removeListener(visibilityListener)
      observedVisibilityInputs.foreach(_.removeListener(visibilityListener))
      observedVisibilityInputs = Vector.empty
      animation.foreach(_.stop())
      animation = None
