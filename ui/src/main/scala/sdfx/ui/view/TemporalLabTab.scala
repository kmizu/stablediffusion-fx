package sdfx.ui.view

import java.io.ByteArrayInputStream
import java.nio.file.{Files, LinkOption, Path, Paths}
import java.util.Base64
import javafx.animation.{KeyFrame, Timeline}
import javafx.beans.property.{SimpleBooleanProperty, SimpleDoubleProperty, SimpleIntegerProperty, SimpleStringProperty}
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.image.{Image, ImageView}
import javafx.scene.layout.{Region, VBox}
import javafx.stage.{DirectoryChooser, FileChooser}
import javafx.util.Duration
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal
import sdfx.core.protocol.*
import sdfx.engine.TemporalLabHandle
import sdfx.ui.Fx
import sdfx.ui.vm.{TemporalLabForms, TemporalLabState, TemporalPreviewCache, TemporalPreviewSlots, TemporalInputInfo, TemporalInputInspection, TemporalInspectionQueue, TemporalLabPresentation}

/** Native local temporal tools. All worker/file operations are asynchronous; the frame preview holds one image. */
final class TemporalLabTab(ctx: UiContext):
  private val state = new TemporalLabState
  private val playback = new TemporalPreviewCache(8, 16L * 1024 * 1024)
  private var closed = false
  private var previewRevision = 0L

  private var active: Option[TemporalLabHandle] = None
  private val reference = new SimpleStringProperty("")
  private val generated = new SimpleStringProperty("")
  private val frames = new SimpleStringProperty("")
  private val style = new SimpleStringProperty("")
  private val videoFrames = new SimpleStringProperty("")
  private val video = new SimpleStringProperty("")
  private val outputParent = new SimpleStringProperty("")
  private val outputName = new SimpleStringProperty("temporal-output")
  private val sourceDenoise = new SimpleIntegerProperty(0)
  private val tolerance = new SimpleDoubleProperty(5)
  private val expand = new SimpleIntegerProperty(0)
  private val smooth = new SimpleIntegerProperty(1)
  private val interDenoise = new SimpleIntegerProperty(9)
  private val interSize = new SimpleIntegerProperty(9)
  private val interFrequency = new SimpleIntegerProperty(3)
  private val interBlur = new SimpleDoubleProperty(3)
  private val refreshFrequency = new SimpleIntegerProperty(0)
  private val preserve = new SimpleDoubleProperty(0)
  private val frameLimit = new SimpleIntegerProperty(0)
  private val previewFrame = new SimpleIntegerProperty(1)
  private val blend = new SimpleDoubleProperty(0)
  private val overlay = new SimpleDoubleProperty(0)
  private val normalize = new SimpleBooleanProperty(false)
  private val fuseStrength = new SimpleDoubleProperty(.5)
  private val fps = new SimpleDoubleProperty(0)
  private val playbackFps = new SimpleIntegerProperty(12)
  private val status = W.label("Select local input and a new output directory name.", "hint")
  status.setId("temporal-status"); status.setWrapText(true); status.setMinHeight(Region.USE_PREF_SIZE)
  private val progress = new ProgressBar(0)
  progress.setId("temporal-progress"); progress.setMaxWidth(Double.MaxValue)
  private val mask = new ImageView()
  mask.setPreserveRatio(true); mask.setFitWidth(350); mask.setFitHeight(250)
  private val previousImage = new ImageView()
  previousImage.setPreserveRatio(true); previousImage.setFitWidth(200); previousImage.setFitHeight(160)
  private val currentImage = new ImageView()
  currentImage.setPreserveRatio(true); currentImage.setFitWidth(200); currentImage.setFitHeight(160)
  private val image = new ImageView()
  image.setPreserveRatio(true); image.setFitWidth(450); image.setFitHeight(300)
  private val previewDescription = W.label("No result yet.", "hint")
  previewDescription.setWrapText(true)
  private val seek = new Slider(0, 0, 0)
  seek.setId("temporal-seek")
  private var changingSeek = false
  private var timeline: Option[Timeline] = None
  private val workTabs = new TabPane()
  workTabs.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  private val cancel: Button = W.button("Cancel", "Request cancellation of this Temporal Lab job") {
    active.foreach { handle =>
      val ticket = activeTicket
      Fx.onFx(handle.interrupt()) { decision =>
        if !closed then ticket.foreach(t => { state.cancelDecision(t, decision); status.setText(state.status) })
      } { e => if !closed then status.setText(s"Cancellation failed: ${Fx.message(e)}") }
    }
  }
  cancel.setId("temporal-cancel"); cancel.setDisable(true)
  private var activeTicket: Option[state.Ticket] = None

  private def pathRow(prop: SimpleStringProperty, title: String, directory: Boolean): javafx.scene.Node =
    val field = W.textField(prop, title)
    field.setId("temporal-" + title.toLowerCase.replace(' ', '-'))
    val choose = W.button("Browse…", title) {
      if directory then
        val chooser = new DirectoryChooser(); chooser.setTitle(title)
        Option(chooser.showDialog(ctx.stage)).foreach(f => prop.set(f.getAbsolutePath))
      else
        val chooser = new FileChooser(); chooser.setTitle(title)
        Option(chooser.showOpenDialog(ctx.stage)).foreach(f => prop.set(f.getAbsolutePath))
    }
    W.hbox(6, W.hgrow(field), choose)
  private def number(p: SimpleIntegerProperty, min: Int, max: Int, step: Int = 1) = W.intSpinner(p, min, max, step)
  private def decimal(p: SimpleDoubleProperty, min: Double, max: Double, step: Double) = W.doubleSpinner(p, min, max, step)
  private def dfiRequest = TemporalLabDfi(reference.get, generated.get, outputParent.get, outputName.get,
    sourceDenoise.get, tolerance.get, expand.get, smooth.get, interDenoise.get, interSize.get, interFrequency.get,
    interBlur.get, refreshFrequency.get, preserve.get, frameLimit.get)
  private def button(label: String, id: String)(action: => Unit): Button =
    val b = W.button(label, label, Seq("accent-button"))(action)
    b.setId(id); b
  private def commit(): Unit = W.commitEnclosingSpinner(ctx.stage.getScene.getFocusOwner)
  private val inspectionExecutor = java.util.concurrent.Executors.newSingleThreadExecutor((r: Runnable) => {
    val thread = new Thread(r, "temporal-input-inspection"); thread.setDaemon(true); thread
  })
  private def makeInspectionQueue() = new TemporalInspectionQueue(
    task => inspectionExecutor.execute(() => task()),
    task => javafx.application.Platform.runLater(() => task()),
    (path, cancelled) => TemporalInputInspection.inspect(path, cancelled))
  private val referenceInspection = makeInspectionQueue()
  private val generatedInspection = makeInspectionQueue()
  private var referenceInfo = Option.empty[Either[String, TemporalInputInfo]]
  private var generatedInfo = Option.empty[Either[String, TemporalInputInfo]]
  private val inputInfo = W.label("Choose reference and generated frame folders to inspect their headers.", "hint")
  inputInfo.setId("temporal-input-info"); inputInfo.setWrapText(true); inputInfo.setMinHeight(Region.USE_PREF_SIZE); inputInfo.setMinWidth(0)
  private def showInputInfo(): Unit =
    inputInfo.setText((referenceInfo, generatedInfo) match
      case (Some(Right(a)), Some(Right(b))) => TemporalLabPresentation.dfiInputs(a, b)
      case (a, b) =>
        def show(label: String, item: Option[Either[String, TemporalInputInfo]]): String = item match
          case Some(Right(i)) => s"$label: ${i.count} frames, ${i.width} × ${i.height}"
          case Some(Left(error)) => s"$label: $error"
          case None => s"$label: awaiting inspection"
        s"${show("Reference", a)}; ${show("Generated", b)}. Header inspection only; worker validates all inputs.")
  private def inspectInput(value: String, queue: TemporalInspectionQueue, set: Option[Either[String, TemporalInputInfo]] => Unit): Unit =
    set(None); showInputInfo()
    // Queue empty input too: it invalidates a previous in-flight scan.
    queue.request(value, result => { if !closed then { set(Some(result)); showInputInfo() } })
  reference.addListener((_, _, value) => inspectInput(value, referenceInspection, x => referenceInfo = x))
  generated.addListener((_, _, value) => inspectInput(value, generatedInspection, x => generatedInfo = x))
  private val dfiControls = W.form()
    .add("Reference frames", pathRow(reference, "Reference frames", true))
    .add("Generated frames", pathRow(generated, "Generated frames", true))
    .add("Source denoise", number(sourceDenoise, 0, 30))
    .add("Tolerance", decimal(tolerance, .5, 20, .5))
    .add("Expand", number(expand, 0, 50))
    .add("Mask smoothing", number(smooth, 1, 99, 2))
    .add("Inter denoise", number(interDenoise, 1, 25))
    .add("Inter size", number(interSize, 1, 25, 2))
    .add("Inter frequency", number(interFrequency, 1, 15))
    .add("Inter blur", decimal(interBlur, 1, 5, .5))
    .add("Refresh frequency", number(refreshFrequency, 0, 30))
    .add("Preserve %", decimal(preserve, 0, 100, 1))
    .add("Frame limit", number(frameLimit, 0, 10000), "0 processes all frames")
    .add("Preview transition", number(previewFrame, 1, 10000))
  private def help(text: String): Label =
    val label = W.label(text, "hint"); label.setWrapText(true); label.setMinWidth(0); label.setMinHeight(Region.USE_PREF_SIZE); label
  private val dfiPane = new VBox(8, inputInfo, help(TemporalLabPresentation.dfiHelp), dfiControls.grid, W.hbox(8,
    button("Preview mask", "temporal-preview") { commit(); submit(TemporalLabForms.preview(dfiRequest, previewFrame.get)) },
    button("Run DFI", "temporal-dfi") { commit(); submit(TemporalLabForms.dfi(dfiRequest)) }))
  private val deflickerControls = W.form()
    .add("Frames", pathRow(frames, "Frames", true))
    .add("Temporal blend", decimal(blend, 0, 1, .05))
    .add("Overlay", decimal(overlay, 0, 1, .05))
    .full(W.check("Normalize brightness", normalize))
  private val deflickerPane = new VBox(8, help(TemporalLabPresentation.deflickerHelp), deflickerControls.grid,
    button("Run deflicker", "temporal-deflicker") { commit(); submit(TemporalLabForms.deflicker(
      TemporalLabDeflicker(frames.get, outputParent.get, outputName.get, blend.get, overlay.get, normalize.get))) })
  private val fuseControls = W.form()
    .add("Style frames", pathRow(style, "Style frames", true))
    .add("Video frames", pathRow(videoFrames, "Video frames", true))
    .add("Strength", decimal(fuseStrength, .1, 1, .05))
  private val fusePane = new VBox(8, fuseControls.grid, W.label("Style frames are resized to video frame dimensions.", "hint"),
    button("Fuse frames", "temporal-fuse") { commit(); submit(TemporalLabForms.fuse(
      TemporalLabFuse(style.get, videoFrames.get, outputParent.get, outputName.get, fuseStrength.get))) })
  private val extractControls = W.form()
    .add("Video file", pathRow(video, "Video file", false))
    .add("FPS", decimal(fps, 0, 1000, 1), "0 uses source FPS; sampling follows frame index")
  private val extractPane = new VBox(8, extractControls.grid,
    button("Extract frames", "temporal-extract") { commit(); submit(TemporalLabForms.extract(
      TemporalLabExtract(video.get, outputParent.get, outputName.get, fps.get))) })
  Seq("DFI" -> dfiPane, "Deflicker" -> deflickerPane, "Fuse" -> fusePane, "Extract" -> extractPane).foreach { (name, content) =>
    content.setPadding(new Insets(10))
    workTabs.getTabs.add(new Tab(name, content))
  }
  private val outputControls = W.form()
    .add("Output parent", pathRow(outputParent, "Output parent", true))
    .add("New folder", W.textField(outputName), "A new folder is created; existing files are never overwritten")
  private val playerControls = W.hbox(6,
    W.button("Play", "Play output frames")(play()),
    W.button("Pause", "Pause playback")(pause()),
    W.button("Stop", "Stop and return to first frame")(stop()),
    W.label("FPS", "hint"), number(playbackFps, 1, 60))
  private val body = new VBox(10,
    W.label("Temporal Lab", "section-title"),
    W.label("Local frame sequences are read-only. Each run creates a new output folder.", "hint"),
    W.card("Output", outputControls.grid), workTabs, W.hbox(8, cancel, progress), status,
    W.card("Mask preview", mask, W.hbox(8, previousImage, currentImage)), W.card("Output preview", image, previewDescription, seek, playerControls))
  body.setPadding(new Insets(12)); body.setMinHeight(Region.USE_PREF_SIZE); body.setMinWidth(0)
  val node: ScrollPane = new ScrollPane(body)
  node.setId("temporal-lab-tool"); node.setFitToWidth(true); node.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  private val previewSlots = new TemporalPreviewSlots[Image](loadPreview)
  private def clearPreview(): Unit =
    previewRevision += 1
    previewSlots.clear()
    pause()
    playback.setFrames(Vector.empty)
    mask.setImage(null); previousImage.setImage(null); currentImage.setImage(null); image.setImage(null)
    previewDescription.setText("Inputs changed. Run preview or processing again.")
    changingSeek = true
    try { seek.setMax(0); seek.setValue(0) } finally changingSeek = false
  ctx.watchImageAccess(node) { () =>
    clearPreview()
    state.result.flatMap(_.outputDir).foreach(loadFrames)
  }
  val invalidateListener: javafx.beans.InvalidationListener = _ => { state.invalidate(); clearPreview() }
  Seq(reference, generated, frames, style, videoFrames, video, outputParent, outputName,
    sourceDenoise, tolerance, expand, smooth, interDenoise, interSize, interFrequency, interBlur,
    refreshFrequency, preserve, frameLimit, previewFrame, blend, overlay, normalize, fuseStrength, fps)
    .foreach(_.addListener(invalidateListener))
  seek.valueProperty.addListener((_, _, value) =>
    if !changingSeek then { playback.seek(value.intValue); showFrame() }
  )

  private def busy(value: Boolean): Unit =
    workTabs.setDisable(value); outputControls.grid.setDisable(value); cancel.setDisable(!value)
  private def submit(req: Either[String, ? <: TemporalLabRequest]): Unit = req match
    case Left(msg) => status.setText(msg)
    case Right(request) =>
      state.start(request) match
        case Left(msg) => status.setText(msg)
        case Right(ticket) =>
          busy(true); status.setText("Queued…"); progress.setProgress(0); activeTicket = Some(ticket)
          clearPreview()
          val session = ctx.services.imageScrambleSession.state
          val handle = ctx.services.client.temporalLab(request, event => event match
            case p: EngineEvent.Progress => Fx.run {
              if !closed then { state.progress(ticket, p.step, p.total); progress.setProgress(state.progressFraction); status.setText(state.status) }
            }
            case _ => (), session.secret)
          active = Some(handle)
          Fx.onFx(handle.result) { result =>
            if !closed then
              state.complete(ticket, result); active = None; activeTicket = None; busy(false)
              if state.result.contains(result) then
                status.setText(TemporalLabPresentation.completion(result))
                progress.setProgress(1)
                if request.isInstanceOf[TemporalLabPreview] then
                  if session.revision == ctx.services.imageScrambleSession.state.revision then showMask(result)
                  else previewDescription.setText("Image access changed. Run preview again to view the result.")
                else result.outputDir.foreach(loadFrames)
          } { error =>
            if !closed then
              state.fail(ticket, Fx.message(error)); active = None; activeTicket = None; busy(false)
              status.setText(state.status)
          }
  private def showMask(result: TemporalLabResult): Unit =
    try
      result.maskPngBase64 match
        case Some(encoded) if encoded.length <= 8 * 1024 * 1024 =>
          val bytes = Base64.getDecoder.decode(encoded)
          require(bytes.length >= 24 && bytes.take(8).toVector == Vector[Byte](-119, 80, 78, 71, 13, 10, 26, 10), "Invalid mask PNG")
          val header = java.nio.ByteBuffer.wrap(bytes)
          val width = header.getInt(16); val height = header.getInt(20)
          require(width > 0 && height > 0 && width <= 1024 && height <= 1024, "Mask dimensions exceed preview limit")
          mask.setImage(new Image(new ByteArrayInputStream(bytes), 350, 250, true, true))
          result.generatedPrevious.foreach(p => showSlot("previous", Paths.get(p), previousImage))
          result.generatedCurrent.foreach(p => showSlot("current", Paths.get(p), currentImage))
          previewDescription.setText(s"Reference transition ${result.frameIndex.getOrElse(0)}: ${result.generatedPrevious.getOrElse("")} → ${result.generatedCurrent.getOrElse("")}")
        case _ => status.setText("No bounded preview image was returned")
    catch case NonFatal(e) => status.setText(s"Preview failed: ${Fx.message(e)}")
  private def loadFrames(path: String): Unit =
    pause(); previewDescription.setText("Loading output frame list…")
    val revision = previewRevision
    Fx.async {
      val dir = Paths.get(path)
      Using.resource(Files.list(dir)) { stream =>
        val entries = stream.iterator.asScala.take(20001).toVector
        require(entries.size <= 20000, "Output directory contains too many entries")
        entries.filter(p => Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS) &&
          p.getFileName.toString.matches("frame[0-9]{6}\\.png")).sortBy(_.getFileName.toString)
      }
    } { paths =>
      if !closed && revision == previewRevision then
        if paths.size > 10000 then previewDescription.setText("Output contains too many frames to preview")
        else
          playback.setFrames(paths)
          seek.setMax((paths.size - 1).max(0)); seek.setValue(0)
          showFrame()
    } { e => if !closed && revision == previewRevision then previewDescription.setText(s"Cannot read output frames: ${Fx.message(e)}") }
  private def loadPreview(slot: String, path: Path, done: Either[String, Image] => Unit): () => Unit =
    var cancelled = false
    val session = ctx.services.imageScrambleSession.state
    // Bound file size and decoded source dimensions off the FX thread before asking JavaFX to load it.
    Fx.async {
      require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS), "Not a regular image file")
      require(Files.size(path) <= 64L * 1024 * 1024, "Image file exceeds preview limit")
      val input = javax.imageio.ImageIO.createImageInputStream(path.toFile)
      require(input != null, "Cannot open image")
      Using.resource(input) { stream =>
        val readers = javax.imageio.ImageIO.getImageReaders(stream)
        require(readers.hasNext, "Unrecognized or corrupt image")
        val reader = readers.next()
        try
          reader.setInput(stream, true, true)
          val width = reader.getWidth(0); val height = reader.getHeight(0)
          require(width > 0 && height > 0 && width.toLong * height <= 16000000L, "Image dimensions exceed preview limit")
        finally reader.dispose()
      }
      ImageFiles.readThumbnail(path, session.secret, 1024).image
    } { candidate =>
      if !closed then
        if !cancelled && session.revision == ctx.services.imageScrambleSession.state.revision then done(Right(candidate))
        else done(Left("Image access changed"))
    } { e =>
      if !closed then done(Left(if cancelled then "Image access changed" else Fx.message(e)))
    }
    () => { cancelled = true }
  private def showSlot(name: String, path: Path, view: ImageView): Unit =
    previewSlots.show(name, path, value => {
      view.setImage(value)
      if name == "frame" then previewDescription.setText(path.toString)
    }, error => {
      view.setImage(null)
      previewDescription.setText(s"Frame preview failed ($path): $error")
    })
  private def showFrame(): Unit =
    playback.current match
      case Some(path) =>
        previewDescription.setText(path.toString)
        changingSeek = true
        try seek.setValue(playback.index)
        finally changingSeek = false
        showSlot("frame", path, image)
      case None => image.setImage(null); previewDescription.setText("No output frames")
  private def play(): Unit =
    pause(); playback.play()
    if playback.playing then
      val t = new Timeline(new KeyFrame(Duration.millis(1000.0 / playbackFps.get), _ => { playback.advance(); showFrame() }))
      t.setCycleCount(javafx.animation.Animation.INDEFINITE); timeline = Some(t); t.play()
  private def pause(): Unit = { timeline.foreach(_.stop()); timeline = None; playback.pause() }
  private def stop(): Unit = { pause(); playback.stop(); showFrame() }
  def dispose(): Unit =
    if !closed then
      closed = true; referenceInspection.dispose(); generatedInspection.dispose(); inspectionExecutor.shutdownNow(); pause(); playback.dispose(); previewSlots.dispose(); state.dispose()
      active.foreach(_.interrupt()); active = None; activeTicket = None
