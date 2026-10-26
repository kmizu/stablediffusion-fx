package sdfx.ui.view

import javafx.application.Platform
import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.input.{Clipboard, ClipboardContent}
import javafx.scene.layout.{Priority, Region, VBox}
import javafx.util.{Duration, StringConverter}
import javafx.animation.{KeyFrame, Timeline}

import java.util.concurrent.{Executors, ThreadFactory}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicReference}
import scala.util.control.NonFatal

/** One shared, read-only inspector for native and unscoped API captures. The store owns all exact
  * documents; history controls retain only bounded labels and keys. No engine call is made here.
  */
final class RequestPayloadPane(ctx: UiContext,
    writeClipboard: String => Unit = RequestPayloadPane.systemClipboard):
  private val model = new RequestPayloadPresentation(ctx.services.payloads, writeClipboard,
    () => System.nanoTime() / 1000000L)
  @volatile private var disposed = false
  private var renderedRevision = -1L
  private var selecting = false

  private val scope = new ComboBox[String](FXCollections.observableArrayList(
    "Txt2Img", "Img2Img", "Inpaint", "All captures"))
  scope.setId("payload-scope")
  scope.setAccessibleText("Capture source mode")
  scope.setMinWidth(144)
  scope.setPrefWidth(144)
  scope.setOnAction(_ => if !selecting then scope.getValue match
    case "Txt2Img" => openMode("txt2img")
    case "Img2Img" => openMode("img2img")
    case "Inpaint" => openMode("inpaint")
    case _ => openAllCaptures())

  private val history = new ComboBox[PayloadHistoryEntry]()
  history.setId("payload-history")
  history.setAccessibleText("Captured jobs and sub-generations")
  history.setMaxWidth(Double.MaxValue)
  history.setConverter(new StringConverter[PayloadHistoryEntry]:
    def toString(entry: PayloadHistoryEntry): String = if entry == null then "" else entry.label
    def fromString(text: String): PayloadHistoryEntry = null)
  history.setOnAction(_ => if !selecting then Option(history.getValue).foreach { entry =>
    model.select(entry.key)
    refresh()
  })

  private val document = new ComboBox[PayloadDocument](FXCollections.observableArrayList(PayloadDocument.values*))
  document.setId("payload-document")
  document.setAccessibleText("Independent payload document")
  document.setConverter(new StringConverter[PayloadDocument]:
    def toString(value: PayloadDocument): String = if value == null then "" else value.label
    def fromString(text: String): PayloadDocument = null)
  document.setOnAction(_ => if !selecting then Option(document.getValue).foreach { selected =>
    model.chooseDocument(selected)
    refresh()
  })

  private val copy = W.button("Copy complete JSON", "Copy the entire selected retained document, not its preview") {
    model.copySelected()
    updateFeedback()
    feedbackTimer.stop()
    feedbackTimer.playFromStart()
  }
  copy.setId("payload-copy")
  private val copyFeedback = W.label("", "muted")
  copyFeedback.setId("payload-copy-feedback")
  copyFeedback.setAccessibleText("Copy result")
  private val feedbackTimer = new Timeline(new KeyFrame(Duration.seconds(3), _ => updateFeedback()))
  private def updateFeedback(): Unit = copyFeedback.setText(model.copyFeedback.getOrElse(""))

  private val includeImages = new CheckBox("Include supported image contents in future captures")
  includeImages.setId("payload-include-images")
  includeImages.setAccessibleText("Include image contents for future API payload captures")
  includeImages.setSelected(ctx.services.settings.apiPayloadIncludeImageContents)
  includeImages.setOnAction(_ => ctx.services.updateSettings(
    _.copy(apiPayloadIncludeImageContents = includeImages.isSelected)))

  private val status = W.label("No payload captured.", "muted")
  status.setId("payload-status")
  status.setWrapText(true)
  status.setMinHeight(Region.USE_PREF_SIZE)
  private val rejected = W.label("", "muted")
  rejected.setId("payload-rejected")
  rejected.setWrapText(true)
  rejected.setMinHeight(Region.USE_PREF_SIZE)
  private val warnings = new TextArea()
  warnings.setId("payload-warnings")
  warnings.setAccessibleText("Retained capture warnings; read-only and scrollable")
  warnings.setEditable(false)
  warnings.setWrapText(true)
  warnings.setPrefHeight(100)
  warnings.setMinHeight(66)
  warnings.setMaxHeight(120)
  warnings.setVisible(false)
  warnings.setManaged(false)
  private val contextFacts = W.label("Origin and model context unavailable.", "muted")
  contextFacts.setId("payload-context-facts")
  contextFacts.setWrapText(true)
  contextFacts.setMinHeight(Region.USE_PREF_SIZE)

  private val raw = new TextArea()
  raw.setId("payload-raw-preview")
  raw.setEditable(false)
  raw.setWrapText(false)
  raw.setAccessibleText("Read-only UTF-8 JSON preview; use Copy for the complete retained document")
  private val rawNotice = W.label("No document selected.", "muted")
  rawNotice.setId("payload-raw-notice")
  rawNotice.setWrapText(true)
  val jsonTree = new PayloadJsonTree()
  val treeHost = new VBox(jsonTree.node)
  treeHost.setId("payload-tree-host")
  VBox.setVgrow(jsonTree.node, Priority.ALWAYS)
  private val views = new TabPane(new Tab("Tree", treeHost),
    new Tab("Raw preview", new VBox(6, rawNotice, raw)))
  views.setId("payload-views")
  views.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  VBox.setVgrow(raw, Priority.ALWAYS)
  VBox.setVgrow(views, Priority.ALWAYS)

  private val help = W.label(
    "Body is the exact retained native protocol v1 JSON. Job and model context separately records selected script and AlwaysOn forms when available, plus the model reported at capture; completion facts, when reported, are separate. Image contents are a diagnostic, not wire image fields or a replay bundle. Init-image bytes are omitted by source policy. The shared image option affects future submissions only.",
    "hint")
  help.setId("payload-help")
  help.setWrapText(true)
  help.setMinHeight(Region.USE_PREF_SIZE)
  val node = new VBox(9, W.label("API payload", "section-title"),
    W.hbox(8, scope, W.hgrow(history)), W.hbox(8, document, copy, copyFeedback),
    includeImages, status, rejected, warnings, contextFacts, views, help)
  node.setId("payload-inspector")
  node.setPadding(new Insets(14))
  VBox.setVgrow(node, Priority.ALWAYS)

  private case class Work(revision: Long, json: Option[String], context: Option[String])
  private case class Result(revision: Long, preview: Option[PayloadRawPreview], facts: String)
  private val pending = new AtomicReference[Work]()
  private val pendingResult = new AtomicReference[Result]()
  private val workerScheduled = new AtomicBoolean()
  private val publicationScheduled = new AtomicBoolean()
  private val refreshScheduled = new AtomicBoolean()
  private val worker = Executors.newSingleThreadExecutor((r => {
    val thread = new Thread(r, "sdfx-payload-preview")
    thread.setDaemon(true)
    thread
  }): ThreadFactory)

  private def queueWork(work: Work): Unit =
    pending.set(work)
    if workerScheduled.compareAndSet(false, true) then worker.execute(() => drainWork())

  private def drainWork(): Unit =
    var work = pending.getAndSet(null)
    while work != null && !disposed do
      val result = Result(work.revision, work.json.map(RequestPayloadPresentation.rawPreview),
        RequestPayloadPane.contextFacts(work.context))
      pendingResult.set(result)
      if publicationScheduled.compareAndSet(false, true) then Platform.runLater(() => publishResult())
      work = pending.getAndSet(null)
    workerScheduled.set(false)
    if !disposed && pending.get() != null && workerScheduled.compareAndSet(false, true) then
      worker.execute(() => drainWork())

  private def publishResult(): Unit =
    publicationScheduled.set(false)
    val result = pendingResult.getAndSet(null)
    if result != null && !disposed && model.acceptsRevision(result.revision) then
      result.preview match
        case Some(preview) =>
          raw.setText(preview.text)
          rawNotice.setText(preview.notice)
        case None =>
          raw.clear()
          rawNotice.setText("Selected document unavailable; nothing to copy")
      contextFacts.setText(result.facts)
    if pendingResult.get() != null && publicationScheduled.compareAndSet(false, true) then
      Platform.runLater(() => publishResult())

  private val unsubscribe = ctx.services.payloads.subscribe(_ => {
    // The subscriber's vector is deliberately discarded. One FX pulse reads current store state.
    if refreshScheduled.compareAndSet(false, true) then Platform.runLater(() => {
      refreshScheduled.set(false)
      if !disposed then refresh()
    })
  })

  def openMode(mode: String): Unit =
    if !disposed then
      model.openMode(mode)
      refresh()

  def openAllCaptures(): Unit =
    if !disposed then
      model.openAllCaptures()
      refresh()

  /** FX-only refresh. It never receives an event snapshot or retains documents in history items. */
  def refresh(): Unit =
    if !disposed then
      model.refresh()
      val entries = model.historyEntries
      selecting = true
      try
        scope.setValue(modelScope)
        history.getItems.setAll(entries*)
        history.setValue(entries.find(e => model.selectedSnapshot.exists(_.key == e.key)).orNull)
        document.setValue(model.selectedDocument)
      finally selecting = false
      status.setText(model.statusText)
      rejected.setText(model.rejectedCaptureText)
      val details = model.warningDetails
      val warningText = details.mkString("\n")
      if warnings.getText != warningText then warnings.setText(warningText)
      warnings.setVisible(details.nonEmpty)
      warnings.setManaged(details.nonEmpty)
      copy.setDisable(!model.copyEnabled)
      includeImages.setSelected(ctx.services.settings.apiPayloadIncludeImageContents)
      updateFeedback()
      val revision = model.selectionRevision
      if revision != renderedRevision then
        renderedRevision = revision
        raw.clear()
        rawNotice.setText("Building bounded preview…")
        contextFacts.setText("Reading submitted context…")
        jsonTree.clear()
        val exact = model.selectedJson
        exact.foreach(jsonTree.showJson)
        queueWork(Work(revision, exact, model.selectedSnapshot.flatMap(_.contextJson)))

  private def modelScope: String =
    // The selected source scope is intentionally explicit, including unscoped MCP/direct captures.
    if model.isAllCaptures then "All captures"
    else model.currentMode match
      case Some("img2img") => "Img2Img"
      case Some("inpaint") => "Inpaint"
      case _ => "Txt2Img"

  def dispose(): Unit =
    if !disposed then
      disposed = true
      unsubscribe()
      feedbackTimer.stop()
      pending.set(null)
      pendingResult.set(null)
      worker.shutdownNow()
      jsonTree.dispose()
      model.dispose()

object RequestPayloadPane:
  def systemClipboard(text: String): Unit =
    val content = new ClipboardContent()
    content.putString(text)
    if !Clipboard.getSystemClipboard.setContent(content) then
      throw new IllegalStateException("System clipboard did not accept JSON")

  /** Context is parsed only on the preview worker. Do not put submitted form JSON in a Label. */
  private def contextFacts(json: Option[String]): String =
    def field(obj: ujson.Value, name: String): Option[ujson.Value] = obj.objOpt.flatMap(_.get(name))
    def string(value: Option[ujson.Value]): Option[String] = value.collect { case ujson.Str(s) => s }
    def label(text: String): String =
      val prefix = if text.length <= 120 then text else text.take(119) + "…"
      prefix.map(ch => if ch < ' ' then ' ' else ch)
    def short(value: ujson.Value): String =
      val text = ujson.write(value)
      if text.length <= 240 then text else text.take(239) + "…"
    json match
      case None => "Origin unavailable; job and model context was not retained."
      case Some(text) =>
        try
          val root = ujson.read(text)
          val submitted = field(root, "submitted")
          val origin = label(submitted.flatMap(s => string(field(s, "origin"))).getOrElse("unavailable"))
          val mode = label(submitted.flatMap(s => string(field(s, "source_mode"))).getOrElse("source mode unavailable"))
          val capture = field(root, "reported_at_capture").flatMap(field(_, "model"))
          val completion = field(root, "reported_at_completion")
          val captureText = capture match
            case Some(ujson.Null) | None => "no model reported"
            case Some(value) => short(value)
          val completionText = completion.fold("no completion model facts reported")(short)
          s"Origin: $origin · Submitted mode: $mode\nReported at capture (snapshot, not execution proof): $captureText\nReported at completion: $completionText"
        catch case NonFatal(_) => "Origin/model context could not be presented; the retained JSON remains available."
