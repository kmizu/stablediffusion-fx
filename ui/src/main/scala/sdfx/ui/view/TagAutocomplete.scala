package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.control.{ButtonType, Dialog, Label, ListCell, ListView, TextArea}
import javafx.scene.input.{InputMethodEvent, KeyCode, KeyEvent, MouseEvent}
import javafx.stage.{FileChooser, Popup}
import sdfx.ui.vm.{CompletionTag, TagCompletion}
import java.nio.file.Paths

/** One dictionary per window; loading never blocks the FX thread. */
final class TagDictionary(ctx: UiContext):
  private var loadedTags: Vector[CompletionTag] = Vector.empty
  def tags: Vector[CompletionTag] = loadedTags
  private var version = 0L
  private var requested: Option[String] = None
  def ensureLoaded(): Unit =
    val path = ctx.services.settings.tagAutocompleteCsv
    if path != requested then
      requested = path
      version += 1
      val ticket = version
      loadedTags = Vector.empty
      path.foreach(p => sdfx.ui.Fx.async(TagCompletion.load(Paths.get(p)))(loaded =>
        if ticket == version then loadedTags = loaded
      )(e => if ticket == version then ctx.error(s"Cannot load tag CSV: ${e.getMessage}")))

  def configure(): Unit =
    val dialog = new Dialog[ButtonType]()
    dialog.initOwner(ctx.stage)
    dialog.setTitle("Tag autocomplete")
    val choose = new ButtonType("Choose CSV…")
    val reload = new ButtonType("Reload")
    val disable = new ButtonType("Disable")
    dialog.getDialogPane.getButtonTypes.addAll(choose, reload, disable, ButtonType.CANCEL)
    val label = new Label("Local UTF-8 CSV: name,type,count,\"aliases\"\n" +
      ctx.services.settings.tagAutocompleteCsv.getOrElse("No file selected") +
      "\nUp/Down: select · Enter/Tab: insert · Escape: close\nNo tag files are downloaded automatically.")
    label.setWrapText(true)
    label.setMaxWidth(600)
    dialog.getDialogPane.setContent(label)
    val result = dialog.showAndWait().orElse(ButtonType.CANCEL)
    if result == choose then
      val chooser = new FileChooser()
      chooser.setTitle("Choose tag CSV")
      chooser.getExtensionFilters.add(new FileChooser.ExtensionFilter("Tag CSV", "*.csv"))
      Option(chooser.showOpenDialog(ctx.stage)).foreach { file =>
        ctx.services.updateSettings(_.copy(tagAutocompleteCsv = Some(file.getAbsolutePath)))
        requested = None
        ensureLoaded()
      }
    else if result == disable then
      ctx.services.updateSettings(_.copy(tagAutocompleteCsv = None))
      ensureLoaded()
    else if result == reload then
      requested = None
      ensureLoaded()

/** Popup does not take focus, so native text editing and IME stay in the TextArea. */
final class TagAutocomplete(area: TextArea, tags: () => Vector[CompletionTag], ensureLoaded: () => Unit):
  def this(area: TextArea, dictionary: TagDictionary) = this(area, () => dictionary.tags, () => dictionary.ensureLoaded())
  private val popup = new Popup()
  private val list = new ListView[CompletionTag]()
  popup.setAutoHide(true)
  popup.setHideOnEscape(false)
  list.setFocusTraversable(false)
  list.setPrefWidth(360)
  list.setCellFactory(_ => new ListCell[CompletionTag]():
    setOnMousePressed(e =>
      if getItem != null then
        list.getSelectionModel.select(getItem)
        accept()
        e.consume()
    )
    override def updateItem(tag: CompletionTag, empty: Boolean): Unit =
      super.updateItem(tag, empty)
      setText(if empty || tag == null then null else s"${tag.name}    ${tag.count}")
  )
  popup.getContent.add(list)
  private var composing = false
  private var imeCommitGuard = false
  private var accepting = false
  private var suppressed = false
  def isComposing: Boolean = composing || imeCommitGuard
  def isShowing: Boolean = popup.isShowing
  def hide(): Unit = popup.hide()

  private def update(): Unit =
    if accepting || suppressed || isComposing || !area.isFocused || area.getScene == null then hide()
    else
      ensureLoaded()
      val matches = TagCompletion.token(area.getText, area.getCaretPosition, area.getSelection.getStart, area.getSelection.getEnd)
        .map(t => TagCompletion.suggestions(tags(), t.query)).getOrElse(Vector.empty)
      if matches.isEmpty then hide()
      else
        list.getItems.setAll(matches*)
        list.getSelectionModel.select(0)
        list.setPrefHeight(math.min(8, matches.size) * 26 + 4)
        val bounds = area.localToScreen(area.getBoundsInLocal)
        if bounds != null then
          list.getStylesheets.setAll(area.getScene.getStylesheets)
          if !popup.isShowing then popup.show(area, bounds.getMinX, bounds.getMaxY)

  private def accept(): Unit =
    val selected = Option(list.getSelectionModel.getSelectedItem)
    val token = TagCompletion.token(area.getText, area.getCaretPosition, area.getSelection.getStart, area.getSelection.getEnd)
    for tag <- selected; t <- token do
      accepting = true
      try
        area.replaceText(t.start, t.end, tag.insertion)
        area.positionCaret(t.start + tag.insertion.length)
      finally accepting = false
    suppressed = true
    hide()

  area.textProperty.addListener((_, _, _) =>
    if !accepting then
      if !isComposing then suppressed = false
      Platform.runLater(() => update())
  )
  area.caretPositionProperty.addListener((_, _, _) => update())
  area.selectionProperty.addListener((_, _, _) => update())
  area.focusedProperty.addListener((_, _, focused) =>
    if focused then ensureLoaded()
    else
      composing = false
      imeCommitGuard = false
      hide()
  )
  area.sceneProperty.addListener((_, _, scene) => if scene == null then hide())
  area.addEventFilter(InputMethodEvent.INPUT_METHOD_TEXT_CHANGED, (e: InputMethodEvent) =>
    composing = !e.getComposed.isEmpty
    imeCommitGuard = true
    suppressed = true
    hide()
    // Never consume IME events; wait for the next ordinary edit after commit.
  )
  area.addEventFilter(KeyEvent.KEY_RELEASED, (_: KeyEvent) => if !composing then imeCommitGuard = false)
  area.addEventFilter(KeyEvent.KEY_PRESSED, (e: KeyEvent) =>
    if !isComposing && popup.isShowing && !e.isControlDown && !e.isMetaDown && !e.isAltDown && !e.isShiftDown then
      e.getCode match
        case KeyCode.DOWN => list.getSelectionModel.select((list.getSelectionModel.getSelectedIndex + 1).min(list.getItems.size - 1)); e.consume()
        case KeyCode.UP => list.getSelectionModel.select((list.getSelectionModel.getSelectedIndex - 1).max(0)); e.consume()
        case KeyCode.ENTER | KeyCode.TAB => accept(); e.consume()
        case KeyCode.ESCAPE => suppressed = true; hide(); e.consume()
        case _ => ()
  )
