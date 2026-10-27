package sdfx.ui.view

import javafx.beans.property.StringProperty
import javafx.geometry.Pos
import javafx.scene.control.*
import javafx.scene.input.{KeyCode, KeyEvent}
import javafx.scene.layout.{StackPane, VBox}
import sdfx.core.model.{LoraSyntax, PromptStyle}
import sdfx.engine.CatalogSnapshot
import sdfx.ui.vm.{PromptEdit, PromptInsertionTarget}

/** Prompt + negative prompt with token counters, Ctrl+Up/Down attention editing, LoRA picker and styles. */
final class PromptEditor(ctx: UiContext, prompt: StringProperty, negative: StringProperty, modeKey: String):
  val promptArea: TextArea = area(prompt, "Prompt — what you want to see (Ctrl+Enter to generate)", 4)
  val negativeArea: TextArea = area(negative, "Negative prompt — what to avoid", 2)
  private var lastFocused: TextArea = promptArea

  Seq(promptArea, negativeArea).foreach(a => a.focusedProperty.addListener((_, _, f) => if f then lastFocused = a))

  private def area(p: StringProperty, hint: String, rows: Int): TextArea =
    val a = new TextArea()
    a.textProperty.bindBidirectional(p)
    a.setPromptText(hint)
    a.setWrapText(true)
    a.setPrefRowCount(rows)
    a.getStyleClass.add("prompt-area")
    a.addEventFilter(KeyEvent.KEY_PRESSED, (e: KeyEvent) =>
      if e.isShortcutDown && !e.isShiftDown && (e.getCode == KeyCode.UP || e.getCode == KeyCode.DOWN) then
        val ed = PromptEdit.adjustWeight(a.getText, a.getSelection.getStart, a.getSelection.getEnd, if e.getCode == KeyCode.UP then 1 else -1)
        if ed.text != a.getText then a.setText(ed.text)
        a.selectRange(ed.selStart, ed.selEnd)
        e.consume()
    )
    a

  private def withCounter(a: TextArea): StackPane =
    val counter = W.label(PromptEdit.tokenCounter(a.getText), "token-counter")
    a.textProperty.addListener((_, _, t) => counter.setText(PromptEdit.tokenCounter(t)))
    W.tip(counter, "Approximate CLIP tokens / current 75-token chunk limit")
    counter.setMouseTransparent(true)
    // prompt-bracket-checker
    val brackets = W.label(PromptEdit.bracketIssues(a.getText), "bracket-warning")
    brackets.setVisible(brackets.getText.nonEmpty)
    a.textProperty.addListener((_, _, t) =>
      brackets.setText(PromptEdit.bracketIssues(t))
      brackets.setVisible(brackets.getText.nonEmpty)
    )
    brackets.setMouseTransparent(true)
    val sp = new StackPane(a, counter, brackets)
    StackPane.setAlignment(counter, Pos.TOP_RIGHT)
    StackPane.setAlignment(brackets, Pos.BOTTOM_RIGHT)
    sp

  // ---- LoRA picker -------------------------------------------------------------------------------------

  private def snap: CatalogSnapshot = ctx.catalog.snapshot.get

  private val loraPopup = new SearchPopup[String](
    "Insert LoRA",
    () => snap.loras.map(_.name),
    identity,
    _ => None,
    name => insert(LoraSyntax.tag(stem(name), 1.0)),
    "No LoRAs found — put files in models/lora or add folders in Settings"
  )

  private def stem(n: String): String =
    val base = n.replace('\\', '/').split('/').last
    val i = base.lastIndexOf('.')
    if i > 0 then base.substring(0, i) else base

  private def insert(text: String): Unit =
    val a = lastFocused
    val ed = PromptEdit.insertAt(a.getText, a.getCaretPosition, text)
    a.setText(ed.text)
    a.requestFocus()
    a.selectRange(ed.selStart, ed.selEnd)

  /** Capture the actual editor before the browser takes focus. */
  def captureInsertionTarget(): PromptInsertionTarget =
    val a = lastFocused
    val captured = a.getText
    val caret = a.getCaretPosition
    PromptInsertionTarget(modeKey, a eq negativeArea, captured, caret,
      () => (a.getText, a.getCaretPosition),
      text =>
        if a.getText == captured && a.getCaretPosition == caret then
          val edit = PromptEdit.insertAt(captured, caret, text)
          a.setText(edit.text)
          a.requestFocus()
          a.selectRange(edit.selStart, edit.selEnd)
    )

  def openNetworks(): Unit = ctx.openModelBrowser("lora", Some(captureInsertionTarget()))

  // ---- styles ------------------------------------------------------------------------------------------

  private val styleBox = new ComboBox[PromptStyle](ctx.styles)
  styleBox.setPromptText("Styles")
  styleBox.setButtonCell(styleCell())
  styleBox.setCellFactory(_ => styleCell())
  W.tip(styleBox, "Prompt styles from config/styles.csv (A1111 compatible)")
  styleBox.setMaxWidth(220)

  private def styleCell(): ListCell[PromptStyle] = new ListCell[PromptStyle]:
    override def updateItem(s: PromptStyle, empty: Boolean): Unit =
      super.updateItem(s, empty)
      setText(if empty || s == null then "Styles" else s.name)

  private def applyStyle(): Unit =
    val selected = Option(styleBox.getValue).flatMap { chosen =>
      val current = ctx.styles.toArray(Array.empty[PromptStyle]).toSeq
      current.find(_ == chosen).orElse {
        val byName = current.filter(_.name == chosen.name)
        if byName.size == 1 then byName.headOption else None
      }
    }
    selected match
      case Some(s) =>
        val (p, n) = s.applyTo(prompt.get, negative.get)
        prompt.set(p)
        negative.set(n)
      case None => ctx.info("Pick a style first")

  private def saveStyle(): Unit =
    val d = new TextInputDialog(Option(styleBox.getValue).map(_.name).getOrElse(""))
    d.initOwner(ctx.stage)
    d.setTitle("Save style")
    d.setHeaderText("Save the current prompt and negative prompt as a style")
    d.setContentText("Name:")
    d.getDialogPane.getStylesheets.setAll(ctx.stage.getScene.getStylesheets)
    d.showAndWait().ifPresent(name =>
      if name.trim.nonEmpty then
        val exists = ctx.styles.toArray(Array.empty[PromptStyle]).exists(_.name == name.trim)
        val overwrite = if !exists then false else
          val confirm = new Alert(Alert.AlertType.CONFIRMATION, s"Replace the prompts in style '${name.trim}'?", ButtonType.CANCEL, ButtonType.OK)
          confirm.initOwner(ctx.stage)
          confirm.setHeaderText("Overwrite existing style")
          confirm.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.OK
        if !exists || overwrite then ctx.saveStyle(PromptStyle(name.trim, prompt.get, negative.get), overwrite)
    )

  val toolbar: javafx.scene.layout.HBox = W.hbox(
    6,
    W.button("✚ LoRA", "Insert a <lora:name:1> tag at the caret (searchable list)")(loraPopup.show(loraButtonAnchor)),
    W.button("Networks…", "Browse network previews")(openNetworks()),
    W.spacer(),
    styleBox,
    W.iconButton("⤓", "Apply the selected style to the prompts")(applyStyle()),
    W.iconButton("✎", "Save the current prompts as a style")(saveStyle()),
    W.button("Styles…", "Edit styles, categories, extended presets and backups")(ctx.openStylesEditor())
  )
  private def loraButtonAnchor = toolbar.getChildren.get(0)

  val view: VBox =
    val v = new VBox(6, withCounter(promptArea), withCounter(negativeArea), toolbar)
    v.getStyleClass.add("prompt-editor")
    W.tip(promptArea, "Ctrl+Up / Ctrl+Down change the attention weight of the selection or the word under the caret")
    v
