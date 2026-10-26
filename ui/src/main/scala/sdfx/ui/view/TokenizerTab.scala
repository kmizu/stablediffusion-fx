package sdfx.ui.view

import javafx.beans.property.SimpleStringProperty
import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.input.{Clipboard, ClipboardContent}
import javafx.scene.layout.{FlowPane, Priority, Region, VBox}
import sdfx.core.protocol.{TokenPiece, TokenStream}
import sdfx.ui.Fx

/** Literal vocabulary inspection, independent of image generation. */
final class TokenizerTab(ctx: UiContext, currentPrompt: () => String):
  private val input = new TextArea()
  input.setId("tokenizer-input")
  input.setWrapText(true)
  input.setPrefRowCount(5)
  input.setMinHeight(Region.USE_PREF_SIZE)
  input.setPromptText("Enter text to inspect with the loaded model's CLIP tokenizer")
  private val mode = new ComboBox[String](FXCollections.observableArrayList("Text", "Token IDs"))
  mode.setId("tokenizer-mode")
  mode.getSelectionModel.selectFirst()
  mode.setAccessibleText("Tokenizer input mode")
  private val status = W.label("Load a checkpoint, then inspect its tokenizer.", "muted")
  status.setId("tokenizer-status")
  status.setWrapText(true)
  status.setMinHeight(Region.USE_PREF_SIZE)
  private val results = new TabPane()
  results.setId("tokenizer-results")
  results.setTabClosingPolicy(TabPane.TabClosingPolicy.UNAVAILABLE)
  results.setMinHeight(Region.USE_PREF_SIZE)
  private val inspectButton = W.button("Inspect tokens", "Inspect actual model vocabulary; no image is generated", Seq("primary"))(inspect())
  inspectButton.setId("tokenizer-inspect")
  private val fromPrompt = W.button("Use current prompt", "Copy the active generation tab's positive prompt") {
    mode.getSelectionModel.selectFirst()
    input.setText(currentPrompt())
  }
  fromPrompt.setId("tokenizer-from-prompt")
  private val help = W.label(
    "Literal CLIP BPE tokens. No automatic BOS/EOS; literal special tokens retained. Attention syntax, schedules, BREAK, LoRA tags and embeddings are not expanded here. " +
    "SDXL shows both tokenizers separately. ID input accepts comma-separated integers.", "muted")
  help.setWrapText(true)
  help.setMinHeight(Region.USE_PREF_SIZE)
  help.setId("tokenizer-help")
  private val content = new VBox(10, W.label("Tokenizer", "section-title"), help,
    W.hbox(8, mode, fromPrompt), input, inspectButton, status, results)
  content.setPadding(new Insets(14))
  // Preserve the natural height of wrapped help, editors and results. At smaller
  // window sizes the page scrolls instead of VBox shrinking text into ellipses.
  content.setMinHeight(Region.USE_PREF_SIZE)
  content.setMinWidth(0)
  private val page = new ScrollPane(content)
  page.setId("tokenizer-page-scroll")
  page.setFitToWidth(true)
  page.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER)
  VBox.setVgrow(page, Priority.ALWAYS)
  val node: VBox = new VBox(page)
  node.setId("tokenizer-tool")

  mode.valueProperty.addListener((_, _, selected) =>
    input.setPromptText(if selected == "Token IDs" then "Comma-separated token IDs, e.g. 9061, 631, 736" else "Text to tokenize literally")
  )

  private def busy(value: Boolean): Unit =
    inspectButton.setDisable(value)
    input.setDisable(value)
    mode.setDisable(value)
    fromPrompt.setDisable(value)

  private def inspect(): Unit =
    val text = input.getText
    if text.length > 65536 then
      status.setText("Shorten the input to at most 65,536 characters.")
    else
      val inputMode = if mode.getValue == "Token IDs" then "ids" else "text"
      busy(true)
      results.getTabs.clear()
      status.setText("Inspecting loaded model… (queued after any running job)")
      Fx.onFx(ctx.services.client.tokenize(text, inputMode)) { result =>
        busy(false)
        result.streams.foreach(s => results.getTabs.add(new Tab(s.name, streamView(s))))
        status.setText(s"${result.streams.size} tokenizer stream(s) · ${if result.inputMode == "ids" then "IDs decoded as supplied" else "No automatic BOS/EOS; literal special tokens retained"}")
      } { error =>
        busy(false)
        status.setText(Fx.message(error))
      }

  private def copy(text: String): Unit =
    val data = new ClipboardContent()
    data.putString(text)
    Clipboard.getSystemClipboard.setContent(data)
    ctx.info("Token IDs copied")

  private def streamView(s: TokenStream): ScrollPane =
    val summary = W.label(s"${s.count} tokens · context content capacity ${s.contentCapacity} · BOS ${s.bosId} / EOS ${s.eosId} / PAD ${s.padId}")
    summary.setWrapText(true)
    summary.setMinHeight(Region.USE_PREF_SIZE)
    val source = W.label(s"Vocabulary: ${s.source}", "muted")
    source.setWrapText(true)
    source.setMinHeight(Region.USE_PREF_SIZE)
    val capacity = W.label("Long input is shown in full; generation may chunk or transform it. Capacity is per encoder context, not a predicted generation chunk count.", "muted")
    capacity.setWrapText(true)
    capacity.setMinHeight(Region.USE_PREF_SIZE)
    val chips = new FlowPane(4, 5)
    chips.setId(s"tokenizer-${s.name}-groups")
    s.groups.take(512).zipWithIndex.foreach { (g, i) =>
      val button = W.button(g.text.replace("\n", "↵"), s"IDs: ${g.ids.mkString(", ")}\nClick to copy these IDs")(copy(g.ids.mkString(", ")))
      button.setId(s"tokenizer-${s.name}-group-$i")
      val colors = Seq("rgba(79,145,220,0.24)", "rgba(104,183,143,0.24)", "rgba(186,128,214,0.24)", "rgba(218,168,85,0.24)")
      button.setStyle(s"-fx-background-color: ${colors(i % colors.size)}; -fx-padding: 4 7 4 7;")
      chips.getChildren.add(button)
    }
    val chipScroll = new ScrollPane(chips)
    chipScroll.setFitToWidth(true)
    chipScroll.setPrefViewportHeight(36)
    chipScroll.setMinHeight(Region.USE_PREF_SIZE)
    chipScroll.setMaxHeight(Region.USE_PREF_SIZE)
    chipScroll.viewportBoundsProperty.addListener((_, previous, bounds) =>
      if previous == null || previous.getWidth != bounds.getWidth then
        chipScroll.setPrefViewportHeight(math.max(36, math.min(120, chips.prefHeight(bounds.getWidth))))
    )
    val previewNote = W.label(if s.groups.size > 512 then "Colored preview: first 512 groups. The ID list and table below contain every token." else "Click a colored group to copy its token IDs.", "muted")
    previewNote.setWrapText(true)
    previewNote.setMinHeight(Region.USE_PREF_SIZE)
    val ids = new TextArea(s.ids.mkString(", "))
    ids.setId(s"tokenizer-${s.name}-ids")
    ids.setEditable(false)
    ids.setWrapText(true)
    ids.setPrefRowCount(2)
    ids.setMinHeight(Region.USE_PREF_SIZE)
    val decoded = new TextArea(s.decoded)
    decoded.setId(s"tokenizer-${s.name}-decoded")
    decoded.setEditable(false)
    decoded.setWrapText(true)
    decoded.setPrefRowCount(2)
    decoded.setMinHeight(Region.USE_PREF_SIZE)
    val table = new TableView[TokenPiece](FXCollections.observableArrayList(s.tokens*))
    table.setId(s"tokenizer-${s.name}-tokens")
    def column(title: String, width: Double, minimum: Double)(value: TokenPiece => String): Unit =
      val c = new TableColumn[TokenPiece, String](title)
      c.setCellValueFactory(v => new SimpleStringProperty(value(v.getValue)))
      c.setPrefWidth(width)
      c.setMinWidth(minimum)
      table.getColumns.add(c)
    column("#", 45, 40)(_.index.toString)
    column("ID", 70, 60)(_.id.toString)
    column("BPE piece", 130, 95)(_.piece)
    column("Display", 100, 75)(_.text)
    column("UTF-8 bytes", 160, 130)(_.bytesHex)
    column("Special", 65, 55)(p => if p.special then "yes" else "")
    table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN)
    table.widthProperty.addListener((_, _, width) =>
      table.setColumnResizePolicy(if width.doubleValue >= 475 then
        TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN else TableView.UNCONSTRAINED_RESIZE_POLICY)
    )
    table.setPlaceholder(W.label("No content tokens"))
    table.setMinHeight(280)
    table.setPrefHeight(320)
    val warning = W.label("A display fragment can show � when a UTF-8 character spans tokens; colored groups above join those bytes. " +
      "BPE piece is the exact vocabulary string. In UTF-8 bytes, 20 marks a word-ending space.", "muted")
    warning.setWrapText(true)
    warning.setMinHeight(Region.USE_PREF_SIZE)
    val copyAll = W.button("Copy all IDs")(copy(s.ids.mkString(", ")))
    copyAll.setId(s"tokenizer-${s.name}-copy-ids")
    val box = new VBox(8, summary, source, capacity, chipScroll, previewNote,
      W.hbox(8, W.label("Token IDs"), copyAll), ids,
      W.label("Decoded by tokenizer"), decoded, table, warning)
    box.setPadding(new Insets(10, 0, 10, 0))
    box.setMinHeight(Region.USE_PREF_SIZE)
    box.setMinWidth(0)
    val scroll = new ScrollPane(box)
    scroll.setId(s"tokenizer-${s.name}-scroll")
    scroll.setFitToWidth(true)
    // Let the outer page own vertical navigation; this retains the stream pane's
    // existing identity while allowing each tab to request its full content height.
    scroll.setMinHeight(Region.USE_PREF_SIZE)
    scroll
