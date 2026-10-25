package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.collections.transformation.FilteredList
import javafx.geometry.{Insets, Pos}
import javafx.scene.Node
import javafx.scene.control.{Label, ListCell, ListView, TextField}
import javafx.scene.input.{KeyCode, KeyEvent, MouseButton}
import javafx.scene.layout.{HBox, Priority, VBox}
import javafx.stage.Popup

/** Searchable picker popup (checkpoints, LoRAs): type to filter, Enter / double click to pick, Esc to close. */
final class SearchPopup[T](
    title: String,
    items: () => Seq[T],
    text: T => String,
    badge: T => Option[String],
    onPick: T => Unit,
    emptyText: String
):
  private val popup = new Popup()
  popup.setAutoHide(true)
  popup.setHideOnEscape(true)

  private val all = FXCollections.observableArrayList[T]()
  private val filtered = new FilteredList[T](all, _ => true)
  private val field = new TextField()
  field.setPromptText("Search…")
  private val list = new ListView[T](filtered)
  list.setPrefSize(420, 340)
  list.setPlaceholder(W.label(emptyText, "muted"))
  list.setCellFactory(_ =>
    new ListCell[T]:
      // only clicks on a filled row pick (scrollbar / empty-area clicks must not load a model)
      setOnMouseClicked(e =>
        if e.getButton == MouseButton.PRIMARY && !isEmpty && getItem != null then
          list.getSelectionModel.select(getItem)
          pick()
      )
      override def updateItem(item: T, empty: Boolean): Unit =
        super.updateItem(item, empty)
        if empty || item == null then
          setText(null)
          setGraphic(null)
        else
          val name = new Label(text(item))
          name.setMaxWidth(Double.MaxValue)
          HBox.setHgrow(name, Priority.ALWAYS)
          val row = new HBox(8, name)
          row.setAlignment(Pos.CENTER_LEFT)
          badge(item).foreach(b => row.getChildren.add(W.label(b, "badge", s"badge-${b.toLowerCase.replaceAll("[^a-z0-9]", "")}")))
          setText(null)
          setGraphic(row)
  )

  private val header = W.label(title, "popup-title")
  private val box = new VBox(8, header, field, list)
  box.getStyleClass.add("search-popup")
  box.setPadding(new Insets(10))
  popup.getContent.add(box)

  field.textProperty.addListener((_, _, q) =>
    val terms = q.toLowerCase.split("\\s+").filter(_.nonEmpty)
    filtered.setPredicate(t => terms.forall(text(t).toLowerCase.contains))
    if !filtered.isEmpty then list.getSelectionModel.select(0)
  )
  private def pick(): Unit =
    Option(list.getSelectionModel.getSelectedItem).foreach { t =>
      popup.hide()
      onPick(t)
    }
  field.addEventFilter(KeyEvent.KEY_PRESSED, (e: KeyEvent) =>
    e.getCode match
      case KeyCode.DOWN =>
        list.getSelectionModel.selectNext()
        list.scrollTo(list.getSelectionModel.getSelectedIndex)
        e.consume()
      case KeyCode.UP =>
        list.getSelectionModel.selectPrevious()
        list.scrollTo(list.getSelectionModel.getSelectedIndex)
        e.consume()
      case KeyCode.ENTER =>
        pick()
        e.consume()
      case _ => ()
  )
  list.setOnKeyPressed(e => if e.getCode == KeyCode.ENTER then pick())

  def show(anchor: Node): Unit =
    all.setAll(items()*)
    field.clear()
    filtered.setPredicate(_ => true)
    val scene = anchor.getScene
    if scene != null then
      box.getStylesheets.setAll(scene.getStylesheets)
      box.getStyleClass.removeAll("theme-dark", "theme-light")
      box.getStyleClass.addAll(scene.getRoot.getStyleClass.filtered(_.startsWith("theme-")))
    val b = anchor.localToScreen(anchor.getBoundsInLocal)
    popup.show(anchor, b.getMinX, b.getMaxY + 4)
    field.requestFocus()
    if !filtered.isEmpty then list.getSelectionModel.select(0)
