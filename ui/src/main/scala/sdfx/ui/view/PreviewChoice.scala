package sdfx.ui.view

import javafx.beans.property.StringProperty
import javafx.collections.FXCollections
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.image.{Image, ImageView}
import javafx.scene.layout.{HBox, VBox, Priority}
import sdfx.core.plugins.api.Control
import sdfx.ui.vm.PreviewChoiceSession

/** Searchable labels plus a single lazy reference image; no full-catalog image allocation. */
object PreviewChoice:
  def render(choice: Control.Choice, value: StringProperty): Node =
    val combo = W.combo(FXCollections.observableArrayList(choice.options*), value)
    combo.setEditable(choice.allowCustom)
    combo.setMaxWidth(Double.MaxValue)
    HBox.setHgrow(combo, Priority.ALWAYS)
    val browse = new Button("Browse examples…")
    browse.getStyleClass.add("preview-choice-browse")
    browse.setAccessibleText(s"Browse ${choice.label} examples")
    val row = new HBox(8, combo, browse)
    browse.setOnAction { _ =>
      val dialog = new Dialog[String]()
      dialog.setTitle(choice.label + " examples")
      if row.getScene != null then dialog.initOwner(row.getScene.getWindow)
      dialog.setResizable(true)
      dialog.getDialogPane.getButtonTypes.addAll(ButtonType.OK, ButtonType.CANCEL)
      val search = new TextField()
      search.setId("preview-choice-search")
      search.setPromptText("Search names")
      search.setAccessibleText(s"Search ${choice.label}")
      val list = new ListView[String](FXCollections.observableArrayList(choice.options*))
      list.setId("preview-choice-list")
      list.setAccessibleText(choice.label + " examples")
      list.setPrefWidth(260)
      val image = new ImageView()
      image.setId("preview-choice-image")
      image.setFitWidth(360)
      image.setFitHeight(360)
      image.setPreserveRatio(true)
      image.setSmooth(true)
      val label = new Label()
      label.setWrapText(true)
      val status = new Label("Select a name to view its reference image.")
      status.setWrapText(true)
      val retry = new Button("Retry preview")
      val preview = new VBox(10, label, image, status, retry)
      val split = new SplitPane(list, preview)
      split.setDividerPositions(0.4)
      val content = new VBox(10, search, split)
      VBox.setVgrow(split, Priority.ALWAYS)
      content.setPrefSize(720, 470)
      dialog.getDialogPane.setContent(content)
      val session = new PreviewChoiceSession(choice.previewResources)
      var loading: Image = null
      def select(name: String): Unit =
        if loading != null then loading.cancel()
        image.setImage(null)
        label.setText(Option(name).getOrElse(""))
        image.setAccessibleText(Option(name).getOrElse("No preview selected"))
        session.begin(Option(name).getOrElse("")) match
          case None =>
            status.setText("No reference image for this option.")
            retry.setDisable(true)
          case Some((lease, uri)) =>
            status.setText("Loading preview…")
            retry.setDisable(false)
            try
              val candidate = new Image(uri, 720, 720, true, true, true)
              loading = candidate
              def update(): Unit =
                if session.accept(lease) then
                  if candidate.isError then
                    status.setText("Preview could not be loaded. Retry or choose another name.")
                    image.setImage(null)
                  else if candidate.getProgress >= 1 then
                    image.setImage(candidate)
                    status.setText("Reference example; results depend on your model and prompt.")
              candidate.progressProperty.addListener((_, _, _) => update())
              candidate.errorProperty.addListener((_, _, _) => update())
              update()
            catch case _: IllegalArgumentException => status.setText("Preview could not be loaded. Retry or choose another name.")
      list.getSelectionModel.selectedItemProperty.addListener((_, _, name) => select(name))
      retry.setOnAction(_ => select(list.getSelectionModel.getSelectedItem))
      search.textProperty.addListener { (_, _, query) =>
        val selected = list.getSelectionModel.getSelectedItem
        list.setItems(FXCollections.observableArrayList(PreviewChoiceSession.filter(choice.options, query)*))
        if selected != null && list.getItems.contains(selected) then list.getSelectionModel.select(selected)
        else list.getSelectionModel.selectFirst()
      }
      list.getSelectionModel.select(value.get)
      list.setOnMouseClicked(event => if event.getClickCount == 2 then
        dialog.setResult(list.getSelectionModel.getSelectedItem)
        dialog.close())
      dialog.setResultConverter(button => if button == ButtonType.OK then list.getSelectionModel.getSelectedItem else null)
      dialog.setOnHidden { _ =>
        session.close()
        if loading != null then loading.cancel()
        image.setImage(null)
        if dialog.getResult != null then value.set(dialog.getResult)
      }
      dialog.setOnShown(_ => search.requestFocus())
      dialog.show()
    }
    row
