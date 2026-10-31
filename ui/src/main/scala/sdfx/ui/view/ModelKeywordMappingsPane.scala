package sdfx.ui.view

import javafx.geometry.Insets
import javafx.scene.control.{Alert, Button, ButtonType, ComboBox, Label, TextField}
import javafx.scene.layout.{HBox, VBox}
import sdfx.core.plugins.api.ChoiceSource
import sdfx.ui.Fx

/** Mapping editor, kept separate from generation controls so Check/Save/Delete use the same model identity. */
final class ModelKeywordMappingsPane(ctx: UiContext, onChoices: (String, Seq[String]) => Unit = (_, _) => (),
    onLoraSelected: String => Unit = _ => ()):
  private val kind = new ComboBox[String]()
  kind.getItems.addAll("Checkpoint", "LoRA")
  kind.getSelectionModel.selectFirst()
  kind.setId("model-keyword-kind")
  private val lora = new ComboBox[String](ctx.catalog.choices(ChoiceSource.Loras))
  lora.setId("model-keyword-lora")
  lora.setPromptText("Select LoRA")
  private val keywords = new TextField()
  keywords.setId("model-keyword-mapping")
  keywords.setPromptText("Keywords separated by |")
  private val status = new Label("Check the selected checkpoint, or select a LoRA.")
  status.setId("model-keyword-mapping-status")
  status.setWrapText(true)
  private val check = new Button("Check")
  private val save = new Button("Save")
  private val delete = new Button("Delete")
  private val installCatalog = new Button("Download upstream LoRA catalog…")
  installCatalog.setId("model-keyword-install-catalog")
  check.setId("model-keyword-check"); save.setId("model-keyword-save"); delete.setId("model-keyword-delete")
  private var revision = 0L
  private var loraChoicesRevision = 0L
  private var checkpointChoicesRevision = 0L
  private def selectedPath: Option[String] =
    if kind.getValue == "Checkpoint" then
      Option(ctx.checkpointName.get).filter(_.nonEmpty).flatMap(ctx.services.catalog.findCheckpoint).map(_.path)
    else Option(lora.getValue).filter(_.nonEmpty).flatMap(ctx.services.catalog.findLora).map(_.path)
  private def updateKind(): Unit =
    revision += 1
    lora.setDisable(kind.getValue != "LoRA")
    if kind.getValue == "LoRA" then onChoices("lora", Seq("None", "< iterate >", "< random >"))
    else onChoices("checkpoint", Seq("keyword1, keyword2", "random", "iterate", "keyword1", "keyword2"))
    if selectedPath.isDefined then request("check")
    else
      Seq(check, save, delete).foreach(_.setDisable(false))
      status.setText("Choose an available model first.")
  kind.valueProperty.addListener((_, _, _) => updateKind())
  lora.valueProperty.addListener((_, _, value) => if kind.getValue == "LoRA" then
    onLoraSelected(Option(value).getOrElse("None"))
    updateKind())
  ctx.checkpointName.addListener((_, _, name) =>
    checkpointChoicesRevision += 1
    val current = checkpointChoicesRevision
    if kind.getValue == "Checkpoint" then updateKind()
    else
      Option(name).filter(_.nonEmpty).flatMap(ctx.services.catalog.findCheckpoint) match
        case None => onChoices("checkpoint", Seq("keyword1, keyword2", "random", "iterate", "keyword1", "keyword2"))
        case Some(entry) =>
          val params = ujson.Obj("action" -> "check", "kind" -> "checkpoint", "model_path" -> entry.path,
            "mapping_dir" -> ctx.services.home.configDir.resolve("model-keyword").toString)
          Fx.onFx(ctx.services.client.raw("model_keyword", params)) { result =>
            if current == checkpointChoicesRevision && ctx.checkpointName.get == name then
              onChoices("checkpoint", result("choices").arr.map(_.str).toSeq)
          } { error =>
            if current == checkpointChoicesRevision then status.setText(s"Checkpoint mapping check failed: ${Fx.message(error)}")
          }
  )
  updateKind()
  def selectLora(name: String): Unit =
    loraChoicesRevision += 1
    val current = loraChoicesRevision
    if name == null || name.isEmpty || name == "None" then lora.getSelectionModel.clearSelection()
    else lora.getSelectionModel.select(name)
    if kind.getValue != "LoRA" then
      val selected = Option(name).filter(n => n.nonEmpty && n != "None").flatMap(ctx.services.catalog.findLora)
      selected match
        case None => onChoices("lora", Seq("None", "< iterate >", "< random >"))
        case Some(entry) =>
          val file = java.nio.file.Path.of(entry.path).getFileName.toString
          val dot = file.lastIndexOf('.')
          val extension = if dot >= 0 then file.substring(dot) else ""
          val params = ujson.Obj("action" -> "check", "kind" -> "lora", "model_path" -> entry.path,
            "mapping_dir" -> ctx.services.home.configDir.resolve("model-keyword").toString,
            "match_name" -> (name.replace('\\', '/') + extension))
          Fx.onFx(ctx.services.client.raw("model_keyword", params)) { result =>
            if current == loraChoicesRevision && lora.getValue == name then
              onChoices("lora", result("choices").arr.map(_.str).toSeq)
          } { error =>
            if current == loraChoicesRevision then status.setText(s"LoRA mapping check failed: ${Fx.message(error)}")
          }
  private def request(action: String): Unit =
    selectedPath match
      case None => status.setText("Choose an available model first.")
      case Some(path) =>
        val requestRevision = revision
        val requestKind = if kind.getValue == "LoRA" then "lora" else "checkpoint"
        val buttons = Seq(check, save, delete)
        buttons.foreach(_.setDisable(true))
        status.setText("Working…")
        val params = ujson.Obj("action" -> action,
          "kind" -> requestKind,
          "model_path" -> path, "mapping_dir" -> ctx.services.home.configDir.resolve("model-keyword").toString,
          "keywords" -> keywords.getText)
        if kind.getValue == "LoRA" then
          val filename = java.nio.file.Path.of(path).getFileName.toString
          val dot = filename.lastIndexOf('.')
          val extension = if dot >= 0 then filename.substring(dot) else ""
          params("match_name") = lora.getValue.replace('\\', '/') + extension
        Fx.onFx(ctx.services.client.raw("model_keyword", params)) { result =>
          if requestRevision == revision then
            buttons.foreach(_.setDisable(false))
            onChoices(requestKind, result("choices").arr.map(_.str).toSeq)
            val words = result("keywords").arr.map(_.str).mkString(" | ")
            val removed = result.obj.get("deleted").collect { case ujson.Bool(v) => v }
            status.setText(s"${result("filename").str} · ${result("hash").str} · ${result("source").str}" +
              removed.map(v => if v then " · custom mapping deleted" else " · no custom mapping to delete").getOrElse("") +
              (if words.nonEmpty then s"\n$words" else " · no keywords") +
              (if !result("default_catalog_available").bool then
                s"\nNo LoRA catalog is installed; only custom mappings are searched. Add lora-keyword.txt to ${ctx.services.home.configDir.resolve("model-keyword")} to search a catalog."
               else ""))
        } { error =>
          if requestRevision == revision then
            buttons.foreach(_.setDisable(false))
            status.setText(s"Mapping action failed: ${Fx.message(error)}")
        }
  check.setOnAction(_ => request("check"))
  save.setOnAction(_ => request("save"))
  delete.setOnAction(_ => request("delete"))
  installCatalog.setOnAction(_ =>
    val confirm = new Alert(Alert.AlertType.CONFIRMATION)
    confirm.setTitle("Download upstream LoRA catalog")
    confirm.setHeaderText("Download the pinned upstream catalog from GitHub?")
    confirm.setContentText("This downloads about 31 MB to your local configuration and does not replace an existing catalog. " +
      "The upstream repository is MIT licensed; its entries include data derived from third-party sources. " +
      "Source: https://github.com/mix1009/model-keyword (commit c3d0833a9434bb41ecd7ea715b5abd5e6b43a5af).")
    if confirm.showAndWait().orElse(ButtonType.CANCEL) == ButtonType.OK then
      installCatalog.setDisable(true)
      status.setText("Downloading and verifying the upstream LoRA catalog…")
      val params = ujson.Obj("action" -> "install_upstream_lora_catalog", "confirmed" -> true,
        "mapping_dir" -> ctx.services.home.configDir.resolve("model-keyword").toString)
      Fx.onFx(ctx.services.client.raw("model_keyword", params)) { result =>
        installCatalog.setDisable(false)
        status.setText(s"Catalog installed: ${result("path").str}")
        if kind.getValue == "LoRA" && selectedPath.isDefined then request("check")
      } { error =>
        installCatalog.setDisable(false)
        status.setText(s"Catalog download failed: ${Fx.message(error)}")
      }
  )
  val node = new VBox(8, new Label("Model keyword mappings"), new HBox(8, kind, lora), keywords,
    new HBox(8, check, save, delete), installCatalog, status)
  node.setPadding(new Insets(12))
