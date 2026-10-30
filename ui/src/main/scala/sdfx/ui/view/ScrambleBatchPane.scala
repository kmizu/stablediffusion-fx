package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.Insets
import javafx.scene.control.*
import javafx.scene.layout.{HBox, Priority, VBox}
import javafx.stage.{DirectoryChooser, Stage}
import sdfx.core.png.ScrambleBatch
import sdfx.ui.Fx
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicBoolean
import scala.util.control.NonFatal

/** Explicit folder conversion; this format is pixel scrambling, not secure encryption. */
final class ScrambleBatchPane(stage: Stage):
  private val source = new TextField()
  source.setId("scramble-source"); source.setPromptText("Folder containing images")
  private val destination = new TextField()
  destination.setId("scramble-destination"); destination.setPromptText("Different output folder")
  private val password = new PasswordField()
  password.setId("scramble-password"); password.setPromptText("Password (not saved)")
  private val mode = new ChoiceBox[String](FXCollections.observableArrayList("Scramble images", "Restore images"))
  mode.setId("scramble-mode"); mode.setValue("Scramble images")
  private val unverified = new CheckBox("Restore legacy images even when the password cannot be verified")
  unverified.setId("scramble-allow-unverified")
  unverified.setWrapText(true)
  unverified.setMaxWidth(Double.MaxValue)
  private val results = new ListView[String]()
  results.setId("scramble-results"); results.setPrefHeight(210)
  private val status = new Label("Choose source and output folders, then enter a password.")
  status.setId("scramble-status"); status.setWrapText(true)
  private val stopFlag = new AtomicBoolean(false)
  private var active = false
  private var alive = true
  private def choose(field: TextField): Unit =
    val chooser = new DirectoryChooser()
    chooser.setTitle("Choose image folder")
    Option(chooser.showDialog(stage)).foreach(f => field.setText(f.toPath.toString))
  private val sourceButton = new Button("Browse…")
  sourceButton.setOnAction(_ => choose(source))
  private val destinationButton = new Button("Browse…")
  destinationButton.setOnAction(_ => choose(destination))
  private val start = new Button("Start")
  start.setId("scramble-start")
  private val stop = new Button("Stop after current image")
  stop.setId("scramble-stop"); stop.setDisable(true)
  stop.setOnAction(_ => { stopFlag.set(true); stop.setDisable(true); status.setText("Stopping after the current image…") })
  private def busy(value: Boolean): Unit =
    active = value
    Seq(source, destination, password, mode, unverified, sourceButton, destinationButton, start).foreach(_.setDisable(value))
    stop.setDisable(!value)
  private def startBatch(): Unit =
    if !active && alive then
      val src = source.getText.trim; val dst = destination.getText.trim; val pass = password.getText
      if src.isEmpty || dst.isEmpty || pass.isEmpty then status.setText("Choose both folders and enter a password.")
      else
        try
          val sourcePath = Path.of(src); val destinationPath = Path.of(dst)
          val operation = if mode.getValue == "Restore images" then ScrambleBatch.Mode.Decode else ScrambleBatch.Mode.Encode
          val allow = unverified.isSelected
          stopFlag.set(false); results.getItems.clear(); busy(true); status.setText("Processing images…")
          Fx.async {
            ScrambleBatch.run(sourcePath, destinationPath, pass, operation, allow,
              () => stopFlag.get(), r => Fx.run {
                if alive then
                  val detail = if r.message.nonEmpty then s" — ${r.message}" else ""
                  results.getItems.add(s"${r.source.getFileName}: ${r.status}$detail")
              })
          } { completed =>
            if alive then
              busy(false)
              val written = completed.count(_.status == ScrambleBatch.Status.Written)
              val prefix = if stopFlag.get() then "Stopped" else "Completed"
              status.setText(s"$prefix: $written written; ${completed.size - written} skipped or failed. See results below.")
          } { error =>
            if alive then { busy(false); status.setText(s"Could not process folder: ${Fx.message(error)}") }
          }
        catch case NonFatal(e) => status.setText(s"Invalid folder: ${Fx.message(e)}")
  start.setOnAction(_ => startBatch())
  private val warning = new Label("Compatibility mode rearranges pixels; it is not secure encryption. Metadata remains readable and can include thumbnails of the original image; some metadata, especially from JPEG or WebP, may be lost. Converted files are PNG; restoring pixels does not restore original JPEG or WebP bytes. Existing output files are skipped.")
  warning.setWrapText(true)
  warning.getStyleClass.add("muted")
  private def folderRow(field: TextField, button: Button): HBox =
    HBox.setHgrow(field, Priority.ALWAYS)
    new HBox(8, field, button)
  val node: VBox = new VBox(10, new Label("Image scrambling — batch conversion"), warning,
    new Label("Source folder"), folderRow(source, sourceButton), new Label("Output folder"), folderRow(destination, destinationButton),
    new Label("Password"), password, mode, unverified, new HBox(8, start, stop), status, results)
  node.setPadding(new Insets(16)); VBox.setVgrow(results, Priority.ALWAYS)
  def dispose(): Unit =
    alive = false; stopFlag.set(true); password.clear()
