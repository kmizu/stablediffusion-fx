package sdfx.ui.view

import javafx.scene.control.{CheckBox, Label, TextArea, TitledPane}
import javafx.scene.layout.VBox
import sdfx.ui.Fx

/** Native settings for future image operations; image provenance never selects the local hardware. */
final class HardwareMetadataPane(ctx: UiContext):
  private var disposed = false
  private var request = 0L
  private val enabled = new CheckBox("Write hardware and time into future images")
  enabled.setId("hardware-info-enabled")
  enabled.setSelected(ctx.services.settings.hardwareInfoEnabled)
  enabled.setOnAction(_ => ctx.services.updateSettings(_.copy(hardwareInfoEnabled = enabled.isSelected)))
  private val remark = new TextArea(ctx.services.settings.hardwareInfoRemark)
  remark.setId("hardware-info-remark")
  remark.setWrapText(true)
  remark.setPrefRowCount(2)
  remark.setPromptText("Optional personal remark (saved locally and included only when enabled)")
  remark.setAccessibleText("Optional remark in future image metadata")
  remark.textProperty.addListener((_, _, value) => ctx.services.updateSettings(_.copy(hardwareInfoRemark = value)))
  private val preview = new Label("Hardware information appears after the engine starts.")
  preview.setId("hardware-info-preview")
  preview.setWrapText(true)
  private val help = W.label("Adds Hardware Info, Time taken and optional Remark to generated and Extras PNGs. Generation time averages each batch; Extras measures its whole operation. Queue wait and model preloading are excluded.", "muted")
  help.setWrapText(true)
  help.setMinHeight(javafx.scene.layout.Region.USE_PREF_SIZE)
  private val content = new VBox(7, enabled, preview, W.label("Remark"), remark, help)
  val node = new TitledPane("Hardware image metadata (opt-in)", content)
  node.setId("hardware-info-settings")
  node.setAnimated(false)
  node.setExpanded(false)
  private def refresh(): Unit =
    if !disposed then
      request += 1
      val ticket = request
      Fx.onFx(ctx.hardwareInfo.profile()) { profile =>
        if !disposed && ticket == request then preview.setText(
          if profile == "unknown" then "Hardware Info: unknown (engine offline or detection unavailable)" else profile)
      } { _ => if !disposed && ticket == request then preview.setText("Hardware information unavailable; the next refresh can retry.") }
  private val removeState = ctx.services.supervisor.addStateListener(_ => Fx.run(refresh()))
  refresh()
  def dispose(): Unit =
    disposed = true
    removeState()
