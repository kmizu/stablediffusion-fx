package sdfx.ui.view

import javafx.geometry.Insets
import javafx.scene.control.{Button, CheckBox, Label, PasswordField}
import javafx.scene.layout.{HBox, VBox}
import sdfx.core.png.ImageScrambleSession

/** Session controls for source-compatible pixel scrambling. The host owns the session lifetime. */
final class ScrambleSessionPane(session: ImageScrambleSession, onChange: () => Unit):
  private val password = new PasswordField()
  password.setId("scramble-session-password"); password.setPromptText("Password (not saved)")
  private val status = new Label()
  status.setId("scramble-session-status"); status.setWrapText(true)
  private val enable = new Button("Enable for this session")
  enable.setId("scramble-session-enable")
  private val lock = new Button("Turn off")
  lock.setId("scramble-session-lock")
  private val plaintextVideo = new CheckBox("Allow plaintext video exports for this session")
  plaintextVideo.setId("scramble-session-plaintext-video")
  private def refresh(): Unit =
    val enabled = session.snapshot.nonEmpty
    status.setText(if enabled then "Enabled for new jobs in this session." else "Off. New jobs save ordinary images.")
    lock.setDisable(!enabled)
    plaintextVideo.setDisable(!enabled)
    plaintextVideo.setSelected(session.snapshot.exists(_.plaintextVideoAllowed))
  enable.setOnAction(_ =>
    if password.getText.isEmpty then status.setText("Enter a password first.")
    else
      session.unlock(password.getText)
      password.clear()
      refresh(); onChange()
  )
  lock.setOnAction(_ => { session.lock(); password.clear(); refresh(); onChange() })
  plaintextVideo.setOnAction(_ => {
    session.setPlaintextVideoAllowed(plaintextVideo.isSelected)
    onChange()
  })
  private val warning = new Label("Compatibility mode rearranges pixels; it is not secure encryption. Metadata remains readable and can include thumbnails of the original image. Temporary input files can remain readable after a crash; exact VAE recovery tensors also remain readable. Custom Code can save readable files. MCP runs separately without this desktop session. Jobs already queued keep their earlier setting.")
  warning.setWrapText(true); warning.getStyleClass.add("muted")
  private val videoWarning = new Label("Videos and their temporary encoding frames are readable; temporary files can remain after a crash. This choice resets when the session is turned off.")
  videoWarning.setWrapText(true); videoWarning.getStyleClass.add("muted")
  val node: VBox = new VBox(10, new Label("Image scrambling — session"), warning, password, new HBox(8, enable, lock), plaintextVideo, videoWarning, status)
  node.setPadding(new Insets(16)); refresh()
  def dispose(): Unit = password.clear()
