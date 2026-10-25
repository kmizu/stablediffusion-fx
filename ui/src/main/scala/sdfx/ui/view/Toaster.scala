package sdfx.ui.view

import javafx.animation.{FadeTransition, PauseTransition, SequentialTransition}
import javafx.geometry.{Insets, Pos}
import javafx.scene.control.{Button, Label}
import javafx.scene.layout.{HBox, Priority, StackPane, VBox}
import javafx.util.Duration

/** Transient notifications in the bottom-right corner (an overlay layer above the main layout). */
final class Toaster:
  val layer = new VBox(8)
  layer.setAlignment(Pos.BOTTOM_RIGHT)
  layer.setPickOnBounds(false)
  layer.setPadding(new Insets(0, 18, 86, 0))
  layer.setMaxWidth(Double.MaxValue)
  layer.setMaxHeight(Double.MaxValue)
  StackPane.setAlignment(layer, Pos.BOTTOM_RIGHT)

  def show(message: String, kind: Toaster.Kind = Toaster.Kind.Info, seconds: Double = -1): Unit =
    val text = new Label(message)
    text.setWrapText(true)
    text.setMaxWidth(420)
    val icon = new Label(kind match
      case Toaster.Kind.Info    => "ℹ"
      case Toaster.Kind.Success => "✓"
      case Toaster.Kind.Error   => "⚠"
    )
    icon.getStyleClass.add("toast-icon")
    val close = new Button("✕")
    close.getStyleClass.addAll("icon-button", "toast-close")
    val box = new HBox(10, icon, text, close)
    HBox.setHgrow(text, Priority.ALWAYS)
    box.setAlignment(Pos.TOP_LEFT)
    box.getStyleClass.addAll("toast", s"toast-${kind.toString.toLowerCase}")
    box.setMaxWidth(480)
    box.setMaxHeight(javafx.scene.layout.Region.USE_PREF_SIZE)
    var seq: SequentialTransition = null
    var dismissed = false
    def dismiss(): Unit =
      if seq != null then seq.stop()
      if !dismissed then
        dismissed = true
        fade()
    def fade(): Unit =
      val f = new FadeTransition(Duration.millis(220), box)
      f.setToValue(0)
      f.setOnFinished(_ => layer.getChildren.remove(box))
      f.play()
    close.setOnAction(_ => dismiss())
    box.setOnMouseClicked(_ => dismiss())
    if layer.getChildren.size >= 4 then layer.getChildren.remove(0)
    layer.getChildren.add(box)
    val secs = if seconds > 0 then seconds else if kind == Toaster.Kind.Error then 9.0 else 4.0
    val fadeIn = new FadeTransition(Duration.millis(160), box)
    fadeIn.setFromValue(0)
    fadeIn.setToValue(1)
    val wait = new PauseTransition(Duration.seconds(secs))
    seq = new SequentialTransition(fadeIn, wait)
    seq.setOnFinished(_ => dismiss())
    box.setOnMouseEntered(_ => seq.pause())
    box.setOnMouseExited(_ => if !dismissed then seq.play())
    seq.play()

object Toaster:
  enum Kind:
    case Info, Success, Error
