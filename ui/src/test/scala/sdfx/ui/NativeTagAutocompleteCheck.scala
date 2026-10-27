package sdfx.ui

import java.util.concurrent.{CountDownLatch, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javafx.application.Platform
import javafx.collections.FXCollections
import javafx.event.Event
import javafx.scene.Scene
import javafx.scene.control.TextArea
import javafx.scene.input.{InputMethodEvent, InputMethodHighlight, InputMethodTextRun, KeyCode, KeyEvent}
import javafx.scene.layout.StackPane
import javafx.stage.Stage
import sdfx.ui.view.TagAutocomplete
import sdfx.ui.vm.TagCompletion

/** Native controls and synthetic key/composition events; no engine or GPU. */
object NativeTagAutocompleteCheck:
  def main(args: Array[String]): Unit =
    val done = new CountDownLatch(1)
    val failure = new AtomicReference[Throwable]()
    Platform.startup(() => {
      val area = new TextArea()
      val tags = TagCompletion.parse(Iterator("blue_hair,0,20,", "blue_eyes,0,10,"))
      val completion = new TagAutocomplete(area, () => tags, () => ())
      val stage = new Stage()
      stage.setScene(new Scene(new StackPane(area), 600, 300))
      stage.show()
      area.requestFocus()
      def key(code: KeyCode, control: Boolean = false, released: Boolean = false): Unit =
        Event.fireEvent(area, new KeyEvent(if released then KeyEvent.KEY_RELEASED else KeyEvent.KEY_PRESSED,
          "", "", code, false, control, false, false))
      def input(value: String): Unit =
        area.setText(value)
        area.positionCaret(value.indexOf("blue") + 4)
      Platform.runLater(() => {
        try
          require(area.isFocused, "TextArea must own focus")
          input("日本語, (blue:1.2), end")
          require(completion.isShowing, "Prefix opens popup")
          key(KeyCode.DOWN)
          key(KeyCode.ENTER)
          require(area.getText == "日本語, (blue eyes:1.2), end", area.getText)
          require(!completion.isShowing, "Accept closes popup")
          input("blue, keep")
          key(KeyCode.ESCAPE)
          require(!completion.isShowing && area.getText == "blue, keep", "Escape only dismisses")
          input("blue, preserved")
          val before = area.getText
          key(KeyCode.ENTER, control = true)
          require(area.getText == before, "Ctrl+Enter must not accept a suggestion")
          val runs = FXCollections.observableArrayList(new InputMethodTextRun("あ", InputMethodHighlight.UNSELECTED_RAW))
          Event.fireEvent(area, new InputMethodEvent(InputMethodEvent.INPUT_METHOD_TEXT_CHANGED, runs, "", 1))
          require(completion.isComposing && !completion.isShowing, "IME hides popup")
          Event.fireEvent(area, new InputMethodEvent(InputMethodEvent.INPUT_METHOD_TEXT_CHANGED,
            FXCollections.observableArrayList[InputMethodTextRun](), "あ", 0))
          require(completion.isComposing, "Commit key remains guarded until release")
          key(KeyCode.ENTER, released = true)
          require(!completion.isComposing, "Composition guard releases")
          input("blue, final")
          require(completion.isShowing, "Ordinary input works after IME commit")
          key(KeyCode.TAB)
          require(area.getText == "blue hair, final", area.getText)
          println("NativeTagAutocompleteCheck: 10 checks passed (synthetic IME events; physical Japanese IME not exercised)")
        catch case e: Throwable => failure.set(e)
        finally
          stage.close()
          done.countDown()
      })
    })
    require(done.await(30, TimeUnit.SECONDS), "Native check timeout")
    Platform.exit()
    if failure.get() != null then throw failure.get()
