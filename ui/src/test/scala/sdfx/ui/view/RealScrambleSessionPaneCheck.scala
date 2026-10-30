package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{Button, CheckBox, Label, PasswordField}
import javafx.stage.Stage
import sdfx.core.png.ImageScrambleSession
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

object RealScrambleSessionPaneCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    val session = new ImageScrambleSession
    var stage: Stage = null
    var pane: ScrambleSessionPane = null
    var changes = 0
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String) = nodes(pane.node).find(_.getId == id).get
    try
      fx {
        stage = new Stage(); pane = new ScrambleSessionPane(session, () => changes += 1)
        stage.setScene(new Scene(pane.node, 700, 260)); stage.show()
        find("scramble-session-password").asInstanceOf[PasswordField].setText("secret")
        find("scramble-session-enable").asInstanceOf[Button].fire()
        require(session.snapshot.exists(_.use(identity) == "secret"))
        require(find("scramble-session-password").asInstanceOf[PasswordField].getText.isEmpty)
        require(changes == 1)
        val permission = find("scramble-session-plaintext-video").asInstanceOf[CheckBox]
        require(!permission.isSelected && !session.snapshot.get.plaintextVideoAllowed)
        permission.fire()
        val captured = session.snapshot.get
        require(captured.plaintextVideoAllowed && changes == 2)
        permission.fire()
        require(!session.snapshot.get.plaintextVideoAllowed && captured.plaintextVideoAllowed && changes == 3)
        find("scramble-session-lock").asInstanceOf[Button].fire()
        require(session.snapshot.isEmpty && changes == 4 && permission.isDisabled && !permission.isSelected)
        find("scramble-session-password").asInstanceOf[PasswordField].setText("secret")
        find("scramble-session-enable").asInstanceOf[Button].fire()
        require(!permission.isSelected && !session.snapshot.get.plaintextVideoAllowed && changes == 5)
        pane.dispose(); stage.close()
      }
      println("PASS native session controls: enable, lock, snapshot and field cleanup")
    finally
      if stage != null then fx(stage.close())
      Platform.exit()
