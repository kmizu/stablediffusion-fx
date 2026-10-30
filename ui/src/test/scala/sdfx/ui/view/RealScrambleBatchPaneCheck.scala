package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.{Node, Parent, Scene}
import javafx.scene.control.{Button, ChoiceBox, Label, PasswordField, TextField}
import javafx.stage.Stage
import sdfx.core.png.{PngMetadata, ScrambledPng}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import scala.jdk.CollectionConverters.*

object RealScrambleBatchPaneCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(30, TimeUnit.SECONDS)
  def main(args: Array[String]): Unit =
    val root = Path.of(args(0)).toAbsolutePath
    Files.createDirectories(root)
    val input = Files.createTempDirectory(root, "input-")
    val output = Files.createTempDirectory(root, "output-")
    val image = new BufferedImage(16, 12, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 12; x <- 0 until 16 do image.setRGB(x, y, (x << 16) | (y << 8) | 7)
    PngMetadata.write(image, input.resolve("image.png"), Seq("parameters" -> "cat"))
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var stage: Stage = null
    var pane: ScrambleBatchPane = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(pane.node).find(_.getId == id).getOrElse(sys.error(s"missing $id"))
    try
      fx {
        stage = new Stage(); pane = new ScrambleBatchPane(stage)
        stage.setScene(new Scene(pane.node, 850, 700)); stage.show()
        find("scramble-source").asInstanceOf[TextField].setText(input.toString)
        find("scramble-destination").asInstanceOf[TextField].setText(output.toString)
        find("scramble-password").asInstanceOf[PasswordField].setText("secret")
        find("scramble-start").asInstanceOf[Button].fire()
      }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      var status = ""
      while !status.startsWith("Completed") && System.nanoTime() < deadline do
        Thread.sleep(50)
        status = fx(find("scramble-status").asInstanceOf[Label].getText)
      require(status.startsWith("Completed"), status)
      val out = output.resolve("image.png")
      require(Files.isRegularFile(out))
      val decoded = ScrambledPng.read(out, Some("secret")).image
      require(image.getRGB(0, 0, 16, 12, null, 0, 16).sameElements(decoded.getRGB(0, 0, 16, 12, null, 0, 16)))
      require(PngMetadata.readTexts(out).get("parameters").contains("cat"))
      val restoreDir = Files.createTempDirectory(root, "restore-")
      fx {
        find("scramble-source").asInstanceOf[TextField].setText(output.toString)
        find("scramble-destination").asInstanceOf[TextField].setText(restoreDir.toString)
        find("scramble-password").asInstanceOf[PasswordField].setText("wrong")
        find("scramble-mode").asInstanceOf[ChoiceBox[String]].setValue("Restore images")
        find("scramble-start").asInstanceOf[Button].fire()
      }
      var retryStatus = ""
      val retryDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !retryStatus.startsWith("Completed") && System.nanoTime() < retryDeadline do
        Thread.sleep(50)
        retryStatus = fx(find("scramble-status").asInstanceOf[Label].getText)
      require(retryStatus.startsWith("Completed: 0 written"), retryStatus)
      require(!Files.exists(restoreDir.resolve("image.png")), "wrong password wrote an output")
      fx {
        val snap = stage.getScene.snapshot(null)
        javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(snap, null), "png", root.resolve("batch-pane.png").toFile)
        find("scramble-password").asInstanceOf[PasswordField].setText("secret")
        find("scramble-start").asInstanceOf[Button].fire()
      }
      var goodStatus = ""
      val goodDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
      while !goodStatus.startsWith("Completed") && System.nanoTime() < goodDeadline do
        Thread.sleep(50)
        goodStatus = fx(find("scramble-status").asInstanceOf[Label].getText)
      require(goodStatus.startsWith("Completed: 1 written"), goodStatus)
      val restored = PngMetadata.readImage(restoreDir.resolve("image.png"))
      require(image.getRGB(0, 0, 16, 12, null, 0, 16).sameElements(restored.getRGB(0, 0, 16, 12, null, 0, 16)))
      fx { pane.dispose(); require(find("scramble-password").asInstanceOf[PasswordField].getText.isEmpty); stage.close() }
      println("PASS native batch panel: encode, wrong/correct-password restore, output pixels and password field cleanup")
    finally
      if stage != null then fx(stage.close())
      Platform.exit()
