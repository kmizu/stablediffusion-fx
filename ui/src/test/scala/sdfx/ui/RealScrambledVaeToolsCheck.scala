package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.image.ImageView
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.png.{PngMetadata, ScrambledPng}
import sdfx.engine.{AppServices, EngineLaunch, EngineState, EngineTimeouts}
import sdfx.mcp.{McpServer, McpTools}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.UiState
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL real native UI/AppServices/MCP acceptance; root executes with real SD15 and SDXL.
  * Args: python checkpoint-dir checkpoint-name original-image output-dir screenshot.png
  */
object RealScrambledVaeToolsCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String, seconds: Int = 90)(condition: => Boolean): Unit =
    val until = System.nanoTime() + seconds.seconds.toNanos
    while !condition && System.nanoTime() < until do Thread.sleep(100)
    require(condition, s"Timed out: $label")
  private def sha(path: Path): String = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString
  def main(args: Array[String]): Unit =
    require(args.length == 5, "python checkpoint-dir checkpoint-name original-image output-dir")
    val Array(python, checkpoints, checkpoint, original, out) = args
    val output = Paths.get(out).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val source = Paths.get(original).toAbsolutePath.normalize()
    val before = sha(source)
    val home = AppHome(Files.createTempDirectory(output, "vae-tools-home-"))
    val engineDir = AppHome.detect().pythonDir
    val services = new AppServices(home, Settings(pythonExe = Some(python), outputDir = output.toString,
      modelDirs = ModelDirs(checkpoints = Seq(Paths.get(checkpoints).toAbsolutePath.toString))), false, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(python, "-m", "sdfx"), engineDir, Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = false, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val started = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); started.countDown() })
    require(started.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def find(root: Node, id: String): Option[Node] =
      if root == null then None
      else if root.getId == id then Some(root)
      else
        val children: Seq[Node] = root match
          case pane: TabPane => pane.getTabs.asScala.flatMap(t => Option(t.getContent)).toSeq
          case pane: TitledPane => Option(pane.getContent).toSeq
          case pane: ScrollPane => Option(pane.getContent).toSeq
          case parent: javafx.scene.Parent => parent.getChildrenUnmodifiable.asScala.toSeq
          case _ => Nil
        children.iterator.flatMap(child => find(child, id)).take(1).toSeq.headOption
    def node[A <: Node](id: String): A = find(window.root, id).getOrElse(throw new IllegalStateException(s"Missing $id")).asInstanceOf[A]
    val protectedSource = output.resolve("native-protected-input.png")
    ScrambledPng.write(ImageIO.read(source.toFile), protectedSource, Nil, Some("vae-native-secret"))
    val protectedBefore = sha(protectedSource)
    def submit(input: Path, operation: String, consent: Boolean): Unit = fx {
      node[TextArea]("vae-tools-report").clear()
      node[TextArea]("vae-tools-input").setText(input.toString)
      node[ComboBox[String]]("vae-tools-operation").setValue(operation)
      node[ComboBox[String]]("vae-tools-representation").setValue("RGBA")
      node[CheckBox]("vae-tools-plaintext-latent").setSelected(consent)
      node[Button]("vae-tools-run").fire()
    }
    def completed(): ujson.Value =
      eventually("VAE report", 600) { fx { !node[Button]("vae-tools-run").isDisabled && node[TextArea]("vae-tools-report").getText.nonEmpty } }
      fx { ujson.read(node[TextArea]("vae-tools-report").getText) }
    def comparePreview(path: Path): Unit =
      val expected = ScrambledPng.read(path, Some("vae-native-secret")).image
      eventually("protected decoded preview") { fx {
        val image = node[ImageView]("vae-tools-preview").getImage
        image != null && image.getPixelReader != null && {
          val actual = SwingFXUtils.fromFXImage(image, null)
          actual.getWidth == expected.getWidth && actual.getHeight == expected.getHeight &&
            actual.getRGB(0, 0, actual.getWidth, actual.getHeight, null, 0, actual.getWidth).sameElements(
              expected.getRGB(0, 0, expected.getWidth, expected.getHeight, null, 0, expected.getWidth))
        }
      } }
    try
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(selectedTab = "vae_tools"), CliOptions())
        w.show(); w.startEngine(); w
      }
      eventually("engine catalog", 600) { services.supervisor.state.isInstanceOf[EngineState.Ready] && fx { window.catalogVM.snapshot.get.checkpoints.exists(_.name == checkpoint) } }
      fx { window.loadCheckpoint(checkpoint) }
      eventually("checkpoint loaded", 600) { services.supervisor.loadedModel.nonEmpty && fx {
        window.topBar.node.lookupAll(".chip-ready").asScala.exists {
          case label: Label => label.getText.startsWith("● Ready")
          case _ => false
        }
      } }
      fx { services.imageScrambleSession.unlock("vae-native-secret"); window.ctx.imageAccessChanged() }
      submit(protectedSource, "encode", false)
      eventually("consent required", 600) { fx {
        !node[Button]("vae-tools-run").isDisabled && node[Label]("vae-tools-status").getText.toLowerCase.contains("consent")
      } }
      require(fx(node[TextArea]("vae-tools-report").getText.isEmpty))
      submit(protectedSource, "encode", true)
      val result = completed()
      val item = result("results")(0)
      require(item("status").str == "complete", result.render())
      require(!fx(node[CheckBox]("vae-tools-plaintext-latent").isSelected), "Consent must reset after submission")
      require(result("warnings").arr.exists(_.str.contains("plaintext")))
      val encoded = Paths.get(item("path").str)
      val preview = Paths.get(item("preview_path").str)
      val tensor = Paths.get(item("latent_path").str)
      require(Files.isRegularFile(tensor) && tensor.toString.endsWith(".safetensors"))
      val texts = PngMetadata.readTexts(encoded)
      require(texts.get("Encrypt").contains("pixel_shuffle_3") && texts.contains("VAE Tools"), texts.toString)
      require(PngMetadata.readTexts(preview).get("Encrypt").contains("pixel_shuffle_3"))
      require(ujson.read(Files.readString(Paths.get(result("report_path").str)))("warnings").arr.exists(_.str.contains("plaintext")))
      comparePreview(preview)
      fx { services.imageScrambleSession.lock(); window.ctx.imageAccessChanged() }
      eventually("lock clears preview") { fx(node[ImageView]("vae-tools-preview").getImage == null) }
      fx { services.imageScrambleSession.unlock("wrong"); window.ctx.imageAccessChanged() }
      eventually("wrong key clears preview") { fx(node[ImageView]("vae-tools-preview").getImage == null) }
      submit(encoded, "decode", false)
      val wrong = completed()
      require(wrong("results")(0)("status").str == "error" && wrong("results")(0)("error").str.toLowerCase.contains("password"), wrong.render())
      fx { services.imageScrambleSession.unlock("vae-native-secret"); window.ctx.imageAccessChanged() }
      submit(encoded, "decode", false)
      val decoded = completed()
      require(decoded("results")(0)("status").str == "complete", decoded.render())
      val decodedPath = Paths.get(decoded("results")(0)("path").str)
      require(PngMetadata.readTexts(decodedPath).get("Encrypt").contains("pixel_shuffle_3"))
      comparePreview(Paths.get(decoded("results")(0)("preview_path").str))
      fx { services.imageScrambleSession.lock(); window.ctx.imageAccessChanged() }
      eventually("decoded lock clears") { fx(node[ImageView]("vae-tools-preview").getImage == null) }
      fx { services.imageScrambleSession.unlock("vae-native-secret"); window.ctx.imageAccessChanged() }
      comparePreview(Paths.get(decoded("results")(0)("preview_path").str))
      require(sha(source) == before && sha(protectedSource) == protectedBefore)
      Files.writeString(output.resolve("native-vae-protected-acceptance.json"), ujson.Obj("encode" -> result, "wrong" -> wrong, "decode" -> decoded).render(indent = 2))
      println("PASS native protected VAE: consent, output metadata, wrong key, lock/unlock, source integrity")
    finally
      try fx { if window != null then { window.dispose(); window.stage.close() } }
      finally
        try services.shutdown()
        finally Platform.exit()
