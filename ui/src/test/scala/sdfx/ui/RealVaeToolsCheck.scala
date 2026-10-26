package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.image.ImageView
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
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
object RealVaeToolsCheck:
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
    require(args.length == 6, "python checkpoint-dir checkpoint-name original-image output-dir screenshot.png")
    val Array(python, checkpoints, checkpoint, original, out, screenshotName) = args
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
    def run(input: String, operation: String, rep: String, format: String = "TIFF"): ujson.Value =
      fx {
        node[TextArea]("vae-tools-report").clear()
        node[TextArea]("vae-tools-input").setText(input)
        node[ComboBox[String]]("vae-tools-operation").setValue(operation)
        node[ComboBox[String]]("vae-tools-representation").setValue(rep)
        node[ComboBox[String]]("vae-tools-cmyk-format").setValue(format)
        node[Button]("vae-tools-run").fire()
      }
      eventually(s"native $operation $rep", 600) { fx { !node[Button]("vae-tools-run").isDisabled && node[TextArea]("vae-tools-report").getText.nonEmpty } }
      val data = fx { ujson.read(node[TextArea]("vae-tools-report").getText) }
      require(data("results").arr.forall(_("status").str == "complete"), data.render())
      val expectedPath = Paths.get(data("results")(0)("preview_path").str).toRealPath()
      // Image canonicalizes file:/// URLs to file:/; compare actual file identity,
      // then wait for background loading and compare every pixel, not URL spelling.
      eventually(s"loaded current $operation $rep preview") { fx {
        val image = node[ImageView]("vae-tools-preview").getImage
        image != null && image.getProgress >= 1 && !image.isError
      } }
      val expected = ImageIO.read(expectedPath.toFile)
      fx {
        val image = node[ImageView]("vae-tools-preview").getImage
        val actualPath = Paths.get(java.net.URI.create(image.getUrl)).toRealPath()
        require(Files.isSameFile(actualPath, expectedPath), s"Same-file rerun retained old preview: ${image.getUrl}; expected $expectedPath")
        val actual = SwingFXUtils.fromFXImage(image, null)
        require(actual.getWidth == expected.getWidth && actual.getHeight == expected.getHeight, "Preview dimensions differ from current result")
        val width = expected.getWidth
        val height = expected.getHeight
        require(actual.getRGB(0, 0, width, height, null, 0, width).sameElements(expected.getRGB(0, 0, width, height, null, 0, width)),
          "Preview pixels differ from current result")
      }
      require(sha(source) == before, "Input bytes changed")
      data
    def snapshot(path: Path): Unit = fx {
      window.root.applyCss()
      window.root.layout()
      Option(path.getParent).foreach(Files.createDirectories(_))
      require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", path.toFile))
    }
    try
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(selectedTab = "vae_tools"), CliOptions())
        w.show()
        w.startEngine()
        w
      }
      eventually("engine startup/catalog") { services.supervisor.state.isInstanceOf[EngineState.Ready] && fx { window.catalogVM.snapshot.get.checkpoints.exists(_.name == checkpoint) } }
      fx { window.loadCheckpoint(checkpoint) }
      eventually("loaded checkpoint and native Ready", 600) { services.supervisor.loadedModel.nonEmpty && fx {
        window.topBar.node.lookupAll(".chip-ready").asScala.exists {
          case label: Label => label.getText.startsWith("● Ready")
          case _ => false
        }
      } }
      fx { require(window.state.selectedTab == "vae_tools") }
      val encoded = run(source.toString, "encode", "L")
      val decoded = run(encoded("results")(0)("path").str, "decode", "L")
      val cmyk = run(source.toString, "encode", "CMYK", "JPEG100")
      val cmykDecoded = run(cmyk("results")(0)("path").str, "decode", "CMYK")
      val rgba = run(source.toString + "\n" + source.toString, "encode", "RGBA")
      require(rgba("results").arr.size == 2)
      val quantized = run(rgba("results")(0)("path").str, "decode", "RGBA")
      fx {
        node[TextArea]("vae-tools-input").setText(output.resolve("missing-file.png").toString)
        node[Button]("vae-tools-run").fire()
        require(!node[Button]("vae-tools-run").isDisabled)
        require(node[Label]("vae-tools-status").getText.contains("existing input"))
        node[CheckBox]("vae-tools-tiled").fire()
      }
      val restored = run(source.toString, "encode+decode", "L")
      // Show the real result and controls at a smaller viewport; page should scroll, not clip.
      fx { window.stage.setWidth(1100); window.stage.setHeight(850); node[ScrollPane]("vae-tools-scroll").setVvalue(0) }
      snapshot(Paths.get(screenshotName).toAbsolutePath.normalize())
      fx { node[ScrollPane]("vae-tools-scroll").setVvalue(1) }
      snapshot(output.resolve("vae-tools-results.png"))
      val mcp = new McpServer(new McpTools(services, McpTools.waitForEngine(services), allowedDirs = Seq(source.getParent)).all)
      val response = mcp.handle(ujson.Obj("jsonrpc" -> "2.0", "id" -> 1, "method" -> "tools/call", "params" -> ujson.Obj(
        "name" -> "vae_tools", "arguments" -> ujson.Obj("input_paths" -> ujson.Arr(source.toString), "operation" -> "encode", "representation" -> "CMYK"))), _ => ()).get
      require(!response.obj.contains("error") && !response("result")("isError").bool, response.render())
      val mcpEncoded = response("result")("structuredContent")
      require(mcpEncoded("results")(0)("path").str.endsWith(".tiff"))
      require(sha(source) == before)
      Files.writeString(output.resolve("native-acceptance.json"), ujson.Obj("success" -> true, "source_sha256" -> before,
        "encoded" -> encoded, "decoded" -> decoded, "cmyk" -> cmyk, "cmyk_decoded" -> cmykDecoded,
        "rgba_batch" -> rgba, "rgba_decoded" -> quantized, "restored" -> restored, "mcp" -> mcpEncoded,
        "checks" -> ujson.Arr("real native startup/load", "selected tab restore", "L PNG decode", "CMYK JPEG original mode", "RGBA batch",
          "validation error and recovery", "tiling", "viewport and snapshots", "MCP safe file path/no image conversion")).render(indent = 2))
      println("VAE Tools native/AppServices/MCP acceptance passed")
    finally
      try fx { if window != null then { window.dispose(); window.stage.close() } }
      finally
        try services.shutdown()
        finally Platform.exit()
