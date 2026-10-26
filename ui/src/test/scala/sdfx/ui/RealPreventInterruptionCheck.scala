package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.event.Event
import javafx.scene.control.*
import javafx.scene.input.{KeyCode, KeyEvent}
import javafx.stage.Stage
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.{InterruptionProtection, Json}
import sdfx.engine.{AppServices, EngineLaunch, Job, JobCancelledException, JobEvent}
import sdfx.mcp.{McpServer, McpTools}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL root-only native acceptance. python output-dir [checkpoint-directory checkpoint-name].
  * Without checkpoint args: actual dry-run JSON Lines process, never a GPU pipeline.
  * With checkpoint args: actual SD15/SDXL generation must be in progress during protected actions.
  */
object RealPreventInterruptionCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)
  private def eventually(label: String)(body: => Boolean): Unit =
    val end = System.nanoTime() + 8.seconds.toNanos
    while !body && System.nanoTime() < end do Thread.sleep(25)
    require(body, "Timed out: " + label)
  def main(args: Array[String]): Unit =
    require(args.length == 2 || args.length == 4, "python output-dir [checkpoint-directory checkpoint-name]")
    val python = Paths.get(args(0)).toAbsolutePath
    val output = Paths.get(args(1)).toAbsolutePath
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "protection-home-"))
    val real = args.length == 4
    val checkpointDir = if real then Paths.get(args(2)).toAbsolutePath else home.root.resolve("models")
    val checkpoint = if real then args(3) else "control-test"
    if !real then
      Files.createDirectories(checkpointDir)
      val header = ujson.Obj("model.diffusion_model.input_blocks.0.0.weight" -> ujson.Obj("dtype" -> "F16",
        "shape" -> ujson.Arr(320, 4, 3, 3), "data_offsets" -> ujson.Arr(0, 2)),
        "cond_stage_model.transformer.text_model.x" -> ujson.Obj("dtype" -> "F16", "shape" -> ujson.Arr(1),
          "data_offsets" -> ujson.Arr(2, 4))).render().getBytes(java.nio.charset.StandardCharsets.UTF_8)
      Files.write(checkpointDir.resolve("control-test.safetensors"), ByteBuffer.allocate(8 + header.length + 4)
        .order(ByteOrder.LITTLE_ENDIAN).putLong(header.length).put(header).putInt(0).array())
    val engineDir = AppHome.detect().pythonDir
    val command = Seq(python.toString, "-m", "sdfx") ++ (if real then Nil else Seq("--dry-run", "--dry-run-delay-ms", "35"))
    val services = new AppServices(home, Settings(pythonExe = Some(python.toString), outputDir = output.resolve("images").toString,
      modelDirs = ModelDirs(checkpoints = Seq(checkpointDir.toString))), dryRun = !real, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(command, engineDir, Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = false, installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var checks = Vector.empty[String]
    var loaded: sdfx.core.protocol.LoadedModel = null
    def toggle: CheckBox =
      val n = window.scene.lookup("#prevent-interruption-toggle")
      require(n != null, "Missing native Prevent interruption toggle")
      n.asInstanceOf[CheckBox]
    def enabled: Boolean = Await.result(services.client.interruptionProtection(), 5.seconds).enabled
    def protect(value: Boolean): Unit =
      fx { if toggle.isSelected != value then toggle.fire() }
      eventually("effective native protection") { enabled == value && fx { !toggle.isDisabled && toggle.isSelected == value } }
    val mcp = new McpServer(new McpTools(services).all)
    def invoke(name: String, params: ujson.Obj): ujson.Value =
      val response = mcp.handle(ujson.Obj("jsonrpc" -> "2.0", "id" -> name, "method" -> "tools/call",
        "params" -> ujson.Obj("name" -> name, "arguments" -> params)), _ => ()).get
      require(!response.obj.contains("error"), response.toString)
      require(!response("result")("isError").bool, response.toString)
      response("result")("structuredContent")
    def runAfterFirstProgress(params: GenerationParams)(action: => Unit) =
      val job = Job(params)
      val progress = new CountDownLatch(1)
      val remove = services.queue.addListener {
        case p: JobEvent.Progress if p.jobId == job.id => progress.countDown()
        case _ => ()
      }
      val result = services.queue.submit(job)
      try
        require(progress.await(90, TimeUnit.SECONDS), "Actual generation never reached first sampling progress")
        require(!result.isCompleted, "Workload completed before protection actions could run")
        action
      finally remove()
      result
    try
      window = fx { val w = new MainWindow(new Stage(), services, UiState(), CliOptions()); w.show(); w }
      val hello = services.start()
      require(hello.dryRun == !real && hello.features.contains("prevent_interrupt"))
      fx { require(!toggle.isSelected) }
      require(!enabled)
      loaded = services.loadCheckpoint(checkpoint, remember = false)
      fx { window.model.checkpoint.set(loaded.name) }
      require(!real || Set("sd15", "sdxl")(loaded.kind))
      protect(true)
      require(invoke("interruption_protection", ujson.Obj())("enabled").bool)
      invoke("interruption_protection", ujson.Obj("enabled" -> false))
      eventually("other frontend state converges within 5s polling") { fx { !toggle.isSelected } }
      protect(true)
      checks :+= "default-off native toggle, acknowledged reader state and other frontend convergence"
      val size = if real && loaded.kind == "sdxl" then 512 else 256
      val params = GenerationParams(prompt = "a small lighthouse at sunset", seed = 66214, width = size, height = size,
        steps = 30, checkpoint = Some(loaded.name))
      val originalPid = services.supervisor.pid
      val complete = runAfterFirstProgress(params) {
        fx {
          require(window.statusBar.generateBtn.getText.contains("Interrupt") && !window.statusBar.generateBtn.isDisabled,
            "Footer was not an actionable Interrupt control")
          window.statusBar.generateBtn.fire()
          Event.fireEvent(window.scene, new KeyEvent(KeyEvent.KEY_PRESSED, "", "", KeyCode.ESCAPE, false, false, false, false))
          val tabs = window.scene.lookup("#main-tabs").asInstanceOf[TabPane]
          tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "system-info-tab").get)
          val interrupt = window.scene.lookup("#system-info-interrupt")
          require(interrupt != null, "Missing native SystemInfo interrupt action")
          require(!interrupt.isDisabled, "SystemInfo interrupt action was disabled")
          interrupt.asInstanceOf[Button].fire()
          window.select(Mode.Txt2Img)
        }
        val result = invoke("interrupt", ujson.Obj())
        require(result("blocked").bool && !result("interrupted").bool)
        fx { window.restartEngine() }
      }
      val images = Await.result(complete, 3.minutes).images
      require(images.size == 1 && Files.isRegularFile(images.head.path))
      require(services.supervisor.pid == originalPid, "Protected explicit restart replaced the owned process")
      checks :+= "actual sampling-progress latch; footer/Esc/SystemInfo/MCP interruption and explicit user restart blocked; output finished"
      eventually("completed image is actually presented before screenshot") {
        fx {
          val image = window.preview.zoom.image
          window.preview.currentPath.contains(images.head.path) && image != null &&
            image.getProgress == 1.0 && !image.isError && image.getWidth > 0 && image.getHeight > 0
        }
      }
      fx {
        window.root.applyCss(); window.root.layout()
        require(ImageIO.write(SwingFXUtils.fromFXImage(window.scene.snapshot(null), null), "png", output.resolve("protected.png").toFile))
      }
      protect(false)
      val cancelled = runAfterFirstProgress(params.copy(seed = 66215)) {
        fx { window.interrupt() }
      }
      try
        Await.result(cancelled, 3.minutes)
        throw new AssertionError("Unlocked generation did not interrupt")
      catch case _: JobCancelledException => ()
      checks :+= "unlock restores ordinary cancellation and truthful native job outcome"
      if real then
        eventually("native job queue drained before standalone benchmark") { services.queue.isIdle }
        val benchmarkProgress = new CountDownLatch(1)
        val benchmark = services.client.systemBenchmark(ujson.Obj("level" -> "quick", "steps" -> "long",
          "warmup" -> false, "width" -> size, "height" -> size), {
          case _: sdfx.core.protocol.EngineEvent.Progress => benchmarkProgress.countDown()
          case _ => ()
        })
        require(benchmarkProgress.await(90, TimeUnit.SECONDS), "Standalone real benchmark never reached sampling progress")
        require(services.queue.isIdle && !benchmark.isCompleted, "Benchmark must be active outside the native job queue")
        fx {
          val tabs = window.scene.lookup("#main-tabs").asInstanceOf[TabPane]
          tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "system-info-tab").get)
          val interrupt = window.scene.lookup("#system-info-interrupt").asInstanceOf[Button]
          require(!interrupt.isDisabled)
          interrupt.fire()
        }
        try
          Await.result(benchmark, 3.minutes)
          throw new AssertionError("Off SystemInfo interrupt did not stop its own non-queue benchmark")
        catch case e: sdfx.core.protocol.EngineError if e.isInterrupted => ()
        checks :+= "Off SystemInfo native control stops its own actual benchmark outside JobQueue"
      if !real then
        protect(true)
        Await.result(services.supervisor.forceRestart(), 30.seconds)
        eventually("new process resets protection off") { !enabled && fx { !toggle.isSelected } }
        require(services.supervisor.pid != originalPid)
        checks :+= "CPU health restart bypass remains functional and new server resets off"
        // Reproduce an actual UI completion ordering without depending on scheduler timing:
        // an initial toggle request had no PID, its old On/revision1 reply was queued on FX,
        // then a new process reached Off/revision0 before that completion was delivered.
        fx {
          val controllerField = classOf[MainWindow].getDeclaredFields.find(_.getType.getName ==
            "sdfx.ui.view.InterruptionControls").getOrElse(sys.error("Missing interruption presentation controller"))
          controllerField.setAccessible(true)
          val controller = controllerField.get(window)
          val publish = controller.getClass.getDeclaredMethods.find(_.getName == "publish").get
          publish.setAccessible(true)
          publish.invoke(controller, InterruptionProtection(enabled = true, revision = 1), None)
          require(!toggle.isSelected, "Stale unbound On/revision1 reply was applied to a new Off/revision0 process")
        }
        eventually("fresh PID-bound query remains authoritative after unbound completion") {
          !enabled && fx { !toggle.isSelected }
        }
        checks :+= "stale unbound UI completion cannot migrate old process revision into the new process"
      Files.writeString(output.resolve("prevent-interruption-report.json"), ujson.write(ujson.Obj("success" -> true,
        "real_model" -> real, "model" -> Json.toJs(loaded), "checks" -> ujson.Arr.from(checks),
        "completed_images" -> ujson.Arr.from(images.map(_.path.toString))), indent = 2))
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown()
      Platform.exit()
