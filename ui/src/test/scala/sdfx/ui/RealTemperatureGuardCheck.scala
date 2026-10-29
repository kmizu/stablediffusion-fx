package sdfx.ui

import javafx.application.Platform
import javafx.stage.Stage
import java.nio.file.{Files, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.model.GenerationParams
import sdfx.engine.{AppServices, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{UiState, Mode, Outcome}

/** Real native Generate/Interrupt, NVIDIA sensor and real checkpoint. */
object RealTemperatureGuardCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit =
    require(args.length == 3, "home python checkpoint")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val checkpoint = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val settings = Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString)))
    val services = new AppServices(home, settings, dryRun = false,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val paused = new CountDownLatch(1)
    val cancelled = new CountDownLatch(1)
    val recovered = new CountDownLatch(1)
    services.queue.addListener {
      case _: JobEvent.JobCancelled => cancelled.countDown()
      case event: JobEvent.JobFinished =>
        require(event.output.images.nonEmpty, "recovery generation must save an image")
        recovered.countDown()
      case event: JobEvent.JobFailed => println(s"NATIVE JOB FAILED: ${event.message}")
      case _ => ()
    }
    services.addLogListener(line => { println(line); if line.contains("GPU temperature pause:") then paused.countDown() })
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    try
      services.start()
      services.loadCheckpoint(checkpoint.toString, remember = false)
      fx {
        val size = if checkpoint.getFileName.toString.toLowerCase.contains("sdxl") then 768 else 512
        val initial = UiState(params = Map("txt2img" -> GenerationParams(steps = 8, seed = 7, width = size, height = size)),
          pluginValues = Map("txt2img" -> Map("gpu_temperature_guard" -> Map(
            "enabled" -> "true", "sleep_temperature" -> "30", "wake_temperature" -> "20",
            "max_sleep" -> "0", "sleep_time" -> "10", "interval" -> "0.1"))))
        window = new MainWindow(new Stage(), services, initial, CliOptions(), _ => ())
        window.show()
        window.model.checkpoint.set(checkpoint.toString)
        window.statusBar.generateBtn.fire()
      }
      require(paused.await(120, TimeUnit.SECONDS), "native generation never reached temperature pause")
      val start = System.nanoTime()
      fx { window.statusBar.generateBtn.fire() }
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
      while !services.queue.isIdle && System.nanoTime() < deadline do Thread.sleep(50)
      require(services.queue.isIdle, "native interruption did not finish")
      require(cancelled.await(5, TimeUnit.SECONDS), "job must end as cancelled rather than failed or finished")
      val interruptSeconds = (System.nanoTime() - start) / 1e9
      fx {
        require(window.progress.get.outcome.contains(Outcome.Cancelled), "native outcome must show cancellation")
        window.plugins(Mode.Txt2Img).form("gpu_temperature_guard").get.props("enabled").set("false")
        window.statusBar.generateBtn.fire()
      }
      require(recovered.await(120, TimeUnit.SECONDS), "native recovery generation did not finish")
      println(s"PASS native temperature interruption $interruptSeconds seconds; recovery saved an image")
    finally
      fx { if window != null then { window.dispose(); window.stage.close() } }
      services.shutdown()
      Platform.exit()
