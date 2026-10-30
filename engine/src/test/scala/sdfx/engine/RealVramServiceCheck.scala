package sdfx.engine

import java.nio.file.{Files, Paths}
import java.util.concurrent.{CountDownLatch, TimeUnit}
import scala.concurrent.Await
import scala.concurrent.duration.*
import sdfx.core.config.{AppHome, Settings, ModelDirs}
import sdfx.core.protocol.{EngineError, EngineEvent, Protocol}

/** Real device acceptance for native service JSON round trips and owned interruption. */
object RealVramServiceCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 3, "home python checkpoint")
    val home = AppHome(Paths.get(args(0)).toAbsolutePath)
    val model = Paths.get(args(2)).toAbsolutePath
    Files.createDirectories(home.pythonDir)
    val services = new AppServices(home, Settings(pythonExe = Some(Paths.get(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(model.getParent.toString))), dryRun = false, builtins = Nil,
      persistSettings = false, installShutdownHook = false)
    try
      services.start()
      services.loadCheckpoint(model.toString, remember = false)
      val estimator = new VramEstimatorService(services)
      val measured = Await.result(estimator.benchmark(ujson.Obj("max_size" -> 320, "max_batch" -> 1)), 180.seconds)
      require(measured("rows").arr.size == 8 && measured("status").str == "complete")
      val reloaded = estimator.history().last
      val exported = home.root.resolve("exported-vram-profile.json")
      estimator.exportProfile(reloaded, exported)
      val imported = Await.result(estimator.importProfile(exported), 20.seconds)
      require(imported.size == 1 && imported.head == reloaded, "export/import changed measurements or duplicated same run")
      val invalid = ujson.read(reloaded.render())
      invalid("identity")("tampered") = ujson.Bool(true)
      val invalidFile = home.root.resolve("invalid-vram-profile.json")
      Files.writeString(invalidFile, invalid.render())
      try
        Await.result(estimator.importProfile(invalidFile), 20.seconds)
        throw new IllegalStateException("tampered profile imported")
      catch case error: EngineError => require(error.code == Protocol.ErrorCodes.InvalidParams)
      require(estimator.history() == imported, "failed import changed history")
      val estimated = Await.result(estimator.estimate(reloaded, "txt2img", 288, 288, 1), 20.seconds)
      require(estimated("available").bool, s"native JSON profile roundtrip failed: $estimated")
      val first = new CountDownLatch(1)
      val running = estimator.benchmark(ujson.Obj("max_size" -> 512, "max_batch" -> 2, "steps" -> 8), {
        case progress: EngineEvent.Progress if progress.pass == "vram_estimator" => first.countDown()
        case _ => ()
      })
      require(first.await(120, TimeUnit.SECONDS), "no native calibration case completed")
      Await.result(services.client.interruptionProtection(Some(true)), 10.seconds)
      val blocked = Await.result(estimator.interrupt(), 10.seconds)
      require(blocked.blocked && !blocked.accepted && estimator.isBenchmarking, "interrupt protection did not preserve benchmark")
      val output = home.outputs(services.settings)
      val foreign = services.client.rawOwned(Protocol.Types.Txt2Img, ujson.Obj("output_dir" -> output.toString,
        "params" -> ujson.Obj("prompt" -> "a ceramic teapot", "seed" -> 7, "steps" -> 2, "width" -> 256, "height" -> 256)),
        Some("foreign-generation"))
      Await.result(services.client.interruptionProtection(Some(false)), 10.seconds)
      val decision = Await.result(estimator.interrupt(), 10.seconds)
      require(decision.accepted && !decision.blocked)
      try
        Await.result(running, 60.seconds)
        throw new IllegalStateException("benchmark completed instead of cancelling")
      catch case error: EngineError => require(error.code == Protocol.ErrorCodes.Interrupted, error.toString)
      val generated = Await.result(foreign, 60.seconds)
      require(Files.isRegularFile(Paths.get(generated("images")(0)("path").str)), "owned stop interrupted foreign generation")
      val history = estimator.history()
      require(history.exists(_("status").str == "complete") && history.exists(_("status").str == "interrupted"), "partial history lost completed run")
      require(!estimator.isBenchmarking, "benchmark owner did not release")
      println(s"PASS native VRAM service: JSON profile roundtrip/estimate, protected stop, owned cancellation/foreign generation, history=${history.size}")
    finally services.shutdown()
