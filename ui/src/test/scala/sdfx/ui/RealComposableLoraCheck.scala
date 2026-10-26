package sdfx.ui

import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import sdfx.core.png.PngMetadata
import sdfx.core.protocol.Json
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, GenerationListener}
import sdfx.plugins.{ComposableLoraPlugin, StandardPlugins}
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL AppServices acceptance, real model and adapter; never part of automated suites.
  * Args: python checkpoint-directory checkpoint-name lora-directory lora-name output-directory
  * Run once per SD1.5/SDXL. Root operator exclusively executes this GPU check.
  */
object RealComposableLoraCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 6, "python checkpoint-directory checkpoint-name lora-directory lora-name output-directory")
    val python = Paths.get(args(0)).toAbsolutePath
    val output = Paths.get(args(5)).toAbsolutePath
    Files.createDirectories(output)
    val engineDir = AppHome.detect().pythonDir
    val home = AppHome(Files.createTempDirectory(output, "composable-app-home-"))
    val settings = Settings(pythonExe = Some(python.toString), outputDir = output.resolve("images").toString,
      modelDirs = ModelDirs(checkpoints = Seq(args(1)), lora = Seq(args(3))), mcpHttpEnabled = false)
    val services = new AppServices(home, settings, dryRun = false, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(python.toString, "-m", "sdfx"), engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))), persistSettings = false,
      timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val logs = scala.collection.mutable.ArrayBuffer.empty[String]
    val temps = scala.collection.mutable.ArrayBuffer.empty[Path]
    val listener = new GenerationListener:
      override def onLog(s: String): Unit = logs += s
      override def onTempFile(p: Path): Unit = temps += p
    def pixels(p: Path): Array[Int] =
      val im = ImageIO.read(p.toFile)
      im.getRGB(0, 0, im.getWidth, im.getHeight, null, 0, im.getWidth)
    def run(p: GenerationParams) = services.generator.generateWith(p, listener)
    try
      val hello = services.start()
      require(!hello.dryRun && hello.features.contains("composable_lora"))
      val loaded = services.loadCheckpoint(args(2), remember = false)
      require(Set("sd15", "sdxl").contains(loaded.kind))
      val size = if loaded.kind == "sdxl" then 768 else 512
      val base = GenerationParams(prompt = "a red fox in a forest", negativePrompt = "lowres", seed = 78212,
        steps = 6, width = size, height = size, checkpoint = Some(args(2)))
      val ordinary = run(base).head
      val plugin = new ComposableLoraPlugin
      val enabled = Values.defaults(plugin.controls).updated("enabled", "true")
      val noAdapter = run(plugin.process(base, enabled)).head
      require(pixels(ordinary.path).sameElements(pixels(noAdapter.path)), "No-adapter native route changed pixels")
      val prompt = s"a red fox <lora:${args(4)}:.65> AND a white owl <lora:${args(4)}:.15>"
      val params = plugin.process(base.copy(prompt = prompt), enabled.updated("plot_weights", "true"))
      val jsonReplay = Json.fromJs[GenerationParams](Json.toJs(params))
      require(jsonReplay == params)
      val prep = services.generator.prepare(jsonReplay)
      require(prep.request.params.prompt == prompt && prep.request.params.loras.size == 1,
        "Native prepare must preserve placement and resolve each adapter once")
      require(!prep.recorded.hooks("composable_lora").obj.contains("output_dir"))
      val generated = run(jsonReplay)
      require(generated.size == 2, "Source plot must append one additional image")
      require(!pixels(generated.head.path).sameElements(pixels(ordinary.path)), "Adapter image equals baseline")
      val savedText = PngMetadata.readParameters(generated.head.path).get
      val replay = Infotext.toParams(Infotext.parse(savedText))
      require(replay.prompt == prompt && replay.hooks("composable_lora") == params.hooks("composable_lora"))
      val replayed = run(replay)
      require(replayed.size == 2 && pixels(generated.head.path).sameElements(pixels(replayed.head.path)),
        "PNG replay changed source image pixels")
      require(!logs.exists(_.contains("composable diffusion (AND) is not supported; treated as plain text")),
        "Active native Composable generation/replay falsely warned that AND was plain text")
      val traces = logs.filter(_.startsWith("Composable LoRA trace: ")).map(s => Paths.get(s.stripPrefix("Composable LoRA trace: ")))
      require(traces.size == 2 && traces.distinct.size == 2)
      traces.foreach { path =>
        require(Files.isRegularFile(path) && path.startsWith(home.outputs(settings)) && !temps.contains(path))
        require(ujson.read(Files.readString(path))("records").arr.nonEmpty, "Trace lacks actual layer evaluations")
      }
      require(temps.forall(p => !Files.exists(p)), "Native image temporary files survived cleanup")
      val stream = Files.list(home.tmpDir(settings))
      try require(stream.iterator().asScala.toSeq.isEmpty, "Orphan files remain in request .tmp")
      finally stream.close()
      val restored = run(base).head
      require(pixels(ordinary.path).sameElements(pixels(restored.path)), "Adapter cleanup changed ordinary pixels")
      Files.writeString(output.resolve("native-composable-report.json"), ujson.write(ujson.Obj(
        "success" -> true, "model" -> Json.toJs(loaded), "params" -> Json.toJs(params),
        "traces" -> ujson.Arr.from(traces.map(_.toString)), "plot_image_count" -> generated.size,
        "checks" -> ujson.Arr("native tags and unique resolution", "JSON/PNG replay", "exact baseline and cleanup pixels",
          "source plot image", "persistent trace paths and records", "no temporary orphans")), indent = 2))
    finally services.shutdown()
