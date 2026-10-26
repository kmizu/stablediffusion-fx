package sdfx.mcp

import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.Values
import sdfx.core.protocol.Json
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, Job}
import sdfx.plugins.{LatentUpscalePlugin, StandardPlugins}

import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*

/** MANUAL actual AppServices script/JSON/infotext replay, one checkpoint per process.
  * Args: python checkpoint-dir checkpoint-name source-image output-dir.
  * Run from the integrated repository; user app settings are never saved.
  */
object RealLatentUpscaleCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 5, "python checkpoint-dir checkpoint-name source-image output-dir")
    val Array(python, checkpointDir, checkpoint, sourceName, outputName) = args
    val output = Paths.get(outputName).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val engineDir = AppHome.detect().pythonDir
    val home = AppHome(Files.createTempDirectory(output, "apphome-"))
    val settings = Settings(pythonExe = Some(python), outputDir = output.resolve("images").toString,
      modelDirs = ModelDirs(checkpoints = Seq(Paths.get(checkpointDir).toAbsolutePath.toString)))
    val services = new AppServices(home, settings, dryRun = false, StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(python, "-m", "sdfx"), engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))),
      persistSettings = false, timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    def rgbHash(path: java.nio.file.Path): String =
      val image = ImageIO.read(path.toFile)
      require(image != null)
      val digest = MessageDigest.getInstance("SHA-256")
      for y <- 0 until image.getHeight; x <- 0 until image.getWidth do
        val pixel = image.getRGB(x, y)
        digest.update((pixel >> 16).toByte)
        digest.update((pixel >> 8).toByte)
        digest.update(pixel.toByte)
      digest.digest().map(b => f"${b & 0xff}%02x").mkString
    try
      val hello = services.start()
      require(!hello.dryRun && hello.features.contains("latent_upscale"))
      val loaded = services.loadCheckpoint(checkpoint, remember = false)
      val source = Paths.get(sourceName).toAbsolutePath.normalize()
      val image = ImageIO.read(source.toFile)
      require(image != null)
      val params = GenerationParams(prompt = "detailed texture, crisp edges", negativePrompt = "blurry", seed = 321,
        steps = 7, cfgScale = 3, width = math.max(64, (image.getWidth * 1.25).toInt / 8 * 8),
        height = math.max(64, (image.getHeight * 1.25).toInt / 8 * 8), samplerName = "Euler", scheduler = "Uniform",
        img2img = Some(Img2ImgParams(source.toString, denoisingStrength = .4)))
      val plugin = new LatentUpscalePlugin
      val values = Values(Map("method" -> "area", "scheduler" -> "normal"))
      def run(p: GenerationParams, script: Boolean, v: Values = values) =
        Await.result(services.queue.submit(Job(Job.newId(), p,
          script = if script then Some(plugin -> v) else None, origin = "manual-latent-check")), 10.minutes).images.head
      val first = run(params, true)
      require(first.params.hooks.contains("latent_upscale"))
      require(first.infotext.contains("Latent upscale method: area"))
      val saved = output.resolve("generation-params.json")
      Files.writeString(saved, Json.toJs(first.params).render(indent = 2))
      val restored = Json.fromJs[GenerationParams](ujson.read(Files.readString(saved)))
      val jsonReplay = run(restored, false)
      val fields = Infotext.parse(first.infotext)
      val replayValues = LatentUpscalePlugin.replay(fields.fields).getOrElse(throw new IllegalStateException("Lost plugin infotext"))
      val infotextParams = Infotext.toParams(fields, params)
      val infotextReplay = run(infotextParams, true, replayValues)
      val hash = rgbHash(first.path)
      require(rgbHash(jsonReplay.path) == hash, "Persisted GenerationParams replay changed RGB pixels")
      require(rgbHash(infotextReplay.path) == hash, "Infotext + script settings replay changed RGB pixels")
      val report = ujson.Obj("success" -> true, "model" -> Json.toJs(loaded), "rgb_sha256" -> hash,
        "first" -> first.path.toString, "json_replay" -> jsonReplay.path.toString,
        "infotext_replay" -> infotextReplay.path.toString, "infotext" -> first.infotext,
        "params" -> Json.toJs(first.params), "isolated_home" -> home.root.toString)
      Files.writeString(output.resolve("native-replay-report.json"), report.render(indent = 2))
      println(s"Latent Upscale native AppServices + JSON/infotext replay passed: ${loaded.kind}")
    finally services.shutdown()
