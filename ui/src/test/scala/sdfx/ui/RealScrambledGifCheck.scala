package sdfx.ui

import sdfx.engine.{AppServices, Job}
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.GenerationParams
import sdfx.core.png.{ImageScrambleSession, PngMetadata, ScrambledPng}
import sdfx.core.plugins.api.{Plugin, Values}
import sdfx.plugins.{SeedTravelPlugin, ShiftAttentionPlugin}
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*

/** Actual model and queue; checks plaintext GIF permission and decoded-frame parity. */
object RealScrambledGifCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 3, "output-dir python checkpoint")
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val home = AppHome(Files.createTempDirectory(root, "gif-check-")); Files.createDirectories(home.pythonDir)
    val checkpoint = Path.of(args(2)).toAbsolutePath
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString),
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString))), dryRun = false,
      builtins = Nil, persistSettings = false, installShutdownHook = false)
    val session = new ImageScrambleSession; session.unlock("test-gif-secret")
    val p = GenerationParams(prompt = "portrait of (woman:0.8~1.2), soft light", negativePrompt = "low quality",
      seed = 42, steps = 3, width = 256, height = 256, checkpoint = Some(checkpoint.toString))
    val cases: Seq[(Plugin, Values)] = Seq(
      new SeedTravelPlugin -> Values(Map("seeds" -> "43", "steps" -> "1", "frame_ms" -> "120")),
      new ShiftAttentionPlugin -> Values(Map("frames" -> "2", "frame_ms" -> "120")))
    def submit(plugin: Plugin, values: Values, secret: Option[sdfx.core.png.ImageScrambleSecret]) =
      Await.result(service.queue.submit(Job(p).copy(script = Some(plugin -> values), imageSecret = secret)), 10.minutes)
    def gifPath(path: Path): Path = path.resolveSibling(path.getFileName.toString.replaceAll("\\.png$", "") + "-anim.gif")
    def frames(path: Path): Vector[Array[Int]] =
      val stream = ImageIO.createImageInputStream(path.toFile)
      val reader = ImageIO.getImageReadersByFormatName("gif").next()
      try
        reader.setInput(stream)
        (0 until reader.getNumImages(true)).map { index =>
          val img = reader.read(index)
          img.getRGB(0, 0, img.getWidth, img.getHeight, null, 0, img.getWidth)
        }.toVector
      finally { reader.dispose(); stream.close() }
    try
      service.start(); service.loadCheckpoint(checkpoint.toString, remember = false)
      for (plugin, values) <- cases do
        val denied = scala.util.Try(submit(plugin, values, session.snapshot))
        require(denied.failed.toOption.exists(e => Option(e.getMessage).exists(_.contains("plaintext video"))))
        val ordinary = submit(plugin, values, None)
        val protectedResult = submit(plugin, values, session.snapshot.map(_.withPlaintextVideoAllowed))
        require(ordinary.images.size == 2 && protectedResult.images.size == 2)
        ordinary.images.zip(protectedResult.images).foreach { (a, b) =>
          require(PngMetadata.readTexts(b.path).get("Encrypt").contains("pixel_shuffle_3"))
          val plain = PngMetadata.readImage(a.path); val decoded = ScrambledPng.read(b.path, Some("test-gif-secret")).image
          require(plain.getRGB(0, 0, 256, 256, null, 0, 256).sameElements(decoded.getRGB(0, 0, 256, 256, null, 0, 256)))
        }
        val baseline = frames(gifPath(ordinary.images.head.path)); val result = frames(gifPath(protectedResult.images.head.path))
        require(baseline.size == 2 && result.size == baseline.size && baseline.zip(result).forall((a, b) => a.sameElements(b)))
        println(s"PASS ${plugin.id}: denied without permission, protected frames and GIF parity")
    finally service.shutdown()
