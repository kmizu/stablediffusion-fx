package sdfx.mcp

import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.plugins.api.*
import sdfx.engine.{AppServices, Job}
import sdfx.plugins.StandardPlugins

import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*

/** Manual real-GPU check (not part of `sbt test`): every AlwaysOn plugin goes through the Scala path
  * (Plugin.process -> JobQueue -> Generator -> Python engine -> GPU) on SD1.5.
  *
  * {{{ sbt "mcp/Test/runMain sdfx.mcp.RealAlwaysOnCheck <python exe> <checkpoint dir> <checkpoint name> <out dir> <face model dir>" }}}
  */
object RealAlwaysOnCheck:
  def main(args: Array[String]): Unit =
    val Array(python, ckptDir, ckpt, out, faceDir) = args.take(5)
    val home = AppHome.detect()
    val settings = Settings(
      pythonExe = Some(python), outputDir = out, lastModel = Some(ckpt),
      modelDirs = ModelDirs(checkpoints = Seq(ckptDir), faceRestore = Seq(faceDir)))
    val services = new AppServices(home, settings, dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false)
    val hello = services.start()
    println(s"engine: ${hello.device}, model loaded: ${services.supervisor.loadedModel.map(_.name)}")
    val base = GenerationParams(prompt = "a cat sitting on a windowsill", negativePrompt = "lowres", seed = 21, steps = 12,
      width = 512, height = 512, samplerName = "DPM++ 2M")
    def run(p: GenerationParams, plugins: Seq[(Plugin, Values)]) =
      Await.result(services.queue.submit(Job(Job.newId(), p, None, plugins, "check")), 10.minutes)

    val ref = run(base, Nil).images.head.path
    val mask = Files.createTempFile(Paths.get(out), "mask", ".png")
    val m = new java.awt.image.BufferedImage(512, 512, java.awt.image.BufferedImage.TYPE_INT_RGB)
    val g = m.createGraphics(); g.setColor(java.awt.Color.WHITE); g.fillRect(128, 128, 256, 256); g.dispose()
    ImageIO.write(m, "png", mask.toFile)

    val extra: Map[String, Map[String, String]] = Map(
      "reference" -> Map("image" -> ref.toString),
      "revision" -> Map("image" -> ref.toString),
      "face_restore" -> Map("model" -> "GFPGANv1.4"),
      "random_resolutions" -> Map("resolutions" -> "640x384, 384x640, 576x448"),
      "neveroom" -> Map("vae_tiling" -> "true", "unet_offload" -> "true"),
      "hypertile" -> Map.empty, "latent_modifier" -> Map("sharpness_multiplier" -> "0.3", "combat_cfg_drift" -> "0.3"))
    // model-file plugins (ip_adapter, controllllite, fooocus_inpaint, photomaker) are verified at engine level
    val skip = Set("ip_adapter", "controllllite", "fooocus_inpaint", "photomaker")
    var okCount = 0
    var total = 0
    StandardPlugins.alwaysOn.filterNot(p => skip.contains(p.id)).foreach { pl =>
      val vals = Values(Values.defaults(pl.controls).map ++ Map("enabled" -> "true") ++ extra.getOrElse(pl.id, Map.empty))
      val p = pl.id match
        case "soft_inpainting" => base.copy(img2img = Some(Img2ImgParams(initImage = ref.toString, mask = Some(mask.toString), maskBlur = 16)))
        case _                 => base
      total += 1
      val t0 = System.nanoTime()
      val res =
        try
          val o = run(p, Seq(pl -> vals)).images.head
          val img = ImageIO.read(o.path.toFile)
          val keys = o.params.extra.keys.filterNot(_ == "Version").take(3).mkString(";")
          okCount += 1
          s"OK  ${img.getWidth}x${img.getHeight} active=${pl.isActive(vals)} infotext=[$keys]"
        catch case e: Throwable => s"ERR ${e.getMessage.take(150)}"
      println(f"${pl.id}%-22s $res (${(System.nanoTime() - t0) / 1e9}%.1fs)")
    }
    println(s"SUMMARY $okCount / $total")
    services.shutdown()
    sys.exit(if okCount == total then 0 else 1)
