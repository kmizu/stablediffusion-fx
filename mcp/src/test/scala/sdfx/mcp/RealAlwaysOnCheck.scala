package sdfx.mcp

import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.model.{GenerationParams, Img2ImgParams}
import sdfx.core.infotext.Infotext
import sdfx.core.json.SnakePickle.{read, write}
import sdfx.core.png.PngMetadata
import sdfx.core.plugins.api.*
import sdfx.engine.{AppServices, Job, JobEvent}
import sdfx.plugins.StandardPlugins

import java.nio.file.{Files, Paths}
import java.util.concurrent.ConcurrentLinkedQueue
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import scala.util.Using

/** Manual real-GPU check (not part of `sbt test`): every AlwaysOn plugin goes through the Scala path
  * (Plugin.process -> JobQueue -> Generator -> Python engine -> GPU) on SD1.5.
  *
  * {{{ sbt "mcp/Test/runMain sdfx.mcp.RealAlwaysOnCheck <python exe> <checkpoint dir> <checkpoint name> <out dir> <face model dir> [plugin ids...]" }}}
  * Select `save_intermediate prompt_fusion` for the native persistence/replay gates. Run separately with SD1.5
  * and SDXL checkpoint names. Evidence is saved under <out dir>/native-alwayson-evidence/. This helper executes
  * real inference only when explicitly run; it is never discovered by `sbt test`.
  */
object RealAlwaysOnCheck:
  def main(args: Array[String]): Unit =
    val Array(python, ckptDir, ckpt, out, faceDir) = args.take(5)
    val only = args.drop(5).toSet
    val home = AppHome.detect()
    val settings = Settings(
      pythonExe = Some(python), outputDir = out, lastModel = Some(ckpt),
      modelDirs = ModelDirs(checkpoints = Seq(ckptDir), faceRestore = Seq(faceDir)))
    val services = new AppServices(home, settings, dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false)
    val outputRoot = home.outputs(settings).toAbsolutePath.normalize()
    val evidence = outputRoot.resolve("native-alwayson-evidence")
    Files.createDirectories(evidence)
    val logs = new ConcurrentLinkedQueue[String]()
    services.queue.addListener {
      case JobEvent.Log(_, message) => logs.add(message); ()
      case _ => ()
    }
    def manifests: Set[java.nio.file.Path] =
      val root = outputRoot.resolve("intermediates")
      if !Files.isDirectory(root) then Set.empty
      else Using.resource(Files.walk(root))(_.iterator().asScala.filter(_.getFileName.toString == "manifest.json").toSet)
    def verifyCapture(before: Set[java.nio.file.Path], expected: Int, label: String): Unit =
      val created = manifests -- before
      require(created.size == 1, s"$label: expected one new persistent manifest, got $created")
      val manifest = created.head
      val data = ujson.read(Files.readString(manifest))
      require(data("status").str == "complete", s"$label: incomplete manifest")
      require(data("settings")("output_dir").str == outputRoot.toString, s"$label: output root was not replaced")
      val frames = data("frames").arr
      require(frames.count(_("pass_name").str == "final") == expected, s"$label: final copy count")
      require(frames.count(_("pass_name").str == "base") == expected * 3, s"$label: cadence/batch count")
      frames.foreach { f =>
        val path = Paths.get(f("path").str).toAbsolutePath.normalize()
        require(path.startsWith(outputRoot.resolve("intermediates")) && Files.isRegularFile(path), s"$label: lost/outside frame $path")
        val text = PngMetadata.readTexts(path)
        require(text.contains("sdfx_intermediate") && text.contains("parameters"), s"$label: PNG metadata missing")
        require(ujson.read(text("sdfx_intermediate"))("seed") == f("seed"), s"$label: PNG seed mismatch")
      }
      require(logs.asScala.exists(_.contains(manifest.toString)), s"$label: typed auxiliary manifest was not logged")
      Files.writeString(evidence.resolve(s"$label-manifest.json"), data.render(indent = 2))
    val hello = services.start()
    println(s"engine: ${hello.device}, model loaded: ${services.supervisor.loadedModel.map(_.name)}")
    val prompt = if only.contains("weight_gradient") then "portrait of a woman wearing {red scarf::0.2-1.4:e}, in a park"
      else if only.contains("negpip") then "portrait of a woman, (black hair:-2), in a park"
      else "a cat sitting on a windowsill"
    val base = GenerationParams(prompt = prompt, negativePrompt = "lowres", seed = 21, steps = 12,
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
      "neutral_prompt" -> Map("cfg_rescale" -> "0.3", "format_enabled" -> "true", "extra_prompt" -> "red scarf",
        "prompt_type" -> "AND_TOPK", "prompt_weight" -> "0.7"),
      "save_intermediate" -> Map("every_n" -> "2", "final_save" -> "true"),
      "prompt_fusion" -> Map("slerp_scale" -> "0.5", "negative_origin" -> "true"),
      "hypertile" -> Map.empty, "latent_modifier" -> Map("sharpness_multiplier" -> "0.3", "combat_cfg_drift" -> "0.3"))
    // model-file plugins (ip_adapter, controllllite, fooocus_inpaint, photomaker) are verified at engine level
    val skip = Set("ip_adapter", "controllllite", "fooocus_inpaint", "photomaker")
    Set("save_intermediate", "prompt_fusion").intersect(only).foreach { id =>
      require(StandardPlugins.alwaysOn.exists(_.id == id), s"requested native gate plugin is not registered: $id")
    }
    var okCount = 0
    var total = 0
    StandardPlugins.alwaysOn.filterNot(p => skip.contains(p.id)).filter(p => only.isEmpty || only.contains(p.id)).foreach { pl =>
      val vals = Values(Values.defaults(pl.controls).map ++ Map("enabled" -> "true") ++ extra.getOrElse(pl.id, Map.empty))
      val p = pl.id match
        case "soft_inpainting" => base.copy(img2img = Some(Img2ImgParams(initImage = ref.toString, mask = Some(mask.toString), maskBlur = 16)))
        case "save_intermediate" => base.copy(steps = 6, batchSize = 2, nIter = 2)
        case "prompt_fusion" => base.copy(prompt = "[a cat:a fox: , ] sitting on a windowsill",
          negativePrompt = "[lowres:blurry::mean]")
        case _                 => base
      total += 1
      val t0 = System.nanoTime()
      val res =
        try
          val before = manifests
          val galleryBefore = services.gallery.scan().map(_.path).toSet
          val generated = run(p, Seq(pl -> vals))
          require(generated.images.size == p.imagesPerJob, s"${pl.id}: final image count changed")
          val o = generated.images.head
          if Set("save_intermediate", "prompt_fusion").contains(pl.id) then
            val embedded = PngMetadata.readParameters(o.path).getOrElse(sys.error("saved PNG parameters missing"))
            require(embedded == o.infotext, s"${pl.id}: stored PNG metadata differs")
            require(Infotext.parse(embedded).prompt == p.prompt, s"${pl.id}: prompt metadata changed")
            Files.writeString(evidence.resolve(s"${pl.id}-parameters.txt"), embedded)
            Files.writeString(evidence.resolve(s"${pl.id}-params.json"), write(o.params, indent = 2))
          if pl.id == "save_intermediate" then
            require(o.params.hooks.contains("save_intermediate"), "intermediate hook missing")
            require(o.infotext.contains("Save intermediate settings:"), "intermediate infotext missing")
            verifyCapture(before, p.imagesPerJob, "save_intermediate")
            val newGallery = services.gallery.scan().map(_.path).toSet -- galleryBefore
            require(newGallery == generated.images.map(_.path).toSet, "intermediate artifacts polluted final gallery")
            // Replay the persisted full params, then inject the raw destination as an MCP client could.
            // No AlwaysOn plugin reprocessing: the Generator itself must overwrite this field.
            val restored = read[GenerationParams](Files.readString(evidence.resolve("save_intermediate-params.json")))
            val outside = outputRoot.resolveSibling(s"outside-intermediate-${java.util.UUID.randomUUID()}")
            val raw = ujson.read(restored.hooks("save_intermediate").render()).obj
            raw("output_dir") = ujson.Str(outside.toString)
            val replay = restored.copy(hooks = restored.hooks.updated("save_intermediate", ujson.Obj.from(raw)))
            val replayBefore = manifests
            val replayGalleryBefore = services.gallery.scan().map(_.path).toSet
            val replayed = run(replay, Nil)
            require(replayed.images.size == p.imagesPerJob, "replay final image count changed")
            require(!Files.exists(outside), "raw hook destination was trusted")
            verifyCapture(replayBefore, p.imagesPerJob, "save_intermediate-replay")
            require((services.gallery.scan().map(_.path).toSet -- replayGalleryBefore) == replayed.images.map(_.path).toSet,
              "replay artifacts polluted final gallery")
            val tmp = home.tmpDir(settings)
            require(Using.resource(Files.walk(tmp))(_.iterator().asScala.forall(p => !Files.isRegularFile(p))), "engine temp files leaked")
          if pl.id == "prompt_fusion" then
            require(o.params.prompt == p.prompt && o.params.negativePrompt == p.negativePrompt, "Fusion grammar was rewritten")
            require(o.params.hooks.contains("prompt_fusion"), "Prompt Fusion hook missing")
            require(o.infotext.contains("Prompt Fusion: True"), "Prompt Fusion infotext missing")
            val restored = read[GenerationParams](Files.readString(evidence.resolve("prompt_fusion-params.json")))
            require(restored.hooks("prompt_fusion") == o.params.hooks("prompt_fusion"), "Fusion replay hook changed")
            val replayed = run(restored, Nil)
            require(replayed.images.size == restored.imagesPerJob, "Fusion replay final count")
            val a = ImageIO.read(o.path.toFile)
            val b = ImageIO.read(replayed.images.head.path.toFile)
            require(a.getWidth == b.getWidth && a.getHeight == b.getHeight &&
              a.getRGB(0, 0, a.getWidth, a.getHeight, null, 0, a.getWidth).sameElements(
                b.getRGB(0, 0, b.getWidth, b.getHeight, null, 0, b.getWidth)), "Fusion persisted-params replay changed final pixels")
          if pl.id == "weight_gradient" then
            require(o.params.hooks.contains("weight_gradient"), "weight gradient hook missing")
            require(o.infotext.contains("Weight Gradient: True"), "weight gradient infotext missing")
          if pl.id == "negpip" then
            require(o.params.hooks.contains("negpip"), "NegPiP hook missing")
            require(o.infotext.contains("NegPiP Active: True"), "NegPiP infotext missing")
          if pl.id == "neutral_prompt" then
            require(o.params.prompt.contains("AND_TOPK red scarf :0.7"), "Neutral Prompt formatter missing")
            require(o.params.hooks.contains("neutral_prompt"), "Neutral Prompt hook missing")
            require(o.infotext.contains("CFG Rescale phi: 0.3"), "Neutral Prompt rescale infotext missing")
            require(o.infotext.contains("Neutral Prompt: True"), "Neutral Prompt infotext missing")
          val img = ImageIO.read(o.path.toFile)
          val keys = o.params.extra.keys.filterNot(_ == "Version").take(3).mkString(";")
          if Set("save_intermediate", "prompt_fusion").contains(pl.id) then
            Files.writeString(evidence.resolve(s"${pl.id}-result.json"), ujson.Obj(
              "status" -> "passed", "checkpoint" -> ckpt, "plugin" -> pl.id,
              "final_images" -> ujson.Arr.from(generated.images.map(_.path.toString)),
              "final_count" -> generated.images.size, "full_params_replay" -> true,
              "png_parameters_checked" -> true).render(indent = 2))
          okCount += 1
          s"OK  ${img.getWidth}x${img.getHeight} active=${pl.isActive(vals)} infotext=[$keys]"
        catch case e: Throwable => s"ERR ${e.getMessage.take(150)}"
      println(f"${pl.id}%-22s $res (${(System.nanoTime() - t0) / 1e9}%.1fs)")
    }
    println(s"SUMMARY $okCount / $total")
    Files.writeString(evidence.resolve("engine-output.log"), logs.asScala.mkString("\n"))
    services.shutdown()
    sys.exit(if okCount == total && total > 0 then 0 else 1)
