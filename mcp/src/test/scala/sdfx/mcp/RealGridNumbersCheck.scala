package sdfx.mcp

import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.json.SnakePickle.{read, write}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.png.PngMetadata
import sdfx.engine.{AppServices, Job}
import sdfx.plugins.{GridNumbersPlugin, StandardPlugins, XyzGridPlugin}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*

/** Parent-run manual GPU/AppServices gate; never discovered by sbt test.
  * sbt "mcp/Test/runMain sdfx.mcp.RealGridNumbersCheck <python> <checkpoint-dir> <checkpoint-name> <output-dir>"
  * Run with separate SD1.5 and SDXL output roots. Inspect the saved batch/XYZ grids and native plugin form.
  */
object RealGridNumbersCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 4, "expected python, checkpoint-dir, checkpoint-name, output-dir")
    val Array(python, directory, checkpoint, output) = args
    val home = AppHome.detect()
    val services = new AppServices(home, Settings(pythonExe = Some(python), outputDir = output,
      lastModel = Some(checkpoint), previewEvery = 0, modelDirs = ModelDirs(checkpoints = Seq(directory))),
      dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false)
    val root = home.outputs(services.settings).toAbsolutePath.normalize()
    val evidence = root.resolve("grid-numbers-evidence")
    Files.createDirectories(evidence)
    def hash(path: Path) = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString
    def pixels(path: Path): Seq[Int] =
      val image = ImageIO.read(path.toFile)
      image.getRGB(0, 0, image.getWidth, image.getHeight, null, 0, image.getWidth).toSeq
    def run(p: GenerationParams, active: Seq[(Plugin, Values)] = Nil, script: Option[(Plugin, Values)] = None) =
      Await.result(services.queue.submit(Job(Job.newId(), p, script, active, "manual-grid-numbers")), 20.minutes)
    def mapping(grid: OutputImage): Seq[String] =
      val text = PngMetadata.readParameters(grid.path).getOrElse(sys.error("grid PNG parameters missing"))
      val parsed = Infotext.parse(text)
      require(parsed.get("Grid image numbers").contains("True"), "grid numbering metadata missing")
      ujson.read(parsed.get("Grid source images").getOrElse(sys.error("source mapping missing"))).arr
        .map(_("filename").str).toSeq
    try
      val hello = services.start()
      require(hello.device != "cpu", "this manual gate is intended for the parent's real-model GPU run")
      val p = GenerationParams(prompt = "a red lighthouse on a rocky island at sunset", negativePrompt = "lowres",
        seed = 123, steps = 6, width = 512, height = 512, batchSize = 2, nIter = 2)
      val plugin = new GridNumbersPlugin
      val on = Values.defaults(plugin.controls).updated("enabled", "true")
      val baseline = run(p)
      require(baseline.images.size == 4 && baseline.grids.isEmpty, "disabled output changed")
      val numbered = run(p, Seq(plugin -> on))
      require(numbered.images.size == 4 && numbered.grids.size == 1, "numbered batch counts wrong")
      baseline.images.zip(numbered.images).foreach((a, b) => require(pixels(a.path) == pixels(b.path), "source final pixels changed"))
      require(mapping(numbered.grids.head) == numbered.images.map(_.path.getFileName.toString), "number/filename mapping wrong")
      val sourceHashes = numbered.images.map(o => o.path -> hash(o.path)).toMap
      Files.writeString(evidence.resolve("batch-params.json"), write(numbered.grids.head.params, indent = 2))
      val restored = read[GenerationParams](Files.readString(evidence.resolve("batch-params.json")))
      val replay = run(restored)
      require(replay.images.size == 4 && replay.grids.size == 1, "persisted batch replay failed")
      require(mapping(replay.grids.head) == replay.images.map(_.path.getFileName.toString), "replay used stale filename numbers")
      sourceHashes.foreach((path, expected) => require(hash(path) == expected, "replay changed an existing source PNG"))
      val xyz = new XyzGridPlugin
      val axes = Values.defaults(xyz.controls).updated("x_type", "Steps").updated("x_values", "4,6")
        .updated("y_type", "CFG Scale").updated("y_values", "5,7").updated("z_type", "Clip skip").updated("z_values", "1,2")
        .updated("include_sub_images", "true").updated("include_sub_grids", "true").updated("draw_legend", "false")
      val axisOutput = run(p.copy(batchSize = 1, nIter = 1), Seq(plugin -> on.updated("xyz_info", "true")), Some(xyz -> axes))
      require(axisOutput.images.size == 8 && axisOutput.grids.size == 3, "XYZ output counts wrong")
      axisOutput.images.foreach { o =>
        require(o.params.extra("Grid X info") == s"Steps: ${o.params.steps}", "X label wrong")
        require(o.params.extra("Grid Y info") == s"CFG Scale: ${Values.fmt(o.params.cfgScale)}", "Y label wrong")
        require(o.params.extra("Grid Z info") == s"Clip skip: ${o.params.clipSkip}", "Z label wrong")
      }
      axisOutput.grids.foreach(mapping)
      val report = ujson.Obj("status" -> "passed", "checkpoint" -> checkpoint,
        "batch_grid" -> numbered.grids.head.path.toString, "replay_grid" -> replay.grids.head.path.toString,
        "xyz_grids" -> ujson.Arr.from(axisOutput.grids.map(o => ujson.Str(o.path.toString))),
        "unchanged_source_hashes" -> ujson.Obj.from(sourceHashes.toSeq.map((p, h) => p.toString -> ujson.Str(h))))
      Files.writeString(evidence.resolve("report.json"), report.render(indent = 2))
      println(s"GRID NUMBERS VERIFIED: ${evidence.resolve("report.json")}")
    finally services.shutdown()
