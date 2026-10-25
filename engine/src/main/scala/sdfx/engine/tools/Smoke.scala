package sdfx.engine.tools

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.model.GenerationParams
import sdfx.core.png.PngMetadata
import sdfx.engine.AppServices

import java.nio.file.{Files, Path}
import scala.util.control.NonFatal

/** End-to-end check against the real Python package in `--dry-run` mode:
  * `sbt "engine/runMain sdfx.engine.tools.Smoke"`. Exits non-zero on failure.
  */
object Smoke:
  def main(args: Array[String]): Unit =
    val home = AppHome.detect()
    val out = Files.createTempDirectory("sdfx-smoke-")
    val settings = Settings.load(home.settingsFile).copy(outputDir = out.toString, lastModel = None, lastVae = None)
    println(s"[smoke] home=${home.root} outputs=$out")
    val services = new AppServices(home, settings, dryRun = true, builtins = Nil, persistSettings = false)
    services.addLogListener(l => println(s"[engine] $l"))
    val failures = scala.collection.mutable.ArrayBuffer.empty[String]
    def check(cond: Boolean, msg: => String): Unit = if !cond then failures += msg
    try
      val hello = services.start()
      println(s"[smoke] hello: $hello")
      check(hello.dryRun, "engine did not report dry_run")
      val params = GenerationParams(prompt = "smoke test, a red apple", negativePrompt = "blurry", seed = 1234L, steps = 4,
        width = 256, height = 192, batchSize = 2)
      val images =
        try services.generator.generate(params)
        catch
          case e: sdfx.core.protocol.EngineError if e.code == sdfx.core.protocol.Protocol.ErrorCodes.NoModel && services.catalog.checkpoints.nonEmpty =>
            val ck = services.catalog.checkpoints.head.name
            println(s"[smoke] engine needs a model; loading $ck")
            services.loadCheckpoint(ck, remember = false)
            services.generator.generate(params)
      check(images.size == 2, s"expected 2 images, got ${images.size}")
      images.zipWithIndex.foreach { (img, i) =>
        check(Files.isRegularFile(img.path) && Files.size(img.path) > 0, s"missing PNG ${img.path}")
        PngMetadata.readParameters(img.path) match
          case None => failures += s"no parameters chunk in ${img.path}"
          case Some(text) =>
            val pi = Infotext.parse(text)
            check(pi.prompt == params.prompt, s"prompt mismatch: '${pi.prompt}'")
            check(pi.negativePrompt == params.negativePrompt, s"negative mismatch: '${pi.negativePrompt}'")
            check(pi.get("Seed").contains((1234L + i).toString), s"seed mismatch: ${pi.get("Seed")} (image $i)")
            check(pi.get("Size").contains("256x192"), s"size mismatch: ${pi.get("Size")}")
            val back = Infotext.toParams(pi)
            check(back.steps == 4, s"steps mismatch: ${back.steps}")
        val dim = try
          val im = PngMetadata.readImage(img.path)
          (im.getWidth, im.getHeight)
        catch case NonFatal(e) => (0, 0)
        check(dim == (256, 192), s"image size $dim")
        println(s"[smoke] image ${img.path}")
      }
    catch
      case NonFatal(e) =>
        e.printStackTrace()
        failures += s"exception: $e"
    finally services.shutdown()
    if failures.isEmpty then
      println("[smoke] OK")
      deleteTree(out)
      sys.exit(0)
    else
      failures.foreach(f => println(s"[smoke] FAIL: $f"))
      sys.exit(1)

  private def deleteTree(p: Path): Unit =
    try
      if Files.isDirectory(p) then
        val s = Files.walk(p)
        try s.sorted(java.util.Comparator.reverseOrder()).forEach(f => Files.deleteIfExists(f))
        finally s.close()
    catch case NonFatal(_) => ()
