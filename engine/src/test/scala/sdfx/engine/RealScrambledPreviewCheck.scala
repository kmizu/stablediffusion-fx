package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.png.{ImageScrambleSession, PngMetadata, ScrambledPng}
import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.concurrent.{Await, ExecutionContext}
import scala.concurrent.duration.*

/** Actual Python dry-run worker preview; no GPU/model. */
object RealScrambledPreviewCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 2, "output directory and Python executable")
    given ExecutionContext = ExecutionContext.global
    val output = Path.of(args(0)).toAbsolutePath
    Files.createDirectories(output)
    val home = AppHome(Files.createTempDirectory(output, "home-"))
    Files.createDirectories(home.pythonDir)
    val service = new AppServices(home, Settings(pythonExe = Some(Path.of(args(1)).toAbsolutePath.toString)),
      dryRun = true, builtins = Nil, persistSettings = false, installShutdownHook = false)
    def fixture(name: String, change: Boolean): (Path, Path) =
      val image = new BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB)
      for y <- 0 until 48; x <- 0 until 64 do
        image.setRGB(x, y, if change && x >= 10 && x < 30 && y >= 5 && y < 20 then 0xf02030 else (x << 16) | (y << 8) | 30)
      val plain = output.resolve(name + ".png")
      val encoded = output.resolve(name + "-scrambled.png")
      PngMetadata.write(image, plain, Nil)
      ScrambledPng.write(image, encoded, Nil, Some("test secret"))
      (plain, encoded)
    val (base, protectedBase) = fixture("base", false)
    val (altered, protectedAltered) = fixture("altered", true)
    val secret = new ImageScrambleSession; secret.unlock("test secret")
    def run(a: Path, b: Path, key: Option[sdfx.core.png.ImageScrambleSecret]) =
      Await.result(InpaintDifferencePreview.run(service, a.toString, b.toString, home.tmpDir(service.settings), ujson.Obj(), key), 60.seconds)
    try
      service.start()
      val ordinary = run(base, altered, None)
      val expected = ImageIO.read(ordinary.mask.toFile)
      val encoded = run(protectedBase, protectedAltered, secret.snapshot)
      val actual = ImageIO.read(encoded.mask.toFile)
      require(expected.getRGB(0, 0, 64, 48, null, 0, 64).sameElements(actual.getRGB(0, 0, 64, 48, null, 0, 64)), "mask mismatch")
      ordinary.cleanup(); encoded.cleanup()
      val wrong = new ImageScrambleSession; wrong.unlock("wrong")
      val refused = scala.util.Try(run(protectedBase, protectedAltered, wrong.snapshot))
      require(refused.isFailure, "wrong password accepted")
      val left = scala.util.Using.resource(Files.list(home.tmpDir(service.settings)))(_.anyMatch(p => p.getFileName.toString.startsWith("sdfx-unscrambled-")))
      require(!left, "plaintext input temporary leaked")
      println("PASS real dry-run difference preview: pixel parity, incorrect password, temporary cleanup")
    finally service.shutdown()
