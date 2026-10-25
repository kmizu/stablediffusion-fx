package sdfx.engine

import sdfx.core.config.{AppHome, Settings}

import java.nio.file.{Files, Path, Paths}
import scala.concurrent.duration.*
import scala.concurrent.{Await, Future}

object FakeEngine:
  lazy val script: Path = Paths.get(getClass.getResource("/fake_engine.py").toURI)

  /** Python discovered the same way the app does (a temp home has no venv, so system Python). */
  lazy val python: Seq[String] =
    val tmpHome = AppHome(Files.createTempDirectory("sdfx-pyprobe-"))
    PythonEnv.locate(tmpHome, Settings()).getOrElse(sys.error("no python found for tests"))

  def launch(extra: String*): EngineLaunch =
    EngineLaunch(python ++ Seq(script.toString) ++ extra, script.getParent, Map.empty, 30.seconds)

  def await[T](f: Future[T], d: FiniteDuration = 30.seconds): T = Await.result(f, d)

  def eventually(timeout: FiniteDuration = 20.seconds)(cond: => Boolean): Boolean =
    val end = System.nanoTime() + timeout.toNanos
    var ok = cond
    while !ok && System.nanoTime() < end do
      Thread.sleep(50)
      ok = cond
    ok

  /** A temp app home with a few fake model files. */
  def tempHome(): AppHome =
    val root = Files.createTempDirectory("sdfx-home-")
    def touch(rel: String): Path =
      val p = root.resolve(rel)
      Files.createDirectories(p.getParent)
      Files.write(p, Array[Byte](1, 2, 3))
    touch("models/checkpoints/foo.safetensors")
    touch("models/checkpoints/sdxl/Bar_XL.safetensors")
    touch("models/lora/My Lora.safetensors")
    touch("models/lora/sub/detail.safetensors")
    touch("models/vae/vae-ft-mse.safetensors")
    touch("models/upscalers/4x-Ultra.pth")
    Files.createDirectories(root.resolve("config"))
    Files.writeString(root.resolve("config/styles.csv"), "name,prompt,negative_prompt\r\nPretty,\"{prompt}, masterpiece\",lowres\r\n")
    AppHome(root)

  def deleteTree(p: Path): Unit =
    try
      val s = Files.walk(p)
      try s.sorted(java.util.Comparator.reverseOrder()).forEach(f => Files.deleteIfExists(f))
      finally s.close()
    catch case _: Exception => ()
