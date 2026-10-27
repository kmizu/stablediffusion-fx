package sdfx.engine

import java.awt.image.BufferedImage
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import scala.concurrent.{Await, ExecutionContext}
import scala.concurrent.duration.*

/** Real worker/server and supervisor, deliberately paused filesystem publication. */
class TemporalPublicationIntegrationSuite extends munit.FunSuite:
  given ExecutionContext = ExecutionContext.global
  override val munitTimeout: Duration = 90.seconds

  test("held Temporal Lab publication reports too-late without blocking control") {
    import FakeEngine.*
    val python = sys.env.getOrElse("SDFX_TEST_PYTHON", sys.error("SDFX_TEST_PYTHON is required"))
    val project = Paths.get("").toAbsolutePath.normalize()
    val helper = Paths.get(this.getClass.getResource("/temporal_real_publication_worker.py").toURI)
    val dir = Files.createTempDirectory("sdfx-temporal-publication-")
    val hold = dir.resolve("hold")
    val ready = dir.resolve("ready")
    val frames = Files.createDirectory(dir.resolve("frames"))
    val output = Files.createDirectory(dir.resolve("output"))
    Files.writeString(hold, "hold")
    ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", frames.resolve("frame0001.png").toFile)
    val launch = () => EngineLaunch(Seq(python, helper.toString, project.resolve("python").toString,
      ready.toString, hold.toString), project, Map("OPENBLAS_NUM_THREADS" -> "1", "OMP_NUM_THREADS" -> "1"), 30.seconds)
    val sup = new EngineSupervisor(launch)
    try
      Await.result(sup.startAsync(), 20.seconds)
      val pending = sup.requestOwned("temporal_lab", ujson.Obj("operation" -> "deflicker",
        "frames_dir" -> frames.toString, "output_parent" -> output.toString, "output_name" -> "result"),
        _ => (), Some("right-owner"))
      val progressed = eventually(15.seconds)(Files.exists(ready) || pending.isCompleted)
      val trace = dir.resolve("trace")
      assert(progressed, s"worker neither reached publication nor completed; trace=${if Files.exists(trace) then Files.readString(trace) else "missing"}")
      if !Files.exists(ready) then
        fail(s"worker completed before publication: ${pending.value}")
      val other = Await.result(sup.captureInterrupt(Some("other-owner"))(), 5.seconds)
      assert(other.accepted && !other.affected && !other.tooLate)
      Await.result(sup.request("interruption_protection", ujson.Obj("enabled" -> true)), 5.seconds)
      val blocked = Await.result(sup.captureInterrupt(Some("right-owner"))(), 5.seconds)
      assert(blocked.blocked && !blocked.tooLate)
      Await.result(sup.request("interruption_protection", ujson.Obj("enabled" -> false)), 5.seconds)
      val late = Await.result(sup.captureInterrupt(Some("right-owner"))(), 5.seconds)
      assert(late.tooLate && !late.accepted && !late.blocked && !late.affected)
      assert(Await.result(sup.request("ping"), 5.seconds).obj.nonEmpty)
      Files.delete(hold)
      val result = Await.result(pending, 10.seconds)
      assertEquals(result("operation").str, "deflicker")
      assert(Files.isDirectory(output.resolve("result")))
    finally
      Files.deleteIfExists(hold)
      sup.close()
      deleteTree(dir)
  }
