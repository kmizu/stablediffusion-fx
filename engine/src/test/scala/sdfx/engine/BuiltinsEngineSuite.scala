package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

import java.nio.file.{Files, Path, Paths}
import scala.concurrent.duration.*

/** Host support for the reForge built-in rewrites: extra model folders, hook model resolution, engineCall. */
class BuiltinsEngineSuite extends munit.FunSuite:
  import FakeEngine.*

  override val munitTimeout: Duration = 120.seconds

  private lazy val home =
    val h = tempHome()
    def touch(rel: String): Path =
      val p = h.root.resolve(rel)
      Files.createDirectories(p.getParent)
      Files.write(p, Array[Byte](1))
    touch("models/face_restore/GFPGANv1.4.pth")
    touch("models/controlnet/ip-adapter_sd15.safetensors")
    touch("models/photomaker/photomaker-v1.bin")
    h

  private lazy val services =
    val s = new AppServices(home, Settings(), dryRun = true, builtins = Nil, launchOverride = Some(() => launch()),
      persistSettings = false)
    s.start()
    s

  override def afterAll(): Unit =
    services.shutdown()
    deleteTree(home.root)

  test("catalog lists the extra model kinds and resolves their paths"):
    val c = services.catalog
    assertEquals(c.choices(ChoiceSource.FaceRestorers), Seq("GFPGANv1.4"))
    assertEquals(c.choices(ChoiceSource.PhotoMakerModels), Seq("photomaker-v1"))
    assertEquals(c.choices(ChoiceSource.SvdModels), Nil)
    assert(c.pathOf(ChoiceSource.FaceRestorers, "gfpganv1.4").exists(_.endsWith("GFPGANv1.4.pth")))
    assert(c.pathOf(ChoiceSource.ControlNets, "ip-adapter_sd15").exists(_.endsWith("ip-adapter_sd15.safetensors")))
    assertEquals(c.pathOf(ChoiceSource.FaceRestorers, "nope"), None)

  test("hook model names are resolved to paths for the engine, infotext keeps names"):
    val warns = Seq.newBuilder[String]
    val hooks = Map[String, ujson.Value](
      "ip_adapter" -> ujson.Obj("model" -> "ip-adapter_sd15", "scale" -> 0.7),
      "face_restore" -> ujson.Obj("model" -> "missing-model"),
      "freeu" -> ujson.Obj("model" -> "not a model field"))
    val out = services.generator.resolveHookModels(hooks, warns += _)
    assert(out("ip_adapter")("model").str.endsWith("ip-adapter_sd15.safetensors"))
    assertEquals(out("ip_adapter")("scale").num, 0.7)
    assertEquals(out("face_restore")("model").str, "missing-model")
    assertEquals(out("freeu")("model").str, "not a model field")
    assertEquals(warns.result(), Seq("face_restore model not found: missing-model"))
    val prepared = services.generator.prepare(GenerationParams(hooks = hooks.view.filterKeys(_ == "ip_adapter").toMap))
    assert(prepared.request.params.hooks("ip_adapter")("model").str.endsWith(".safetensors"))
    assertEquals(prepared.recorded.hooks("ip_adapter")("model").str, "ip-adapter_sd15")

  test("engineCall fills output_dir; inside a job its image paths are temp files removed at the end"):
    val r = services.generator.engineCall("caption", ujson.Obj("image_path" -> "x.png", "method" -> "blip"))
    assertEquals(r("caption").str, "fake caption")
    val st = await(services.client.raw("get_state"))
    assert(st("last_extra")("params")("output_dir").str.nonEmpty)

    var produced: Option[Path] = None
    val script = new Plugin:
      def id = "restore_test"
      def name = "restore test"
      def kind = PluginKind.Script
      override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
        val res = ctx.engineCall("restore_faces", ujson.Obj("image_path" -> "in.png", "model" -> "m.pth"))
        produced = Some(Paths.get(res("path").str))
        assert(Files.exists(produced.get))
        JobOutput(Nil)
    await(services.queue.submit(Job(Job.newId(), GenerationParams(), Some(script -> Values.empty), Nil, "test")), 60.seconds)
    assert(produced.isDefined)
    assert(!Files.exists(produced.get), "engine output registered as a job temp file must be deleted")

  test("custom code opt-in reaches the engine environment"):
    assertEquals(PythonEnv.engineEnv(Settings()).get("SDFX_ALLOW_CODE"), None)
    assertEquals(PythonEnv.engineEnv(Settings(allowCustomCode = true)).get("SDFX_ALLOW_CODE"), Some("1"))
