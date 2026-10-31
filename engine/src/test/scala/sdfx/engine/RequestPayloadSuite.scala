package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.model.GenerationParams
import sdfx.core.protocol.*
import scala.concurrent.duration.*

class RequestPayloadSuite extends munit.FunSuite:
  import FakeEngine.*
  override val munitTimeout: Duration = 120.seconds

  private def withServices(f: AppServices => Unit): Unit =
    val home = tempHome()
    val services = new AppServices(home, Settings(lastModel = Some("foo")), dryRun = true, builtins = Nil,
      launchOverride = Some(() => launch()), persistSettings = false, installShutdownHook = false)
    try
      services.start()
      f(services)
    finally
      services.shutdown()
      deleteTree(home.root)

  test("generation password travels outside captured request parameters") {
    withServices { s =>
      val req = GenerationRequest(GenerationParams(seed = 7, steps = 1, width = 64, height = 64),
        s.home.tmpDir(s.settings).toString)
      val session = new sdfx.core.png.ImageScrambleSession
      session.unlock("transport-secret")
      var captured = ""
      await(s.client.generateOwned(req, GenerationMode.Txt2Img, _ => (), None,
        body => captured = body.bodyJson, imageSecret = session.snapshot).result)
      val actual = await(s.client.raw("get_state"))
      assertEquals(actual("last_generation")("request"), ujson.read(captured))
      assertEquals(actual("password_received").bool, true)
      assert(!captured.contains("transport-secret"))
      assert(!actual.render().contains("transport-secret"))
      assertEquals(actual("plaintext_video_allowed").bool, false)
      await(s.client.generateOwned(req, GenerationMode.Txt2Img, _ => (), None,
        body => captured = body.bodyJson, imageSecret = session.snapshot.map(_.withPlaintextVideoAllowed)).result)
      val permitted = await(s.client.raw("get_state"))
      assertEquals(permitted("plaintext_video_allowed").bool, true)
      assert(!captured.contains("allow_plaintext_video"))
      assert(!session.snapshot.get.plaintextVideoAllowed)
    }
  }

  test("editor preview uses protected generation transport without capturing its draft") {
    withServices { s =>
      val req = GenerationRequest(GenerationParams(seed = 7, steps = 1, width = 64, height = 64),
        s.home.tmpDir(s.settings).toString)
      val session = new sdfx.core.png.ImageScrambleSession
      session.unlock("editor-secret")
      var observed = false
      val draft = ujson.Obj("name" -> "cat", "vectors" -> ujson.Obj("clip_l" -> ujson.Arr(ujson.Arr(3.0))))
      await(s.client.generateOwned(req, GenerationMode.Txt2Img, _ => (), None,
        _ => observed = true, imageSecret = session.snapshot, editorPreview = Some(EditorPreviewPayload(draft.render()))).result)
      val state = await(s.client.raw("get_state"))
      assert(!observed)
      assertEquals(state("password_received").bool, true)
      assertEquals(state("last_generation")("request")("embedding_editor_preview"), draft)
      assert(!state.render().contains("editor-secret"))
      assert(!Job(GenerationParams()).copy(editorPreview = Some(EditorPreviewPayload(draft.render()))).toString.contains("clip_l"))
    }
  }

  test("observed body equals actual fake-process input and mutable hooks cannot change either") {
    withServices { s =>
      var draws = 0
      val random = new scala.util.Random(0L):
        override def nextLong(): Long =
          draws += 1
          100L + draws
      val generator = new Generator(s.client, s.catalog, s.home, () => s.settings, random = random)
      val hook = ujson.Obj("nested" -> ujson.Arr(ujson.Obj("literal" -> "<b>雪</b>")))
      val prep = generator.prepare(GenerationParams(seed = -1, subseed = -1, steps = 1, width = 64, height = 64,
        hooks = Map("diagnostic_fixture" -> hook)))
      var captured: Option[GenerationBody] = None
      val before = System.currentTimeMillis()
      val handle = s.client.generateOwned(prep.request, prep.mode, _ => (), None, body => {
        captured = Some(body)
        hook("nested")(0)("literal") = "MUTATED"
      })
      val result = await(handle.result)
      val capture = captured.getOrElse(fail("body observer not invoked"))
      val actual = await(s.client.raw("get_state"))("last_generation")
      assertEquals(ujson.read(capture.bodyJson), actual("request"))
      assertEquals(capture.requestType, "txt2img")
      assert(capture.capturedAtMillis >= before)
      assertEquals(draws, 2)
      val body = ujson.read(capture.bodyJson)
      assertEquals(body("params")("hooks")("diagnostic_fixture")("nested")(0)("literal").str, "<b>雪</b>")
      assertEquals(body("params")("seed").num.toLong, 101L)
      assertEquals(body("params")("subseed").num.toLong, 102L)
      assertEquals(body("params")("img2img"), ujson.Null)
      assertEquals(body("params")("subseed_strength").num, 0.0)
      assertEquals(body("params")("hires")("resize_x").num, 0.0)
      assertEquals(body("preview_every").num, s.settings.previewEvery.toDouble)
      assert(!body.obj.contains("outputDir"))
      assertEquals(result.images.head.seed, 101L)
    }
  }

  test("throwing body observer leaves a valid generation and received body unchanged") {
    withServices { s =>
      val req = GenerationRequest(GenerationParams(seed = 7, subseed = 8, steps = 1, width = 64, height = 64),
        s.home.tmpDir(s.settings).toString)
      val result = await(s.client.generateOwned(req, GenerationMode.Img2Img, _ => (), None,
        _ => throw new IllegalStateException("viewer failed")).result)
      assertEquals(result.images.head.seed, 7L)
      val actual = await(s.client.raw("get_state"))("last_generation")
      assertEquals(actual("type").str, "img2img")
      assertEquals(actual("request"), Json.toJs(req))
    }
  }

  test("required body serialization errors are not hidden as successful capture or dispatch") {
    withServices { s =>
      val before = await(s.client.raw("get_state"))("last_generation")
      var observed = false
      val broken = new scala.collection.mutable.ArrayBuffer[ujson.Value]:
        override def foreach[U](f: ujson.Value => U): Unit =
          throw new IllegalStateException("required body conversion failed")
      broken += ujson.Str("never serialized")
      val invalid = GenerationRequest(GenerationParams(seed = 1, hooks = Map("broken" -> new ujson.Arr(broken))),
        s.home.tmpDir(s.settings).toString)
      val error = intercept[IllegalStateException] {
        await(s.client.generateOwned(invalid, GenerationMode.Txt2Img, _ => (), None, _ => observed = true).result)
      }
      assertEquals(error.getMessage, "required body conversion failed")
      assert(!observed)
      assertEquals(await(s.client.raw("get_state"))("last_generation"), before)
    }
  }

  test("an interrupted optional observer cannot cancel the actual generation") {
    withServices { s =>
      val req = GenerationRequest(GenerationParams(seed = 17, subseed = 18, steps = 1, width = 64, height = 64),
        s.home.tmpDir(s.settings).toString)
      val result = try await(s.client.generateOwned(req, GenerationMode.Txt2Img, _ => (), None,
        _ => throw new InterruptedException("optional diagnostic interrupted")).result)
      catch case _: InterruptedException => fail("Optional observer interruption escaped into generation")
      assertEquals(result.images.head.seed, 17L)
      assertEquals(await(s.client.raw("get_state"))("last_generation")("request"), Json.toJs(req))
    }
  }
