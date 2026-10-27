package sdfx.ui

import munit.FunSuite
import sdfx.core.model.*
import sdfx.core.payload.SubmittedPayloadContext
import sdfx.core.plugins.api.*
import sdfx.engine.{CaptureOptions, Job, JobLimits}
import scala.collection.immutable.ListMap

class FrozenJobEvidenceSuite extends FunSuite:
  private class Script extends Plugin:
    def id = "script"
    def name = "script"
    def kind = PluginKind.Script
  private class Marker extends Plugin:
    def id = "marker"
    def name = "marker"
    def kind = PluginKind.AlwaysOn
  private val script = new Script
  private val marker = new Marker
  private val marker2 = new Marker
  private def fixture: Job = Job(
    id = "job-001",
    params = GenerationParams(
      prompt = "raw\r\nprompt", negativePrompt = "negative", seed = 123456789L, subseed = 987654321L,
      subseedStrength = 0.3, samplerName = "sampler", scheduler = "scheduler", steps = 23, cfgScale = 6.5,
      width = 640, height = 768, batchSize = 2, nIter = 3, clipSkip = 2, eta = Some(0.2),
      hires = HiresParams(true, 1.5, "up", 8, 960, 1152, 0.4, Some("hs"), Some("hc"), "hp", "hn", Some(4.5)),
      img2img = Some(Img2ImgParams("C:\\image.png", Some("mask"), 0.6, 2, 7, 3, true, 40, true)),
      loras = Seq(LoraRef("one", 0.8, Some(0.7), Some("lbw"), Some(0.1), Some(0.9)), LoraRef("two")),
      controlnet = Seq(ControlNetUnit(true, "model", Some("image"), "prep", 0.6, 0.2, 0.8, ListMap("z" -> 2.0, "a" -> 1.0)), ControlNetUnit(model = "second")),
      hooks = ListMap("freeu" -> ujson.Obj("enabled" -> false, "nested" -> ujson.Arr(ujson.Obj("x" -> 1), ujson.Null)), "other" -> ujson.Str("raw")),
      styles = Seq("first", "second"), checkpoint = Some("model"), vae = Some("vae"), extra = ListMap("z" -> "last", "a" -> "first")),
    script = Some(script -> Values(ListMap("z" -> "last", "evidence" -> "retained"))),
    alwaysOn = Seq(marker -> Values(Map("evidence" -> "retained")), marker2 -> Values(Map("other" -> "2"))),
    origin = "api", limits = JobLimits(4096, 12, 20, 12345678L),
    submittedContext = Some(SubmittedPayloadContext("source", Some("img2img"), Some(" {\"x\": 1}\r\n"), Some(true))),
    captureOptions = Some(CaptureOptions(false)))

  private def fields(p: Product): Set[String] = p.productElementNames.toSet
  test("schema guards detect added or removed product fields") {
    val j = fixture; val p = j.params
    assertEquals(fields(j), Set("id", "params", "script", "alwaysOn", "origin", "limits", "submittedContext", "captureOptions"))
    assertEquals(fields(p), Set("prompt", "negativePrompt", "seed", "subseed", "subseedStrength", "samplerName", "scheduler", "steps", "cfgScale", "width", "height", "batchSize", "nIter", "clipSkip", "eta", "hires", "img2img", "loras", "controlnet", "hooks", "styles", "checkpoint", "vae", "extra"))
    assertEquals(fields(p.hires), Set("enabled", "scale", "upscaler", "secondPassSteps", "resizeX", "resizeY", "denoisingStrength", "samplerName", "scheduler", "prompt", "negativePrompt", "cfgScale"))
    assertEquals(fields(p.img2img.get), Set("initImage", "mask", "denoisingStrength", "resizeMode", "maskBlur", "inpaintingFill", "inpaintFullRes", "inpaintFullResPadding", "inpaintingMaskInvert"))
    assertEquals(fields(p.loras.head), Set("name", "weight", "unet", "lbw", "start", "stop"))
    assertEquals(fields(p.controlnet.head), Set("enabled", "model", "image", "preprocessor", "weight", "guidanceStart", "guidanceEnd", "preprocessorArgs"))
    assertEquals(fields(j.limits), Set("maxSide", "maxImagesPerGeneration", "maxGenerations", "maxPixelsPerGeneration"))
    assertEquals(fields(j.submittedContext.get), Set("origin", "sourceMode", "configurationJson", "uiVariationEnabled"))
    assertEquals(fields(j.captureOptions.get), Set("includeImageContents"))
    assertEquals(fields(j.script.get._2), Set("map"))
  }

  test("snapshot records complete values and raw context without engine payload projection") {
    val evidence = FrozenJobEvidence.freeze(fixture)
    val json = ujson.read(evidence.snapshotJson)
    assertEquals(json.obj.keySet.toSet, Set("id", "params", "script", "alwaysOn", "origin", "limits", "submittedContext", "captureOptions"))
    assertEquals(json("id").str, "job-001")
    assertEquals(json("params")("prompt").str, "raw\r\nprompt")
    assertEquals(json("params")("hooks")("freeu")("nested")(1), ujson.Null)
    assertEquals(json("script")("pluginId").str, "script")
    assertEquals(json("script")("pluginClass").str, script.getClass.getName)
    assertEquals(json("script")("values")("evidence").str, "retained")
    assertEquals(json("alwaysOn").arr.map(_("pluginId").str).toSeq, Seq("marker", "marker"))
    assertEquals(json("submittedContext")("configurationJson").str, " {\"x\": 1}\r\n")
    assertEquals(json("captureOptions")("includeImageContents").bool, false)
    assertEquals(json("params")("seed").num.toLong, 123456789L)
  }

  test("each job and recursively nested parameter field changes evidence and hash") {
    val j = fixture; val p = j.params; val h = p.hires; val i = p.img2img.get; val l = p.loras.head; val c = p.controlnet.head
    val variants: Seq[(String, Job)] = Seq(
      "id" -> j.copy(id = "different"), "origin" -> j.copy(origin = "different"),
      "script absent" -> j.copy(script = None), "script values" -> j.copy(script = Some(script -> Values(Map("evidence" -> "changed")))),
      "alwaysOn absent" -> j.copy(alwaysOn = Nil), "alwaysOn order" -> j.copy(alwaysOn = j.alwaysOn.reverse),
      "alwaysOn values" -> j.copy(alwaysOn = Seq(marker -> Values(Map("evidence" -> "changed")), j.alwaysOn(1))),
      "limits maxSide" -> j.copy(limits = j.limits.copy(maxSide = 4000)), "limits images" -> j.copy(limits = j.limits.copy(maxImagesPerGeneration = 11)),
      "limits generations" -> j.copy(limits = j.limits.copy(maxGenerations = 19)), "limits pixels" -> j.copy(limits = j.limits.copy(maxPixelsPerGeneration = 12345677L)),
      "context absent" -> j.copy(submittedContext = None), "context origin" -> j.copy(submittedContext = j.submittedContext.map(_.copy(origin = "other"))),
      "context sourceMode" -> j.copy(submittedContext = j.submittedContext.map(_.copy(sourceMode = None))),
      "context raw configuration" -> j.copy(submittedContext = j.submittedContext.map(_.copy(configurationJson = Some("{\"x\":1}")))),
      "context variation" -> j.copy(submittedContext = j.submittedContext.map(_.copy(uiVariationEnabled = Some(false)))),
      "capture absent" -> j.copy(captureOptions = None), "capture value" -> j.copy(captureOptions = Some(CaptureOptions(true)))
    ) ++ Seq[(String, GenerationParams)](
      "prompt" -> p.copy(prompt = "other"), "negativePrompt" -> p.copy(negativePrompt = "other"), "seed" -> p.copy(seed = 124),
      "subseed" -> p.copy(subseed = 123), "subseedStrength" -> p.copy(subseedStrength = 0.4), "samplerName" -> p.copy(samplerName = "other"),
      "scheduler" -> p.copy(scheduler = "other"), "steps" -> p.copy(steps = 24), "cfgScale" -> p.copy(cfgScale = 7),
      "width" -> p.copy(width = 648), "height" -> p.copy(height = 776), "batchSize" -> p.copy(batchSize = 1), "nIter" -> p.copy(nIter = 4),
      "clipSkip" -> p.copy(clipSkip = 3), "eta" -> p.copy(eta = None),
      "hires enabled" -> p.copy(hires = h.copy(enabled = false)), "hires scale" -> p.copy(hires = h.copy(scale = 2)),
      "hires upscaler" -> p.copy(hires = h.copy(upscaler = "other")), "hires steps" -> p.copy(hires = h.copy(secondPassSteps = 9)),
      "hires resizeX" -> p.copy(hires = h.copy(resizeX = 968)), "hires resizeY" -> p.copy(hires = h.copy(resizeY = 1160)),
      "hires denoising" -> p.copy(hires = h.copy(denoisingStrength = 0.5)), "hires sampler" -> p.copy(hires = h.copy(samplerName = None)),
      "hires scheduler" -> p.copy(hires = h.copy(scheduler = None)), "hires prompt" -> p.copy(hires = h.copy(prompt = "other")),
      "hires negative" -> p.copy(hires = h.copy(negativePrompt = "other")), "hires cfg" -> p.copy(hires = h.copy(cfgScale = None)),
      "img2img absent" -> p.copy(img2img = None), "img init" -> p.copy(img2img = Some(i.copy(initImage = "other"))),
      "img mask" -> p.copy(img2img = Some(i.copy(mask = None))), "img denoising" -> p.copy(img2img = Some(i.copy(denoisingStrength = 0.7))),
      "img resize" -> p.copy(img2img = Some(i.copy(resizeMode = 1))), "img blur" -> p.copy(img2img = Some(i.copy(maskBlur = 8))),
      "img fill" -> p.copy(img2img = Some(i.copy(inpaintingFill = 2))), "img full" -> p.copy(img2img = Some(i.copy(inpaintFullRes = false))),
      "img padding" -> p.copy(img2img = Some(i.copy(inpaintFullResPadding = 41))), "img invert" -> p.copy(img2img = Some(i.copy(inpaintingMaskInvert = false))),
      "lora name" -> p.copy(loras = Seq(l.copy(name = "other"), p.loras(1))), "lora weight" -> p.copy(loras = Seq(l.copy(weight = 0.9), p.loras(1))),
      "lora unet" -> p.copy(loras = Seq(l.copy(unet = None), p.loras(1))), "lora lbw" -> p.copy(loras = Seq(l.copy(lbw = None), p.loras(1))),
      "lora start" -> p.copy(loras = Seq(l.copy(start = None), p.loras(1))), "lora stop" -> p.copy(loras = Seq(l.copy(stop = None), p.loras(1))),
      "lora order" -> p.copy(loras = p.loras.reverse), "control enabled" -> p.copy(controlnet = Seq(c.copy(enabled = false), p.controlnet(1))),
      "control model" -> p.copy(controlnet = Seq(c.copy(model = "other"), p.controlnet(1))), "control image" -> p.copy(controlnet = Seq(c.copy(image = None), p.controlnet(1))),
      "control preprocessor" -> p.copy(controlnet = Seq(c.copy(preprocessor = "other"), p.controlnet(1))), "control weight" -> p.copy(controlnet = Seq(c.copy(weight = 0.7), p.controlnet(1))),
      "control start" -> p.copy(controlnet = Seq(c.copy(guidanceStart = 0.3), p.controlnet(1))), "control end" -> p.copy(controlnet = Seq(c.copy(guidanceEnd = 0.9), p.controlnet(1))),
      "control args" -> p.copy(controlnet = Seq(c.copy(preprocessorArgs = Map("a" -> 3.0)), p.controlnet(1))), "control order" -> p.copy(controlnet = p.controlnet.reverse),
      "hooks nested" -> p.copy(hooks = Map("freeu" -> ujson.Obj("nested" -> ujson.Arr(2)))), "styles order" -> p.copy(styles = p.styles.reverse),
      "checkpoint" -> p.copy(checkpoint = None), "vae" -> p.copy(vae = None), "extra" -> p.copy(extra = Map("a" -> "changed"))
    ).map((name, params) => name -> j.copy(params = params))
    val original = FrozenJobEvidence.freeze(j)
    variants.foreach { (name, variant) =>
      val changed = FrozenJobEvidence.freeze(variant)
      assert(!original.sameJobAs(changed), clues(name))
      assertNotEquals(original.canonicalHash, changed.canonicalHash, clues(name))
    }
  }

  test("same id and class do not conceal different plugin instances") {
    val j = fixture; val initial = FrozenJobEvidence.freeze(j)
    val changedScript = FrozenJobEvidence.freeze(j.copy(script = Some(new Script -> j.script.get._2)))
    assertEquals(initial.canonicalHash, changedScript.canonicalHash)
    assert(!initial.sameJobAs(changedScript))
    val changedAlways = FrozenJobEvidence.freeze(j.copy(alwaysOn = Seq(new Marker -> j.alwaysOn.head._2, j.alwaysOn(1))))
    assertEquals(initial.canonicalHash, changedAlways.canonicalHash)
    assert(!initial.sameJobAs(changedAlways))
    assert(initial.sameJobAs(FrozenJobEvidence.freeze(j.copy())))
  }

  test("frozen snapshot and hash survive mutation of nested JSON graph") {
    val j = fixture; val frozen = FrozenJobEvidence.freeze(j)
    val beforeJson = frozen.snapshotJson; val beforeHash = frozen.canonicalHash
    j.params.hooks("freeu")("nested")(0)("x") = ujson.Num(99)
    assertEquals(frozen.snapshotJson, beforeJson)
    assertEquals(frozen.canonicalHash, beforeHash)
    assert(!frozen.sameJobAs(FrozenJobEvidence.freeze(j)))
  }

  test("canonical hash ignores only object map order, not array order") {
    val j = fixture; val p = j.params
    val reordered = j.copy(params = p.copy(extra = ListMap("a" -> "first", "z" -> "last"),
      hooks = ListMap("other" -> ujson.Str("raw"), "freeu" -> ujson.Obj("nested" -> ujson.Arr(ujson.Obj("x" -> 1), ujson.Null), "enabled" -> false)),
      controlnet = Seq(p.controlnet.head.copy(preprocessorArgs = ListMap("a" -> 1.0, "z" -> 2.0)), p.controlnet(1))),
      script = Some(script -> Values(ListMap("evidence" -> "retained", "z" -> "last"))))
    val a = FrozenJobEvidence.freeze(j); val b = FrozenJobEvidence.freeze(reordered)
    assertEquals(a.canonicalHash, b.canonicalHash)
    assert(a.sameJobAs(b))
    assertNotEquals(a.canonicalHash, FrozenJobEvidence.freeze(j.copy(params = p.copy(styles = p.styles.reverse))).canonicalHash)
  }

  test("context boolean distinguishes None, Some(false), and Some(true)") {
    val j = fixture
    val snapshots = Seq(None, Some(false), Some(true)).map { value =>
      FrozenJobEvidence.freeze(j.copy(submittedContext = j.submittedContext.map(_.copy(uiVariationEnabled = value))))
    }
    assertEquals(snapshots.map(_.canonicalHash).distinct.size, 3)
    assert(!snapshots(0).sameJobAs(snapshots(1)))
    assert(!snapshots(1).sameJobAs(snapshots(2)))
  }

  test("mutating JSON parsed by a caller cannot change frozen evidence") {
    val frozen = FrozenJobEvidence.freeze(fixture)
    val json = frozen.snapshotJson; val hash = frozen.canonicalHash
    val callerCopy = ujson.read(json)
    callerCopy("params")("hooks")("freeu")("enabled") = ujson.Bool(true)
    assertEquals(frozen.snapshotJson, json)
    assertEquals(frozen.canonicalHash, hash)
    assertEquals(ujson.read(frozen.snapshotJson)("params")("hooks")("freeu")("enabled").bool, false)
  }

  test("full-width Long values retain their exact decimal digits") {
    val j = fixture.copy(params = fixture.params.copy(seed = 9007199254740993L))
    val a = FrozenJobEvidence.freeze(j)
    val b = FrozenJobEvidence.freeze(j.copy(params = j.params.copy(seed = 9007199254740992L)))
    assert(a.snapshotJson.contains("9007199254740993"))
    assertNotEquals(a.canonicalHash, b.canonicalHash)
    assert(!a.sameJobAs(b))
  }

  test("absent and present empty option strings remain distinct") {
    val j = fixture
    val none = FrozenJobEvidence.freeze(j.copy(submittedContext = j.submittedContext.map(_.copy(configurationJson = None))))
    val empty = FrozenJobEvidence.freeze(j.copy(submittedContext = j.submittedContext.map(_.copy(configurationJson = Some("")))))
    assertNotEquals(none.canonicalHash, empty.canonicalHash)
    assert(!none.sameJobAs(empty))
  }

  test("snapshot object keys cover each current nested product schema") {
    val json = ujson.read(FrozenJobEvidence.freeze(fixture).snapshotJson)
    val p = json("params")
    assertEquals(p.obj.keySet.toSet, fields(fixture.params))
    assertEquals(p("hires").obj.keySet.toSet, fields(fixture.params.hires))
    assertEquals(p("img2img").obj.keySet.toSet, fields(fixture.params.img2img.get))
    assertEquals(p("loras")(0).obj.keySet.toSet, fields(fixture.params.loras.head))
    assertEquals(p("controlnet")(0).obj.keySet.toSet, fields(fixture.params.controlnet.head))
    assertEquals(json("limits").obj.keySet.toSet, fields(fixture.limits))
    assertEquals(json("submittedContext").obj.keySet.toSet, fields(fixture.submittedContext.get))
    assertEquals(json("captureOptions").obj.keySet.toSet, fields(fixture.captureOptions.get))
    assertEquals(json("script").obj.keySet.toSet, Set("pluginId", "pluginClass", "values"))
    assertEquals(json("alwaysOn")(0).obj.keySet.toSet, Set("pluginId", "pluginClass", "values"))
  }

  test("typed signed zero remains distinct in snapshot, equality, and hash") {
    val j = fixture
    val negative = FrozenJobEvidence.freeze(j.copy(params = j.params.copy(eta = Some(-0.0))))
    val positive = FrozenJobEvidence.freeze(j.copy(params = j.params.copy(eta = Some(0.0))))
    assert(negative.snapshotJson.contains("\"eta\":-0.0"))
    assertNotEquals(negative.canonicalHash, positive.canonicalHash)
    assert(!negative.sameJobAs(positive))
  }

  test("recursive hook signed zero remains distinct in snapshot, equality, and hash") {
    val j = fixture
    def withZero(n: Double) = j.copy(params = j.params.copy(hooks = Map(
      "freeu" -> ujson.Obj("nested" -> ujson.Arr(ujson.Obj("value" -> ujson.Num(n)))))))
    val negative = FrozenJobEvidence.freeze(withZero(-0.0))
    val positive = FrozenJobEvidence.freeze(withZero(0.0))
    assert(negative.snapshotJson.contains("\"value\":-0.0"))
    assertNotEquals(negative.canonicalHash, positive.canonicalHash)
    assert(!negative.sameJobAs(positive))
  }

  test("nonfinite typed numbers reject the snapshot with the precise field path") {
    val j = fixture; val p = j.params
    val cases: Seq[(String, Double => GenerationParams)] = Seq(
      "params.eta" -> ((v: Double) => p.copy(eta = Some(v))),
      "params.hires.scale" -> ((v: Double) => p.copy(hires = p.hires.copy(scale = v))),
      "params.img2img.denoisingStrength" -> ((v: Double) => p.copy(img2img = p.img2img.map(_.copy(denoisingStrength = v)))),
      "params.loras[0].weight" -> ((v: Double) => p.copy(loras = Seq(p.loras.head.copy(weight = v)))),
      "params.controlnet[0].preprocessorArgs.a" -> ((v: Double) => p.copy(controlnet = Seq(p.controlnet.head.copy(preprocessorArgs = Map("a" -> v)))))
    )
    cases.foreach { (path, change) =>
      Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity).foreach { value =>
        val error = intercept[IllegalArgumentException] { FrozenJobEvidence.freeze(j.copy(params = change(value))) }
        assert(error.getMessage.contains(path), clues(path, value, error.getMessage))
      }
    }
  }

  test("nonfinite nested hook numbers reject the snapshot with the array and field path") {
    val j = fixture
    Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity).foreach { value =>
      val hook = ujson.Obj("nested" -> ujson.Arr(ujson.Str("ok"), ujson.Obj("value" -> ujson.Num(value))))
      val error = intercept[IllegalArgumentException] {
        FrozenJobEvidence.freeze(j.copy(params = j.params.copy(hooks = Map("freeu" -> hook))))
      }
      assert(error.getMessage.contains("params.hooks.freeu.nested[1].value"), clues(value, error.getMessage))
    }
  }

  test("ordinary strings spelling nonfinite numbers remain strings and do not fail") {
    val j = fixture
    val hook = ujson.Obj("nested" -> ujson.Arr("NaN", "Infinity", "-Infinity"))
    val frozen = FrozenJobEvidence.freeze(j.copy(params = j.params.copy(hooks = Map("freeu" -> hook))))
    val values = ujson.read(frozen.snapshotJson)("params")("hooks")("freeu")("nested").arr
    assertEquals(values.map(_.str).toSeq, Seq("NaN", "Infinity", "-Infinity"))
  }
