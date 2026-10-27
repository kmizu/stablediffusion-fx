package sdfx.core

import munit.FunSuite
import sdfx.core.protocol.*

class TemporalLabProtocolSuite extends FunSuite:
  test("typed requests preserve each operation discriminator and fields") {
    val requests: Seq[TemporalLabRequest] = Seq(
      TemporalLabDfi("r", "g", "o", "n", tolerance = 8, frameLimit = 3),
      TemporalLabPreview("r", "g", frameIndex = 2, expand = 4),
      TemporalLabDeflicker("f", "o", "n", blend = 0.5, normalize = true),
      TemporalLabFuse("s", "v", "o", "n", strength = 0.75),
      TemporalLabExtract("v.avi", "o", "n", fps = 12)
    )
    val json = requests.map(_.params)
    assertEquals(json.map(_("operation").str), Seq("dfi", "preview", "deflicker", "fuse", "extract"))
    assertEquals(json(0)("reference_dir").str, "r")
    assertEquals(json(0)("tolerance").num, 8.0)
    assertEquals(json(0)("frame_limit").num, 3.0)
    assertEquals(json(1)("frame_index").num, 2.0)
    assertEquals(json(1)("expand").num, 4.0)
    assertEquals(json(2)("normalize").bool, true)
    assertEquals(json(3)("style_dir").str, "s")
    assertEquals(json(3)("strength").num, 0.75)
    assertEquals(json(4)("video_path").str, "v.avi")
    assertEquals(json(4)("fps").num, 12.0)
    assertEquals(Protocol.Types.TemporalLab, "temporal_lab")
  }

  test("typed results decode output and preview without conflating dry-run") {
    val result = Json.fromJs[TemporalLabResult](ujson.Obj(
      "operation" -> "dfi", "output_dir" -> "o/n", "frame_count" -> 2,
      "dimensions" -> ujson.Arr(512, 512), "reference_aligned" -> true))
    assertEquals(result.outputDir, Some("o/n"))
    assertEquals(result.frameCount, Some(2))
    assertEquals(result.dimensions, Seq(512, 512))
    assertEquals(result.referenceAligned, Some(true))
    val preview = Json.fromJs[TemporalLabResult](ujson.Obj(
      "operation" -> "preview", "mask_png_base64" -> "AA==", "frame_index" -> 1,
      "generated_previous" -> "a", "generated_current" -> "b", "dimensions" -> ujson.Arr(2, 2)))
    assertEquals(preview.maskPngBase64, Some("AA=="))
    assertEquals(preview.generatedPrevious, Some("a"))
    val fake = Json.fromJs[TemporalLabResult](ujson.Obj("operation" -> "dfi", "dry_run" -> true))
    assert(fake.dryRun)
    assertEquals(fake.outputDir, None)
  }
