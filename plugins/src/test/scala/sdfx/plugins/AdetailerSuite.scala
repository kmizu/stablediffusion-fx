package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*

class AdetailerSuite extends munit.FunSuite:
  private val p = new AdetailerPlugin

  test("sections carry no values; defaults give one face unit"):
    val d = Values.defaults(p.controls)
    assert(!d.map.contains("u1_main") && !d.map.contains("u2_advanced"))
    val units = p.hookFields(d)("units").arr
    assertEquals(units.size, 1)
    assertEquals(units(0)("model").str, "face_yolov8n.pt")
    assertEquals(units(0)("color_correction").bool, true)

  test("second unit, custom model, and inactive without units"):
    val v = Values.defaults(p.controls).updated("enabled", "true").updated("u2_model", "hand_yolov8n.pt")
      .updated("u1_custom_model", "C:/m/face.pt")
    val units = p.hookFields(v)("units").arr
    assertEquals(units.map(_("model").str).toSeq, Seq("C:/m/face.pt", "hand_yolov8n.pt"))
    assert(p.isActive(v))
    assert(!p.isActive(v.updated("u1_model", "None").updated("u1_custom_model", "").updated("u2_model", "None")))

  test("infotext uses the extension's keys, with ' 2nd' for the second unit"):
    val v = Values.defaults(p.controls).updated("enabled", "true").updated("u2_model", "hand_yolov8n.pt")
      .updated("u1_prompt", "smile")
    val extra = p.process(GenerationParams(), v).extra
    assertEquals(extra.get("ADetailer model"), Some("face_yolov8n.pt"))
    assertEquals(extra.get("ADetailer model 2nd"), Some("hand_yolov8n.pt"))
    assertEquals(extra.get("ADetailer prompt"), Some("smile"))
    assertEquals(extra.get("ADetailer denoising strength"), Some("0.4"))

  test("enabledBy names existing controls"):
    val keys = p.controls.map(_.key).toSet
    p.enabledBy.foreach((k, gate) => assert(keys(k) && keys(gate), s"$k -> $gate"))
