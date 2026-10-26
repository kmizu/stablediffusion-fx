package sdfx.core

import sdfx.core.config.Settings
import sdfx.core.infotext.{HardwareInfo, Infotext}
import sdfx.core.json.SnakePickle
import sdfx.core.protocol.GeneratedImage

class HardwareInfoSuite extends munit.FunSuite:
  test("hardware metadata opt-in and Unicode remark persist independently of System Info selection") {
    val defaults = Settings()
    assert(!defaults.hardwareInfoEnabled)
    assertEquals(defaults.hardwareInfoRemark, "")
    val configured = defaults.copy(hardwareInfoEnabled = true, hardwareInfoRemark = "机, quiet\n日本語",
      systemInfoMetadataFields = Seq("platform.python"))
    assertEquals(SnakePickle.read[Settings](SnakePickle.write(configured)), configured)
    assertEquals(configured.systemInfoMetadataFields, Seq("platform.python"))
  }

  test("per-image metadata remains associated through typed protocol roundtrip and safely quoted infotext") {
    val record = HardwareInfo.Record("RTX 3060 12GB, i5-10400F, 64GB RAM", "1 min. 2.5 sec.", Some("quiet, 日本語\nsecond line"))
    val image = GeneratedImage("image.png", seed = 19L, hardwareInfo = Some(record))
    assertEquals(SnakePickle.read[GeneratedImage](SnakePickle.write(image)), image)
    val fields = HardwareInfo.fields(Some(record))
    assertEquals(fields.keys.toSeq, Seq("Hardware Info", "Remark", "Time taken"))
    val text = Infotext.render("cat", "bad", fields)
    val parsed = Infotext.parse(text)
    assertEquals(parsed.get("Hardware Info"), Some(record.hardwareInfo))
    assertEquals(parsed.get("Remark"), record.remark)
    assertEquals(parsed.get("Time taken"), Some(record.timeTaken))
    assertEquals(HardwareInfo.fields(None).size, 0)
    assert(!HardwareInfo.fields(Some(record.copy(remark = Some("")))).contains("Remark"))
  }

  test("paste differences are informative only and capped at two mismatches per session") {
    val notices = new HardwareInfo.PasteNotices
    val local = "RTX 3060 12GB, i5-10400F, 64GB RAM"
    assertEquals(notices.pasted(local, None), None)
    assertEquals(notices.pasted(local, Some("RTX 3060 12GB, different CPU, 8GB RAM")), None)
    assertEquals(notices.pasted(local, Some("unknown")), None)
    val first = notices.pasted(local, Some("RTX 4090 24GB, other CPU, 64GB RAM")).get
    assertEquals(first.currentDevice, "RTX 3060 12GB")
    assertEquals(first.imageDevice, "RTX 4090 24GB")
    assert(notices.pasted(local, Some("RTX 4060 8GB, CPU, RAM")).nonEmpty)
    assertEquals(notices.pasted(local, Some("RTX 4080 16GB, CPU, RAM")), None)
    val fresh = new HardwareInfo.PasteNotices
    assertEquals(fresh.pasted("unknown", Some(local)), None)
    assert(fresh.pasted(local, Some("RTX 4080 16GB, CPU, RAM")).nonEmpty)
  }

  test("CPU-only first segments keep the same bounded provenance comparison without asserting a physical GPU") {
    val notices = new HardwareInfo.PasteNotices
    val difference = notices.pasted("i5-10400F, 64GB RAM", Some("i7-13700, 32GB RAM")).get
    assertEquals(difference.currentDevice, "i5-10400F")
    assertEquals(difference.imageDevice, "i7-13700")
  }
