package sdfx.core

import sdfx.core.modelpresets.*
import java.nio.file.Files

class ModelPresetsSuite extends munit.FunSuite with TempDir:
  test("upstream document and sharing preserve unknown multiline infotext") {
    val raw = "猫, a portrait\nNegative prompt: fog\nSteps: 20, Future field: xyz"
    val doc = ModelPresetDocument.empty.save("日本語", raw).setDefault("日本語")
      .copy(url = "https://civitai.com/models/1", triggerWords = Vector("a.b", "猫"))
    assertEquals(ModelPresetDocument.parse(doc.json), doc)
    assertEquals(ModelPresetDocument.parseShared("description\n" + doc.share + "\nfooter"), doc)
    assertEquals(doc.rename("日本語", "new").defaultPreset, "new")
    assertEquals(doc.delete("日本語").defaultPreset, "default")
    intercept[IllegalArgumentException](doc.rename("日本語", "default"))
  }
  test("bounded malformed input and literal trigger removal") {
    intercept[IllegalArgumentException](ModelPresetDocument.parse("{}"))
    intercept[IllegalArgumentException](ModelPresetDocument.parse("x" * (ModelPresetDocument.MaxBytes + 1)))
    assertEquals(ModelPresetDocument.toggleTrigger("a.b cat axb", "a.b", false), "cat axb")
    assertEquals(ModelPresetDocument.toggleTrigger("cat", "a.b", true), "a.b cat")
    assertEquals(ModelPresetDocument.empty.delete("default"), ModelPresetDocument.empty)
  }
  test("trigger selection and removal respect complete Unicode words and comma boundaries") {
    assertEquals(ModelPresetDocument.toggleTrigger("cart, art", "art", false), "cart")
    assertEquals(ModelPresetDocument.toggleTrigger("cart", "art", true), "art cart")
    assert(!ModelPresetDocument.containsTrigger("cart", "art"))
    assertEquals(ModelPresetDocument.toggleTrigger("portrait, art", "art", false), "portrait")
    assertEquals(ModelPresetDocument.toggleTrigger("portrait", "art", true), "art portrait")
    assertEquals(ModelPresetDocument.toggleTrigger("art, portrait", "art", false), "portrait")
    assertEquals(ModelPresetDocument.toggleTrigger("portrait, art, landscape", "art", false), "portrait, landscape")
    assertEquals(ModelPresetDocument.toggleTrigger("fine art portrait", "art", false), "fine portrait")
    assertEquals(ModelPresetDocument.toggleTrigger("猫娘, 猫", "猫", false), "猫娘")
    assertEquals(ModelPresetDocument.toggleTrigger("xa.by, a.b", "a.b", false), "xa.by")
  }
  tmp.test("revision bound atomic store isolates full model identities and rejects stale edits") { root =>
    val store = new ModelPresetStore(root)
    val a = "a" * 64
    val b = "a" * 10 + "b" * 54
    val first = store.load(a)
    val written = store.save(a, ModelPresetDocument.empty.save("one", "portrait"), first.revision)
    assertEquals(store.load(b).document, ModelPresetDocument.empty)
    intercept[IllegalStateException](store.save(a, ModelPresetDocument.empty, first.revision))
    assertEquals(store.load(a), written)
    intercept[IllegalArgumentException](store.load("../escape"))
    val exported = root.resolve("shared.json")
    store.exportDocument(written.document, exported)
    intercept[java.nio.file.FileAlreadyExistsException](store.exportDocument(written.document, exported))
    assertEquals(ModelPresetDocument.parse(Files.readString(exported)), written.document)
  }

  test("duplicate keys, invalid UTF-8 and deep JSON are rejected before interpretation") {
    intercept[IllegalArgumentException](ModelPresetDocument.parse(ModelPresetDocument.empty.json.replace("\"url\": \"\"", "\"url\": \"\", \"url\": \"changed\"")))
    intercept[IllegalArgumentException](ModelPresetDocument.parse("[" * 128 + "0" + "]" * 128))
    intercept[IllegalArgumentException](ModelPresetDocument.parseBytes(Array[Byte](0xc3.toByte, 0x28)))
  }

  tmp.test("import rejects malformed UTF-8 rather than changing the shared prompt") { root =>
    val path = root.resolve("invalid.json")
    val prefix = ModelPresetDocument.empty.json.replace("\"url\": \"\"", "\"url\": \"BAD\"").getBytes("UTF-8")
    val start = new String(prefix, "UTF-8").indexOf("BAD")
    prefix(start) = 0xc3.toByte; prefix(start + 1) = 0x28.toByte
    Files.write(path, prefix)
    intercept[Exception](new ModelPresetStore(root).importDocument(path))
  }
