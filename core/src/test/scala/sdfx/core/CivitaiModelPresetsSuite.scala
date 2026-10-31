package sdfx.core

import sdfx.core.modelpresets.*
import java.net.URI
import java.util.concurrent.CancellationException
import scala.collection.mutable.ArrayBuffer

class CivitaiModelPresetsSuite extends munit.FunSuite:
  test("hash lookup downloads words and first image, preserving local presets without shared data") {
    val calls = ArrayBuffer.empty[String]
    val client = new CivitaiPresetClient((uri, limit, check) => {
      check(); calls += uri.toString
      if uri.getPath.contains("by-hash") then """{"modelId":12,"trainedWords":["cat"],"images":[{"url":"https://image.civitai.com/first.png"}]}""".getBytes("UTF-8")
      else if uri.getPath.endsWith("/12") then """{"description":"no shared presets"}""".getBytes("UTF-8")
      else Array[Byte](1, 2, 3)
    })
    val local = ModelPresetDocument.empty.save("mine", "portrait")
    val result = client.fetch("a" * 64, local, () => ())
    assertEquals(result.document.presets, local.presets)
    assertEquals(result.document.triggerWords, Vector("cat"))
    assertEquals(result.document.url, "https://civitai.com/models/12")
    assertEquals(result.thumbnail.toVector.flatMap(_.toVector), Vector[Byte](1, 2, 3))
    assert(calls.head.endsWith("a" * 64))
  }
  test("manual model/version URL wins and quoted braces survive HTML shared import") {
    val shared = ModelPresetDocument.empty.save("x", "a {brace} and \"quote\"")
    val description = "<p>" + shared.share.replace("&", "&amp;").replace("\"", "&quot;").replace("\n", "<br>") + "</p>"
    val calls = ArrayBuffer.empty[String]
    val client = new CivitaiPresetClient((uri, limit, check) => {
      calls += uri.toString
      if uri.getPath.endsWith("/99") then ujson.Obj("modelId" -> 12, "trainedWords" -> ujson.Arr(), "images" -> ujson.Arr()).render().getBytes("UTF-8")
      else ujson.Obj("description" -> description).render().getBytes("UTF-8")
    })
    val result = client.fetch("a" * 64, ModelPresetDocument.empty.copy(url = "https://civitai.com/models/12?modelVersionId=99"), () => ())
    assertEquals(result.document, shared)
    assertEquals(calls.head, "https://civitai.com/api/v1/model-versions/99")
  }
  test("cancellation and non-Civitai fetch fail before transport") {
    var calls = 0
    val client = new CivitaiPresetClient((uri, limit, check) => { calls += 1; Array.emptyByteArray })
    intercept[CancellationException](client.fetch("a" * 64, ModelPresetDocument.empty, () => throw new CancellationException()))
    intercept[IllegalArgumentException](client.fetch("a" * 64, ModelPresetDocument.empty.copy(url = "file:///secret"), () => ()))
    assertEquals(calls, 0)
  }

  test("HTML sharing preserves repeated spaces inside the stored prompt") {
    val doc = ModelPresetDocument.empty.save("spaced", "a  b   c")
    val html = "<p>" + doc.share.replace("\"", "&quot;") + "</p>"
    assertEquals(ModelPresetDocument.parseShared(CivitaiPresetClient.plainHtml(html)), doc)
  }
