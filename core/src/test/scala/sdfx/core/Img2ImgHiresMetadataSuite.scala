package sdfx.core

import sdfx.core.infotext.Img2ImgHiresMetadata

class Img2ImgHiresMetadataSuite extends munit.FunSuite:
  test("PNG parameters restore refinement and clear a previous image's active refinement") {
    import sdfx.core.infotext.Infotext
    import sdfx.core.model.GenerationParams
    val text = "cat\nSteps: 10, Seed: 5, Size: 64x64, img2img Hires Fix: \"{'scale':2,'upscaler':'Lanczos','denoise':0.4}\""
    val restored = Infotext.toParams(Infotext.parse(text))
    assertEquals(restored.hooks("img2img_hires_fix")("denoise").num, .4)
    val plain = Infotext.toParams(Infotext.parse("dog\nSteps: 10, Seed: 5, Size: 64x64"), restored)
    assert(!plain.hooks.contains("img2img_hires_fix"))
  }
  test("source numeric and explicit size metadata restore all independent settings") {
    val h = Img2ImgHiresMetadata.parse("{'scale':'1024x768','upscaler':'Lanczos','denoise':0.4,'steps':12,'cfg':3,'prompt':'dog','negative_prompt':'blur','sampler':'Euler','scheduler':'Karras'}").get
    assertEquals(h("width").num, 1024.0)
    assertEquals(h("height").num, 768.0)
    assertEquals(h("prompt").str, "dog")
    assertEquals(h("scheduler").str, "Karras")
    val roundtrip = Img2ImgHiresMetadata.parse(Img2ImgHiresMetadata.encode(h)).get
    assertEquals(roundtrip, h)
    assertEquals(Img2ImgHiresMetadata.parse("{\"scale\":1.5,\"upscaler\":\"Lanczos\",\"denoise\":0.33}").get("ratio").num, 1.5)
  }
  test("malformed source settings never partially enable refinement") {
    Seq("[]", "{}", "{'scale':'oops'}", "{'scale':0}", "{'scale':2,'denoise':4}",
      "{'scale':2,'steps':-1}", "x" * 65537).foreach(s => assertEquals(Img2ImgHiresMetadata.parse(s), None))
  }
