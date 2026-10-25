package sdfx.core

import sdfx.core.infotext.Infotext
import sdfx.core.png.PngMetadata

import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO

class PngMetadataSuite extends munit.FunSuite with TempDir:
  import TestFiles.*

  val latinParams = "masterpiece, 1girl\nNegative prompt: lowres\nSteps: 28, Sampler: Euler a, Seed: 1, Lora hashes: \"a: 1, b: 2\""
  val jaParams = "桜の木の下で微笑む少女\nNegative prompt: 低品質\nSteps: 20, Sampler: Euler a, Seed: 2, Model: 日本語モデル"

  tmp.test("Latin-1 text is written as tEXt and read back") { dir =>
    val f = dir.resolve("a.png")
    PngMetadata.write(image(16, 8), f, Seq("parameters" -> latinParams))
    val chunks = chunkTypes(f)
    assert(chunks.contains("tEXt"), chunks)
    assert(!chunks.contains("iTXt"), chunks)
    assertEquals(chunks.head, "IHDR")
    assertEquals(chunks.last, "IEND")
    assertEquals(PngMetadata.readParameters(f), Some(latinParams))
  }

  tmp.test("Latin-1 accented text stays tEXt (PIL rule: latin-1 encodable)") { dir =>
    val f = dir.resolve("a.png")
    val s = "café, naïve, ©, ±"
    PngMetadata.write(image(4, 4), f, Seq("parameters" -> s))
    assert(chunkTypes(f).contains("tEXt"))
    assertEquals(PngMetadata.readParameters(f), Some(s))
  }

  tmp.test("Japanese text is written as iTXt and read back") { dir =>
    val f = dir.resolve("j.png")
    PngMetadata.write(image(16, 8), f, Seq("parameters" -> jaParams))
    val chunks = chunkTypes(f)
    assert(chunks.contains("iTXt"), chunks)
    assert(!chunks.contains("tEXt"), chunks)
    assertEquals(PngMetadata.readParameters(f), Some(jaParams))
  }

  tmp.test("mixed entries: each goes to the right chunk, all readable") { dir =>
    val f = dir.resolve("m.png")
    PngMetadata.write(image(8, 8), f, Seq("parameters" -> jaParams, "sdfx" -> "{\"v\":1}", "comment" -> "ok"))
    val chunks = chunkTypes(f)
    assert(chunks.contains("iTXt") && chunks.contains("tEXt"), chunks)
    assertEquals(PngMetadata.readTexts(f), Map("parameters" -> jaParams, "sdfx" -> "{\"v\":1}", "comment" -> "ok"))
  }

  tmp.test("pixels survive, including alpha") { dir =>
    val f = dir.resolve("p.png")
    val img = new BufferedImage(3, 2, BufferedImage.TYPE_INT_ARGB)
    img.setRGB(0, 0, 0x80ff0000)
    img.setRGB(2, 1, 0xff00ff00)
    PngMetadata.write(img, f, Seq("parameters" -> "x"))
    val back = PngMetadata.readImage(f)
    assertEquals((back.getWidth, back.getHeight), (3, 2))
    assertEquals(back.getRGB(0, 0), 0x80ff0000)
    assertEquals(back.getRGB(2, 1), 0xff00ff00)
  }

  tmp.test("no texts -> plain PNG, readParameters None") { dir =>
    val f = dir.resolve("n.png")
    PngMetadata.write(image(4, 4), f, Nil)
    assertEquals(PngMetadata.readParameters(f), None)
    assertEquals(PngMetadata.readTexts(f), Map.empty[String, String])
  }

  tmp.test("empty parameters value reads as None") { dir =>
    val f = dir.resolve("e.png")
    PngMetadata.write(image(4, 4), f, Seq("parameters" -> ""))
    assertEquals(PngMetadata.readParameters(f), None)
  }

  tmp.test("write creates parent dirs, overwrites, and leaves no temp files") { dir =>
    val f = dir.resolve("sub/deeper/o.png")
    PngMetadata.write(image(4, 4), f, Seq("parameters" -> "first"))
    PngMetadata.write(image(4, 4), f, Seq("parameters" -> "second"))
    assertEquals(PngMetadata.readParameters(f), Some("second"))
    val names = Files.list(f.getParent).toArray.map(_.toString).toSeq
    assertEquals(names.size, 1, names)
  }

  tmp.test("copyWithParameters from a JPEG source") { dir =>
    val jpg = dir.resolve("src.jpg")
    assert(ImageIO.write(image(20, 10, 0xff8800), "jpg", jpg.toFile))
    assertEquals(PngMetadata.readParameters(jpg), None)
    val dst = dir.resolve("out.png")
    PngMetadata.copyWithParameters(jpg, dst, jaParams, Seq("extra" -> "1"))
    assertEquals(PngMetadata.readParameters(dst), Some(jaParams))
    assertEquals(PngMetadata.readTexts(dst).get("extra"), Some("1"))
    val img = PngMetadata.readImage(dst)
    assertEquals((img.getWidth, img.getHeight), (20, 10))
    assertEquals(chunkTypes(dst).head, "IHDR")
  }

  tmp.test("copyWithParameters replaces the parameters of a PNG source") { dir =>
    val a = dir.resolve("a.png")
    PngMetadata.write(image(4, 4), a, Seq("parameters" -> "old"))
    val b = dir.resolve("b.png")
    PngMetadata.copyWithParameters(a, b, "new")
    assertEquals(PngMetadata.readParameters(b), Some("new"))
    assertEquals(PngMetadata.readParameters(a), Some("old"))
  }

  tmp.test("copyWithParameters rejects non-images") { dir =>
    val t = dir.resolve("x.txt")
    Files.writeString(t, "hello")
    intercept[IllegalArgumentException](PngMetadata.copyWithParameters(t, dir.resolve("o.png"), "p"))
    intercept[IllegalArgumentException](PngMetadata.readImage(t))
  }

  tmp.test("non-PNG / non-image / missing / truncated files -> None") { dir =>
    val jpg = dir.resolve("a.jpg")
    ImageIO.write(image(4, 4), "jpg", jpg.toFile)
    val bmp = dir.resolve("a.bmp")
    ImageIO.write(image(4, 4), "bmp", bmp.toFile)
    val txt = dir.resolve("a.txt")
    Files.writeString(txt, "Steps: 20, Sampler: Euler, Seed: 1")
    val empty = Files.createFile(dir.resolve("empty.png"))
    val good = dir.resolve("good.png")
    PngMetadata.write(image(4, 4), good, Seq("parameters" -> "p"))
    val trunc = dir.resolve("trunc.png")
    Files.write(trunc, Files.readAllBytes(good).take(20))
    Seq(jpg, bmp, txt, empty, dir.resolve("missing.png"), trunc).foreach { f =>
      assertEquals(PngMetadata.readParameters(f), None, f.toString)
    }
    assertEquals(PngMetadata.readTexts(txt), Map.empty[String, String])
  }

  test("PNG written by Python PIL (A1111 save path), tEXt") {
    val f = resource("pil_latin1.png")
    assert(chunkTypes(f).contains("tEXt"))
    val p = PngMetadata.readParameters(f)
    assert(p.isDefined)
    val pi = Infotext.parse(p.get)
    assertEquals(pi.prompt, "masterpiece, 1girl")
    assertEquals(pi.negativePrompt, "lowres, bad hands")
    assertEquals(pi.get("Lora hashes"), Some("foo: abcdef123456, bar: 0123456789ab"))
    assertEquals(pi.get("Version"), Some("f1.0.0v2-v1.10.1RC"))
    val img = PngMetadata.readImage(f)
    assertEquals((img.getWidth, img.getHeight), (64, 48))
  }

  test("PNG written by Python PIL (A1111 save path), iTXt with Japanese") {
    val f = resource("pil_japanese.png")
    assert(chunkTypes(f).contains("iTXt"))
    val pi = Infotext.parse(PngMetadata.readParameters(f).get)
    assertEquals(pi.prompt, "桜の木の下で微笑む少女, 着物, 夕焼け")
    assertEquals(pi.negativePrompt, "低品質, ぼやけ")
    assertEquals(pi.get("Model"), Some("日本語モデル"))
    assertEquals(Infotext.toParams(pi).seed, 987654321L)
  }

  tmp.test("our output is re-readable after copy (PIL fixture -> sdfx PNG)") { dir =>
    val src = resource("pil_japanese.png")
    val dst = dir.resolve("copy.png")
    val params = PngMetadata.readParameters(src).get
    PngMetadata.copyWithParameters(src, dst, params)
    assertEquals(PngMetadata.readParameters(dst), Some(params))
  }
