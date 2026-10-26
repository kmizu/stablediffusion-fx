package sdfx.engine

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import java.util.Base64
import java.util.zip.CRC32
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*

/** These tests exercise the actual collector, file/PNG boundaries and unchanged frozen body.
  * SourceAccess replaces only the OS boundary to make file-growth timing deterministic.
  */
class PayloadAssetsSuite extends munit.FunSuite:
  private def withDir(f: Path => Unit): Unit =
    val dir = Files.createTempDirectory("sdfx-payload-assets-")
    try f(dir)
    finally
      val walk = Files.walk(dir)
      try walk.iterator().asScala.toVector.sortBy(_.getNameCount).reverse.foreach(Files.deleteIfExists(_))
      finally walk.close()

  private def image(path: Path, w: Int = 3, h: Int = 2): BufferedImage =
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
    for y <- 0 until h; x <- 0 until w do
      img.setRGB(x, y, if x == 0 then 0xffff2000 else if x == 1 then 0xff0055ff else 0x7020ee40)
    assert(ImageIO.write(img, "png", path.toFile))
    img

  private def body(params: ujson.Obj): String = ujson.write(ujson.Obj("params" -> params,
    "output_dir" -> "ignored.png", "lora_paths" -> ujson.Obj("fake" -> "weights.png")))
  private def assets(r: PayloadAssets.Result): Vector[ujson.Value] = ujson.read(r.assetsJson)("assets").arr.toVector
  private def collectMask(path: Path, include: Boolean = true): PayloadAssets.Result =
    PayloadAssets.collect(body(ujson.Obj("img2img" -> ujson.Obj("init_image" -> "never-open-init.png", "mask" -> path.toString))), include)
  private def mask(r: PayloadAssets.Result): ujson.Value = assets(r).find(_("role").str == "inpaint_mask").get
  private def sourceHash(bytes: Array[Byte]): String = MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 255}%02x").mkString
  private def pngBytes(v: ujson.Value): Array[Byte] =
    val data = v("png_data_url").str
    assert(data.startsWith("data:image/png;base64,"))
    Base64.getDecoder.decode(data.stripPrefix("data:image/png;base64,"))

  private def denyIO: PayloadAssets.SourceAccess = new PayloadAssets.SourceAccess:
    def isRegularFile(path: Path): Boolean = fail(s"unexpected file inspection: $path")
    def size(path: Path): Long = fail(s"unexpected file size: $path")
    def open(path: Path): InputStream = fail(s"unexpected file open: $path")

  test("default diagnostics enumerate only exact image fields in stable original array order without file IO") {
    val frozen = body(ujson.Obj(
      "img2img" -> ujson.Obj("init_image" -> "literal/../初期.png", "mask" -> "mask.png"),
      "controlnet" -> ujson.Arr(ujson.Obj("image" -> "c0.png", "enabled" -> false), ujson.Obj("image" -> "c1.png")),
      "hooks" -> ujson.Obj(
        "reference" -> ujson.Obj("image" -> "ref.png"), "revision" -> ujson.Obj("image" -> "rev.png"),
        "ip_adapter" -> ujson.Obj("image" -> "ip.png", "image_encoder" -> "encoder.png", "model" -> "weight.png"),
        "controllllite" -> ujson.Obj("image" -> "ll.png"),
        "photomaker" -> ujson.Obj("images" -> ujson.Arr("id.png", "id.png"), "model" -> "model.png"),
        "unknown" -> ujson.Obj("image" -> "never.png")),
      "extra" -> ujson.Obj("image" -> "ignored.png"), "checkpoint" -> "model.png"))
    val before = frozen
    val r = PayloadAssets.collectWith(frozen, false, PayloadAssets.Limits(), denyIO)
    val a = assets(r)
    assertEquals(a.map(_("json_path").str), Vector("$.params.img2img.init_image", "$.params.img2img.mask",
      "$.params.controlnet[0].image", "$.params.controlnet[1].image", "$.params.hooks.reference.image",
      "$.params.hooks.revision.image", "$.params.hooks.ip_adapter.image", "$.params.hooks.controllllite.image",
      "$.params.hooks.photomaker.images[0]", "$.params.hooks.photomaker.images[1]"))
    assertEquals(a.map(_("role").str), Vector("init_image", "inpaint_mask", "controlnet_image", "controlnet_image",
      "reference_image", "revision_image", "ip_adapter_image", "controllllite_image", "photomaker_id_image", "photomaker_id_image"))
    assertEquals(a.head("path").str, "literal/../初期.png")
    assertEquals(a(2)("enabled").bool, false)
    assert(a.forall(v => !v.obj.contains("png_data_url") && !v.obj.contains("source_sha256")))
    assertEquals(a.head("status").str, "omitted_source_init_images_policy")
    assertEquals(a(2)("status").str, "omitted_inactive_configuration")
    assertEquals(a(3)("status").str, "omitted_disabled_by_preference")
    assertEquals(frozen, before)
    assertEquals(r.warnings, Vector.empty[String])
  }

  test("init image and ControlNet or LLLite fallbacks remain omitted even when contents are requested") {
    val r = PayloadAssets.collectWith(body(ujson.Obj("img2img" -> ujson.Obj("init_image" -> "init.png"),
      "controlnet" -> ujson.Arr(ujson.Obj("image" -> ujson.Null), ujson.Obj("image" -> "")),
      "hooks" -> ujson.Obj("controllllite" -> ujson.Obj("image" -> "")))), true, PayloadAssets.Limits(), denyIO)
    val a = assets(r)
    assertEquals(a.map(_("status").str), Vector("omitted_source_init_images_policy", "omitted_init_image_fallback",
      "omitted_init_image_fallback", "omitted_pipeline_input_fallback"))
    assertEquals(a(1)("derived_from_json_path").str, "$.params.img2img.init_image")
    assertEquals(a.last("fallback").str, "pipeline_input_pixels_not_captured")
    assert(a.forall(v => !v.obj.contains("png_data_url") && !v.obj.contains("source_sha256")))
  }

  test("explicitly disabled configured images are visible and never inspected") {
    val r = PayloadAssets.collectWith(body(ujson.Obj("controlnet" -> ujson.Arr(ujson.Obj("image" -> "disabled.png", "enabled" -> false)),
      "hooks" -> ujson.Obj("photomaker" -> ujson.Obj("images" -> ujson.Arr("id.png"), "enabled" -> false),
        "reference" -> ujson.Obj("image" -> "ref.png", "enabled" -> false)))), true, PayloadAssets.Limits(), denyIO)
    assertEquals(assets(r).size, 3)
    assert(assets(r).forall(_("status").str == "omitted_inactive_configuration"))
  }

  test("opt-in PNG preserves decoded RGB and alpha with source-byte hash and exact literal path association") {
    withDir { dir =>
      val p = dir.resolve("色.png")
      val expected = image(p)
      val before = Files.readAllBytes(p)
      val r = collectMask(p)
      val m = mask(r)
      assertEquals(m("status").str, "included")
      assertEquals(m("path").str, p.toString)
      assertEquals(m("source_sha256").str, sourceHash(before))
      assertEquals(m("mime").str, "image/png")
      assertEquals(m("width").num.toInt, 3)
      assertEquals(m("height").num.toInt, 2)
      val encoded = pngBytes(m)
      assertEquals(m("png_sha256").str, sourceHash(encoded))
      val decoded = ImageIO.read(new ByteArrayInputStream(encoded))
      for y <- 0 until 2; x <- 0 until 3 do assertEquals(decoded.getRGB(x, y), expected.getRGB(x, y))
      assertEquals(Files.readAllBytes(p).toVector, before.toVector)
      assert(r.assetsJson.contains("diagnostic_capture_only"))
      Files.delete(p)
      assertEquals(ImageIO.read(new ByteArrayInputStream(pngBytes(mask(r)))).getRGB(0, 0), 0xffff2000)
    }
  }

  test("every non-init whitelist field embeds independently including duplicate PhotoMaker paths") {
    withDir { dir =>
      val p = dir.resolve("input.png")
      image(p)
      val hooks = ujson.Obj.from(Vector("reference", "revision", "ip_adapter", "controllllite").map(k => k -> ujson.Obj("image" -> p.toString)))
      hooks("photomaker") = ujson.Obj("images" -> ujson.Arr(p.toString, p.toString))
      val r = PayloadAssets.collect(body(ujson.Obj("controlnet" -> ujson.Arr(ujson.Obj("image" -> p.toString)), "hooks" -> hooks)), true)
      assertEquals(assets(r).size, 7)
      assert(assets(r).forall(_("status").str == "included"))
      assertEquals(assets(r).takeRight(2).map(_("path").str), Vector(p.toString, p.toString))
    }
  }

  test("recognized malformed shapes produce bounded diagnostics without arbitrary field discovery") {
    val malformedFields = ujson.Obj("controlnet" -> ujson.Arr(7, ujson.Obj("image" -> 5, "enabled" -> "bad")),
      "hooks" -> ujson.Obj("reference" -> "bad", "revision" -> ujson.Obj("image" -> ujson.Arr("x.png")),
        "photomaker" -> ujson.Obj("images" -> ujson.Arr("", 2, ujson.Obj("image" -> "never.png")))))
    val malformed = Vector("null", "[]", "{", body(ujson.Obj("img2img" -> "bad", "controlnet" -> "bad", "hooks" -> "bad")),
      body(malformedFields), body(ujson.Obj("hooks" -> ujson.Obj("photomaker" -> ujson.Obj("images" -> "not-an-array")))))
    malformed.foreach { frozen =>
      val r = PayloadAssets.collectWith(frozen, true, PayloadAssets.Limits(), denyIO)
      assert(r.warnings.nonEmpty, frozen)
      assert(r.warnings.forall(_.length <= 512))
      assert(!r.assetsJson.contains("png_data_url"))
      ujson.read(r.assetsJson)
    }
  }

  test("missing corrupt and directory inputs fail only diagnostics") {
    withDir { dir =>
      val bad = Files.writeString(dir.resolve("bad.png"), "not an image")
      Vector(dir.resolve("missing.png"), dir, bad).foreach { p =>
        val r = collectMask(p)
        assertEquals(mask(r)("status").str, "unavailable")
        assert(r.warnings.nonEmpty)
        assert(!mask(r).obj.contains("png_data_url"))
      }
      assertEquals(Files.readString(bad), "not an image")
    }
  }

  test("source limit rejects more than 16 MiB before opening a handle") {
    withDir { dir =>
      val p = dir.resolve("large.png")
      val file = new java.io.RandomAccessFile(p.toFile, "rw")
      try file.setLength(16L * 1024 * 1024 + 1) finally file.close()
      var opened = false
      val io = new PayloadAssets.SourceAccess:
        def isRegularFile(path: Path): Boolean = Files.isRegularFile(path)
        def size(path: Path): Long = Files.size(path)
        def open(path: Path): InputStream = { opened = true; Files.newInputStream(path) }
      val r = PayloadAssets.collectWith(body(ujson.Obj("hooks" -> ujson.Obj("reference" -> ujson.Obj("image" -> p.toString)))),
        true, PayloadAssets.Limits(), io)
      assertEquals(assets(r).head("status").str, "unavailable")
      assertEquals(assets(r).head("reason").str, "source_file_limit")
      assert(!opened)
    }
  }

  test("one bounded source handle detects growth after size inspection and never reads past limit plus one") {
    var opens = 0
    var reads = 0
    var closes = 0
    val io = new PayloadAssets.SourceAccess:
      def isRegularFile(path: Path): Boolean = true
      def size(path: Path): Long = 1L
      def open(path: Path): InputStream =
        opens += 1
        new ByteArrayInputStream(Array.fill[Byte](1024)(42)):
          override def read(b: Array[Byte], off: Int, len: Int): Int =
            val n = super.read(b, off, len)
            if n > 0 then reads += n
            n
          override def close(): Unit = { closes += 1; super.close() }
    val r = PayloadAssets.collectWith(body(ujson.Obj("hooks" -> ujson.Obj("reference" -> ujson.Obj("image" -> "growing.png")))),
      true, PayloadAssets.Limits(maxSourceBytes = 100), io)
    assertEquals(assets(r).head("reason").str, "source_file_limit")
    assertEquals(opens, 1)
    assertEquals(reads, 101)
    assertEquals(closes, 1)
  }

  // Valid PNG header with no IDAT. If the collector tries raster decoding first, ImageIO fails for
  // missing data instead of returning the dimension-limit diagnostic asserted below.
  private def pngHeader(w: Int, h: Int): Array[Byte] =
    val out = new ByteArrayOutputStream()
    val d = new java.io.DataOutputStream(out)
    d.write(Array[Byte](-119, 80, 78, 71, 13, 10, 26, 10))
    val chunk = new ByteArrayOutputStream()
    val hd = new java.io.DataOutputStream(chunk)
    hd.writeBytes("IHDR"); hd.writeInt(w); hd.writeInt(h); hd.write(Array[Byte](8, 2, 0, 0, 0)); hd.close()
    val bytes = chunk.toByteArray
    val crc = new CRC32(); crc.update(bytes)
    d.writeInt(13); d.write(bytes); d.writeInt(crc.getValue.toInt); d.close()
    out.toByteArray

  test("oversized image dimensions and pixel product are rejected from header before raster decode") {
    withDir { dir =>
      Vector((8193, 1), (1, 8193), (8000, 4001), (Int.MaxValue, Int.MaxValue)).zipWithIndex.foreach { case ((w, h), i) =>
        val p = Files.write(dir.resolve(s"header-$i.png"), pngHeader(w, h))
        val r = collectMask(p)
        assertEquals(mask(r)("status").str, "unavailable")
        assertEquals(mask(r)("reason").str, "image_dimensions_limit")
      }
    }
  }

  test("PNG encoding is capped while encoding without publishing partial data URLs") {
    withDir { dir =>
      val p = dir.resolve("input.png")
      image(p, 30, 30)
      val r = PayloadAssets.collectWith(body(ujson.Obj("hooks" -> ujson.Obj("reference" -> ujson.Obj("image" -> p.toString)))),
        true, PayloadAssets.Limits(maxPngBytes = 32), PayloadAssets.fileSource)
      assertEquals(assets(r).head("reason").str, "encoded_png_limit")
      assert(!assets(r).head.obj.contains("png_data_url"))
    }
  }

  test("4 MiB PNG encoding cap is enforced for a real incompressible configured image") {
    withDir { dir =>
      val p = dir.resolve("noise.png")
      val img = new BufferedImage(1200, 1200, BufferedImage.TYPE_INT_RGB)
      val rng = new java.util.Random(17L)
      for y <- 0 until 1200; x <- 0 until 1200 do img.setRGB(x, y, rng.nextInt(0x1000000))
      assert(ImageIO.write(img, "png", p.toFile))
      assert(Files.size(p) > 4L * 1024 * 1024)
      val r = collectMask(p)
      assertEquals(mask(r)("reason").str, "encoded_png_limit")
      assert(!mask(r).obj.contains("png_data_url"))
    }
  }

  test("final asset document bounds include UTF-8 paths JSON overhead and base64 expansion") {
    withDir { dir =>
      val p = dir.resolve("input.png")
      image(p)
      val r = PayloadAssets.collectWith(body(ujson.Obj("hooks" -> ujson.Obj("reference" -> ujson.Obj("image" -> p.toString)))),
        true, PayloadAssets.Limits(maxDocumentBytes = 600), PayloadAssets.fileSource)
      assert(r.assetsJson.getBytes(UTF_8).length <= 600)
      assertEquals(ujson.read(r.assetsJson)("status").str, "unavailable_document_limit")
      assert(r.warnings.nonEmpty)
      val hugeLiteral = "日" * (6 * 1024 * 1024)
      val large = PayloadAssets.collectWith(body(ujson.Obj("hooks" -> ujson.Obj("reference" -> ujson.Obj("image" -> hugeLiteral)))),
        false, PayloadAssets.Limits(), denyIO)
      assert(large.assetsJson.getBytes(UTF_8).length <= 16 * 1024 * 1024)
      assertEquals(ujson.read(large.assetsJson)("status").str, "unavailable_document_limit")
    }
  }

  test("descriptor and warning overflow remains visible bounded and does not inspect omitted entries") {
    val p = body(ujson.Obj("hooks" -> ujson.Obj("photomaker" -> ujson.Obj("images" -> ujson.Arr.from((0 until 2000).map(_ => ujson.Num(9)))))))
    val r = PayloadAssets.collectWith(p, true, PayloadAssets.Limits(), denyIO)
    assert(assets(r).size <= 256)
    assert(r.warnings.size <= 33)
    assert(r.warnings.forall(_.length <= 512))
    val doc = ujson.read(r.assetsJson)
    assert(doc("omitted_asset_count").num > 0)
    assert(doc("omitted_warning_count").num > 0)
    assert(r.warnings.last.contains("omitted"))
  }

  /** A virtual long string makes accidental whole-input conversion fail deterministically.
    * Counters prove bounded work without relying on heap exhaustion or wall-clock timing.
    */
  private class RepeatedChars(char: Char, size: Int) extends CharSequence:
    var reads = 0
    def length(): Int = size
    def charAt(i: Int): Char =
      require(i >= 0 && i < size)
      reads += 1
      char
    def subSequence(start: Int, end: Int): CharSequence = fail("bounded writer must not materialize a subsequence")
    override def toString: String = fail("bounded writer must not materialize its complete input")

  test("bounded descriptor writer stops escaped Unicode paths at budget plus one before materializing the value") {
    Vector('日', '\u0000', '"', '\\').foreach { c =>
      val path = new RepeatedChars(c, 100000000)
      val out = new PayloadAssets.BoundedJsonWriter(1024)
      out.appendRaw("{\"path\":")
      out.appendQuoted(path)
      out.appendRaw("}")
      assertEquals(out.result, None)
      assertEquals(out.bytesSeen, 1025)
      assert(path.reads > 0 && path.reads <= 1025, s"unbounded path work for $c: ${path.reads}")
    }
    val exact = new PayloadAssets.BoundedJsonWriter(21)
    exact.appendQuoted("日\u0000\"\\😀")
    assertEquals(exact.result.map(ujson.read(_).str), Some("日\u0000\"\\😀"))
    assertEquals(exact.bytesSeen, 19)
    Vector("\uD800x", "x\uDC00", "\uD800\uDFFF", "\n\t\r").foreach { literal =>
      val unicode = new PayloadAssets.BoundedJsonWriter(64)
      unicode.appendQuoted(literal)
      assertEquals(unicode.result.map(ujson.read(_).str), Some(literal))
    }
  }

  test("bounded writer shares aggregate budget across JSON descriptor and data URL expansion") {
    val out = new PayloadAssets.BoundedJsonWriter(1024)
    out.appendRaw("{\"assets\":[")
    (0 until 3).foreach { _ =>
      out.appendRaw("{\"path\":")
      out.appendQuoted("日" * 50)
      out.appendRaw(",\"png_data_url\":")
      out.appendQuoted("data:image/png;base64," + "A" * 60)
      out.appendRaw("},")
    }
    val encoded = new RepeatedChars('A', 100000000)
    out.appendRaw("{\"png_data_url\":")
    out.appendQuoted(encoded)
    assertEquals(out.result, None)
    assertEquals(out.bytesSeen, 1025)
    assert(encoded.reads > 0 && encoded.reads < 250)
  }

  test("body admission counts bounded UTF-8 before invoking JSON parser without converting the complete input") {
    val oversized = new RepeatedChars('日', 100000000)
    var parses = 0
    val parsed = PayloadAssets.admitBody(oversized, 1024) { parses += 1; ujson.Obj() }
    assertEquals(parsed, None)
    assertEquals(parses, 0)
    assert(oversized.reads > 0 && oversized.reads <= 343)
    val admitted = PayloadAssets.admitBody("{}", 2) { parses += 1; ujson.Obj() }
    assertEquals(admitted, Some(ujson.Obj()))
    assertEquals(parses, 1)
    val frozen = body(ujson.Obj("hooks" -> ujson.Obj("reference" -> ujson.Obj("image" -> ("日\u0000" * 2000)))))
    val r = PayloadAssets.collectWith(frozen, false, PayloadAssets.Limits(maxInputBodyBytes = 1024), denyIO)
    assertEquals(ujson.read(r.assetsJson)("status").str, "unavailable_document_limit")
    assert(r.warnings.exists(_.contains("input body")))
    assertEquals(ujson.read(frozen)("params")("hooks")("reference")("image").str, "日\u0000" * 2000)
  }

  test("image diagnostics reject excessive structural work before invoking the input parser") {
    assertEquals(PayloadAssets.Limits().maxInputBodyBytes, 16 * 1024 * 1024)
    // 100001 lexical tokens in about 200 KiB, well below the approved byte-admission limit.
    val wide = "[" + Vector.fill(100000)("0").mkString(",") + "]"
    val deep = ("[" * 513) + "0" + ("]" * 513)
    Seq("nodes_limit" -> wide, "depth_limit" -> deep).foreach { (reason, input) =>
      var parses = 0
      val parsed = PayloadAssets.admitBody(input, PayloadAssets.Limits().maxInputBodyBytes) {
        parses += 1
        ujson.Obj()
      }
      assertEquals(parsed, None, reason)
      assertEquals(parses, 0, reason)
      val frozen = "{\"params\":{\"hooks\":{\"unrecognized\":" + input + "}}}"
      val result = PayloadAssets.collectWith(frozen, false, PayloadAssets.Limits(), denyIO)
      assertEquals(ujson.read(result.assetsJson)("status").str, "unavailable_document_limit")
      assertEquals(ujson.read(result.assetsJson)("assets").arr.size, 0)
      assert(result.warnings.exists(_.contains(reason)), result.warnings.mkString("; "))
    }
  }

  test("aggregate asset budget stops further file reads after first excessive base64 descriptor") {
    withDir { dir =>
      val p = dir.resolve("input.png")
      image(p)
      var opens = 0
      val io = new PayloadAssets.SourceAccess:
        def isRegularFile(path: Path): Boolean = Files.isRegularFile(path)
        def size(path: Path): Long = Files.size(path)
        def open(path: Path): InputStream = { opens += 1; Files.newInputStream(path) }
      val frozen = body(ujson.Obj("hooks" -> ujson.Obj("photomaker" -> ujson.Obj("images" -> ujson.Arr.from(Vector.fill(20)(p.toString))))))
      val r = PayloadAssets.collectWith(frozen, true, PayloadAssets.Limits(maxDocumentBytes = 3000), io)
      val doc = ujson.read(r.assetsJson)
      assertEquals(doc("status").str, "unavailable_document_limit")
      assertEquals(doc("omitted_asset_count").num.toInt, 20)
      assert(opens > 1 && opens < 20)
      assert(r.assetsJson.getBytes(UTF_8).length <= 3000)
      assertEquals(doc("assets").arr.size, 0)
    }
  }
