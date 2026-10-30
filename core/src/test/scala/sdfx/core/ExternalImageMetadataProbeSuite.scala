package sdfx.core

import sdfx.core.png.ExternalImageMetadataProbe
import java.nio.file.Files
import java.io.ByteArrayOutputStream

class ExternalImageMetadataProbeSuite extends munit.FunSuite with TempDir:
  private def bytes(values: Int*): Array[Byte] = values.map(_.toByte).toArray
  private def segment(marker: Int, payload: Array[Byte]): Array[Byte] =
    val size = payload.length + 2
    bytes(0xff, marker, size >> 8, size & 255) ++ payload
  private def ascii(s: String): Array[Byte] = s.getBytes(java.nio.charset.StandardCharsets.ISO_8859_1)
  private def le(n: Int): Array[Byte] = bytes(n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >> 24) & 255)
  private def chunk(name: String, payload: Array[Byte]): Array[Byte] =
    ascii(name) ++ le(payload.length) ++ payload ++ (if payload.length % 2 == 1 then bytes(0) else Array.emptyByteArray)
  private def webp(chunks: Array[Byte]*): Array[Byte] =
    val body = ascii("WEBP") ++ chunks.flatten
    ascii("RIFF") ++ le(body.length) ++ body

  tmp.test("JPEG recognizes EXIF ICC and JFIF density and reports scan boundary") { root =>
    val p = root.resolve("misleading.webp")
    val data = bytes(0xff, 0xd8) ++ segment(0xe0, ascii("JFIF\u0000") ++ bytes(1, 2, 1, 0, 144, 0, 144, 0, 0)) ++
      segment(0xe1, ascii("Exif\u0000\u0000") ++ bytes(1)) ++
      segment(0xe2, ascii("ICC_PROFILE\u0000") ++ bytes(1, 1)) ++ bytes(0xff, 0xda)
    Files.write(p, data)
    val result = ExternalImageMetadataProbe.inspect(p)
    assertEquals(result.format(), "jpeg")
    assertEquals(result.kinds().toArray.toSet, Set[AnyRef]("density", "exif", "icc"))
    assertEquals(result.complete(), false)
    assert(Files.readAllBytes(p).sameElements(data))
  }

  tmp.test("WebP detects odd sized chunks and declared metadata") { root =>
    val p = root.resolve("image.webp")
    val data = webp(chunk("VP8X", bytes(0x2c, 0, 0, 0, 0, 0, 0, 0, 0, 0)), chunk("EXIF", bytes(1, 2, 3)))
    Files.write(p, data)
    val result = ExternalImageMetadataProbe.inspect(p)
    assertEquals(result.format(), "webp")
    assertEquals(result.kinds().toArray.toSet, Set[AnyRef]("icc", "exif", "xmp"))
    assertEquals(result.complete(), true)
    assert(Files.readAllBytes(p).sameElements(data))
  }

  tmp.test("truncated and invalid sizes fail explicitly") { root =>
    val p = root.resolve("bad")
    Files.write(p, bytes(0xff, 0xd8, 0xff, 0xe1, 0, 20, 1))
    intercept[java.io.IOException](ExternalImageMetadataProbe.inspect(p))
    Files.write(p, ascii("RIFF") ++ le(14) ++ ascii("WEBP") ++ ascii("EXIF") ++ le(50) ++ bytes(0, 0))
    intercept[java.io.IOException](ExternalImageMetadataProbe.inspect(p))
  }
