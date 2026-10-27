package sdfx.core

import sdfx.core.config.Settings
import java.io.{ByteArrayInputStream, InputStream}
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

class CardPreviewBoundedInputSuite extends munit.FunSuite:
  private val limit = 16 * 1024 * 1024

  private final class UnendingStream extends InputStream:
    var consumed = 0L
    var closed = false
    override def read(): Int =
      consumed += 1
      if consumed > limit.toLong + 1 then throw new java.io.IOException("read budget exceeded")
      'x'.toInt
    override def read(b: Array[Byte], off: Int, len: Int): Int =
      consumed += len
      if consumed > limit.toLong + 1 then throw new java.io.IOException("read budget exceeded")
      java.util.Arrays.fill(b, off, off + len, 'x'.toByte)
      len
    override def close(): Unit = { closed = true }

  private final class CountingStream(bytes: Array[Byte]) extends ByteArrayInputStream(bytes):
    var consumed = 0L
    var closed = false
    override def read(): Int =
      val n = super.read()
      if n >= 0 then consumed += 1
      n
    override def read(b: Array[Byte], off: Int, len: Int): Int =
      val n = super.read(b, off, len)
      if n > 0 then consumed += n
      n
    override def close(): Unit = { closed = true; super.close() }

  test("oversized or unending input stops at the limit plus one sentinel byte and closes") {
    val stream = new UnendingStream
    val result = Settings.loadResultFromStream(() => stream)
    assert(result.recovery.exists(_.reason.nonEmpty))
    assertEquals(result.settings.nsfwCardBlurDefault, "Hide")
    assertEquals(result.recovery.flatMap(_.originalSha256), None)
    assert(stream.consumed <= limit.toLong + 1L, s"consumed ${stream.consumed} bytes")
    assert(stream.closed)
  }

  test("finite recoverable input is read to EOF, hashed and closed") {
    val bytes = """{"theme":"light","nsfw_card_blur_default":"bad"}""".getBytes(StandardCharsets.UTF_8)
    val stream = new CountingStream(bytes)
    val result = Settings.loadResultFromStream(() => stream)
    val hash = MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 0xff}%02x").mkString
    assertEquals(result.settings.theme, "light")
    assertEquals(result.settings.nsfwCardBlurDefault, "Hide")
    assertEquals(result.recovery.flatMap(_.originalSha256), Some(hash))
    assertEquals(stream.consumed, bytes.length.toLong)
    assert(stream.closed)
  }

  test("exactly 16 MiB remains admissible and its stream closes") {
    val prefix = "{\"padding\":\"".getBytes(StandardCharsets.UTF_8)
    val suffix = "\",\"theme\":\"light\"}".getBytes(StandardCharsets.UTF_8)
    val bytes = Array.fill[Byte](limit)('a'.toByte)
    System.arraycopy(prefix, 0, bytes, 0, prefix.length)
    System.arraycopy(suffix, 0, bytes, bytes.length - suffix.length, suffix.length)
    val stream = new CountingStream(bytes)
    val result = Settings.loadResultFromStream(() => stream)
    assertEquals(result.recovery, None)
    assertEquals(result.settings.theme, "light")
    assert(stream.closed)
  }
