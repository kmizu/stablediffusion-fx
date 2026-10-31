package sdfx.mcp

import java.nio.file.Files
import java.util.Base64

class ImageMaterializeCleanupSuite extends munit.FunSuite:
  test("failed and unavailable PNG writes remove their owned temporary file") {
    val dir = Files.createTempDirectory("image-cleanup-")
    val bytes = Base64.getDecoder.decode(TestEnv.pngBase64(8, 8))
    def temp(suffix: String) = Files.createTempFile(dir, "input-", suffix)
    def count(): Long =
      val stream = Files.list(dir)
      try stream.count() finally stream.close()
    intercept[Exception] {
      Images.writeDecoded(bytes, "image", temp, (_, path) => { Files.write(path, Array[Byte](1)); throw new java.io.IOException("failed") })
    }
    assertEquals(count(), 0L)
    intercept[Exception] {
      Images.writeDecoded(bytes, "image", temp, (_, path) => { Files.write(path, Array[Byte](1)); false })
    }
    assertEquals(count(), 0L)
  }
