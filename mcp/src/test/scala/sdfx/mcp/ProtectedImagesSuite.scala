package sdfx.mcp

import java.awt.image.BufferedImage
import java.nio.file.Files
import java.util.Base64
import sdfx.core.png.{PngMetadata, ScrambledPng}

class ProtectedImagesSuite extends munit.FunSuite:
  test("MCP rejects protected paths and inline images before marker stripping or preview encoding") {
    val dir = Files.createTempDirectory("mcp-protected-")
    val path = dir.resolve("protected.png")
    val image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB)
    ScrambledPng.write(image, path, Nil, Some("private-password"))
    val bytes = Files.readAllBytes(path)
    val policy = new PathPolicy(() => Seq(dir))
    val temp: String => java.nio.file.Path = suffix => Files.createTempFile(dir, "input-", suffix)
    val encoded = Base64.getEncoder.encodeToString(bytes)
    try
      for spec <- Seq(path.toString, encoded, "data:image/png;base64," + encoded) do
        val error = intercept[ToolError](Images.materialize(spec, "init_image", policy, dir, temp))
        assert(error.getMessage.contains("password"), error.getMessage)
        assert(!error.getMessage.contains("private-password"))
      for side <- Seq(0, 4) do
        val error = intercept[ToolError](Images.encode(path, side))
        assert(error.getMessage.contains("password"), error.getMessage)
      intercept[ToolError](Images.rejectProtectedPng(path, "input_paths"))
      assert(java.util.Arrays.equals(bytes, Files.readAllBytes(path)))
      val stream = Files.list(dir)
      try assertEquals(stream.count(), 1L)
      finally stream.close()
    finally
      Files.deleteIfExists(path)
      Files.deleteIfExists(dir)
  }

  test("unmarked images remain usable even when they carry a legacy password verifier") {
    val dir = Files.createTempDirectory("mcp-unmarked-")
    val path = dir.resolve("ordinary.png")
    val image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB)
    PngMetadata.write(image, path, Seq("EncryptPwdSha" -> "legacy"))
    val policy = new PathPolicy(() => Seq(dir))
    val temp: String => java.nio.file.Path = suffix => Files.createTempFile(dir, "input-", suffix)
    try
      assertEquals(Images.materialize(path.toString, "init_image", policy, dir, temp)._1, path.toRealPath())
      val inline = Images.materialize(Base64.getEncoder.encodeToString(Files.readAllBytes(path)), "init_image", policy, dir, temp)
      try assertEquals(Images.size(inline._1), (8, 8))
      finally inline._2.foreach(Files.deleteIfExists(_))
      assertEquals(Images.encode(path, 4).width, 4)
    finally
      Files.deleteIfExists(path)
      Files.deleteIfExists(dir)
  }
