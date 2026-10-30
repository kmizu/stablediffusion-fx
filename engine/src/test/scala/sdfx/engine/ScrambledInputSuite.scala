package sdfx.engine

import sdfx.core.png.{ImageScrambleSession, PngMetadata, ScrambledPng}
import java.awt.image.BufferedImage
import java.nio.file.Files

class ScrambledInputSuite extends munit.FunSuite:
  test("legacy passwords emit warnings for synchronous and asynchronous worker inputs") {
    import scala.concurrent.{Await, ExecutionContext, Future}
    import scala.concurrent.duration.*
    given ExecutionContext = ExecutionContext.global
    val dir = Files.createTempDirectory("scrambled-legacy-warning")
    val legacy = dir.resolve("legacy.png")
    val verified = dir.resolve("verified.png")
    val image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB)
    val encoded = sdfx.core.png.PixelScramble.encode(image, "actual")
    PngMetadata.write(encoded, legacy, Seq("Encrypt" -> "pixel_shuffle_3"))
    ScrambledPng.write(image, verified, Nil, Some("wrong"))
    val session = new ImageScrambleSession; session.unlock("wrong")
    val warnings = scala.collection.mutable.ArrayBuffer.empty[String]
    ScrambledInput.withPlain(legacy, dir, session.snapshot, warnings += _) { _ => () }
    assertEquals(warnings.toSeq, Seq(ScrambledInput.LegacyWarning))
    warnings.clear()
    ScrambledInput.withPlain(verified, dir, session.snapshot, warnings += _) { _ => () }
    assert(warnings.isEmpty)
    Await.result(ScrambledInput.withObjectAsync(ujson.Obj("image_path" -> legacy.toString), Seq("image_path"),
      dir, session.snapshot, warnings += _)(_ => Future.successful(())), 10.seconds)
    assertEquals(warnings.toSeq, Seq(ScrambledInput.LegacyWarning))
    Files.delete(legacy); Files.delete(verified); Files.delete(dir)
  }
  test("decoded input is scoped to the operation and removed even on failure") {
    val dir = Files.createTempDirectory("scrambled-input-test")
    val src = dir.resolve("protected.png")
    val image = new BufferedImage(6, 5, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until 5; x <- 0 until 6 do image.setRGB(x, y, x * 1024 + y * 32)
    ScrambledPng.write(image, src, Seq("parameters" -> "cat"), Some("password"))
    val before = Files.readAllBytes(src).toSeq
    val session = new ImageScrambleSession
    session.unlock("password")
    var temp: java.nio.file.Path = src
    intercept[IllegalStateException] {
      ScrambledInput.withPlain(src, dir, session.snapshot) { path =>
        temp = path
        assertNotEquals(path, src)
        val decoded = PngMetadata.readImage(path)
        assert(image.getRGB(0, 0, 6, 5, null, 0, 6).sameElements(decoded.getRGB(0, 0, 6, 5, null, 0, 6)))
        throw new IllegalStateException("test failure")
      }
    }
    assert(!Files.exists(temp))
    assertEquals(Files.readAllBytes(src).toSeq, before)
    intercept[IllegalArgumentException](ScrambledInput.withPlain(src, dir, None)(_ => ()))
    assertEquals(Files.readAllBytes(src).toSeq, before)
    Files.delete(src); Files.delete(dir)
  }

  test("ordinary input stays at its original path") {
    val dir = Files.createTempDirectory("ordinary-input-test")
    val src = dir.resolve("ordinary.png")
    PngMetadata.write(new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), src, Nil)
    ScrambledInput.withPlain(src, dir, None)(p => assertEquals(p, src))
    assert(Files.exists(src))
    Files.delete(src); Files.delete(dir)
  }


  test("failure on a later input removes the earlier decoded temporary") {
    val dir = Files.createTempDirectory("scrambled-multiple-test")
    val first = dir.resolve("first.png"); val second = dir.resolve("second.png")
    val img = new BufferedImage(3, 3, BufferedImage.TYPE_INT_RGB)
    ScrambledPng.write(img, first, Nil, Some("first"))
    ScrambledPng.write(img, second, Nil, Some("second"))
    val session = new ImageScrambleSession; session.unlock("first")
    val params = sdfx.core.model.GenerationParams(img2img = Some(sdfx.core.model.Img2ImgParams(first.toString, mask = Some(second.toString))))
    intercept[IllegalArgumentException](ScrambledInput.withParams(params, dir, session.snapshot)(_ => fail("should not dispatch")))
    val files = scala.util.Using.resource(Files.list(dir))(_.toArray.toSeq.map(_.asInstanceOf[java.nio.file.Path].getFileName.toString).toSet)
    assertEquals(files, Set("first.png", "second.png"))
    Files.delete(first); Files.delete(second); Files.delete(dir)
  }


  test("disabled ControlNet image is not accessed or decoded") {
    val dir = Files.createTempDirectory("scrambled-disabled-test")
    val source = dir.resolve("other-password.png")
    ScrambledPng.write(new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), source, Nil, Some("other"))
    val p = sdfx.core.model.GenerationParams(controlnet = Seq(sdfx.core.model.ControlNetUnit(enabled = false, image = Some(source.toString))))
    ScrambledInput.withParams(p, dir, None)(result => assertEquals(result.controlnet.head.image, Some(source.toString)))
    Files.delete(source); Files.delete(dir)
  }


  test("asynchronous path leases remain until completion and are removed on success or failure") {
    import scala.concurrent.{Await, ExecutionContext, Future, Promise}
    import scala.concurrent.duration.*
    given ExecutionContext = ExecutionContext.global
    val dir = Files.createTempDirectory("scrambled-async-test")
    val source = dir.resolve("input.png")
    val image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB)
    ScrambledPng.write(image, source, Nil, Some("secret"))
    val session = new ImageScrambleSession; session.unlock("secret")
    def exercise(fail: Boolean): Unit =
      val received = Promise[java.nio.file.Path]()
      val pending = Promise[String]()
      val result = ScrambledInput.withObjectAsync(ujson.Obj("image_path" -> source.toString), Seq("image_path"), dir, session.snapshot) { values =>
        val path = java.nio.file.Path.of(values("image_path").str)
        received.success(path)
        pending.future
      }
      val path = Await.result(received.future, 10.seconds)
      assertNotEquals(path, source)
      assert(Files.exists(path))
      if fail then pending.failure(new IllegalStateException("failure")) else pending.success("ok")
      if fail then intercept[IllegalStateException](Await.result(result, 10.seconds))
      else assertEquals(Await.result(result, 10.seconds), "ok")
      assert(!Files.exists(path))
    exercise(false); exercise(true)
    val failed = ScrambledInput.withObjectAsync(ujson.Obj("image_path" -> source.toString), Seq("image_path"), dir, None)(_ => Future.successful("unexpected"))
    intercept[IllegalArgumentException](Await.result(failed, 10.seconds))
    Files.delete(source); Files.delete(dir)
  }


  test("asynchronous setup and synchronous dispatch failure clean all acquired inputs") {
    import scala.concurrent.{Await, ExecutionContext, Future}
    import scala.concurrent.duration.*
    given ExecutionContext = ExecutionContext.global
    val dir = Files.createTempDirectory("scrambled-async-failure")
    val first = dir.resolve("first.png"); val second = dir.resolve("second.png")
    val image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB)
    ScrambledPng.write(image, first, Nil, Some("first"))
    ScrambledPng.write(image, second, Nil, Some("second"))
    val session = new ImageScrambleSession; session.unlock("first")
    val bad = ScrambledInput.withObjectAsync(ujson.Obj("a" -> first.toString, "b" -> second.toString), Seq("a", "b"), dir, session.snapshot)(_ => Future.successful("bad"))
    intercept[IllegalArgumentException](Await.result(bad, 10.seconds))
    val thrown = ScrambledInput.withObjectAsync[String](ujson.Obj("a" -> first.toString), Seq("a"), dir, session.snapshot)(_ => throw new IllegalStateException("dispatch failed"))
    intercept[IllegalStateException](Await.result(thrown, 10.seconds))
    val files = scala.util.Using.resource(Files.list(dir))(_.toArray.toSeq.map(_.asInstanceOf[java.nio.file.Path].getFileName.toString).toSet)
    assertEquals(files, Set("first.png", "second.png"))
    Files.delete(first); Files.delete(second); Files.delete(dir)
  }
