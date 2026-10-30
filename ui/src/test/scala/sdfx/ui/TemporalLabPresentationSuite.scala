package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO
import sdfx.core.protocol.TemporalLabResult
import sdfx.ui.vm.{TemporalInputInfo, TemporalInputInspection, TemporalLabPresentation}

class TemporalLabPresentationSuite extends munit.FunSuite:
  test("inspection counts supported frames and reports mixed dimensions and invalid path") {
    val dir = Files.createTempDirectory("temporal-inspection-")
    try
      def image(name: String, width: Int, height: Int): Unit =
        assert(ImageIO.write(new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB), "png", dir.resolve(name).toFile))
      image("one.png", 20, 10); image("two.png", 20, 10)
      Files.writeString(dir.resolve("notes.txt"), "ignored")
      assertEquals(TemporalInputInspection.inspect(dir.toString), Right(TemporalInputInfo(2, 20, 10)))
      image("three.png", 10, 10)
      assert(TemporalInputInspection.inspect(dir.toString).isLeft)
      assert(TemporalInputInspection.inspect(dir.resolve("missing").toString).isLeft)
    finally
      val stream = Files.list(dir)
      try stream.forEach(p => Files.delete(p)) finally stream.close()
      Files.delete(dir)
  }

  test("DFI input summary reports both counts and conditional center crop and resize") {
    val same = TemporalLabPresentation.dfiInputs(TemporalInputInfo(3, 512, 512), TemporalInputInfo(3, 512, 512))
    assert(same.contains("3") && same.contains("512") && !same.toLowerCase.contains("crop"))
    val different = TemporalLabPresentation.dfiInputs(TemporalInputInfo(3, 640, 480), TemporalInputInfo(2, 512, 512))
    assert(different.contains("3") && different.contains("2"))
    assert(different.toLowerCase.contains("crop") && different.toLowerCase.contains("resize"))
    assert(different.toLowerCase.contains("count"))
  }

  test("extraction completion includes actual source and output rates and duration") {
    val result = TemporalLabResult("extract", outputDir = Some("result"), frameCount = Some(3),
      fps = Some(5), sourceFps = Some(10), duration = Some(0.6))
    val text = TemporalLabPresentation.completion(result)
    assert(text.contains("result") && text.contains("3") && text.contains("5") && text.contains("10") && text.contains("0.6"))
  }

  test("completion displays unverified legacy password warning") {
    val result = TemporalLabResult("preview", warnings = Seq("Legacy password is unverified"))
    assert(TemporalLabPresentation.completion(result).contains("unverified"))
  }

  test("help exposes stage order, overlay first pair duplication, and DFI blur quantization") {
    val dfi = TemporalLabPresentation.dfiHelp.toLowerCase
    val stages = TemporalLabPresentation.deflickerHelp.toLowerCase
    assert(dfi.contains("round") && dfi.contains("odd") && dfi.contains("inter"))
    assert(stages.indexOf("blend") < stages.indexOf("overlay") && stages.indexOf("overlay") < stages.indexOf("normalize"))
    assert(stages.contains("first") && stages.contains("twice") && stages.contains("copy"))
  }

  test("inspection coalesces rapid path changes, ignores stale results and stops after disposal") {
    val jobs = scala.collection.mutable.ArrayBuffer.empty[() => Unit]
    val delivered = scala.collection.mutable.ArrayBuffer.empty[String]
    val queue = new sdfx.ui.vm.TemporalInspectionQueue(
      (task: () => Unit) => jobs += task,
      (task: () => Unit) => task(),
      (path: String, cancelled: () => Boolean) => Right(TemporalInputInfo(path.length, 10, 10)))
    queue.request("first", r => delivered += r.toOption.get.count.toString)
    queue.request("second", r => delivered += r.toOption.get.count.toString)
    queue.request("third", r => delivered += r.toOption.get.count.toString)
    assertEquals(jobs.size, 1)
    jobs.remove(0)()
    assertEquals(delivered.toVector, Vector.empty)
    assertEquals(jobs.size, 1)
    jobs.remove(0)()
    assertEquals(delivered.toVector, Vector("5"))
    queue.request("fourth", r => delivered += r.toOption.get.count.toString)
    queue.dispose()
    jobs.remove(0)()
    assertEquals(delivered.toVector, Vector("5"))
  }

  test("missing Java reader is reported as advisory unavailable, not corrupt") {
    val dir = Files.createTempDirectory("temporal-webp-")
    val file = dir.resolve("frame.webp")
    Files.write(file, Array[Byte](1))
    try
      val defaultMessage = TemporalInputInspection.inspect(dir.toString).swap.toOption.get.toLowerCase
      assert(defaultMessage.contains("unavailable") && defaultMessage.contains("worker") && !defaultMessage.contains("corrupt"))
      val read = (_: java.nio.file.Path) => Left("inspection unavailable: no Java image reader; worker will validate")
      val result = TemporalInputInspection.inspect(dir.toString, () => false, read)
      assert(result.isLeft)
      val message = result.swap.toOption.get.toLowerCase
      assert(message.contains("unavailable") && message.contains("worker") && !message.contains("corrupt"))
    finally { Files.delete(file); Files.delete(dir) }
  }

  test("cancelled scan stops between image headers") {
    val dir = Files.createTempDirectory("temporal-cancel-scan-")
    val files = (1 to 4).map(i => Files.write(dir.resolve(s"frame$i.png"), Array[Byte](1)))
    try
      var reads = 0
      val read = (_: java.nio.file.Path) => { reads += 1; Right((20, 10)) }
      val result = TemporalInputInspection.inspect(dir.toString, () => reads >= 1, read)
      assert(result.isLeft)
      assertEquals(reads, 1)
    finally { files.foreach(Files.delete(_)); Files.delete(dir) }
  }
