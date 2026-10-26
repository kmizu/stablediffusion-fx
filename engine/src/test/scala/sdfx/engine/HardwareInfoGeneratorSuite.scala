package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.infotext.{HardwareInfo, Infotext}
import sdfx.core.model.GenerationParams
import sdfx.core.png.PngMetadata
import java.awt.image.BufferedImage
import java.nio.file.Files
import scala.collection.immutable.ListMap

/** The stdlib fixture supplies synthetic records; source timing has separate controlled Python tests. */
class HardwareInfoGeneratorSuite extends munit.FunSuite:
  import FakeEngine.*
  private def withServices(enabled: Boolean, remark: String = "")(check: AppServices => Unit): Unit =
    val home = tempHome()
    val services = new AppServices(home, Settings(lastModel = Some("foo"), hardwareInfoEnabled = enabled,
      hardwareInfoRemark = remark), dryRun = true, builtins = Nil, launchOverride = Some(() => launch()),
      persistSettings = false, installShutdownHook = false)
    try
      services.start()
      check(services)
    finally
      services.shutdown()
      deleteTree(home.root)

  test("Generator injects trusted native hardware options without changing sampling or original prompt") {
    withServices(true, "机, quiet\nsecond line") { services =>
      val original = GenerationParams(prompt = "a cat", seed = 7, subseed = 8, steps = 3, width = 64, height = 64)
      val prepared = services.generator.prepare(original)
      assertEquals(prepared.request.hardwareInfo, HardwareInfo.Options(true, "机, quiet\nsecond line"))
      assertEquals(prepared.recorded, original)
      assertEquals(prepared.request.params.seed, 7L)
      assertEquals(prepared.request.params.prompt, "a cat")
    }
  }

  test("disabled generation cannot retain a previous machine timing or remark from imported extras") {
    withServices(false) { services =>
      val imported = GenerationParams(prompt = "cat", seed = 7, extra = ListMap("Hardware Info" -> "old GPU",
        "Remark" -> "old remark", "Time taken" -> "999.0 sec.", "Custom field" -> "keep"))
      val prepared = services.generator.prepare(imported)
      assertEquals(prepared.recorded.extra, ListMap("Custom field" -> "keep"))
      assertEquals(prepared.request.params.extra, ListMap("Custom field" -> "keep"))
      assertEquals(prepared.request.hardwareInfo, HardwareInfo.Options())
    }
  }

  test("each saved batch image embeds its own typed server record with literal remark and unchanged final count") {
    withServices(true, "日本語, literal\nnext") { services =>
      val images = services.generator.generate(GenerationParams(seed = 51, steps = 1, width = 64, height = 64,
        batchSize = 2, nIter = 2))
      assertEquals(images.size, 4)
      images.zipWithIndex.foreach { (image, index) =>
        val fields = Infotext.parse(PngMetadata.readParameters(image.path).get)
        assertEquals(fields.get("Hardware Info"), Some("Fixture GPU, Fixture CPU, 32GB RAM"))
        assertEquals(fields.get("Remark"), Some("日本語, literal\nnext"))
        assertEquals(fields.get("Time taken"), Some(s"${index + 1}.0 sec."))
        assertEquals(fields.get("Seed"), Some((51 + index).toString))
      }
    }
  }

  test("Extras writes actual output provenance without replacing source bytes or claiming new diffusion conditioning") {
    withServices(true, "Extras remark") { services =>
      val source = services.home.root.resolve("input.png")
      val input = new BufferedImage(40, 30, BufferedImage.TYPE_INT_RGB)
      PngMetadata.write(input, source, Seq("parameters" -> "original diffusion prompt\nSteps: 4, Seed: 1, Size: 40x30",
        "Comment" -> "source metadata stays in the original"))
      val before = Files.readAllBytes(source)
      val output = services.generator.upscaleImage(source, "Lanczos", 2)
      assert(output != source)
      assert(Files.readAllBytes(source).sameElements(before))
      val pixels = PngMetadata.readImage(output)
      assertEquals(pixels.getRGB(0, 0) & 0xffffff, 0x010203) // actual stdlib fixture output
      val parsed = Infotext.parse(PngMetadata.readParameters(output).getOrElse(fail("Extras output is missing embedded parameters")))
      assertEquals(parsed.prompt, "")
      assertEquals(parsed.get("Hardware Info"), Some("Fixture GPU, Fixture CPU, 32GB RAM"))
      assertEquals(parsed.get("Remark"), Some("Extras remark"))
      assertEquals(parsed.get("Time taken"), Some("7.5 sec."))
      assertEquals(parsed.get("Postprocess upscaler"), Some("Lanczos"))
      assertEquals(parsed.get("Size"), Some(s"${pixels.getWidth}x${pixels.getHeight}"))
    }
  }
