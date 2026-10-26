package sdfx.engine

import java.nio.file.{Files, Path}
import scala.concurrent.Future

/** Files go directly to Python: image loaders must not flatten RGBA, CMYK or L data. */
final class VaeToolsService(services: AppServices):
  def run(input: Seq[Path], operation: String, representation: String, approximation: String,
          output: Path, seed: Long = 0, tiled: Boolean = false, cmykFormat: String = "TIFF",
          allowDownload: Boolean = false, approxNnPath: String = "",
          listener: sdfx.core.protocol.EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    require(input.nonEmpty && input.size <= 64 && input.forall(Files.isRegularFile(_)), "Choose 1 to 64 existing input files")
    require(Set("encode", "decode", "encode+decode")(operation), "Unknown VAE operation")
    require(Set("L", "RGBA", "CMYK")(representation), "Unknown latent representation")
    require(Set("Full", "Approx NN", "Approx cheap", "TAESD")(approximation), "Unknown VAE approximation")
    require(seed >= 0 && seed <= 0xffffffffL, "Seed must be 0 to 4294967295")
    Files.createDirectories(output)
    services.client.vaeTools(ujson.Obj(
      "input_paths" -> ujson.Arr.from(input.map(_.toAbsolutePath.toString)), "operation" -> operation,
      "representation" -> representation, "approximation" -> approximation, "output_dir" -> output.toAbsolutePath.toString,
      "seed" -> seed.toDouble, "tiled" -> tiled, "cmyk_format" -> cmykFormat, "allow_download" -> allowDownload,
      "approx_nn_path" -> approxNnPath,
      "cache_dir" -> services.home.root.resolve("models/vae-tools").toAbsolutePath.toString), listener)
