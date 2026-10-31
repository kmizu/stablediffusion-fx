package sdfx.core

import sdfx.core.modelpresets.*
import java.nio.file.{Files, Path}

/** Explicit network acceptance against a public model; no weights or inference dependencies. */
object LiveCivitaiModelPresetsCheck:
  def main(args: Array[String]): Unit =
    require(args.length == 1 || args.length == 2)
    val output = Path.of(args(0)).toAbsolutePath
    Files.createDirectory(output)
    val hash = if args.length == 2 then args(1) else "0" * 64
    val local = if args.length == 2 then ModelPresetDocument.empty else
      ModelPresetDocument.empty.copy(url = "https://civitai.com/models/4384")
    val result = new CivitaiPresetClient().fetch(hash, local, () => ())
    result.document.validate()
    require(result.document.url.nonEmpty)
    require(result.thumbnail.exists(_.nonEmpty))
    Files.writeString(output.resolve("document.json"), result.document.json)
    Files.write(output.resolve("thumbnail.bin"), result.thumbnail.get)
    println(s"PASS real Civitai Java client: ${result.document.url}; thumbnail bytes=${result.thumbnail.get.length}")
