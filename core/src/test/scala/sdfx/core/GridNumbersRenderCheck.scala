package sdfx.core

import sdfx.core.image.{Grid, GridNumbers}
import java.awt.Color
import java.awt.image.BufferedImage
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO

/** Manual CPU renderer contact sheet. No Python engine or model is launched.
  * sbt "core/Test/runMain sdfx.core.GridNumbersRenderCheck outputs/grid-numbers-render-check"
  */
object GridNumbersRenderCheck:
  def main(args: Array[String]): Unit =
    val root = Paths.get(args.headOption.getOrElse("outputs/grid-numbers-render-check"))
    Files.createDirectories(root)
    def source(width: Int, height: Int): BufferedImage =
      val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      for x <- 0 until width; y <- 0 until height do
        image.setRGB(x, y, new Color(70 + x * 100 / width, 100 + y * 80 / height, 140).getRGB)
      image
    val full = source(512, 512)
    val options = Map("Grid image numbers" -> "True", "Grid XYZ info" -> "True")
    val regular = GridNumbers.render(full, "00421-123.png", options,
      Map("X" -> "Steps: 20", "Y" -> "CFG Scale: 7", "Z" -> "Checkpoint name: Model A"))
    val unicode = GridNumbers.render(full, "00422-124.png", options,
      Map("X" -> "Prompt: 桜の木", "Y" -> "Clip skip: 2", "Z" -> "Sampler: DPM++ 2M"))
    val small = GridNumbers.render(source(180, 128), "00423-125.png", options,
      Map("X" -> "Sampler: DPM++ 2M", "Y" -> ("very long label " * 20 + ": 7")))
    val tiny = GridNumbers.render(source(16, 16), "00424-126.png", options, Map("X" -> "Steps: 20"))
    Seq("source" -> full, "four-corners" -> regular, "unicode" -> unicode, "small" -> small, "tiny" -> tiny)
      .foreach((name, image) => ImageIO.write(image, "png", root.resolve(s"$name.png").toFile))
    val contact = Grid.labeled(Seq(full, regular, unicode, small, tiny), 3,
      Seq("Untouched source", "Number + XYZ", "Unicode"), Nil)
    ImageIO.write(contact, "png", root.resolve("contact.png").toFile)
    println(s"GRID NUMBER RENDER CHECK: ${root.toAbsolutePath.resolve("contact.png")}")
