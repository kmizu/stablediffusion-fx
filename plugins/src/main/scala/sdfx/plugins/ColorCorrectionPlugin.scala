package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import java.awt.image.BufferedImage
import java.nio.file.Paths

/** Reference color transfer; shared engine algorithms also serve the native Extras workspace. */
class ColorCorrectionPlugin extends ExtrasPlugin:
  def id = "color_correction"
  def name = "Color correction"
  override def description = "Match reference colors. An empty reference uses the original input. A1111 needs scikit-image and blendmodes. Outputs are RGB."
  override def controls: Seq[Control] = Seq(
    ImageFile("reference", "Reference image (optional)"),
    Choice("method", "Method", Seq("A1111", "Wavelet", "AdaIN"), "A1111"),
    Slider("strength", "Strength", 0, 1, 0.05, 1),
    Check("all_methods", "Also output other methods")
  )

  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    val reference = v.str("reference").trim
    val fallback = Option.when(reference.isEmpty)(Support.writeTemp(ctx, img))
    val referencePath = fallback.map(_.toString).getOrElse(reference)
    try
      val method = v.str("method", "A1111")
      val r = ImageTools.run(ctx, "color_correction", img,
        "reference_path" -> ujson.Str(referencePath), "method" -> ujson.Str(method),
        "strength" -> ujson.Num(v.double("strength", 1)), "all_methods" -> ujson.Bool(v.bool("all_methods")))
      try
        val info = Seq("Color Correction" -> method, "Color Correction strength" -> Values.fmt(v.double("strength", 1)),
          "Color Correction reference" -> (if reference.isEmpty then "original input" else Paths.get(reference).getFileName.toString))
        val comparisons = r.outputs.toSeq.sortBy(_._1).map { (name, path) =>
          (Support.readImage(path), info.filterNot(_._1 == "Color Correction") :+ ("Color Correction" -> name))
        }
        (r.image, info) +: comparisons
      finally Support.deleteQuietly(r.outputs.values.toSeq*)
    finally fallback.foreach(path => Support.deleteQuietly(path))
