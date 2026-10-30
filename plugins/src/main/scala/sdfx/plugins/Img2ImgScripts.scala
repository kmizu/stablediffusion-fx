package sdfx.plugins

import sdfx.core.image.Grid
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.core.png.PngMetadata
import sdfx.plugins.imageops.{ImageOps, Tile}

import java.awt.image.BufferedImage
import java.nio.file.{Path, Paths}
import scala.collection.immutable.ListMap

object LoopbackPlugin:
  /** A1111 `calculate_denoising_strength(loop)`. */
  def denoising(loop: Int, loops: Int, initial: Double, finalStrength: Double, curve: String): Double =
    if loops <= 1 then initial
    else
      val progress = loop.toDouble / (loops - 1)
      val s = curve match
        case "Aggressive" => math.sin(progress * math.Pi * 0.5)
        case "Lazy"       => 1 - math.cos(progress * math.Pi * 0.5)
        case _            => progress
      initial + (finalStrength - initial) * s

/** A1111 "Loopback": feeds each output back as the next init image.
  *
  * A1111 semantics: `n_iter` outer repetitions; every iteration processes the whole batch (member k uses seed + k)
  * and the next iteration uses seed + 1. Img2ImgParams holds one init image, so each batch member is its own
  * generation (batch size 1) chaining its own output.
  */
class LoopbackPlugin extends Plugin:
  def id = "loopback"
  def name = "Loopback"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Runs img2img repeatedly, using each output as the next input."
  override def controls: Seq[Control] = Seq(
    IntField("loops", "Loops", 1, 32, 4),
    Slider("final_denoising_strength", "Final denoising strength", 0, 1, 0.01, 0.5),
    Choice("denoising_curve", "Denoising strength curve", Seq("Aggressive", "Linear", "Lazy"), "Linear"),
    Check("make_grid", "Make a grid of all iterations", true)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val loops = math.max(1, v.int("loops", 4))
    val finalS = v.double("final_denoising_strength", 0.5)
    val curve = v.str("denoising_curve", "Linear")
    val i0 = p0.img2img.get
    val initial = i0.denoisingStrength
    val extra = p0.extra ++ Seq(
      "Loopback loops" -> loops.toString,
      "Final denoising strength" -> Values.fmt(finalS),
      "Denoising curve" -> curve
    )
    val batchSize = math.max(1, p0.batchSize)
    val nIter = math.max(1, p0.nIter)
    val total = nIter * loops * batchSize
    var seed = Support.fixSeed(p0.seed)
    // one row per chain (n, k), one column per iteration
    val rows = scala.collection.mutable.ArrayBuffer[scala.collection.mutable.ArrayBuffer[OutputImage]]()
    var done = 0
    var running = true
    var n = 0
    while running && n < nIter do
      val chains = Seq.fill(batchSize)(scala.collection.mutable.ArrayBuffer[OutputImage]())
      rows ++= chains
      val inits = Array.fill(batchSize)(i0.initImage)
      var i = 0
      while running && i < loops do
        val ii0 = i0.copy(
          denoisingStrength = LoopbackPlugin.denoising(i, loops, initial, finalS, curve),
          inpaintingFill = if i > 0 then 1 else i0.inpaintingFill // A1111: "original" after the first loop
        )
        var k = 0
        while running && k < batchSize do
          val p = p0.copy(img2img = Some(ii0.copy(initImage = inits(k))), seed = seed + k, batchSize = 1, nIter = 1, extra = extra)
          ctx.status(s"Loopback ${n + 1}/$nIter, iteration ${i + 1}/$loops", done, total)
          Support.step(ctx)(ctx.generate(p)).flatMap(_.headOption) match
            case Some(o) =>
              chains(k) += o
              inits(k) = o.path.toString
              done += 1
            case None => running = false
          k += 1
        seed += 1
        i += 1
      n += 1
    val images = rows.toSeq.flatMap(_.toSeq)
    val grids =
      if v.bool("make_grid", true) && images.size > 1 then
        val filled = rows.toSeq.filter(_.nonEmpty)
        val cols = filled.map(_.size).max
        val cells = filled.flatMap(r => r.toSeq.map(o => Support.gridTile(ctx, o)) ++ Seq.fill(cols - r.size)(null))
        val gp = Support.gridParams(p0.copy(extra = extra + ("Script" -> "Loopback")), images)
        Seq(ctx.saveImage(Grid.labeled(cells, cols, Nil, Nil), "grids", gp, Support.infotext(gp, images.head.seed)))
      else Nil
    JobOutput(images, grids)

/** A1111 "SD upscale": upscale, then img2img every tile and blend them back. Tile generations are transient: their
  * files are deleted after reading (the core PluginContext has no non-persisting generate).
  */
class SdUpscalePlugin extends Plugin:
  def id = "sd_upscale"
  def name = "SD upscale"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Upscales the image, then refines it tile by tile (tile size = width x height)."
  override def controls: Seq[Control] = Seq(
    Note("help", "Will upscale the image by the selected scale factor; use width and height to set tile size"),
    IntField("overlap", "Tile overlap", 0, 256, 64),
    DynamicChoice("upscaler", "Upscaler", ChoiceSource.Upscalers, "None", allowNone = true),
    Slider("scale", "Scale Factor", 1, 4, 0.05, 2.0)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val i0 = p0.img2img.get
    val overlap = v.int("overlap", 64)
    val upscaler = v.str("upscaler", "None")
    val scale = v.double("scale", 2.0)
    val upscaledPath = Support.step(ctx)(ctx.upscale(Paths.get(i0.initImage), upscaler, scale)) match
      case Some(path) => path
      case None       => return JobOutput(Nil)
    val img =
      try Support.readImage(ctx, upscaledPath)
      finally if upscaledPath.toString != i0.initImage then Support.deleteQuietly(upscaledPath)
    val tiles = ImageOps.splitTiles(img.getWidth, img.getHeight, p0.width, p0.height, overlap)
    val upscaleCount = math.max(1, p0.nIter) // A1111: batch count = number of upscales, batch size groups tiles
    val total = tiles.size * upscaleCount
    ctx.log(s"SD upscaling will process ${tiles.size} tiles x $upscaleCount upscale(s)")
    val seed0 = Support.fixSeed(p0.seed)
    val extra = p0.extra ++ Seq("SD upscale overlap" -> overlap.toString, "SD upscale upscaler" -> upscaler)
    val results = scala.collection.mutable.ArrayBuffer[OutputImage]()
    var done = 0
    var running = true
    var n = 0
    while running && n < upscaleCount do
      val startSeed = seed0 + n
      val processed = scala.collection.mutable.ArrayBuffer[(Tile, BufferedImage)]()
      val it = tiles.zipWithIndex.iterator
      while running && it.hasNext do
        val (t, ti) = it.next()
        val tilePath = Support.writeTemp(ctx, ImageOps.crop(img, t))
        val p = p0.copy(
          img2img = Some(i0.copy(initImage = tilePath.toString, mask = None, resizeMode = 0)),
          width = t.w,
          height = t.h,
          seed = startSeed + ti, // A1111: p.seed = processed.seed + 1 after every tile
          batchSize = 1,
          nIter = 1,
          extra = extra
        )
        ctx.status(s"SD upscale tile ${done + 1}/$total", done, total)
        try
          Support.step(ctx)(ctx.generateTransient(p)) match
            case Some(outs) =>
              outs.headOption.foreach(o => processed += t -> Support.readImage(ctx, o.path))
              Support.deleteQuietly(outs.map(_.path)*) // transient tiles: the caller deletes them
              done += 1
            case None => running = false
        finally Support.deleteQuietly(tilePath)
      if running && processed.size == tiles.size then
        val merged = ImageOps.mergeTiles(img.getWidth, img.getHeight, processed.toSeq, overlap)
        val gp = p0.copy(width = img.getWidth, height = img.getHeight, seed = startSeed, extra = extra + ("Script" -> "SD upscale"))
        results += ctx.saveImage(merged, "img2img", gp, Support.infotext(gp, startSeed))
      n += 1
    JobOutput(results.toSeq)

/** Poor man's outpainting: expands the canvas and inpaints the new borders in one pass. */
class OutpaintPlugin extends Plugin:
  def id = "outpaint"
  def name = "Poor man's outpainting"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Extends the image in the chosen directions and inpaints the new area."
  val fills: Seq[String] = Seq("fill", "original", "latent noise", "latent nothing")
  override def controls: Seq[Control] = Seq(
    IntField("pixels", "Pixels to expand", 8, 256, 128),
    IntField("mask_blur", "Mask blur", 0, 64, 4),
    Choice("fill", "Masked content", fills, "fill"),
    Check("left", "Left", true),
    Check("right", "Right", true),
    Check("up", "Up", true),
    Check("down", "Down", true)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val i0 = p0.img2img.get
    val img = Support.readImage(ctx, i0.initImage)
    val blur = v.int("mask_blur", 4)
    val e = ImageOps.expansion(
      img.getWidth, img.getHeight, v.int("pixels", 128),
      v.bool("left", true), v.bool("right", true), v.bool("up", true), v.bool("down", true)
    )
    val canvas = Support.writeTemp(ctx, ImageOps.expandCanvas(img, e))
    val mask = Support.writeTemp(ctx, ImageOps.outpaintMask(e, blur))
    val fillIdx = math.max(0, fills.indexOf(v.str("fill", "fill")))
    val dirs = Seq("left", "right", "up", "down").filter(v.bool(_, true))
    val p = p0.copy(
      width = e.canvasW,
      height = e.canvasH,
      img2img = Some(i0.copy(initImage = canvas.toString, mask = Some(mask.toString), maskBlur = blur, inpaintingFill = fillIdx,
        inpaintFullRes = false, inpaintingMaskInvert = false, resizeMode = 0)),
      extra = p0.extra ++ Seq("Script" -> "Poor man's outpainting", "Outpaint pixels" -> v.int("pixels", 128).toString,
        "Outpaint directions" -> dirs.mkString(" "))
    )
    ctx.status("Outpainting", 0, 1)
    try JobOutput(Support.step(ctx)(ctx.generate(p)).getOrElse(Nil))
    finally Support.deleteQuietly(canvas, mask)

/** Upscales the img2img init image, or every txt2img result, with an upscaler model (A1111 "Extras" as a script). */
class UpscaleExtrasPlugin extends Plugin:
  def id = "upscale_extras"
  def name = "Upscale (extras)"
  def kind = PluginKind.Script
  override def description = "img2img: upscales the init image. txt2img: generates, then upscales every image."
  override def controls: Seq[Control] = Seq(
    DynamicChoice("upscaler", "Upscaler", ChoiceSource.Upscalers, "", allowNone = false),
    Slider("scale", "Scale by", 1, 8, 0.05, 2.0)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val upscaler = v.str("upscaler")
    val scale = v.double("scale", 2.0)
    val post = ListMap("Postprocess upscale by" -> Values.fmt(scale), "Postprocess upscaler" -> upscaler)
    def postText(base: Option[String]): String =
      val line = post.map((k, x) => s"$k: $x").mkString(", ")
      base.map(_.trim).filter(_.nonEmpty).fold(line)(b => s"$b, $line")
    def up(src: Path, params: GenerationParams, info: Option[String]): Option[OutputImage] =
      Support.step(ctx)(ctx.upscale(src, upscaler, scale)).map { tmp =>
        val img =
          try Support.readImage(ctx, tmp)
          finally if tmp != src then Support.deleteQuietly(tmp)
        val gp = params.copy(width = img.getWidth, height = img.getHeight, extra = params.extra ++ post)
        ctx.saveImage(img, "extras", gp, postText(info))
      }
    p0.img2img match
      case Some(i) =>
        val src = Paths.get(i.initImage)
        val info = scala.util.Try(PngMetadata.readParameters(src)).toOption.flatten
        ctx.status("Upscaling", 0, 1)
        JobOutput(up(src, p0, info).toSeq)
      case None =>
        val gen = Support.step(ctx)(ctx.generate(p0)).getOrElse(Nil)
        val outs = scala.collection.mutable.ArrayBuffer[OutputImage]()
        val it = gen.zipWithIndex.iterator
        var running = true
        while running && it.hasNext do
          val (o, i) = it.next()
          ctx.status(s"Upscaling ${i + 1}/${gen.size}", i, gen.size)
          up(o.path, o.params, Some(o.infotext)) match
            case Some(u) => outs += u
            case None    => running = false
        JobOutput(outs.toSeq)
