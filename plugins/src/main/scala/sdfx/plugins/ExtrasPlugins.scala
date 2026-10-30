package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.core.png.PngMetadata
import sdfx.plugins.imageops.ImageOps
import sdfx.plugins.imageops.ImageOps.Expansion

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import scala.util.Random

/** Pure image helpers for the extras / script plugins (unit tested). */
object ExtrasOps:
  case class Box(x: Int, y: Int, w: Int, h: Int)

  def flip(img: BufferedImage, horizontal: Boolean, vertical: Boolean): BufferedImage =
    val w = img.getWidth
    val h = img.getHeight
    val out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until h; x <- 0 until w do
      out.setRGB(if horizontal then w - 1 - x else x, if vertical then h - 1 - y else y, img.getRGB(x, y))
    out

  /** A1111 split_oversized: pieces along the long side when its ratio to the short side exceeds 1/threshold. */
  def splitBoxes(w: Int, h: Int, threshold: Double, overlap: Double): Seq[Box] =
    val inverse = w < h
    val (long, short) = if inverse then (h, w) else (w, h)
    val ratio = short.toDouble / long
    if ratio >= threshold || short <= 0 then Seq(Box(0, 0, w, h))
    else
      val n = math.ceil(long.toDouble / short / (1 - overlap) - overlap / (1 - overlap)).toInt.max(2)
      val step = (long - short).toDouble / (n - 1)
      (0 until n).map { i =>
        val o = math.round(i * step).toInt
        if inverse then Box(0, o, w, short) else Box(o, 0, short, h)
      }

  /** Largest box with the aspect ratio of tw:th that fits w x h, centred on (cx, cy) and clamped to the image. */
  def cropBox(w: Int, h: Int, tw: Int, th: Int, cx: Double, cy: Double): Box =
    val scale = math.min(w.toDouble / tw, h.toDouble / th)
    val cw = math.max(1, math.min(w, math.round(tw * scale).toInt))
    val ch = math.max(1, math.min(h, math.round(th * scale).toInt))
    val x = math.max(0, math.min(w - cw, math.round(cx - cw / 2.0).toInt))
    val y = math.max(0, math.min(h - ch, math.round(cy - ch / 2.0).toInt))
    Box(x, y, cw, ch)

  /** Auto-sized crop (processing_autosized_crop): dimensions divisible by 64 within [minDim, maxDim] whose area is
    * closest to `targetArea` and whose aspect is closest to the image's.
    */
  def autoSize(w: Int, h: Int, minDim: Int, maxDim: Int, targetArea: Int): (Int, Int) =
    val ar = w.toDouble / h
    val dims = (minDim to maxDim by 64).toSeq
    val cands = for cw <- dims; ch <- dims yield (cw, ch)
    cands.minBy((cw, ch) => (math.abs(math.log(cw.toDouble / ch / ar)), math.abs(cw * ch - targetArea).toDouble / targetArea))

  /** Focal point from edge magnitude and local entropy (A1111 focal crop without the face detector). */
  def focalPoint(img: BufferedImage, edgeWeight: Double, entropyWeight: Double): (Double, Double) =
    val small = ImageOps.resize(img, math.max(1, img.getWidth / 4), math.max(1, img.getHeight / 4))
    val w = small.getWidth
    val h = small.getHeight
    def lum(x: Int, y: Int): Double =
      val c = small.getRGB(math.max(0, math.min(w - 1, x)), math.max(0, math.min(h - 1, y)))
      0.299 * ((c >> 16) & 255) + 0.587 * ((c >> 8) & 255) + 0.114 * (c & 255)
    var sx, sy, sw = 0.0
    val win = 4
    for y <- 0 until h; x <- 0 until w do
      val gx = lum(x + 1, y) - lum(x - 1, y)
      val gy = lum(x, y + 1) - lum(x, y - 1)
      val edge = math.sqrt(gx * gx + gy * gy)
      val ent =
        if entropyWeight <= 0 then 0.0
        else
          val hist = new Array[Int](16)
          for dy <- -win to win by 2; dx <- -win to win by 2 do hist((lum(x + dx, y + dy) / 16).toInt.min(15)) += 1
          val n = hist.sum.toDouble
          -hist.filter(_ > 0).map(c => c / n * math.log(c / n)).sum
      val wgt = edge * edgeWeight + ent * entropyWeight * 20
      sx += x * wgt; sy += y * wgt; sw += wgt
    if sw <= 0 then (img.getWidth / 2.0, img.getHeight / 2.0)
    else (sx / sw * img.getWidth / w, sy / sw * img.getHeight / h)

  def crop(img: BufferedImage, b: Box): BufferedImage =
    val out = new BufferedImage(b.w, b.h, BufferedImage.TYPE_INT_RGB)
    val g = out.createGraphics()
    g.drawImage(img, -b.x, -b.y, null)
    g.dispose()
    out

  /** Outpainting mk2 fill: the new border is the smeared edge colour plus image-statistics noise that grows with the
    * distance from the original (fall-off exponent), so the sampler starts from plausible colours.
    */
  def noiseFill(img: BufferedImage, e: Expansion, colorVariation: Double, falloff: Double, seed: Long): BufferedImage =
    val canvas = ImageOps.expandCanvas(img, e)
    val w = canvas.getWidth
    val h = canvas.getHeight
    val rnd = new Random(seed)
    val px = img.getRGB(0, 0, img.getWidth, img.getHeight, null, 0, img.getWidth)
    def chan(k: Int) = px.map(c => ((c >> (16 - 8 * k)) & 255).toDouble)
    val std = (0 until 3).map { k =>
      val a = chan(k)
      val m = a.sum / a.length
      math.sqrt(a.map(v => (v - m) * (v - m)).sum / a.length)
    }
    val maxDist = math.max(1, Seq(e.left, e.right, e.top, e.bottom).max)
    for y <- 0 until h; x <- 0 until w do
      val inside = x >= e.left && x < e.left + img.getWidth && y >= e.top && y < e.top + img.getHeight
      if !inside then
        val dx = if x < e.left then e.left - x else if x >= e.left + img.getWidth then x - (e.left + img.getWidth - 1) else 0
        val dy = if y < e.top then e.top - y else if y >= e.top + img.getHeight then y - (e.top + img.getHeight - 1) else 0
        val d = math.pow(math.min(1.0, math.max(dx, dy).toDouble / maxDist), math.max(0.1, falloff))
        val c = canvas.getRGB(x, y)
        def ch(k: Int) =
          val base = ((c >> (16 - 8 * k)) & 255).toDouble
          math.max(0, math.min(255, math.round(base + rnd.nextGaussian() * std(k) * colorVariation * d).toInt))
        canvas.setRGB(x, y, (ch(0) << 16) | (ch(1) << 8) | ch(2))
    canvas

private[plugins] object Extras:
  def initImage(p: GenerationParams, plugin: String, ctx: PluginContext): (Path, BufferedImage) =
    Support.requireImg2Img(p, plugin)
    val path = Paths.get(p.img2img.get.initImage)
    (path, Support.readImage(ctx, path))

  def infoOf(path: Path): Option[String] = scala.util.Try(PngMetadata.readParameters(path)).toOption.flatten

  def save(ctx: PluginContext, img: BufferedImage, p: GenerationParams, base: Option[String], post: Seq[(String, String)]): OutputImage =
    val line = post.map((k, v) => s"$k: $v").mkString(", ")
    val text = base.map(_.trim).filter(_.nonEmpty).fold(line)(b => if line.isEmpty then b else s"$b, $line")
    ctx.saveImage(img, "extras", p.copy(width = img.getWidth, height = img.getHeight, extra = p.extra ++ post), text)

  def jsonPath(r: ujson.Value, key: String): Option[String] = r match
    case o: ujson.Obj => o.value.get(key).collect { case ujson.Str(s) => s }
    case _            => None

/** Outpainting mk2 (outpainting_mk_2.py): noise-matched fill + inpaint. */
class OutpaintMk2Plugin extends Plugin:
  def id = "outpaint_mk2"
  def name = "Outpainting mk2"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def description = "Extends the canvas with colour-matched noise and inpaints it (use high denoising, e.g. 0.8)."
  override def controls: Seq[Control] = Seq(
    IntField("pixels", "Pixels to expand", 8, 256, 128),
    IntField("mask_blur", "Mask blur", 0, 64, 8),
    Check("left", "Left", true), Check("right", "Right", true), Check("up", "Up", false), Check("down", "Down", false),
    Slider("falloff", "Fall-off exponent (lower = higher detail)", 0, 4, 0.01, 1.0),
    Slider("color_variation", "Color variation", 0, 1, 0.01, 0.05)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val (_, img) = Extras.initImage(p0, name, ctx)
    val i0 = p0.img2img.get
    val blur = v.int("mask_blur", 8)
    val e = ImageOps.expansion(img.getWidth, img.getHeight, v.int("pixels", 128),
      v.bool("left", true), v.bool("right", true), v.bool("up"), v.bool("down"))
    val canvas = Support.writeTemp(ctx, ExtrasOps.noiseFill(img, e, v.double("color_variation", 0.05), v.double("falloff", 1.0),
      Support.fixSeed(p0.seed)))
    val mask = Support.writeTemp(ctx, ImageOps.outpaintMask(e, blur))
    val p = p0.copy(width = e.canvasW, height = e.canvasH,
      img2img = Some(i0.copy(initImage = canvas.toString, mask = Some(mask.toString), maskBlur = blur, inpaintingFill = 1,
        inpaintFullRes = false, inpaintingMaskInvert = false, resizeMode = 0)),
      extra = p0.extra ++ Seq("Script" -> "Outpainting mk2", "Outpaint pixels" -> v.int("pixels", 128).toString))
    ctx.status("Outpainting mk2", 0, 1)
    try JobOutput(Support.step(ctx)(ctx.generate(p)).getOrElse(Nil))
    finally Support.deleteQuietly(canvas, mask)

/** img2img alternative test (img2imgalt.py): DDIM-inverts the init image to noise, then samples from it. */
class Img2ImgAltPlugin extends Plugin:
  def id = "img2img_alt"
  def name = "img2img alternative test"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def requiresEngineFeatures: Seq[String] = Seq("img2imgalt")
  override def description = "DDIM noise inversion: estimates the init image's noise so a changed prompt edits it (denoising 1.0)."
  override def controls: Seq[Control] = Seq(
    Note("inversion_help", "Inversion is approximate. Lower CFG and more decode steps can improve reconstruction. " +
      "Randomness 0 keeps the recovered noise; 1 uses fresh seeded noise."),
    Text("override_prompt", "Original prompt (for inversion; empty = current prompt)"),
    Text("override_negative", "Original negative prompt"),
    Slider("decode_cfg_scale", "Decode CFG scale", 0, 15, 0.1, 1.0),
    IntField("inversion_steps", "Decode steps (0 = sampling steps)", 0, 150, 50),
    Slider("randomness", "Randomness", 0, 1, 0.01, 0.0)
  )
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    Support.requireImg2Img(p0, name)
    val hook = ujson.Obj("override_prompt" -> v.str("override_prompt"), "override_negative" -> v.str("override_negative"),
      "decode_cfg_scale" -> v.double("decode_cfg_scale", 1.0), "inversion_steps" -> v.int("inversion_steps", 50),
      "randomness" -> v.double("randomness", 0.0))
    val p = p0.copy(hooks = p0.hooks.updated("img2imgalt", hook),
      extra = p0.extra ++ Seq("Script" -> "img2img alternative test",
        "Decode prompt" -> Option(v.str("override_prompt")).filter(_.nonEmpty).getOrElse(p0.prompt),
        "Decode negative prompt" -> Option(v.str("override_negative")).filter(_.nonEmpty).getOrElse(p0.negativePrompt),
        "Decode steps" -> (if v.int("inversion_steps", 50) > 0 then v.int("inversion_steps", 50) else p0.steps).toString,
        "Decode CFG scale" -> Values.fmt(v.double("decode_cfg_scale", 1.0)),
        "Randomness" -> Values.fmt(v.double("randomness", 0.0))))
    JobOutput(Support.step(ctx)(ctx.generate(p)).getOrElse(Nil))

/** Custom code (custom_code.py). Disabled unless Settings > Allow custom code (like A1111 --allow-code). */
class CustomCodePlugin extends Plugin:
  def id = "custom_code"
  def name = "Custom code"
  def kind = PluginKind.Script
  override def requiresEngineFeatures: Seq[String] = Seq("custom_code")
  override def description = "Runs Python in the engine (engine, pipe, torch, Image, params available; set `result` / `images`)."
  override def controls: Seq[Control] = Seq(
    Note("warning", "Only enabled when 'Allow custom code' is on in Settings. Code runs with your user's rights."),
    Note("scrambling_warning", "Image scrambling cannot protect files written by custom Python. Code can save readable images or other data anywhere it can access; these writes cannot be intercepted by the app."),
    Text("code", "Python code", "result = str(pipe)", multiline = true)
  )
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    if ctx.imageScramblingEnabled then
      ctx.log("Warning: custom Python can write readable files; image scrambling cannot intercept or protect those writes.")
    val r = ctx.engineCall("run_code", ujson.Obj("code" -> v.str("code")))
    val stdout = Extras.jsonPath(r, "stdout").getOrElse("")
    val result = Extras.jsonPath(r, "result").getOrElse("")
    if stdout.nonEmpty then ctx.log(stdout)
    if result.nonEmpty then ctx.log(s"result: $result")
    val imgs = r match
      case o: ujson.Obj => o.value.get("images").collect { case a: ujson.Arr => a.value.collect { case ujson.Str(s) => s } }.getOrElse(Nil)
      case _            => Nil
    JobOutput(imgs.toSeq.map(s => ctx.saveImage(Support.readImage(ctx, s), "custom", p0, s"Script: Custom code, Result: ${Infotext1(result)}")))

private[plugins] object Infotext1:
  def apply(s: String): String = s.replace("\n", " ").take(200)

/** Stable Video Diffusion (sd_forge_svd): image to animated GIF. */
class SvdPlugin extends Plugin:
  def id = "svd"
  def name = "Stable Video Diffusion"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def requiresEngineFeatures: Seq[String] = Seq("img2vid")
  override def description = "Animates the init image with an SVD model (models/svd). Saves a GIF plus the first frame."
  override def controls: Seq[Control] = Seq(
    DynamicChoice("model", "SVD model", ChoiceSource.SvdModels, "", allowNone = false),
    IntField("width", "Width", 64, 2048, 1024), IntField("height", "Height", 64, 2048, 576),
    IntField("frames", "Video frames", 2, 64, 14), IntField("fps", "FPS", 1, 60, 6),
    IntField("motion_bucket_id", "Motion bucket id", 1, 1023, 127),
    Slider("noise_aug_strength", "Augmentation level", 0, 10, 0.01, 0.02),
    IntField("steps", "Sampling steps", 1, 100, 25),
    Slider("min_cfg", "Min CFG", 0, 15, 0.1, 1.0), Slider("max_cfg", "Max CFG", 0, 15, 0.1, 3.0),
    IntField("decode_chunk_size", "Decode chunk size", 1, 64, 4)
  )
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    ctx.requirePlaintextVideoAllowed()
    val (src, _) = Extras.initImage(p0, name, ctx)
    val model = ctx.modelPath(ChoiceSource.SvdModels, v.str("model")).getOrElse(throw new IllegalArgumentException(s"SVD model not found: ${v.str("model")}"))
    val seed = Support.fixSeed(p0.seed)
    val params = ujson.Obj("model" -> model, "image_path" -> src.toString, "seed" -> seed)
    Seq("width", "height", "frames", "fps", "motion_bucket_id", "steps", "decode_chunk_size").foreach(k => params(k) = v.int(k))
    Seq("noise_aug_strength", "min_cfg", "max_cfg").foreach(k => params(k) = v.double(k))
    ctx.status("Stable Video Diffusion", 0, 1)
    val r = Support.step(ctx)(ctx.engineCall("img2vid", params))
    r.fold(JobOutput(Nil)) { res =>
      val gif = Extras.jsonPath(res, "path").map(Paths.get(_))
      val preview = Extras.jsonPath(res, "preview").map(Paths.get(_))
      val info = s"Script: Stable Video Diffusion, Model: ${v.str("model")}, Seed: $seed, Frames: ${v.int("frames")}, FPS: ${v.int("fps")}, " +
        s"Motion bucket: ${v.int("motion_bucket_id")}"
      val out = preview.map(pv => ctx.saveImage(Support.readImage(ctx, pv), "videos", p0.copy(seed = seed), info))
      for o <- out; g <- gif do Files.copy(g, o.path.resolveSibling(o.path.getFileName.toString.replaceAll("\\.png$", ".gif")),
        StandardCopyOption.REPLACE_EXISTING)
      JobOutput(out.toSeq)
    }

/** Zero123 novel view (sd_forge_z123). */
class Zero123Plugin extends Plugin:
  def id = "zero123"
  def name = "Zero123 novel view"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Img2Img)
  override def requiresEngineFeatures: Seq[String] = Seq("novel_view")
  override def description = "Renders the object in the init image from another angle (diffusers Zero123 model folder in models/zero123)."
  override def controls: Seq[Control] = Seq(
    DynamicChoice("model", "Zero123 model", ChoiceSource.Zero123Models, "", allowNone = false),
    Slider("elevation", "Elevation", -90, 90, 1, 10), Slider("azimuth", "Azimuth", -180, 180, 1, 90),
    IntField("steps", "Steps", 1, 150, 50), Slider("cfg_scale", "CFG", 0, 15, 0.1, 3.0)
  )
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val (src, _) = Extras.initImage(p0, name, ctx)
    val model = ctx.modelPath(ChoiceSource.Zero123Models, v.str("model")).getOrElse(throw new IllegalArgumentException(s"Zero123 model not found: ${v.str("model")}"))
    val seed = Support.fixSeed(p0.seed)
    val r = Support.step(ctx)(ctx.engineCall("novel_view", ujson.Obj("model" -> model, "image_path" -> src.toString,
      "elevation" -> v.double("elevation", 10), "azimuth" -> v.double("azimuth", 90), "steps" -> v.int("steps", 50),
      "cfg_scale" -> v.double("cfg_scale", 3.0), "seed" -> seed)))
    JobOutput(r.flatMap(Extras.jsonPath(_, "path")).toSeq.map { p =>
      ctx.saveImage(Support.readImage(ctx, p), "img2img", p0.copy(seed = seed),
        s"Script: Zero123, Model: ${v.str("model")}, Elevation: ${Values.fmt(v.double("elevation", 10))}, Azimuth: ${Values.fmt(v.double("azimuth", 90))}, Seed: $seed")
    })

/** Base for extras that transform the img2img init image (or each txt2img result) without diffusion. */
abstract class ExtrasPlugin extends Plugin:
  def kind = PluginKind.Script
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext): Seq[(BufferedImage, Seq[(String, String)])]

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val sources: Seq[(BufferedImage, GenerationParams, Option[String])] = p0.img2img match
      case Some(i) =>
        val path = Paths.get(i.initImage)
        Seq((Support.readImage(ctx, path), p0, Extras.infoOf(path)))
      case None => Support.step(ctx)(ctx.generate(p0)).getOrElse(Nil).map(o => (Support.readImage(ctx, o.path), o.params, Some(o.infotext)))
    val outs = scala.collection.mutable.ArrayBuffer[OutputImage]()
    var running = true
    val it = sources.zipWithIndex.iterator
    while running && it.hasNext do
      val ((img, p, info), i) = it.next()
      ctx.status(s"$name ${i + 1}/${sources.size}", i, sources.size)
      Support.step(ctx)(transform(img, p, v, ctx)) match
        case Some(res) => res.foreach((im, post) => outs += Extras.save(ctx, im, p, info, post))
        case None      => running = false
    JobOutput(outs.toSeq)

class FlippedCopiesPlugin extends ExtrasPlugin:
  def id = "flipped_copies"
  def name = "Create flipped copies"
  override def description = "Saves horizontally / vertically flipped copies (postprocessing_create_flipped_copies)."
  override def controls: Seq[Control] = Seq(Check("horizontal", "Horizontal", true), Check("vertical", "Vertical"),
    Check("both", "Both"))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    Seq(("horizontal", true, false), ("vertical", false, true), ("both", true, true)).filter((k, _, _) => v.bool(k, k == "horizontal"))
      .map((k, h, vv) => (ExtrasOps.flip(img, h, vv), Seq("Flipped" -> k)))

class SplitOversizedPlugin extends ExtrasPlugin:
  def id = "split_oversized"
  def name = "Split oversized images"
  override def description = "Splits very wide / tall images into overlapping pieces of the target size (postprocessing_split_oversized)."
  override def controls: Seq[Control] = Seq(Slider("threshold", "Split threshold", 0, 1, 0.05, 0.5),
    Slider("overlap", "Overlap ratio", 0, 0.9, 0.05, 0.2))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    ExtrasOps.splitBoxes(img.getWidth, img.getHeight, v.double("threshold", 0.5), v.double("overlap", 0.2)).zipWithIndex.map { (b, i) =>
      (ImageOps.resize(ExtrasOps.crop(img, b), p.width, p.height), Seq("Split part" -> (i + 1).toString))
    }

class AutosizedCropPlugin extends ExtrasPlugin:
  def id = "autosized_crop"
  def name = "Auto-sized crop"
  override def description = "Picks a 64-divisible size close to the target area and the image's aspect, then crops and resizes."
  override def controls: Seq[Control] = Seq(IntField("min_dim", "Dimension lower bound", 64, 2048, 384),
    IntField("max_dim", "Dimension upper bound", 64, 2048, 768),
    IntField("target_area", "Target area (pixels, 0 = width x height)", 0, 16777216, 0))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    val area = if v.int("target_area") > 0 then v.int("target_area") else p.width * p.height
    val lo = math.min(v.int("min_dim", 384), v.int("max_dim", 768))
    val (tw, th) = ExtrasOps.autoSize(img.getWidth, img.getHeight, lo, math.max(lo, v.int("max_dim", 768)), area)
    val b = ExtrasOps.cropBox(img.getWidth, img.getHeight, tw, th, img.getWidth / 2.0, img.getHeight / 2.0)
    Seq((ImageOps.resize(ExtrasOps.crop(img, b), tw, th), Seq("Auto-sized crop" -> s"${tw}x$th")))

class FocalCropPlugin extends ExtrasPlugin:
  def id = "focal_crop"
  def name = "Auto focal point crop"
  override def description = "Crops to the target width x height around the most detailed region (edges / entropy)."
  override def controls: Seq[Control] = Seq(Slider("edges_weight", "Edges weight", 0, 1, 0.05, 0.5),
    Slider("entropy_weight", "Entropy weight", 0, 1, 0.05, 0.15))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    val (cx, cy) = ExtrasOps.focalPoint(img, v.double("edges_weight", 0.5), v.double("entropy_weight", 0.15))
    val b = ExtrasOps.cropBox(img.getWidth, img.getHeight, p.width, p.height, cx, cy)
    Seq((ImageOps.resize(ExtrasOps.crop(img, b), p.width, p.height), Seq("Focal crop" -> s"${math.round(cx)},${math.round(cy)}")))

class FaceRestoreExtrasPlugin extends ExtrasPlugin:
  def id = "face_restore_extras"
  def name = "Restore faces (extras)"
  override def requiresEngineFeatures: Seq[String] = Seq("face_restore")
  override def description = "GFPGAN / CodeFormer on the init image (or on each generated image)."
  override def controls: Seq[Control] = Seq(DynamicChoice("model", "Model", ChoiceSource.FaceRestorers, "", allowNone = false),
    Slider("visibility", "Visibility", 0, 1, 0.01, 1.0), Slider("codeformer_weight", "CodeFormer weight", 0, 1, 0.01, 0.5))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    val model = ctx.modelPath(ChoiceSource.FaceRestorers, v.str("model")).getOrElse(throw new IllegalArgumentException(s"face restoration model not found: ${v.str("model")}"))
    val src = Support.writeTemp(ctx, img)
    try
      val r = ctx.engineCall("restore_faces", ujson.Obj("image_path" -> src.toString, "model" -> model,
        "visibility" -> v.double("visibility", 1.0), "codeformer_weight" -> v.double("codeformer_weight", 0.5)))
      Extras.jsonPath(r, "path").toSeq.map(pth => (Support.readImage(ctx, pth), Seq("Face restoration" -> v.str("model"),
        "Face restoration visibility" -> Values.fmt(v.double("visibility", 1.0)))))
    finally Support.deleteQuietly(src)

class CaptionPlugin extends ExtrasPlugin:
  def id = "caption"
  def name = "Caption (BLIP / WD14 tags)"
  override def requiresEngineFeatures: Seq[String] = Seq("caption")
  override def description = "Captions the image, saves a copy with the caption in its infotext and a .txt next to it."
  override def controls: Seq[Control] = Seq(Choice("method", "Method", Seq("blip", "wd14"), "blip"),
    Slider("threshold", "Tag threshold (wd14)", 0, 1, 0.01, 0.35))
  def transform(img: BufferedImage, p: GenerationParams, v: Values, ctx: PluginContext) =
    val src = Support.writeTemp(ctx, img)
    try
      val r = ctx.engineCall("caption", ujson.Obj("image_path" -> src.toString, "method" -> v.str("method", "blip"),
        "threshold" -> v.double("threshold", 0.35)))
      val text = Extras.jsonPath(r, "caption").getOrElse("")
      ctx.log(s"caption: $text")
      Seq((img, Seq("Caption" -> text)))
    finally Support.deleteQuietly(src)

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val out = super.run(p0, v, ctx)
    out.images.foreach { o =>
      o.params.extra.get("Caption").foreach(c => Files.writeString(o.path.resolveSibling(o.path.getFileName.toString.replaceAll("\\.png$", ".txt")), c))
    }
    out

