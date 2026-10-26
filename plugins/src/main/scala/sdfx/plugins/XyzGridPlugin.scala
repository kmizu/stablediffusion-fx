package sdfx.plugins

import sdfx.core.image.{Grid, GridNumbers}
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.plugins.imageops.ImageOps

import java.awt.image.BufferedImage
import scala.util.matching.Regex

/** One X/Y/Z axis definition. `cost` orders the loops: higher cost axes change least often (outermost). */
case class AxisDef(
    name: String,
    kind: ValueKind,
    apply: (GenerationParams, String) => GenerationParams,
    choices: Option[ChoiceSource] = None,
    cost: Int = 0,
    showLabel: Boolean = true
):
  def label(v: String): String =
    if kind == ValueKind.Permutations then XyzValues.csv(v).mkString(", ")
    else if showLabel then s"$name: $v"
    else v

object XyzGridPlugin:
  private def int(s: String) = s.trim.toDouble.toInt
  private def long(s: String) = s.trim.toDouble.toLong
  private def dbl(s: String) = s.trim.toDouble
  private val SizeRe = """\s*(\d+)\s*[xX×]\s*(\d+)\s*""".r

  /** `apply_order` from xyz_grid.py: reorders the given tokens inside the prompt. */
  def applyOrder(prompt: String, tokens: Seq[String]): String =
    val order = tokens.map(t => (prompt.indexOf(t), t)).sortBy(_._1)
    var rest = prompt
    val parts = order.map { (_, t) =>
      val n = rest.indexOf(t)
      if n < 0 then ""
      else
        val part = rest.substring(0, n)
        rest = rest.substring(n + t.length)
        part
    }
    parts.zip(tokens).map((part, t) => part + t).mkString + rest

  /** Replaces the weight of `<lora:name:...>` in the prompt (appends the tag if absent). */
  def setLoraWeight(p: GenerationParams, value: String): GenerationParams =
    val i = value.lastIndexOf(':')
    if i <= 0 then throw new IllegalArgumentException(s"LoRA weight value must be name:weight (got '$value')")
    val name = value.substring(0, i).trim
    val w = value.substring(i + 1).trim
    val wd = w.toDouble
    val re = ("""<(lora|lyco):""" + Regex.quote(name) + """(?::[^:>]*)?((?::[^>]*)?)>""").r
    val prompt =
      if re.findFirstIn(p.prompt).isDefined then re.replaceAllIn(p.prompt, m => Regex.quoteReplacement(s"<${m.group(1)}:$name:$w${m.group(2)}>"))
      else s"${p.prompt} <lora:$name:$w>".trim
    p.copy(prompt = prompt, loras = p.loras.map(l => if l.name == name then l.copy(weight = wd) else l))

  def builtinAxes: Seq[AxisDef] = Seq(
    AxisDef("Nothing", ValueKind.Str, (p, _) => p, showLabel = false),
    AxisDef("Seed", ValueKind.IntNum, (p, v) => p.copy(seed = long(v))),
    AxisDef("Var. seed", ValueKind.IntNum, (p, v) => p.copy(subseed = long(v))),
    AxisDef("Var. strength", ValueKind.FloatNum, (p, v) => p.copy(subseedStrength = dbl(v))),
    AxisDef("Steps", ValueKind.IntNum, (p, v) => p.copy(steps = int(v))),
    AxisDef("Hires steps", ValueKind.IntNum, (p, v) => p.copy(hires = p.hires.copy(secondPassSteps = int(v)))),
    AxisDef("CFG Scale", ValueKind.FloatNum, (p, v) => p.copy(cfgScale = dbl(v))),
    AxisDef("Prompt S/R", ValueKind.Str, (p, v) => p, showLabel = false), // replaced per job (needs the search term)
    AxisDef("Prompt order", ValueKind.Permutations, (p, v) => p.copy(prompt = applyOrder(p.prompt, XyzValues.csv(v)))),
    AxisDef("Sampler", ValueKind.Str, (p, v) => p.copy(samplerName = v), Some(ChoiceSource.Samplers), showLabel = false),
    AxisDef("Hires sampler", ValueKind.Str, (p, v) => p.copy(hires = p.hires.copy(samplerName = Some(v))), Some(ChoiceSource.Samplers)),
    AxisDef("Schedule type", ValueKind.Str, (p, v) => p.copy(scheduler = v), Some(ChoiceSource.Schedulers)),
    AxisDef("Checkpoint name", ValueKind.Str, (p, v) => p.copy(checkpoint = Some(v)), Some(ChoiceSource.Checkpoints), cost = 3, showLabel = false),
    AxisDef(
      "VAE",
      ValueKind.Str,
      (p, v) => p.copy(vae = if v.equalsIgnoreCase("Automatic") then None else Some(v)),
      Some(ChoiceSource.Vaes),
      cost = 2
    ),
    AxisDef(
      "Denoising",
      ValueKind.FloatNum,
      (p, v) =>
        p.img2img match
          case Some(i) => p.copy(img2img = Some(i.copy(denoisingStrength = dbl(v))))
          case None    => p.copy(hires = p.hires.copy(denoisingStrength = dbl(v)))
    ),
    AxisDef("Hires upscaler", ValueKind.Str, (p, v) => p.copy(hires = p.hires.copy(upscaler = v)), Some(ChoiceSource.HiresUpscalers)),
    AxisDef("Hires upscale", ValueKind.FloatNum, (p, v) => p.copy(hires = p.hires.copy(scale = dbl(v), resizeX = 0, resizeY = 0))),
    AxisDef("Clip skip", ValueKind.IntNum, (p, v) => p.copy(clipSkip = int(v))),
    AxisDef(
      "Size",
      ValueKind.Str,
      (p, v) =>
        v match
          case SizeRe(w, h) => p.copy(width = w.toInt, height = h.toInt)
          case _            => throw new IllegalArgumentException(s"size must be WxH (got '$v')")
    ),
    AxisDef("LoRA weight", ValueKind.Str, setLoraWeight, Some(ChoiceSource.Loras), cost = 1)
  )

  def fromXyzAxis(a: XyzAxis): AxisDef =
    AxisDef(a.name, if a.numeric then ValueKind.FloatNum else ValueKind.Str, (p, v) => a.apply(p, v), a.choices)

  /** A planned cell: indices into the x/y/z value lists. */
  case class Cell(x: Int, y: Int, z: Int)

  /** Generation order: axes with higher cost are iterated outermost (ties: Z outside Y outside X). */
  def order(costs: (Int, Int, Int), sizes: (Int, Int, Int)): Seq[Cell] =
    val (cx, cy, cz) = costs
    val (nx, ny, nz) = sizes
    // (axis id, cost, tie rank) sorted outermost first
    val axes = Seq((0, cx, 0), (1, cy, 1), (2, cz, 2)).sortBy(a => (-a._2, -a._3)).map(_._1)
    val n = Array(nx, ny, nz)
    for
      a <- 0 until n(axes(0))
      b <- 0 until n(axes(1))
      c <- 0 until n(axes(2))
    yield
      val idx = Array(0, 0, 0)
      idx(axes(0)) = a
      idx(axes(1)) = b
      idx(axes(2)) = c
      Cell(idx(0), idx(1), idx(2))

/** A1111 "X/Y/Z plot". */
class XyzGridPlugin(extraAxes: () => Seq[XyzAxis] = () => Nil) extends Plugin:
  import XyzGridPlugin.*

  def id = "xyz_grid"
  def name = "X/Y/Z plot"
  def kind = PluginKind.Script
  override def description = "Generates a grid of images varying up to three parameters."

  def axes: Seq[AxisDef] =
    val builtins = builtinAxes
    builtins ++ extraAxes().filterNot(a => builtins.exists(_.name == a.name)).map(fromXyzAxis)

  private def axisNames: Seq[String] = axes.map(_.name)

  override def controls: Seq[Control] =
    val names = axisNames
    Seq(
      Choice("x_type", "X type", names, "Seed"),
      Text("x_values", "X values", "", placeholder = "1-5, 7, 9 or 1-10 (+2) or 0.5-1.5 [5]"),
      Choice("y_type", "Y type", names, "Nothing"),
      Text("y_values", "Y values"),
      Choice("z_type", "Z type", names, "Nothing"),
      Text("z_values", "Z values"),
      Check("draw_legend", "Draw legend", true),
      Check("keep_random_seed", "Keep -1 for seeds"),
      Check("include_sub_images", "Include sub images"),
      Check("include_sub_grids", "Include sub grids"),
      IntField("margin", "Grid margins (px)", 0, 500, 0)
    )

  private def findAxis(n: String): AxisDef =
    axes.find(_.name == n).orElse(axes.find(_.name.equalsIgnoreCase(n))).getOrElse(throw new IllegalArgumentException(s"unknown X/Y/Z axis '$n'"))

  /** Resolved axis: definition, its values, and an apply function bound to the base params. */
  case class Resolved(axis: AxisDef, values: Seq[String], applyFn: (GenerationParams, String) => GenerationParams):
    def labels: Seq[String] = values.map(axis.label)

  def resolve(typeName: String, text: String, base: GenerationParams, keepRandom: Boolean): Resolved =
    val axis = findAxis(typeName)
    if axis.name == "Nothing" then Resolved(axis, Seq(""), (p, _) => p)
    else
      var values = XyzValues.parse(text, axis.kind)
      if values.isEmpty then throw new IllegalArgumentException(s"no values for axis '${axis.name}'")
      if !keepRandom && (axis.name == "Seed" || axis.name == "Var. seed") then
        values = values.map(v => if v.trim.toLong < 0 then Support.fixSeed(-1).toString else v)
      val applyFn: (GenerationParams, String) => GenerationParams =
        if axis.name == "Prompt S/R" then
          val search = values.head
          if !base.prompt.contains(search) && !base.negativePrompt.contains(search) then
            throw new IllegalArgumentException(s"Prompt S/R did not find '$search' in prompt or negative prompt")
          (p, v) => p.copy(prompt = p.prompt.replace(search, v), negativePrompt = p.negativePrompt.replace(search, v))
        else axis.apply
      values.foreach(v => applyFn(base, v)) // validate early (bad sizes, numbers...)
      Resolved(axis, values, applyFn)

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val keepRandom = v.bool("keep_random_seed")
    val base = if keepRandom then p0 else p0.copy(seed = Support.fixSeed(p0.seed), subseed = Support.fixSeed(p0.subseed))
    val rx = resolve(v.str("x_type", "Nothing"), v.str("x_values"), base, keepRandom)
    val ry = resolve(v.str("y_type", "Nothing"), v.str("y_values"), base, keepRandom)
    val rz = resolve(v.str("z_type", "Nothing"), v.str("z_values"), base, keepRandom)
    val (nx, ny, nz) = (rx.values.size, ry.values.size, rz.values.size)
    val cells = order((rx.axis.cost, ry.axis.cost, rz.axis.cost), (nx, ny, nz))
    ctx.log(s"X/Y/Z plot will create ${cells.size} cells (${nx}x${ny}x$nz), ${cells.size * base.imagesPerJob} images")

    val results = scala.collection.mutable.Map[Cell, (Seq[OutputImage], GenerationParams)]()
    var done = 0
    val it = cells.iterator
    var running = true
    while running && it.hasNext do
      val c = it.next()
      val p = rz.applyFn(ry.applyFn(rx.applyFn(base, rx.values(c.x)), ry.values(c.y)), rz.values(c.z))
      ctx.status(s"X/Y/Z plot ${done + 1}/${cells.size}", done, cells.size)
      Support.step(ctx)(ctx.generate(p)) match
        case Some(outs) =>
          // Attach axis captions to the returned descriptors, not the already-saved source PNGs. The grid's
          // source mapping persists these captions. Loop order can change for model-switch cost; cell indices do not.
          val axisInfo = Seq(("X", rx, c.x), ("Y", ry, c.y), ("Z", rz, c.z))
            .filter(_._2.axis.name != "Nothing").map { (name, r, index) =>
              val value = if r.axis.kind == ValueKind.Permutations then XyzValues.csv(r.values(index)).mkString(", ") else r.values(index)
              s"Grid $name info" -> s"${r.axis.name}: $value"
            }.toMap
          val labeled = outs.map(o => if GridNumbers.enabled(o.params.extra) then o.copy(params = o.params.copy(extra = o.params.extra ++ axisInfo)) else o)
          results(c) = (labeled, p)
          done += 1
        case None => running = false
    ctx.status("X/Y/Z plot", done, cells.size)

    val allImages = cells.flatMap(c => results.get(c).toSeq.flatMap(_._1))
    if allImages.isEmpty then return JobOutput(Nil)

    val legend = v.bool("draw_legend", true)
    val margin = v.int("margin", 0)
    val cellImgs: Map[Cell, BufferedImage] =
      results.view.mapValues(r => Support.cellImage(r._1)).collect { case (k, Some(i)) => k -> ImageOps.withMargin(i, margin) }.toMap
    val firstParams = cells.flatMap(results.get).head._2
    val firstSeed = allImages.head.seed

    val xyzExtra: Seq[(String, String)] =
      Seq("Script" -> "X/Y/Z plot") ++
        Seq(("X", rx), ("Y", ry), ("Z", rz)).filter(_._2.axis.name != "Nothing").flatMap { (n, r) =>
          Seq(s"$n Type" -> r.axis.name, s"$n Values" -> XyzValues.toCsv(r.values))
        }
    def sources(z: Int): Seq[OutputImage] =
      (for y <- 0 until ny; x <- 0 until nx yield results.get(Cell(x, y, z)).flatMap(_._1.headOption)).flatten
    def save(img: BufferedImage, p: GenerationParams, seed: Long, visible: Seq[OutputImage]): OutputImage =
      val gp = Support.gridParams(p.copy(extra = p.extra ++ xyzExtra), visible)
      ctx.saveImage(img, "grids", gp, Support.infotext(gp, seed))

    val zGrids: Seq[(BufferedImage, Int)] = (0 until nz).flatMap { z =>
      val imgs = for y <- 0 until ny; x <- 0 until nx yield cellImgs.getOrElse(Cell(x, y, z), null)
      if imgs.forall(_ == null) then None
      else
        val colLabels = if legend && rx.axis.name != "Nothing" then rx.labels else Nil
        val rowLabels = if legend && ry.axis.name != "Nothing" then ry.labels else Nil
        Some(Grid.labeled(imgs, nx, colLabels, rowLabels) -> z)
    }

    val grids: Seq[OutputImage] =
      if nz == 1 then zGrids.map((g, z) => save(g, firstParams, firstSeed, sources(z)))
      else
        val zLabels = if legend then zGrids.map((_, z) => rz.labels(z)) else Nil
        val combined = Grid.labeled(zGrids.map(_._1), zGrids.size, zLabels, Nil)
        val main = save(combined, firstParams, firstSeed, zGrids.flatMap((_, z) => sources(z)))
        val subs =
          if v.bool("include_sub_grids") then
            zGrids.map { (g, z) =>
              val zp = cells.filter(_.z == z).flatMap(results.get).headOption.map(_._2).getOrElse(firstParams)
              save(g, zp, firstSeed, sources(z))
            }
          else Nil
        main +: subs

    JobOutput(if v.bool("include_sub_images") then allImages else Nil, grids)
