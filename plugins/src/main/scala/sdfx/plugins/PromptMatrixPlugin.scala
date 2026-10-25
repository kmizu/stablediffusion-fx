package sdfx.plugins

import sdfx.core.image.Grid
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.plugins.imageops.ImageOps

object PromptMatrixPlugin:
  /** One combination: its index (bit n = part n+1 included) and the resulting prompt. */
  case class Combo(index: Int, prompt: String, included: Seq[String])

  /** All 2^(parts-1) prompts of `a|b|c`, in A1111 order. */
  def combinations(prompt: String, putAtStart: Boolean, delimiter: String): Seq[Combo] =
    val parts = prompt.split("\\|", -1).toSeq
    val variable = parts.tail.map(_.trim.stripPrefix(",").stripSuffix(",").trim)
    (0 until (1 << variable.size)).map { c =>
      val selected = variable.zipWithIndex.collect { case (t, n) if (c & (1 << n)) != 0 => t }
      val all = if putAtStart then selected :+ parts.head.trim else parts.head.trim +: selected
      Combo(c, all.filter(_.nonEmpty).mkString(delimiter), selected)
    }

  /** Grid layout used by A1111: rows = 2^(k/2) (k = variable part count), columns = the rest. */
  def layout(variableParts: Int): (Int, Int) =
    val rows = 1 << (variableParts / 2)
    val total = 1 << variableParts
    (total / rows, rows)

/** A1111 "Prompt matrix". */
class PromptMatrixPlugin extends Plugin:
  import PromptMatrixPlugin.*
  def id = "prompt_matrix"
  def name = "Prompt matrix"
  def kind = PluginKind.Script
  override def description = "Generates every combination of the `|`-separated prompt parts."
  override def controls: Seq[Control] = Seq(
    Note("help", "Separate prompt parts with |. The first part is always used, the others are toggled."),
    Check("put_at_start", "Put variable parts at start of prompt"),
    Check("different_seeds", "Use different seed for each picture"),
    Choice("prompt_type", "Select prompt", Seq("positive", "negative"), "positive"),
    Choice("delimiter", "Select joining char", Seq("comma", "space"), "comma"),
    IntField("margin", "Grid margins (px)", 0, 500, 0)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val negative = v.str("prompt_type", "positive") == "negative"
    val delimiter = if v.str("delimiter", "comma") == "space" then " " else ", "
    val source = if negative then p0.negativePrompt else p0.prompt
    val combos = combinations(source, v.bool("put_at_start"), delimiter)
    val variable = source.split("\\|", -1).toSeq.tail.map(_.trim.stripPrefix(",").stripSuffix(",").trim)
    val seed = Support.fixSeed(p0.seed)
    val different = v.bool("different_seeds")
    ctx.log(s"Prompt matrix will create ${combos.size} images")
    val outs = scala.collection.mutable.ArrayBuffer[(Combo, Seq[OutputImage])]()
    val it = combos.iterator
    var running = true
    while running && it.hasNext do
      val c = it.next()
      val p =
        (if negative then p0.copy(negativePrompt = c.prompt) else p0.copy(prompt = c.prompt))
          .copy(seed = seed + (if different then c.index else 0), batchSize = 1, nIter = 1)
      ctx.status(s"Prompt matrix ${c.index + 1}/${combos.size}", c.index, combos.size)
      Support.step(ctx)(ctx.generate(p)) match
        case Some(o) => outs += c -> o
        case None    => running = false
    val images = outs.toSeq.flatMap(_._2)
    if images.isEmpty then return JobOutput(Nil)

    val margin = v.int("margin", 0)
    val cellsByIdx = outs.toMap.view.mapValues(o => Support.cellImage(o).map(ImageOps.withMargin(_, margin)).orNull).toMap
    val (cols, rows) = layout(variable.size)
    val cells = combos.map(c => outs.find(_._1 == c).flatMap(o => Option(cellsByIdx(o._1))).orNull)
    val colBits = variable.size - variable.size / 2
    def label(bits: Seq[Int], mask: Int): String =
      val inc = bits.zipWithIndex.collect { case (b, i) if (mask & (1 << i)) != 0 => variable(b) }
      if inc.isEmpty then "-" else inc.mkString(", ")
    val colLabels = if variable.isEmpty then Nil else (0 until cols).map(c => label(0 until colBits, c))
    val rowLabels = if rows > 1 then (0 until rows).map(r => label(colBits until variable.size, r)) else Nil
    val grid = Grid.labeled(cells, cols, colLabels, rowLabels)
    val gp = p0.copy(seed = images.head.seed, extra = p0.extra + ("Script" -> "Prompt matrix"))
    val saved = ctx.saveImage(grid, "grids", gp, Support.infotext(gp, images.head.seed))
    JobOutput(images, Seq(saved))
