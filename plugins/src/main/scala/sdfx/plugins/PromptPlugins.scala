package sdfx.plugins

import sdfx.core.config.AppHome
import sdfx.core.image.{Gif, Grid}
import sdfx.core.model.{GenerationParams, LoraSyntax}
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.plugins.imageops.ImageOps
import sdfx.plugins.prompt.DynamicPrompt

import java.nio.file.{Files, Path, Paths}
import scala.util.Random

/** Helpers for scripts that run a list of generations and optionally build a grid / GIF. */
private[plugins] object Sequence:
  final case class Item(params: GenerationParams, label: String)

  def run(ctx: PluginContext, items: Seq[Item], title: String): Seq[(Item, Seq[OutputImage])] =
    val out = scala.collection.mutable.ArrayBuffer[(Item, Seq[OutputImage])]()
    val it = items.zipWithIndex.iterator
    var running = true
    while running && it.hasNext do
      val (item, i) = it.next()
      ctx.status(s"$title ${i + 1}/${items.size}", i, items.size)
      Support.step(ctx)(ctx.generate(item.params)) match
        case Some(o) => out += item -> o
        case None    => running = false
    out.toSeq

  def grid(ctx: PluginContext, done: Seq[(Item, Seq[OutputImage])], base: GenerationParams, script: String,
      cols: Int = 0, labels: Boolean = true): Seq[OutputImage] =
    val cells = done.flatMap((item, outs) => Support.cellImage(outs).map(_ -> item.label))
    if cells.size < 2 then Nil
    else
      val c = if cols > 0 then cols else Grid.autoColumns(cells.size)
      val img =
        if labels then Grid.labeled(cells.map(_._1), c, cells.map(_._2).take(c), Nil, None)
        else Grid.plain(cells.map(_._1), c)
      val gp = base.copy(seed = done.head._2.head.seed, extra = base.extra + ("Script" -> script))
      Seq(ctx.saveImage(img, "grids", gp, Support.infotext(gp, gp.seed)))

  /** Saves an animated GIF next to the first saved image (`<name>.gif`) and returns its path. */
  def gif(done: Seq[(Item, Seq[OutputImage])], frameMillis: Int, pingPong: Boolean): Option[Path] =
    val frames = done.flatMap(_._2.headOption).map(o => Support.readImage(o.path))
    if frames.size < 2 then None
    else
      val seq = if pingPong then frames ++ frames.reverse.tail.init else frames
      val first = done.head._2.head.path
      Some(Gif.write(seq, first.resolveSibling(first.getFileName.toString.replaceAll("\\.png$", "") + "-anim.gif"), frameMillis))

private[plugins] object Wildcards:
  def dirs(extra: String): Seq[Path] =
    val home = AppHome.detect().root
    (Seq(home.resolve("wildcards"), home.resolve("models").resolve("wildcards")) ++
      extra.split("[;\\r\\n]+").map(_.trim).filter(_.nonEmpty).map(Paths.get(_))).distinct

// -------------------------------------------------------------------------------------------- Dynamic Prompts
/** sd-dynamic-prompts + stable-diffusion-webui-wildcards: `{a|b}`, `{2$$a|b|c}`, `__wildcard__`, random or
  * combinatorial, one generation per image so every image gets its own prompt (as the original). */
class DynamicPromptsPlugin extends Plugin:
  def id = "dynamic_prompts"
  def name = "Dynamic Prompts / Wildcards"
  def kind = PluginKind.Script
  override def description = "Template prompts: {a|b|c}, {2$$a|b}, weights 2::a, and __wildcards__ from wildcards/*.txt."
  override def controls: Seq[Control] = Seq(
    Choice("mode", "Mode", Seq("random", "combinatorial"), "random"),
    IntField("max_images", "Max images (combinatorial)", 1, 1000, 100),
    Check("fixed_seed", "Fixed seed (same seed for every prompt)"),
    Check("unlink_negative", "Expand the negative prompt independently"),
    Text("wildcard_dirs", "Extra wildcard folders (; separated)"),
    Note("help", "Wildcard files live in stablediffusion-fx/wildcards (one option per line, # comments).")
  )

  def expand(p0: GenerationParams, v: Values): Seq[GenerationParams] =
    val wc = new DynamicPrompt.Wildcards(Wildcards.dirs(v.str("wildcard_dirs")))
    val seed = Support.fixSeed(p0.seed)
    val n = p0.batchSize * p0.nIter
    val fixed = v.bool("fixed_seed")
    val prompts =
      if v.str("mode", "random") == "combinatorial" then DynamicPrompt.combinatorial(p0.prompt, wc, v.int("max_images", 100))
      else (0 until n).map(i => DynamicPrompt.random(p0.prompt, new Random(seed + i), wc))
    prompts.zipWithIndex.map { (pr, i) =>
      val sd = if fixed then seed else seed + i
      val neg = DynamicPrompt.random(p0.negativePrompt, new Random(if v.bool("unlink_negative") then sd else seed), wc)
      p0.copy(prompt = pr, negativePrompt = neg, seed = sd, batchSize = 1, nIter = 1,
        extra = p0.extra + ("Template" -> p0.prompt.take(500)))
    }

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val items = expand(p0, v).map(p => Sequence.Item(p, p.prompt))
    ctx.log(s"Dynamic Prompts: ${items.size} prompt(s)")
    JobOutput(Sequence.run(ctx, items, "Dynamic Prompts").flatMap(_._2))

// ------------------------------------------------------------------------------------------------- Randomize
/** Randomize (sd-webui-randomize): random sampler / scheduler / steps / CFG / size per generation. */
class RandomizePlugin extends Plugin:
  def id = "randomize"
  def name = "Randomize"
  def kind = PluginKind.AlwaysOn
  override def description = "Picks sampler, schedule, steps, CFG and size randomly per generation (seeded)."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Text("samplers", "Samplers (comma separated, empty = keep)", "Euler a, DPM++ 2M, DPM++ SDE"),
    Text("schedulers", "Schedule types", ""),
    Text("steps", "Steps range (min-max)", "20-30"),
    Text("cfg", "CFG range (min-max)", "5-8"),
    Text("sizes", "Sizes (WxH, comma separated)", "")
  )
  private def range(s: String): Option[(Double, Double)] =
    s.split("-").map(_.trim).filter(_.nonEmpty).flatMap(_.toDoubleOption) match
      case Array(a, b) => Some((math.min(a, b), math.max(a, b)))
      case Array(a)    => Some((a, a))
      case _           => None
  private def list(s: String) = s.split(",").map(_.trim).filter(_.nonEmpty).toSeq

  override def process(p: GenerationParams, v: Values): GenerationParams =
    val rnd = new Random(if p.seed >= 0 then p.seed else System.nanoTime())
    def pick(xs: Seq[String]) = Option.when(xs.nonEmpty)(xs(rnd.nextInt(xs.size)))
    val sampler = pick(list(v.str("samplers"))).getOrElse(p.samplerName)
    val sched = pick(list(v.str("schedulers"))).getOrElse(p.scheduler)
    val steps = range(v.str("steps")).map((a, b) => a.toInt + rnd.nextInt(b.toInt - a.toInt + 1)).getOrElse(p.steps)
    val cfg = range(v.str("cfg")).map((a, b) => math.round((a + rnd.nextDouble() * (b - a)) * 2) / 2.0).getOrElse(p.cfgScale)
    val sizes = new RandomResolutionsPlugin().parse(v.str("sizes"))
    val (w, h) = if sizes.isEmpty then (p.width, p.height) else sizes(rnd.nextInt(sizes.size))
    p.copy(samplerName = sampler, scheduler = sched, steps = steps, cfgScale = cfg, width = w, height = h,
      extra = p.extra + ("Randomize" -> "True"))

// ---------------------------------------------------------------------------------------------- Prompt Format
object PromptText:
  /** prompt-format extensions: collapse whitespace and commas, trim, remove empty tags. */
  def format(s: String, dedupe: Boolean): String =
    val lines = s.split("\n", -1).map { line =>
      val tags = line.split(",").map(_.replaceAll("\\s+", " ").trim).filter(_.nonEmpty)
      val unique = if dedupe then tags.distinctBy(_.toLowerCase) else tags
      unique.mkString(", ")
    }
    lines.mkString("\n").replaceAll("\\(\\s+", "(").replaceAll("\\s+\\)", ")").trim

  /** Novel AI emphasis to A1111: {x} = 1.05x, [x] = 1/1.05x (nested multiply), `\{` literals kept. */
  def fromNovelAI(s: String): String =
    // stack of (opening bracket, text collected inside it); index 0 is the top level
    val frames = scala.collection.mutable.ArrayBuffer[(Char, StringBuilder)](('-', new StringBuilder))
    s.foreach {
      case c @ ('{' | '[') => frames += ((c, new StringBuilder))
      case c @ ('}' | ']') if frames.size > 1 && frames.last._1 == (if c == '}' then '{' else '[') =>
        val (open, inner) = frames.remove(frames.size - 1)
        frames.last._2.append(reweight(inner.toString, if open == '{' then 1.05 else 1 / 1.05))
      case c => frames.last._2.append(c)
    }
    // unbalanced openers are kept literally
    while frames.size > 1 do
      val (open, inner) = frames.remove(frames.size - 1)
      frames.last._2.append(open).append(inner)
    frames.head._2.toString

  private val Weighted = """^\((.*):([0-9.]+)\)$""".r
  private def reweight(inner: String, w: Double): String = inner match
    case Weighted(t, x) => s"($t:${fmt(x.toDouble * w)})"
    case t              => s"($t:${fmt(w)})"
  private def fmt(d: Double) = BigDecimal(d).setScale(3, BigDecimal.RoundingMode.HALF_UP).bigDecimal.stripTrailingZeros.toPlainString

  /** Wraps the whole text with a weight (negative prompt weight). */
  def weightAll(s: String, w: Double): String = if s.trim.isEmpty || w == 1.0 then s else s"(${s.trim}:${fmt(w)})"

class PromptFormatPlugin extends Plugin:
  def id = "prompt_format"
  def name = "Prompt Format"
  def kind = PluginKind.AlwaysOn
  override def description = "Removes duplicate spaces / commas (and optionally duplicate tags) before generating."
  override def controls: Seq[Control] = Seq(Check("enabled", "Enabled"), Check("dedupe", "Remove duplicate tags", true),
    Check("negative", "Also format the negative prompt", true))
  override def process(p: GenerationParams, v: Values): GenerationParams =
    p.copy(prompt = PromptText.format(p.prompt, v.bool("dedupe", true)),
      negativePrompt = if v.bool("negative", true) then PromptText.format(p.negativePrompt, v.bool("dedupe", true)) else p.negativePrompt)

/** Prompt filters / Diffusion Defender: banned words and find-replace rules. */
class PromptFilterPlugin extends Plugin:
  def id = "prompt_filter"
  def name = "Prompt Filter"
  def kind = PluginKind.AlwaysOn
  override def description = "Removes banned words and applies find => replace rules to both prompts."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Text("banned", "Banned words (comma separated)", ""),
    Text("replace", "Replace rules (one per line: find => replace)", "", multiline = true),
    Check("dedupe", "Remove duplicate tags", true)
  )
  def filter(s: String, v: Values): String =
    val banned = v.str("banned").split(",").map(_.trim.toLowerCase).filter(_.nonEmpty).toSet
    val rules = v.str("replace").split("[\\r\\n]+").flatMap { l =>
      l.split("=>", 2) match
        case Array(a, b) if a.trim.nonEmpty => Some((a.trim, b.trim))
        case _                              => None
    }
    val replaced = rules.foldLeft(s)((acc, r) => acc.replace(r._1, r._2))
    val tags = replaced.split(",").map(_.trim).filter(t => t.nonEmpty && !banned.exists(b => t.toLowerCase.contains(b)))
    (if v.bool("dedupe", true) then tags.distinctBy(_.toLowerCase) else tags).mkString(", ")
  override def process(p: GenerationParams, v: Values): GenerationParams =
    p.copy(prompt = filter(p.prompt, v), negativePrompt = filter(p.negativePrompt, v))

class NegativePromptWeightPlugin extends Plugin:
  def id = "negative_prompt_weight"
  def name = "Negative Prompt Weight"
  def kind = PluginKind.AlwaysOn
  override def description = "Applies one global weight to the whole negative prompt."
  override def controls: Seq[Control] = Seq(Check("enabled", "Enabled"), Slider("weight", "Weight", 0, 2, 0.01, 1.0))
  override def process(p: GenerationParams, v: Values): GenerationParams =
    val w = v.double("weight", 1.0)
    p.copy(negativePrompt = PromptText.weightAll(p.negativePrompt, w), extra = p.extra + ("NPW_weight" -> Values.fmt(w)))
  override def xyzAxes: Seq[XyzAxis] = Seq(XyzAxis("Negative prompt weight", isNumeric = true)((p, s) =>
    p.copy(negativePrompt = PromptText.weightAll(p.negativePrompt, s.trim.toDoubleOption.getOrElse(1.0)))))

class NovelAIPromptPlugin extends Plugin:
  def id = "novelai_prompt"
  def name = "NovelAI prompt converter"
  def kind = PluginKind.AlwaysOn
  override def description = "Converts NovelAI emphasis {tag} / [tag] into A1111 weights before generating."
  override def controls: Seq[Control] = Seq(Check("enabled", "Enabled"))
  override def process(p: GenerationParams, v: Values): GenerationParams =
    p.copy(prompt = PromptText.fromNovelAI(p.prompt), negativePrompt = PromptText.fromNovelAI(p.negativePrompt))

/** Style variables: `{style:Name}` in a prompt is replaced by that saved style's prompt. */
class StyleVariablesPlugin(styles: () => Seq[sdfx.core.model.PromptStyle] = () => StyleVariablesPlugin.load()) extends Plugin:
  def id = "style_variables"
  def name = "Style variables"
  def kind = PluginKind.AlwaysOn
  override def description = "Use saved styles inside prompts with {style:Name} (or $Name)."
  override def controls: Seq[Control] = Seq(Check("enabled", "Enabled"))
  private val Re = """\{style:([^}]+)\}|\$([A-Za-z0-9_\-]+)""".r
  def substitute(s: String, negative: Boolean): String =
    val st = styles()
    Re.replaceAllIn(s, m => {
      val n = Option(m.group(1)).getOrElse(m.group(2)).trim
      st.find(_.name.equalsIgnoreCase(n)).map(x => (if negative then x.negativePrompt else x.prompt).replace("{prompt}", ""))
        .map(scala.util.matching.Regex.quoteReplacement).getOrElse(scala.util.matching.Regex.quoteReplacement(m.matched))
    })
  override def process(p: GenerationParams, v: Values): GenerationParams =
    p.copy(prompt = substitute(p.prompt, false), negativePrompt = substitute(p.negativePrompt, true))
object StyleVariablesPlugin:
  def load(): Seq[sdfx.core.model.PromptStyle] = sdfx.core.model.Styles.load(AppHome.detect().stylesFile)

// ----------------------------------------------------------------------------------------- variation scripts
/** Test my prompt!: removes each comma-separated part once to show its effect. */
class TestMyPromptPlugin extends Plugin:
  def id = "test_my_prompt"
  def name = "Test my prompt!"
  def kind = PluginKind.Script
  override def description = "Generates the prompt with each part removed in turn (same seed) and a labelled grid."
  override def controls: Seq[Control] = Seq(Check("include_full", "Include the full prompt first", true))
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val seed = Support.fixSeed(p0.seed)
    val parts = p0.prompt.split(",").map(_.trim).filter(_.nonEmpty).toSeq
    val items = (if v.bool("include_full", true) then Seq(Sequence.Item(p0.copy(seed = seed, batchSize = 1, nIter = 1), "full prompt")) else Nil) ++
      parts.indices.map(i => Sequence.Item(p0.copy(prompt = parts.patch(i, Nil, 1).mkString(", "), seed = seed, batchSize = 1, nIter = 1),
        s"-${parts(i)}"))
    val done = Sequence.run(ctx, items, "Test my prompt")
    JobOutput(done.flatMap(_._2), Sequence.grid(ctx, done, p0, "Test my prompt!"))

/** ScramblePrompts [m9]: variations by shuffling / dropping tags. */
class ScramblePromptsPlugin extends Plugin:
  def id = "scramble_prompts"
  def name = "Scramble prompts"
  def kind = PluginKind.Script
  override def description = "Creates prompt variations by shuffling and randomly dropping comma separated tags."
  override def controls: Seq[Control] = Seq(IntField("count", "Variations", 1, 100, 8),
    Slider("drop", "Drop probability", 0, 0.9, 0.05, 0.1), Check("shuffle", "Shuffle order", true),
    IntField("keep_first", "Keep the first N tags in place", 0, 50, 1))
  def variation(prompt: String, v: Values, rnd: Random): String =
    val tags = prompt.split(",").map(_.trim).filter(_.nonEmpty).toSeq
    val keep = math.min(v.int("keep_first", 1), tags.size)
    val (fixed, rest) = tags.splitAt(keep)
    val kept = rest.filter(_ => rnd.nextDouble() >= v.double("drop", 0.1))
    (fixed ++ (if v.bool("shuffle", true) then rnd.shuffle(kept) else kept)).mkString(", ")
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val seed = Support.fixSeed(p0.seed)
    val items = (0 until v.int("count", 8)).map { i =>
      val pr = variation(p0.prompt, v, new Random(seed + i))
      Sequence.Item(p0.copy(prompt = pr, seed = seed, batchSize = 1, nIter = 1), s"#${i + 1}")
    }
    val done = Sequence.run(ctx, items, "Scramble prompts")
    JobOutput(done.flatMap(_._2), Sequence.grid(ctx, done, p0, "Scramble prompts"))

/** TweakWeights [m9]: variations by changing the weight of tags that match a pattern. */
class TweakWeightsPlugin extends Plugin:
  def id = "tweak_weights"
  def name = "Tweak weights"
  def kind = PluginKind.Script
  override def description = "Creates variations by randomly re-weighting tags matching a pattern (all tags if empty)."
  override def controls: Seq[Control] = Seq(IntField("count", "Variations", 1, 100, 8), Text("pattern", "Tags containing (regex)", ""),
    Slider("min", "Min weight", 0, 2, 0.05, 0.8), Slider("max", "Max weight", 0, 2, 0.05, 1.3))
  def variation(prompt: String, v: Values, rnd: Random): String =
    val re = Option(v.str("pattern")).filter(_.nonEmpty).map(_.r)
    prompt.split(",").map(_.trim).filter(_.nonEmpty).map { t =>
      if re.forall(_.findFirstIn(t).isDefined) && !t.startsWith("<") then
        val w = v.double("min", 0.8) + rnd.nextDouble() * (v.double("max", 1.3) - v.double("min", 0.8))
        PromptText.weightAll(t.stripPrefix("(").stripSuffix(")").replaceAll(":[0-9.]+$", ""), math.round(w * 100) / 100.0)
      else t
    }.mkString(", ")
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val seed = Support.fixSeed(p0.seed)
    val items = (0 until v.int("count", 8)).map(i =>
      Sequence.Item(p0.copy(prompt = variation(p0.prompt, v, new Random(seed + i)), seed = seed, batchSize = 1, nIter = 1), s"#${i + 1}"))
    val done = Sequence.run(ctx, items, "Tweak weights")
    JobOutput(done.flatMap(_._2), Sequence.grid(ctx, done, p0, "Tweak weights"))

// -------------------------------------------------------------------------------------------- queue scripts
/** Lora Queue Helper: the same prompt once per LoRA (`<lora:name:w>` appended). */
class LoraQueuePlugin extends Plugin:
  def id = "lora_queue"
  def name = "LoRA queue"
  def kind = PluginKind.Script
  override def description = "Generates the prompt once per selected LoRA (one per line, or all when empty) and a grid."
  override def controls: Seq[Control] = Seq(Text("loras", "LoRAs (one per line; empty = all)", "", multiline = true),
    Slider("weight", "Weight", -2, 3, 0.05, 1.0), Check("grid", "Make a grid", true))
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val chosen = v.str("loras").split("[\\r\\n,]+").map(_.trim).filter(_.nonEmpty).toSeq
    val names = if chosen.nonEmpty then chosen else ctx.choices(ChoiceSource.Loras)
    val seed = Support.fixSeed(p0.seed)
    val items = names.map(n => Sequence.Item(p0.copy(prompt = s"${p0.prompt} ${LoraSyntax.tag(n, v.double("weight", 1.0))}".trim,
      seed = seed, batchSize = 1, nIter = 1), n))
    val done = Sequence.run(ctx, items, "LoRA queue")
    JobOutput(done.flatMap(_._2), if v.bool("grid", true) then Sequence.grid(ctx, done, p0, "LoRA queue") else Nil)

/** Batch Checkpoints and Prompts: run on several checkpoints, each with an optional prompt template. */
class BatchCheckpointsPromptsPlugin extends Plugin:
  def id = "batch_checkpoints_prompts"
  def name = "Batch checkpoints and prompts"
  def kind = PluginKind.Script
  override def description = "One generation per line `checkpoint | prompt template` ({prompt} = current prompt)."
  override def controls: Seq[Control] = Seq(Text("lines", "checkpoint | prompt template (one per line)", "", multiline = true),
    Check("grid", "Make a grid", true))
  def parse(text: String, p0: GenerationParams): Seq[Sequence.Item] =
    text.split("[\\r\\n]+").map(_.trim).filter(_.nonEmpty).toSeq.map { line =>
      val (ck, tpl) = line.split("\\|", 2) match
        case Array(a, b) => (a.trim, b.trim)
        case Array(a)    => (a.trim, "{prompt}")
      val prompt = if tpl.isEmpty then p0.prompt else tpl.replace("{prompt}", p0.prompt)
      Sequence.Item(p0.copy(checkpoint = Some(ck).filter(_.nonEmpty), prompt = prompt, batchSize = 1, nIter = 1), ck)
    }
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val seed = Support.fixSeed(p0.seed)
    val done = Sequence.run(ctx, parse(v.str("lines"), p0.copy(seed = seed)), "Checkpoints")
    JobOutput(done.flatMap(_._2), if v.bool("grid", true) then Sequence.grid(ctx, done, p0, "Batch checkpoints and prompts") else Nil)

// ----------------------------------------------------------------------------------------- travel scripts
/** Seed travel: interpolates the noise between seeds with variation seed strength 0 -> 1. */
class SeedTravelPlugin extends Plugin:
  def id = "seed_travel"
  def name = "Seed travel"
  def kind = PluginKind.Script
  override def tabs: Set[Tab] = Set(Tab.Txt2Img)
  override def description = "Frames between seeds (slerp of the noise); saves the frames and an animated GIF."
  override def controls: Seq[Control] = Seq(Text("seeds", "Destination seeds (comma separated)", "1, 2"),
    IntField("steps", "Frames between seeds", 1, 200, 10), IntField("frame_ms", "GIF frame time (ms)", 20, 2000, 120),
    Check("loop", "Loop back to the first seed", false))
  def frames(p0: GenerationParams, v: Values): Seq[Sequence.Item] =
    val seeds = Support.fixSeed(p0.seed) +: v.str("seeds").split(",").flatMap(_.trim.toLongOption).toSeq
    val path = if v.bool("loop") then seeds :+ seeds.head else seeds
    val n = v.int("steps", 10)
    path.sliding(2).toSeq.flatMap {
      case Seq(a, b) => (0 until n).map(k => Sequence.Item(p0.copy(seed = a, subseed = b, subseedStrength = k.toDouble / n,
        batchSize = 1, nIter = 1), f"$a->$b ${k.toDouble / n}%.2f"))
      case _ => Nil
    } :+ Sequence.Item(p0.copy(seed = path.last, subseedStrength = 0, batchSize = 1, nIter = 1), path.last.toString)
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val done = Sequence.run(ctx, frames(p0, v), "Seed travel")
    Sequence.gif(done, v.int("frame_ms", 120), pingPong = false).foreach(g => ctx.log(s"Seed travel GIF: $g"))
    JobOutput(done.flatMap(_._2))

/** Shift attention: animates the weight of `(word:a~b)` tokens across frames. */
class ShiftAttentionPlugin extends Plugin:
  def id = "shift_attention"
  def name = "Shift attention"
  def kind = PluginKind.Script
  override def description = "Write (word:0.5~1.5) in the prompt; frames sweep the weight. Saves frames and a GIF."
  override def controls: Seq[Control] = Seq(IntField("frames", "Frames", 2, 300, 12), IntField("frame_ms", "GIF frame time (ms)", 20, 2000, 120),
    Check("ping_pong", "Ping-pong GIF", false))
  private val Re = """\(([^():~]+):([0-9.]+)~([0-9.]+)\)""".r
  def promptAt(prompt: String, t: Double): String =
    Re.replaceAllIn(prompt, m => {
      val a = m.group(2).toDouble; val b = m.group(3).toDouble
      scala.util.matching.Regex.quoteReplacement(s"(${m.group(1)}:${math.round((a + (b - a) * t) * 1000) / 1000.0})")
    })
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val seed = Support.fixSeed(p0.seed)
    val n = v.int("frames", 12)
    val items = (0 until n).map { k =>
      val t = if n == 1 then 0.0 else k.toDouble / (n - 1)
      Sequence.Item(p0.copy(prompt = promptAt(p0.prompt, t), seed = seed, batchSize = 1, nIter = 1), f"$t%.2f")
    }
    val done = Sequence.run(ctx, items, "Shift attention")
    Sequence.gif(done, v.int("frame_ms", 120), v.bool("ping_pong")).foreach(g => ctx.log(s"Shift attention GIF: $g"))
    JobOutput(done.flatMap(_._2))

// ------------------------------------------------------------------------------------------ prompt generators
/** One Button Prompt: builds a whole random prompt from built-in word lists (own lists, seeded). */
class OneButtonPromptPlugin extends Plugin:
  def id = "one_button_prompt"
  def name = "One Button Prompt"
  def kind = PluginKind.Script
  override def description = "Generates complete random prompts (subject, style, setting, lighting...) and renders them."
  override def controls: Seq[Control] = Seq(IntField("count", "Prompts", 1, 100, 4),
    Choice("subject", "Subject", Seq("all", "humanoid", "animal", "landscape", "object", "concept"), "all"),
    Slider("complexity", "Complexity", 1, 10, 1, 5), Text("prefix", "Prefix", ""), Text("suffix", "Suffix", ""),
    Check("keep_prompt", "Prepend the current prompt", false))
  import OneButtonPromptPlugin.Words.*
  def generate(rnd: Random, v: Values): String =
    def one(xs: Seq[String]) = xs(rnd.nextInt(xs.size))
    val subj = v.str("subject", "all") match
      case "all"   => one(Seq("humanoid", "animal", "landscape", "object", "concept"))
      case other   => other
    val core = subj match
      case "humanoid"  => s"${one(adjectives)} ${one(people)}, ${one(clothing)}, ${one(poses)}"
      case "animal"    => s"${one(adjectives)} ${one(animals)}, ${one(poses)}"
      case "landscape" => s"${one(adjectives)} ${one(landscapes)}"
      case "object"    => s"${one(adjectives)} ${one(objects)}"
      case _           => s"${one(concepts)}"
    val extras = Seq(settings, lighting, styles, moods, cameras, colors, artMedia)
    val n = math.max(1, v.int("complexity", 5) / 2)
    val parts = core +: rnd.shuffle(extras).take(n).map(one)
    (Seq(v.str("prefix")) ++ parts ++ Seq(one(quality), v.str("suffix"))).map(_.trim).filter(_.nonEmpty).mkString(", ")
  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val seed = Support.fixSeed(p0.seed)
    val items = (0 until v.int("count", 4)).map { i =>
      val pr = generate(new Random(seed + 7919L * i), v)
      Sequence.Item(p0.copy(prompt = if v.bool("keep_prompt") && p0.prompt.trim.nonEmpty then s"${p0.prompt}, $pr" else pr,
        seed = seed + i, batchSize = 1, nIter = 1), pr.take(40))
    }
    items.foreach(i => ctx.log(s"prompt: ${i.params.prompt}"))
    JobOutput(Sequence.run(ctx, items, "One Button Prompt").flatMap(_._2))
object OneButtonPromptPlugin:
  object Words:
    val adjectives = Seq("ancient", "mysterious", "elegant", "futuristic", "whimsical", "majestic", "tiny", "giant", "ethereal", "rugged",
      "cheerful", "melancholic", "ornate", "minimalist", "glowing", "weathered", "serene", "fierce")
    val people = Seq("knight", "witch", "astronaut", "samurai", "scientist", "dancer", "pirate captain", "street musician", "queen",
      "monk", "detective", "farmer", "cyborg", "elf ranger", "chef")
    val animals = Seq("fox", "owl", "tiger", "whale", "cat", "dragon", "wolf", "octopus", "red panda", "stag", "koi fish", "hummingbird")
    val landscapes = Seq("mountain valley", "desert canyon", "floating islands", "coral reef", "misty forest", "volcanic coast",
      "snowy tundra", "rice terraces", "neon city skyline", "lavender fields")
    val objects = Seq("pocket watch", "lantern", "teapot", "spaceship", "violin", "treasure chest", "bonsai tree", "typewriter",
      "crystal sphere", "vintage camera")
    val concepts = Seq("the passage of time", "a dream within a dream", "the sound of rain", "forgotten memories", "the birth of a star",
      "solitude", "harmony of nature and machine")
    val clothing = Seq("flowing robe", "leather armor", "kimono", "space suit", "ball gown", "trench coat", "hooded cloak", "school uniform")
    val poses = Seq("standing", "sitting", "looking back", "running", "reaching out", "sleeping", "dancing", "close-up portrait")
    val settings = Seq("in a library", "at sunset", "under the stars", "in the rain", "inside a cathedral", "on a rooftop",
      "in a greenhouse", "at the bottom of the sea")
    val lighting = Seq("volumetric lighting", "rim light", "soft diffused light", "golden hour", "dramatic chiaroscuro", "neon glow",
      "candlelight", "moonlight")
    val styles = Seq("oil painting", "watercolor", "anime style", "photorealistic", "art nouveau", "ukiyo-e", "concept art", "pixel art",
      "3d render", "charcoal sketch")
    val moods = Seq("peaceful", "epic", "eerie", "nostalgic", "dreamy", "dramatic", "playful")
    val cameras = Seq("wide angle", "telephoto", "macro shot", "bird's eye view", "low angle", "shallow depth of field")
    val colors = Seq("pastel colors", "vibrant colors", "monochrome", "teal and orange", "muted tones", "gold and black")
    val artMedia = Seq("digital art", "gouache", "ink illustration", "stained glass", "papercut", "linocut print")
    val quality = Seq("highly detailed", "masterpiece", "intricate details", "sharp focus", "award winning")
