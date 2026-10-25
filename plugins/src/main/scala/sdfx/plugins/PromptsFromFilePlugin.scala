package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

object PromptsFromFilePlugin:
  /** POSIX `shlex.split`: whitespace separated, '...' literal, "..." with backslash escapes, \x outside quotes. */
  def shlex(line: String): Seq[String] =
    val out = scala.collection.mutable.ArrayBuffer[String]()
    val cur = new StringBuilder
    var has = false
    var i = 0
    val n = line.length
    while i < n do
      val c = line.charAt(i)
      if c.isWhitespace then
        if has then
          out += cur.toString
          cur.clear()
          has = false
        i += 1
      else if c == '\'' then
        val end = line.indexOf('\'', i + 1)
        if end < 0 then throw new IllegalArgumentException("No closing quotation")
        cur.append(line.substring(i + 1, end))
        has = true
        i = end + 1
      else if c == '"' then
        i += 1
        var closed = false
        while i < n && !closed do
          val d = line.charAt(i)
          if d == '"' then
            closed = true
            i += 1
          else if d == '\\' && i + 1 < n && "\"\\$`\n".contains(line.charAt(i + 1)) then
            cur += line.charAt(i + 1)
            i += 2
          else
            cur += d
            i += 1
        if !closed then throw new IllegalArgumentException("No closing quotation")
        has = true
      else if c == '\\' then
        if i + 1 >= n then throw new IllegalArgumentException("No escaped character")
        cur += line.charAt(i + 1)
        has = true
        i += 2
      else
        cur += c
        has = true
        i += 1
    if has then out += cur.toString
    out.toSeq

  enum TagType:
    case Str, IntT, FloatT, BoolT

  import TagType.*
  val tags: Map[String, TagType] = Map(
    "sd_model" -> Str, "prompt" -> Str, "negative_prompt" -> Str, "styles" -> Str, "seed" -> IntT,
    "subseed_strength" -> FloatT, "subseed" -> IntT, "sampler_name" -> Str, "scheduler" -> Str, "batch_size" -> IntT,
    "n_iter" -> IntT, "steps" -> IntT, "cfg_scale" -> FloatT, "width" -> IntT, "height" -> IntT, "clip_skip" -> IntT,
    "denoising_strength" -> FloatT, "outpath_samples" -> Str, "outpath_grids" -> Str, "prompt_for_display" -> Str,
    "seed_resize_from_h" -> IntT, "seed_resize_from_w" -> IntT, "sampler_index" -> IntT, "restore_faces" -> BoolT,
    "tiling" -> BoolT, "do_not_save_samples" -> BoolT, "do_not_save_grid" -> BoolT
  )

  /** A1111 `cmdargs`: `--tag value` pairs; `--prompt`/`--negative_prompt` swallow words up to the next `--`. */
  def cmdargs(line: String): Either[String, Map[String, String]] =
    try
      val args = shlex(line)
      var pos = 0
      val res = scala.collection.mutable.LinkedHashMap[String, String]()
      while pos < args.size do
        val arg = args(pos)
        if !arg.startsWith("--") then return Left(s"must start with \"--\": $arg")
        if pos + 1 >= args.size then return Left(s"missing argument for command line option $arg")
        val tag = arg.drop(2)
        if tag == "prompt" || tag == "negative_prompt" then
          pos += 1
          val sb = new StringBuilder(args(pos))
          pos += 1
          while pos < args.size && !args(pos).startsWith("--") do
            sb.append(' ').append(args(pos))
            pos += 1
          res(tag) = sb.toString
        else
          tags.get(tag) match
            case None => return Left(s"unknown commandline option: $arg")
            case Some(t) =>
              val v = args(pos + 1)
              val ok = t match
                case IntT   => v.toLongOption.isDefined
                case FloatT => v.toDoubleOption.isDefined
                case BoolT  => Set("true", "false").contains(v.toLowerCase)
                case Str    => true
              if !ok then return Left(s"bad value for $arg: $v")
              res(tag) = v
              pos += 2
      Right(res.toMap)
    catch case e: IllegalArgumentException => Left(e.getMessage)

  /** Parses a line: plain text is a prompt; lines with `--` are option lists (falling back to a prompt on error). */
  def parseLine(line: String): Map[String, String] =
    if line.contains("--") then cmdargs(line).getOrElse(Map("prompt" -> line)) else Map("prompt" -> line)

  /** Applies parsed options to params. `samplers` maps names case-insensitively (A1111 samplers_map). */
  def applyArgs(p: GenerationParams, args: Map[String, String], samplers: Seq[String]): GenerationParams =
    args.foldLeft(p) { case (q, (k, v)) =>
      k match
        case "prompt"           => q.copy(prompt = v)
        case "negative_prompt"  => q.copy(negativePrompt = v)
        case "seed"             => q.copy(seed = v.toLong)
        case "subseed"          => q.copy(subseed = v.toLong)
        case "subseed_strength" => q.copy(subseedStrength = v.toDouble)
        case "sampler_name"     => q.copy(samplerName = samplers.find(_.equalsIgnoreCase(v)).getOrElse(v))
        case "sampler_index"    => samplers.lift(v.toInt).fold(q)(s => q.copy(samplerName = s))
        case "scheduler"        => q.copy(scheduler = v)
        case "batch_size"       => q.copy(batchSize = v.toInt)
        case "n_iter"           => q.copy(nIter = v.toInt)
        case "steps"            => q.copy(steps = v.toInt)
        case "cfg_scale"        => q.copy(cfgScale = v.toDouble)
        case "width"            => q.copy(width = v.toInt)
        case "height"           => q.copy(height = v.toInt)
        case "clip_skip"        => q.copy(clipSkip = v.toInt)
        case "sd_model"         => q.copy(checkpoint = Some(v))
        case "styles"           => q.copy(styles = v.split(",").map(_.trim).filter(_.nonEmpty).toSeq)
        case "denoising_strength" =>
          q.img2img.fold(q.copy(hires = q.hires.copy(denoisingStrength = v.toDouble)))(i =>
            q.copy(img2img = Some(i.copy(denoisingStrength = v.toDouble)))
          )
        case _ => q // accepted for compatibility, no effect here
    }

/** A1111 "Prompts from file or textbox". */
class PromptsFromFilePlugin extends Plugin:
  import PromptsFromFilePlugin.*
  def id = "prompts_from_file"
  def name = "Prompts from file or textbox"
  def kind = PluginKind.Script
  override def description = "Runs one generation per line; lines may use --prompt/--steps/... options."
  override def controls: Seq[Control] = Seq(
    Check("iterate_seed", "Iterate seed every line"),
    Check("same_seed", "Use same random seed for all lines"),
    Choice("prompt_position", "Insert prompts at the", Seq("start", "end"), "start"),
    Text("prompts", "List of prompt inputs", multiline = true, placeholder = "one prompt per line, or --prompt \"...\" --steps 20")
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val lines = v.str("prompts").linesIterator.map(_.trim).filter(_.nonEmpty).toSeq
    val jobs = lines.map(parseLine)
    val iterate = v.bool("iterate_seed")
    val atStart = v.str("prompt_position", "start") == "start"
    var base = if (iterate || v.bool("same_seed")) && p0.seed < 0 then p0.copy(seed = Support.fixSeed(-1)) else p0
    lazy val samplers = ctx.choices(ChoiceSource.Samplers)
    ctx.log(s"Will process ${lines.size} lines")
    val out = scala.collection.mutable.ArrayBuffer[OutputImage]()
    val it = jobs.zipWithIndex.iterator
    var running = true
    while running && it.hasNext do
      val (args, i) = it.next()
      var p = applyArgs(base, args, if args.contains("sampler_name") || args.contains("sampler_index") then samplers else Nil)
      def join(a: String, b: String) = if atStart then s"$a $b" else s"$b $a"
      if args.get("prompt").exists(_.nonEmpty) && base.prompt.nonEmpty then p = p.copy(prompt = join(args("prompt"), base.prompt))
      if args.get("negative_prompt").exists(_.nonEmpty) && base.negativePrompt.nonEmpty then
        p = p.copy(negativePrompt = join(args("negative_prompt"), base.negativePrompt))
      ctx.status(s"Prompts from file ${i + 1}/${jobs.size}", i, jobs.size)
      Support.step(ctx)(ctx.generate(p)) match
        case Some(o) => out ++= o
        case None    => running = false
      if iterate then base = base.copy(seed = base.seed + base.batchSize * base.nIter)
    JobOutput(out.toSeq)
