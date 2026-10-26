package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

import java.nio.file.{Path, Paths}

/** Helpers shared by the checkpoint tools. */
private[plugins] object ModelTools:
  /** A safe file name: no directories, no reserved characters, `.safetensors` extension. */
  def fileName(raw: String, fallback: String): String =
    val base = Option(raw).map(_.trim).filter(_.nonEmpty).getOrElse(fallback)
      .replaceAll("""(?i)\.(safetensors|ckpt|pt)$""", "")
      .replaceAll("""[\\/:*?"<>|]+""", "_").replaceAll("""^\.+""", "").trim
    (if base.isEmpty then fallback else base) + ".safetensors"

  def stem(name: String): String = Paths.get(name).getFileName.toString.replaceAll("""\.[^.]+$""", "")

  def path(ctx: PluginContext, source: ChoiceSource, name: String, what: String): Path =
    if name.isEmpty || name == "None" then throw new IllegalArgumentException(s"choose $what")
    ctx.modelPath(source, name).map(Paths.get(_)).getOrElse(throw new IllegalArgumentException(s"$what not found: $name"))

  def runTool(ctx: PluginContext, kind: String, params: ujson.Obj): ujson.Value =
    ctx.status(s"$kind…", 0, 1)
    val r = ctx.engineCall(kind, params)
    ctx.status(s"$kind done", 1, 1)
    r

/** Checkpoint merger (SuperMerger / Merge Block Weighted / Merge Board / Model Mixer / Untitled Merger, re-implemented):
  * weighted sum, add difference, sum twice, triple sum; optional per-block weights; optional test generation.
  */
class ModelMergePlugin extends Plugin:
  def id = "model_merge"
  def name = "Checkpoint merger"
  def kind = PluginKind.Script
  override def description =
    "Merges two or three checkpoints into a new .safetensors next to model A, then (optionally) generates with it. " +
      "Use Generate to run the merge."
  override def enabledBy: Map[String, String] = Map("block_weights" -> "use_blocks", "block_preset" -> "use_blocks")
  override def controls: Seq[Control] = Seq(
    DynamicChoice("model_a", "Model A", ChoiceSource.Checkpoints, "None"),
    DynamicChoice("model_b", "Model B", ChoiceSource.Checkpoints, "None"),
    DynamicChoice("model_c", "Model C (difference / triple)", ChoiceSource.Checkpoints, "None"),
    Choice("mode", "Merge mode", Seq("Weight sum", "Add difference", "Sum twice", "Triple sum"), "Weight sum"),
    Slider("alpha", "Alpha (B)", -1, 2, 0.01, 0.5),
    Slider("beta", "Beta (C, sum twice / triple sum)", -1, 2, 0.01, 0.25),
    Note("formula", "Weight sum: A·(1−α) + B·α · Add difference: A + (B − C)·α · Sum twice: (A·(1−α) + B·α)·(1−β) + C·β · " +
      "Triple sum: A·(1−α−β) + B·α + C·β"),
    Section("blocks", "Block weights", expanded = false),
    Check("use_blocks", "Use per-block alpha"),
    Text("block_weights", "Block alphas", placeholder = "SD1.x: 25 values IN00..IN11, M00, OUT00..OUT11 (26 with BASE first) · SDXL: 19 (20)"),
    Choice("block_preset", "Fill from preset", BlockPresets.names, "custom"),
    Section("output", "Output", expanded = true),
    Text("output_name", "File name", placeholder = "empty = A_x_B_alpha"),
    Choice("precision", "Precision", Seq("fp16", "bf16", "fp32"), "fp16"),
    Choice("vae_source", "VAE", Seq("merge", "A", "B"), "merge"),
    Check("overwrite", "Overwrite an existing file"),
    Check("generate", "Generate with the merged model afterwards", true)
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val a = ModelTools.path(ctx, ChoiceSource.Checkpoints, v.str("model_a"), "model A")
    val b = ModelTools.path(ctx, ChoiceSource.Checkpoints, v.str("model_b"), "model B")
    val mode = v.str("mode", "Weight sum")
    val c = if mode == "Weight sum" then None else Some(ModelTools.path(ctx, ChoiceSource.Checkpoints, v.str("model_c"), "model C"))
    val alpha = v.double("alpha", 0.5)
    val blocks =
      if !v.bool("use_blocks") then ""
      else
        val preset = v.str("block_preset", "custom")
        if preset != "custom" then s"preset:$preset" else v.str("block_weights")
    val out = a.resolveSibling(ModelTools.fileName(v.str("output_name"),
      s"${ModelTools.stem(a.toString)}_x_${ModelTools.stem(b.toString)}_${Values.fmt(alpha)}"))
    ctx.log(s"merging ${a.getFileName} + ${b.getFileName}${c.map(x => s" + ${x.getFileName}").getOrElse("")} ($mode, α=$alpha) -> $out")
    val r = ModelTools.runTool(ctx, "merge_models", ujson.Obj(
      "model_a" -> a.toString, "model_b" -> b.toString, "model_c" -> c.map(x => ujson.Str(x.toString)).getOrElse(ujson.Null),
      "mode" -> mode, "alpha" -> alpha, "beta" -> v.double("beta", 0.25), "block_weights" -> blocks,
      "precision" -> v.str("precision", "fp16"), "vae_source" -> v.str("vae_source", "merge"),
      "overwrite" -> v.bool("overwrite"), "output" -> out.toString))
    ctx.log(s"merged model written: $out (${r.obj.get("keys").map(_.num.toInt).getOrElse(0)} tensors)")
    if !v.bool("generate", true) then JobOutput(Nil)
    else
      val p = p0.copy(checkpoint = Some(out.toString),
        extra = p0.extra ++ Seq("Merge" -> s"$mode ${ModelTools.stem(a.toString)} + ${ModelTools.stem(b.toString)} α=${Values.fmt(alpha)}"))
      Support.step(ctx)(ctx.generate(p)).map(JobOutput(_)).getOrElse(JobOutput(Nil))

/** Block weight presets of the merger (expanded by the engine for the model's layout, python/sdfx/model_tools.py). */
object BlockPresets:
  val names: Seq[String] = Seq("custom", "GRAD_V", "GRAD_A", "FLAT_25", "FLAT_50", "FLAT_75", "IN_ONLY", "OUT_ONLY", "MID_ONLY")

/** Checkpoint converter (Model Converter / Model toolkit, re-implemented). */
class ModelConverterPlugin extends Plugin:
  def id = "model_converter"
  def name = "Checkpoint converter"
  def kind = PluginKind.Script
  override def description =
    "Writes a cleaned copy of a checkpoint: precision, EMA weights, junk keys, CLIP position_ids, baked VAE. Use Generate to run."
  override def controls: Seq[Control] = Seq(
    DynamicChoice("model", "Model", ChoiceSource.Checkpoints, "None"),
    Choice("precision", "Precision", Seq("fp16", "bf16", "fp32", "keep"), "fp16"),
    Choice("ema", "EMA weights", Seq("no-ema", "ema-only", "keep"), "no-ema"),
    Check("fix_clip", "Fix broken CLIP position_ids", true),
    Check("drop_junk", "Remove training leftovers (optimizer, loss, ...)", true),
    DynamicChoice("bake_vae", "Bake in VAE", ChoiceSource.Vaes, "None"),
    Text("output_name", "File name", placeholder = "empty = <model>-<precision>"),
    Check("overwrite", "Overwrite an existing file")
  )

  override def run(p0: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val src = ModelTools.path(ctx, ChoiceSource.Checkpoints, v.str("model"), "model")
    val vae = Option(v.str("bake_vae", "None")).filter(n => n.nonEmpty && n != "None").map(n => ModelTools.path(ctx, ChoiceSource.Vaes, n, "VAE"))
    val precision = v.str("precision", "fp16")
    val out = src.resolveSibling(ModelTools.fileName(v.str("output_name"), s"${ModelTools.stem(src.toString)}-$precision"))
    val r = ModelTools.runTool(ctx, "convert_model", ujson.Obj(
      "model" -> src.toString, "output" -> out.toString, "precision" -> precision, "ema" -> v.str("ema", "no-ema"),
      "fix_clip" -> v.bool("fix_clip", true), "drop_junk" -> v.bool("drop_junk", true),
      "bake_vae" -> vae.map(x => ujson.Str(x.toString)).getOrElse(ujson.Null), "overwrite" -> v.bool("overwrite")))
    def n(k: String) = r.obj.get(k).map(_.num.toInt).getOrElse(0)
    ctx.log(s"converted model written: $out (${n("keys")} tensors, ${n("dropped")} removed, ${n("clip_fixed")} position_ids fixed)")
    JobOutput(Nil)
