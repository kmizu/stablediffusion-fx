package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*
import sdfx.plugins.prompt.{StylePile, StylePileCatalog, StylePileTools}

final class StylePilePlugin(customRoot: Option[java.nio.file.Path] = None) extends Plugin:
  private lazy val catalogRoot = customRoot.getOrElse(sdfx.core.config.AppHome.detect().root.resolve("stylepile"))
  private lazy val catalog = StylePileCatalog.withOverrides(catalogRoot)
  val id = "stylepile"
  val name = "StylePile"
  val kind = PluginKind.Script
  override val description = "Mix weighted image types, artists and movements, with sequential and random prompt variables."
  private def plain(key: String, label: String, field: String, default: String = "Not set"): Control =
    Choice(key, label, Seq("Not set", "Random") ++ StylePileCatalog.literal(field).arr.toSeq.map(_.str), default)
  private def visual(key: String, label: String, group: String, custom: Boolean = false): Control =
    val options = catalog.choices(group)
    Choice(key, label, Seq("Not set", "Random") ++ options, "Not set",
      previewResources = options.flatMap(name => catalog.preview(group, name).map(url => name -> url.toExternalForm)).toMap,
      allowCustom = custom)
  private def strength(key: String): Control = Slider(key + "_strength", "Influence", 0, 2, 0.05, 1.3)
  override lazy val controls: Seq[Control] = Seq(
    Note("help", "[X] cycles sequential lines; [Y] cycles sublines inside each copy. [A]/[B]/[C] choose random lines. Random choices are reproducible from the resolved seed. Up to 10000 images; blank sublines remain meaningful."),
    Note("custom_catalog", s"Add reference images under $catalogRoot/<category>; disabled.txt hides names. Restart to refresh the catalog. " + catalog.warnings.mkString("; ")),
    plain("concept", "Concept", "ElementConcept"), plain("image_type", "Image type", "ResultNames"), strength("type"),
    visual("direction", "View / direction", "Directions"), strength("direction"),
    visual("mood", "Mood", "Moods"), strength("mood"), visual("color", "Colors", "Colors"), strength("color"),
    Choice("preset", "Preset", StylePileCatalog.literal("Preset").obj.keys.toSeq, "None"),
    Section("artists", "Artists and movements", expanded = false)
  ) ++ (1 to 3).flatMap(i => Seq(visual(s"artist$i", s"Artist $i", "Artists", true), strength(s"artist$i"))) ++
    (1 to 3).flatMap(i => Seq(visual(s"movement$i", s"Movement $i", "Art Movements", true), strength(s"movement$i"))) ++ Seq(
      Section("variables", "Prompt variables", expanded = false),
      Text("sequential", "Sequential [X]", multiline = true), Text("subsequential", "Subsequence [Y]", multiline = true),
      Text("random_a", "Random [A] (also legacy [R])", multiline = true),
      Text("random_b", "Random [B]", multiline = true), Text("random_c", "Random [C]", multiline = true),
      Check("override_count", "Set batch count from sequential lines", true),
      Check("increase_seed", "Increase seed within each sequence row", true),
      Check("show_tips", "Show a usage tip after generation"),
      Note("seed_help", "Seed increments across copies and sublines, then resets for the next X row. Noise tokens: [xs] six digits, [s] five characters, [m] ten, [l] two groups of five, [xl] two groups of ten; uppercase variants are separate choices."),
      Section("helpers", "Prompt insertion tools", expanded = false),
      Note("insert_help", "Example and Inspire me replace the positive prompt. Append tools add syntax using the fields below. Negative prompt stays unchanged."),
      Text("tool_text", "Text to weight"), Slider("tool_strength", "Text influence", 0, 2, .05, 1.3),
      Text("tool_from", "First text"), Text("tool_to", "Second text"), Slider("tool_position", "Switch position", 0, 1, .01, .5)

    )
  override val promptActions: Seq[PromptAction] = Seq(
    PromptAction("default", "Insert example prompt", true, _ => "Portrait of an attractive young lady,flower field background, square ratio"),
    PromptAction("inspire", "Inspire me", true, _ => StylePileTools.inspire(System.nanoTime(), catalog)),
    PromptAction("weight", "Append weighted text", false, v => StylePileTools.weight(v.str("tool_text"), v.double("tool_strength", 1.3))),
    PromptAction("morph", "Append step switch", false, v => StylePileTools.morph(v.str("tool_from"), v.str("tool_to"), v.double("tool_position", .5))),
    PromptAction("bounce", "Append alternating text", false, v => StylePileTools.bounce(v.str("tool_from"), v.str("tool_to")))
  )
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    catalog.warnings.foreach(ctx.log)
    val items = StylePile.expand(p, v, catalog).map(params => Sequence.Item(params, params.prompt))
    ctx.log(s"StylePile: ${items.size} images")
    val outputs = Sequence.run(ctx, items, "StylePile").flatMap(_._2)
    if v.bool("show_tips") && outputs.nonEmpty then
      ctx.log("StylePile tip: compare styles with the same seed, try image types separately, and inspect the saved expanded prompt when a combination works well.")
    JobOutput(outputs)
