package sdfx.engine.testplugins

import sdfx.core.image.Grid
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.png.PngMetadata

import scala.collection.mutable.ArrayBuffer

/** Loaded through ServiceLoader from a jar built by the registry test. */
class ExtPlugin extends Plugin:
  def id = "ext-test"
  def name = "External test plugin"
  def kind = PluginKind.AlwaysOn
  override def xyzAxes: Seq[XyzAxis] = Seq(XyzAxis("Ext axis")((p, v) => p.copy(extra = p.extra.updated("Ext", v))))

/** Same id as a builtin to check precedence. */
class DuplicateIdPlugin extends Plugin:
  def id = "repeat"
  def name = "Should be shadowed"
  def kind = PluginKind.Script

/** Script that calls ctx.generate `n` times, optionally saving a grid. */
class RepeatScript extends Plugin:
  def id = "repeat"
  def name = "Repeat"
  def kind = PluginKind.Script
  override def controls = Seq(Control.IntField("n", "Times", 1, 50, 3), Control.Check("grid", "Grid"))
  override def xyzAxes: Seq[XyzAxis] = Seq(XyzAxis("Repeat axis", isNumeric = true)((p, v) => p))
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val n = v.int("n", 3)
    val out = ArrayBuffer.empty[OutputImage]
    var i = 0
    while i < n && !ctx.isInterrupted do
      ctx.status(s"repeat ${i + 1}/$n", i, n)
      out ++= ctx.generate(p.copy(seed = 1000L + i))
      i += 1
    val grids =
      if v.bool("grid") && out.nonEmpty then
        val img = Grid.plain(out.map(o => PngMetadata.readImage(o.path)).toSeq)
        Seq(ctx.saveImage(img, "grids", p.copy(seed = 1000L), out.head.infotext))
      else Nil
    JobOutput(out.toSeq, grids)

/** Script that throws the given throwable (fatal errors included). */
class ThrowingScript(t: => Throwable) extends Plugin:
  def id = "throwing"
  def name = "Throwing"
  def kind = PluginKind.Script
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput = throw t

/** Script that generates until interrupted by the engine, then saves a partial grid and rethrows. */
class PartialGridScript extends Plugin:
  @volatile var sawInterrupted = false
  def id = "partial-grid"
  def name = "Partial grid"
  def kind = PluginKind.Script
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val out = ArrayBuffer.empty[OutputImage]
    try
      out ++= ctx.generate(p.copy(seed = 1L))
      out ++= ctx.generate(p.copy(seed = 2L, extra = p.extra.updated("force_error", "interrupted")))
      JobOutput(out.toSeq)
    catch
      case e: sdfx.core.protocol.EngineError if e.isInterrupted =>
        sawInterrupted = ctx.isInterrupted
        ctx.saveImage(Grid.plain(out.map(o => PngMetadata.readImage(o.path)).toSeq), "grids", p.copy(seed = 1L), "partial")
        throw e

/** Script that only makes transient generations and returns them as-is. */
class TransientScript extends Plugin:
  def id = "transient"
  def name = "Transient"
  def kind = PluginKind.Script
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    JobOutput(ctx.generateTransient(p.copy(seed = 77L, batchSize = 2)))

/** Creates temp files through every ctx path, keeps one transient image in its output, optionally fails. */
class TempUsingScript(fail: Boolean) extends Plugin:
  @volatile var created: Seq[java.nio.file.Path] = Nil
  @volatile var kept: Option[java.nio.file.Path] = None
  def id = "temp-using"
  def name = "Temp using"
  def kind = PluginKind.Script
  override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
    val tr = ctx.generateTransient(p.copy(seed = 5L, batchSize = 2))
    val up = ctx.upscale(tr.head.path, "Lanczos", 2)
    val tf = ctx.tempFile(".bin")
    java.nio.file.Files.write(tf, Array[Byte](1))
    created = tr.map(_.path) ++ Seq(up, tf)
    if fail then throw new IllegalStateException("script failed on purpose")
    kept = Some(tr.last.path)
    JobOutput(Seq(tr.last))

/** AlwaysOn that tags every generation. */
class MarkerAlwaysOn extends Plugin:
  def id = "marker"
  def name = "Marker"
  def kind = PluginKind.AlwaysOn
  override def controls = Seq(Control.Check("enabled", "Enabled"), Control.Text("value", "Value", "m"))
  override def process(p: GenerationParams, v: Values): GenerationParams =
    p.copy(extra = p.extra.updated("Marker", v.str("value")), hooks = p.hooks.updated("marker", ujson.Str(v.str("value"))))
