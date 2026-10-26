package sdfx.plugins

import sdfx.core.infotext.{ImageFacts, Infotext}
import sdfx.core.image.GridNumbers
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.{OutputImage, PluginContext}
import sdfx.core.protocol.EngineError

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path, Paths}
import javax.imageio.ImageIO
import scala.util.Random

/** Small helpers shared by the standard plugins. */
object Support:
  private val rnd = new Random()

  /** A1111 `get_fixed_seed`: -1 (or any negative) becomes a random seed. */
  def fixSeed(seed: Long): Long = if seed < 0 then rnd.nextLong(4294967294L) else seed

  def readImage(path: Path): BufferedImage =
    val img = ImageIO.read(path.toFile)
    if img == null then throw new IllegalStateException(s"cannot read image: $path") else img

  def readImage(path: String): BufferedImage = readImage(Paths.get(path))

  def writeTemp(ctx: PluginContext, img: BufferedImage): Path =
    val p = ctx.tempFile(".png")
    Option(p.getParent).foreach(Files.createDirectories(_))
    ImageIO.write(img, "png", p.toFile)
    p

  /** Infotext for a derived image (grid, merged upscale...) built from params plus a seed. */
  def infotext(p: GenerationParams, seed: Long): String = Infotext.write(p, ImageFacts(seed = seed))

  def requireImg2Img(p: GenerationParams, plugin: String): Unit =
    if p.img2img.isEmpty then throw new IllegalArgumentException(s"$plugin needs an img2img init image")

  /** Image of a cell: the first output, like A1111 (`processed.images[0]`); other batch images stay sub images. */
  def gridTile(o: OutputImage): BufferedImage = GridNumbers.render(readImage(o.path), o.path.getFileName.toString, o.params.extra)
  def cellImage(outs: Seq[OutputImage]): Option[BufferedImage] = outs.headOption.map(gridTile)

  /** Keep native display controls and the exact saved-source correspondence on derived grid PNGs. */
  def gridParams(base: GenerationParams, outs: Seq[OutputImage]): GenerationParams =
    val settings = outs.headOption.toSeq.flatMap(_.params.extra.filter((k, _) => GridNumbers.settingKeys.contains(k))).toMap
    val extras = base.extra ++ settings
    if !GridNumbers.enabled(extras) then base
    else
      val sources = ujson.Arr.from(outs.map { o =>
        ujson.Obj("filename" -> o.path.getFileName.toString, "seed" -> o.seed,
          "number" -> GridNumbers.number(o.path.getFileName.toString).map(ujson.Str(_)).getOrElse(ujson.Null),
          "axes" -> ujson.Obj.from(Seq("X", "Y", "Z").flatMap(a => o.params.extra.get(s"Grid $a info").map(a -> ujson.Str(_)))))
      })
      base.copy(extra = extras.updated(GridNumbers.SourcesKey, sources.render()))

  /** True for the error the host throws when the job is interrupted (or any error once the job is interrupted). */
  def isInterruption(e: Throwable, ctx: PluginContext): Boolean = e match
    case ee: EngineError => ee.isInterrupted || ctx.isInterrupted
    case _               => false

  /** Runs one step of a script loop; returns None (and the loop should stop) when the job was interrupted, either
    * by the flag before the step or by the host throwing `EngineError(interrupted)` during it.
    */
  def step[T](ctx: PluginContext)(body: => T): Option[T] =
    if ctx.isInterrupted then None
    else
      try Some(body)
      catch case e: Throwable if isInterruption(e, ctx) => None

  /** Best-effort delete of temporary / transient files. */
  def deleteQuietly(paths: Path*): Unit =
    paths.foreach(p => try Files.deleteIfExists(p) catch case _: Exception => ())

/** Engine-hook helpers for AlwaysOn plugins. X/Y/Z axes store per-field overrides under `<hook>_xyz`; `process`
  * merges those on top of the control values, so axis values win over the UI sliders (forge `_freeu_xyz` style).
  */
object Hooks:
  def overridesKey(hook: String): String = s"${hook}_xyz"

  def setOverride(p: GenerationParams, hook: String, field: String, value: ujson.Value): GenerationParams =
    val k = overridesKey(hook)
    val cur = p.hooks.get(k).collect { case o: ujson.Obj => o.value.toSeq }.getOrElse(Nil)
    p.copy(hooks = p.hooks.updated(k, ujson.Obj.from(cur.filterNot(_._1 == field) :+ (field -> value))))

  /** Writes `hook` = base fields overridden by any X/Y/Z overrides and returns the merged object. */
  def apply(p: GenerationParams, hook: String, base: ujson.Obj): (GenerationParams, ujson.Obj) =
    val overrides = p.hooks.get(overridesKey(hook)).collect { case o: ujson.Obj => o.value.toSeq }.getOrElse(Nil)
    val merged = ujson.Obj.from(base.value.toSeq.map((k, v) => k -> overrides.find(_._1 == k).map(_._2).getOrElse(v)))
    // The override key is consumed here so it never reaches the engine (the host applies process once per generate).
    (p.copy(hooks = p.hooks.removed(overridesKey(hook)).updated(hook, merged)), merged)

  def num(v: ujson.Value): String = v match
    case ujson.Num(d)  => sdfx.core.plugins.api.Values.fmt(d)
    case ujson.Bool(b) => if b then "True" else "False"
    case ujson.Str(s)  => s
    case other         => other.render()

  /** Numeric axis value as JSON. */
  def numValue(s: String): ujson.Value = s.trim.toDoubleOption.map(ujson.Num(_)).getOrElse(ujson.Str(s.trim))
