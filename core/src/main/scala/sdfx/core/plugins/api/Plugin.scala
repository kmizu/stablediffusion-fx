package sdfx.core.plugins.api

import sdfx.core.model.GenerationParams

import java.awt.image.BufferedImage
import java.nio.file.Path

enum Tab:
  case Txt2Img, Img2Img

/** AlwaysOn plugins tweak every generation (FreeU, Dynamic Thresholding...). Script plugins take over the run;
  * only one script can be selected at a time (X/Y/Z grid, Loopback...), mirroring A1111's "Script" dropdown.
  */
enum PluginKind:
  case AlwaysOn, Script

/** Dynamic choice lists resolved by the host at runtime. */
enum ChoiceSource:
  case Samplers, Schedulers, Checkpoints, Vaes, Loras, Upscalers, HiresUpscalers, Preprocessors, ControlNets,
    FaceRestorers, SvdModels, Zero123Models, PhotoMakerModels

/** Declarative UI controls; the JavaFX host renders them, MCP exposes them as JSON schema. */
sealed trait Control:
  def key: String
  def label: String
  def default: String
  /** False for layout-only controls (notes, sections) that carry no value. */
  def hasValue: Boolean = true

object Control:
  case class Slider(key: String, label: String, min: Double, max: Double, step: Double, defaultValue: Double) extends Control:
    def default: String = Values.fmt(defaultValue)
  case class IntField(key: String, label: String, min: Int, max: Int, defaultValue: Int) extends Control:
    def default: String = defaultValue.toString
  case class Check(key: String, label: String, defaultValue: Boolean = false) extends Control:
    def default: String = defaultValue.toString
  case class Choice(key: String, label: String, options: Seq[String], defaultValue: String,
      previewResources: Map[String, String] = Map.empty, allowCustom: Boolean = false) extends Control:
    def default: String = defaultValue
  case class DynamicChoice(key: String, label: String, source: ChoiceSource, defaultValue: String = "", allowNone: Boolean = true) extends Control:
    def default: String = defaultValue
  case class Text(key: String, label: String, defaultValue: String = "", multiline: Boolean = false, placeholder: String = "") extends Control:
    def default: String = defaultValue
  /** An image file path (the host offers a file chooser / drop zone). `multiple` = newline separated paths. */
  case class ImageFile(key: String, label: String, multiple: Boolean = false) extends Control:
    def default: String = ""
  case class Note(key: String, label: String) extends Control:
    def default: String = ""
    override def hasValue: Boolean = false
  /** Starts a collapsible group: the controls after it (up to the next Section) are shown inside it. */
  case class Section(key: String, label: String, expanded: Boolean = true) extends Control:
    def default: String = ""
    override def hasValue: Boolean = false

/** String-typed control values with lenient typed accessors. */
final case class Values(map: Map[String, String]):
  def str(k: String, d: String = ""): String = map.getOrElse(k, d)
  def int(k: String, d: Int = 0): Int = map.get(k).flatMap(v => v.trim.toIntOption.orElse(v.trim.toDoubleOption.map(_.toInt))).getOrElse(d)
  def double(k: String, d: Double = 0): Double = map.get(k).flatMap(_.trim.toDoubleOption).getOrElse(d)
  def bool(k: String, d: Boolean = false): Boolean = map.get(k).map(_.trim.toLowerCase).map(v => v == "true" || v == "1" || v == "yes").getOrElse(d)
  def updated(k: String, v: String): Values = Values(map.updated(k, v))

object Values:
  val empty: Values = Values(Map.empty)
  def defaults(controls: Seq[Control]): Values = Values(controls.collect { case c if c.hasValue => c.key -> c.default }.toMap)
  def fmt(d: Double): String =
    if d.isNaN || d.isInfinite then d.toString
    else if d == math.rint(d) && math.abs(d) < 1e15 then d.toLong.toString
    else BigDecimal(d).bigDecimal.stripTrailingZeros.toPlainString

/** An image produced by the host, already saved with its infotext. */
case class OutputImage(path: Path, params: GenerationParams, infotext: String, seed: Long)

/** The result of a whole job: individual images plus optional grids, in display order. */
case class JobOutput(images: Seq[OutputImage], grids: Seq[OutputImage] = Nil):
  def all: Seq[OutputImage] = grids ++ images

/** Services the host offers to plugins. Calls are blocking and run on the job thread. */
trait PluginContext:
  /** Whether this job opted into recording consumed local resource hashes. */
  def resourceHashesEnabled: Boolean = false
  /** Runs one generation (txt2img or img2img depending on params) and saves the images. */
  def generate(p: GenerationParams): Seq[OutputImage]
  /** Runs one generation without persisting it as a user output: images stay in the temp dir,
    * no gallery/ImageSaved event, caller deletes them.
    */
  def generateTransient(p: GenerationParams): Seq[OutputImage] = generate(p)

  /** Calls an engine request that is not a generation (`restore_faces`, `caption`, `img2vid`, `novel_view`,
    * `run_code`, ...). The host fills in `output_dir` (a temp dir) when absent; returned image paths are temporary
    * and must be saved with [[saveImage]] (or copied) by the caller.
    */
  def engineCall(kind: String, params: ujson.Obj): ujson.Value =
    throw new UnsupportedOperationException(s"engine request '$kind' is not available in this host")

  /** Resolves a model choice (as listed by [[choices]]) to its file path. */
  def modelPath(source: ChoiceSource, name: String): Option[String] = None
  def upscale(image: Path, upscaler: String, scale: Double): Path
  /** Saves an arbitrary image (e.g. a grid) to outputs/<kind>/ with the given infotext. */
  def saveImage(img: BufferedImage, kind: String, params: GenerationParams, infotext: String): OutputImage
  def tempFile(suffix: String): Path
  def choices(source: ChoiceSource): Seq[String]
  def log(message: String): Unit
  def status(text: String, done: Int, total: Int): Unit
  def isInterrupted: Boolean

/** An X/Y/Z grid axis. `apply` sets one value on the params; values are user typed strings. */
trait XyzAxis:
  def name: String
  def choices: Option[ChoiceSource] = None
  def numeric: Boolean = false
  def apply(p: GenerationParams, value: String): GenerationParams

object XyzAxis:
  def apply(axisName: String, isNumeric: Boolean = false, source: Option[ChoiceSource] = None)(f: (GenerationParams, String) => GenerationParams): XyzAxis =
    new XyzAxis:
      def name = axisName
      override def numeric = isNumeric
      override def choices = source
      def apply(p: GenerationParams, value: String) = f(p, value)

/** An explicit user action that proposes text for the visible positive prompt. */
final case class PromptAction(id: String, label: String, replace: Boolean, text: Values => String)

trait Plugin:
  def id: String
  def name: String
  def description: String = ""
  /** Control key -> key of the checkbox that enables it; the host greys the control out while the box is off. */
  def enabledBy: Map[String, String] = Map.empty
  def kind: PluginKind
  def tabs: Set[Tab] = Set(Tab.Txt2Img, Tab.Img2Img)
  def controls: Seq[Control] = Nil
  def promptActions: Seq[PromptAction] = Nil

  /** AlwaysOn: whether the plugin applies for these values (by convention the `enabled` checkbox). */
  def isActive(v: Values): Boolean = kind == PluginKind.Script || v.bool("enabled")

  /** AlwaysOn: adjust params before each generation (engine hooks, extra infotext keys...). */
  def process(p: GenerationParams, v: Values): GenerationParams = p

  /** Script: run the whole job. Default just generates once. */
  def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput = JobOutput(ctx.generate(p))

  /** Axes this plugin contributes to the X/Y/Z grid. */
  def xyzAxes: Seq[XyzAxis] = Nil

  /** Python-side features this plugin needs (checked against the engine hello `features`). */
  def requiresEngineFeatures: Seq[String] = Nil
