package sdfx.ui

import sdfx.core.model.*
import sdfx.core.payload.SubmittedPayloadContext
import sdfx.core.plugins.api.{Plugin, Values}
import sdfx.engine.{CaptureOptions, Job, JobLimits}

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/** Immutable evidence of a queued Job. JSON retains all data, while plugin references are held
  * separately because equal plugin IDs and class names do not imply the same plugin object.
  */
final class FrozenJobEvidence private (
    val snapshotJson: String,
    private val canonicalJson: String,
    private val scriptRef: Option[Plugin],
    private val alwaysOnRefs: Vector[Plugin]
):
  val canonicalHash: String = MessageDigest.getInstance("SHA-256")
    .digest(canonicalJson.getBytes(StandardCharsets.UTF_8)).map(b => f"${b & 255}%02x").mkString

  def sameJobAs(other: FrozenJobEvidence): Boolean =
    canonicalJson == other.canonicalJson &&
      ((scriptRef, other.scriptRef) match
        case (None, None) => true
        case (Some(a), Some(b)) => a eq b
        case _ => false) &&
      alwaysOnRefs.size == other.alwaysOnRefs.size &&
      alwaysOnRefs.zip(other.alwaysOnRefs).forall((a, b) => a eq b)

object FrozenJobEvidence:
  // Render directly so Long values are not rounded by ujson's Double-backed Num. Only object
  // keys are sorted for canonical output; strings, numbers and array order remain untouched.
  private sealed trait Node:
    def render(canonical: Boolean): String
  private final case class Obj(entries: Vector[(String, Node)]) extends Node:
    def render(canonical: Boolean): String =
      val ordered = if canonical then entries.sortBy(_._1) else entries
      ordered.iterator.map((k, v) => ujson.write(ujson.Str(k)) + ":" + v.render(canonical)).mkString("{", ",", "}")
  private final case class Arr(values: Vector[Node]) extends Node:
    def render(canonical: Boolean): String = values.iterator.map(_.render(canonical)).mkString("[", ",", "]")
  private final case class Atom(encoded: String) extends Node:
    def render(canonical: Boolean): String = encoded

  private def obj(fields: (String, Node)*): Node = Obj(fields.toVector)
  private def string(s: String): Node = Atom(ujson.write(ujson.Str(s)))
  private def int(n: Int): Node = Atom(n.toString)
  private def long(n: Long): Node = Atom(n.toString)
  private def double(n: Double, path: String): Node =
    require(java.lang.Double.isFinite(n), s"Non-finite number at $path")
    Atom(java.lang.Double.toString(n))
  private def bool(b: Boolean): Node = Atom(if b then "true" else "false")
  private val nul: Node = Atom("null")
  private def option[A](v: Option[A])(f: A => Node): Node = v.fold(nul)(f)
  private def seq[A](v: Seq[A])(f: A => Node): Node = Arr(v.iterator.map(f).toVector)
  private def map[A](v: Map[String, A])(f: A => Node): Node = Obj(v.iterator.map((k, value) => k -> f(value)).toVector)

  private def childPath(parent: String, key: String): String =
    if key.matches("[A-Za-z_][A-Za-z0-9_]*") then s"$parent.$key"
    else s"$parent[${ujson.write(ujson.Str(key))}]"

  private def json(value: ujson.Value, path: String): Node = value match
    case o: ujson.Obj => Obj(o.value.iterator.map((k, v) => k -> json(v, childPath(path, k))).toVector)
    case a: ujson.Arr => Arr(a.value.iterator.zipWithIndex.map((v, i) => json(v, s"$path[$i]")).toVector)
    case n: ujson.Num => double(n.num, path)
    case scalar => Atom(ujson.write(scalar))

  private def hires(h: HiresParams, path: String): Node = obj(
    "enabled" -> bool(h.enabled), "scale" -> double(h.scale, s"$path.scale"), "upscaler" -> string(h.upscaler),
    "secondPassSteps" -> int(h.secondPassSteps), "resizeX" -> int(h.resizeX), "resizeY" -> int(h.resizeY),
    "denoisingStrength" -> double(h.denoisingStrength, s"$path.denoisingStrength"), "samplerName" -> option(h.samplerName)(string),
    "scheduler" -> option(h.scheduler)(string), "prompt" -> string(h.prompt), "negativePrompt" -> string(h.negativePrompt),
    "cfgScale" -> option(h.cfgScale)(v => double(v, s"$path.cfgScale")))

  private def img2img(i: Img2ImgParams, path: String): Node = obj(
    "initImage" -> string(i.initImage), "mask" -> option(i.mask)(string), "denoisingStrength" -> double(i.denoisingStrength, s"$path.denoisingStrength"),
    "resizeMode" -> int(i.resizeMode), "maskBlur" -> int(i.maskBlur), "inpaintingFill" -> int(i.inpaintingFill),
    "inpaintFullRes" -> bool(i.inpaintFullRes), "inpaintFullResPadding" -> int(i.inpaintFullResPadding),
    "inpaintingMaskInvert" -> bool(i.inpaintingMaskInvert))

  private def lora(l: LoraRef, path: String): Node = obj(
    "name" -> string(l.name), "weight" -> double(l.weight, s"$path.weight"), "unet" -> option(l.unet)(v => double(v, s"$path.unet")),
    "lbw" -> option(l.lbw)(string), "start" -> option(l.start)(v => double(v, s"$path.start")), "stop" -> option(l.stop)(v => double(v, s"$path.stop")), "dyn" -> option(l.dyn)(int))

  private def control(c: ControlNetUnit, path: String): Node = obj(
    "enabled" -> bool(c.enabled), "model" -> string(c.model), "image" -> option(c.image)(string),
    "preprocessor" -> string(c.preprocessor), "weight" -> double(c.weight, s"$path.weight"), "guidanceStart" -> double(c.guidanceStart, s"$path.guidanceStart"),
    "guidanceEnd" -> double(c.guidanceEnd, s"$path.guidanceEnd"), "preprocessorArgs" -> Obj(c.preprocessorArgs.iterator.map((k, v) => k -> double(v, childPath(s"$path.preprocessorArgs", k))).toVector))

  private def params(p: GenerationParams): Node =
    val path = "params"
    obj(
    "prompt" -> string(p.prompt), "negativePrompt" -> string(p.negativePrompt), "seed" -> long(p.seed),
    "subseed" -> long(p.subseed), "subseedStrength" -> double(p.subseedStrength, s"$path.subseedStrength"), "samplerName" -> string(p.samplerName),
    "scheduler" -> string(p.scheduler), "steps" -> int(p.steps), "cfgScale" -> double(p.cfgScale, s"$path.cfgScale"),
    "width" -> int(p.width), "height" -> int(p.height), "batchSize" -> int(p.batchSize), "nIter" -> int(p.nIter),
    "clipSkip" -> int(p.clipSkip), "eta" -> option(p.eta)(v => double(v, s"$path.eta")), "hires" -> hires(p.hires, s"$path.hires"),
    "img2img" -> option(p.img2img)(v => img2img(v, s"$path.img2img")),
    "loras" -> Arr(p.loras.iterator.zipWithIndex.map((v, i) => lora(v, s"$path.loras[$i]")).toVector),
    "controlnet" -> Arr(p.controlnet.iterator.zipWithIndex.map((v, i) => control(v, s"$path.controlnet[$i]")).toVector),
    "hooks" -> Obj(p.hooks.iterator.map((k, v) => k -> json(v, childPath(s"$path.hooks", k))).toVector), "styles" -> seq(p.styles)(string), "checkpoint" -> option(p.checkpoint)(string),
    "vae" -> option(p.vae)(string), "extra" -> map(p.extra)(string))

  private def plugin(p: (Plugin, Values)): Node = obj(
    "pluginId" -> string(p._1.id), "pluginClass" -> string(p._1.getClass.getName), "values" -> map(p._2.map)(string))

  private def limits(l: JobLimits): Node = obj(
    "maxSide" -> int(l.maxSide), "maxImagesPerGeneration" -> int(l.maxImagesPerGeneration),
    "maxGenerations" -> int(l.maxGenerations), "maxPixelsPerGeneration" -> long(l.maxPixelsPerGeneration))

  private def context(c: SubmittedPayloadContext): Node = obj(
    "origin" -> string(c.origin), "sourceMode" -> option(c.sourceMode)(string),
    "configurationJson" -> option(c.configurationJson)(string), "uiVariationEnabled" -> option(c.uiVariationEnabled)(bool))

  private def capture(c: CaptureOptions): Node = obj("includeImageContents" -> bool(c.includeImageContents))

  def freeze(job: Job): FrozenJobEvidence =
    val node = obj(
      "id" -> string(job.id), "params" -> params(job.params), "script" -> option(job.script)(plugin),
      "alwaysOn" -> seq(job.alwaysOn)(plugin), "origin" -> string(job.origin), "limits" -> limits(job.limits),
      "submittedContext" -> option(job.submittedContext)(context), "captureOptions" -> option(job.captureOptions)(capture))
    new FrozenJobEvidence(node.render(false), node.render(true), job.script.map(_._1), job.alwaysOn.iterator.map(_._1).toVector)
