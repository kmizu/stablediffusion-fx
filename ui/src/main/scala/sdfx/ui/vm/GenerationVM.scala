package sdfx.ui.vm

import javafx.beans.property.*
import javafx.collections.{FXCollections, ObservableList}
import sdfx.core.infotext.Infotext
import sdfx.core.model.*

import java.nio.file.{Files, Paths}

/** Model selection shared by all tabs (top bar): checkpoint, VAE and clip skip. */
final class ModelVM:
  val checkpoint = new SimpleStringProperty("")
  val vae = new SimpleStringProperty("Automatic")
  val clipSkip = new SimpleIntegerProperty(1)

/** One ControlNet unit editor. */
final class ControlNetUnitVM:
  val enabled = new SimpleBooleanProperty(false)
  val model = new SimpleStringProperty("")
  val image = new SimpleStringProperty("")
  val preprocessor = new SimpleStringProperty("none")
  val weight = new SimpleDoubleProperty(1.0)
  val guidanceStart = new SimpleDoubleProperty(0.0)
  val guidanceEnd = new SimpleDoubleProperty(1.0)

  def isUsable: Boolean = enabled.get && model.get.nonEmpty && model.get != "None"

  def toUnit: ControlNetUnit =
    ControlNetUnit(
      enabled.get,
      model.get,
      Option(image.get).filter(_.nonEmpty),
      Option(preprocessor.get).filter(_.nonEmpty).getOrElse("none"),
      weight.get,
      guidanceStart.get,
      guidanceEnd.get
    )

  def load(u: Option[ControlNetUnit]): Unit =
    val x = u.getOrElse(ControlNetUnit(enabled = false))
    enabled.set(x.enabled)
    model.set(x.model)
    image.set(x.image.getOrElse(""))
    preprocessor.set(x.preprocessor)
    weight.set(x.weight)
    guidanceStart.set(x.guidanceStart)
    guidanceEnd.set(x.guidanceEnd)

object GenerationVM:
  val ControlNetUnits = 3
  val SizePresets: Seq[(Int, Int)] =
    Seq(512 -> 512, 512 -> 768, 768 -> 512, 768 -> 768, 832 -> 1216, 896 -> 1152, 1024 -> 1024, 1152 -> 896, 1216 -> 832)
  val ResizeModes: Seq[String] = Seq("Just resize", "Crop and resize", "Resize and fill")
  val MaskedContent: Seq[String] = Seq("fill", "original", "latent noise", "latent nothing")

/** All editable generation parameters of one tab as JavaFX properties (works without a running toolkit). */
final class GenerationVM(val mode: Mode, val shared: ModelVM = new ModelVM):
  val prompt = new SimpleStringProperty("")
  val negativePrompt = new SimpleStringProperty("")
  val sampler = new SimpleStringProperty("DPM++ 2M")
  val scheduler = new SimpleStringProperty("Automatic")
  val steps = new SimpleIntegerProperty(20)
  val cfgScale = new SimpleDoubleProperty(7.0)
  val width = new SimpleIntegerProperty(512)
  val height = new SimpleIntegerProperty(512)
  val batchCount = new SimpleIntegerProperty(1)
  val batchSize = new SimpleIntegerProperty(1)
  val seed = new SimpleLongProperty(-1L)
  val subseed = new SimpleLongProperty(-1L)
  val subseedStrength = new SimpleDoubleProperty(0.0)
  val variationEnabled = new SimpleBooleanProperty(true)

  // hires fix (txt2img only)
  val hiresEnabled = new SimpleBooleanProperty(false)
  val hiresUpscaler = new SimpleStringProperty("Latent")
  val hiresScale = new SimpleDoubleProperty(2.0)
  val hiresSteps = new SimpleIntegerProperty(0)
  val hiresDenoise = new SimpleDoubleProperty(0.55)
  val hiresResizeX = new SimpleIntegerProperty(0)
  val hiresResizeY = new SimpleIntegerProperty(0)
  val hiresSampler = new SimpleStringProperty("") // "" = same as first pass
  val hiresScheduler = new SimpleStringProperty("")
  val hiresPrompt = new SimpleStringProperty("")
  val hiresNegative = new SimpleStringProperty("")
  val hiresCfg = new SimpleDoubleProperty(0.0) // 0 = same as first pass

  // img2img / inpaint
  val initImage = new SimpleStringProperty("")
  val maskImage = new SimpleStringProperty("") // exported mask PNG (set by the view right before toParams)
  val denoise = new SimpleDoubleProperty(0.75)
  val resizeMode = new SimpleIntegerProperty(0)
  val maskBlur = new SimpleIntegerProperty(4)
  val inpaintingFill = new SimpleIntegerProperty(1)
  val inpaintFullRes = new SimpleBooleanProperty(false)
  val inpaintPadding = new SimpleIntegerProperty(32)
  val maskInvert = new SimpleBooleanProperty(false)
  val hasMask = new SimpleBooleanProperty(false) // whether anything is painted (maintained by the mask view)
  // a mask exported for another image must never be sent with a new one
  initImage.addListener((_, _, _) => maskImage.set(""))

  val controlNet: Vector[ControlNetUnitVM] = Vector.fill(GenerationVM.ControlNetUnits)(new ControlNetUnitVM)

  /** Problems from the last validation, shown inline above the form. */
  val problems: ObservableList[String] = FXCollections.observableArrayList()

  def hires: HiresParams =
    HiresParams(
      enabled = mode == Mode.Txt2Img && hiresEnabled.get,
      scale = hiresScale.get,
      upscaler = hiresUpscaler.get,
      secondPassSteps = hiresSteps.get,
      resizeX = hiresResizeX.get,
      resizeY = hiresResizeY.get,
      denoisingStrength = hiresDenoise.get,
      samplerName = Option(hiresSampler.get).filter(_.nonEmpty),
      scheduler = Option(hiresScheduler.get).filter(_.nonEmpty),
      prompt = hiresPrompt.get,
      negativePrompt = hiresNegative.get,
      cfgScale = Option(hiresCfg.get).filter(_ > 0)
    )

  /** Hires target size for the current first-pass size. */
  def hiresTarget: (Int, Int) = hires.targetSize(width.get, height.get)

  def img2img: Option[Img2ImgParams] =
    Option.when(mode.usesInitImage)(
      Img2ImgParams(
        initImage = initImage.get,
        mask = Option.when(mode == Mode.Inpaint)(maskImage.get).filter(_.nonEmpty),
        denoisingStrength = denoise.get,
        resizeMode = resizeMode.get,
        maskBlur = maskBlur.get,
        inpaintingFill = inpaintingFill.get,
        inpaintFullRes = inpaintFullRes.get,
        inpaintFullResPadding = inpaintPadding.get,
        inpaintingMaskInvert = maskInvert.get
      )
    )

  def toParams: GenerationParams =
    GenerationParams(
      prompt = prompt.get,
      negativePrompt = negativePrompt.get,
      seed = seed.get,
      subseed = subseed.get,
      subseedStrength = if variationEnabled.get then subseedStrength.get else 0.0,
      samplerName = sampler.get,
      scheduler = scheduler.get,
      steps = steps.get,
      cfgScale = cfgScale.get,
      width = width.get,
      height = height.get,
      batchSize = batchSize.get,
      nIter = batchCount.get,
      clipSkip = shared.clipSkip.get,
      hires = hires,
      img2img = img2img,
      controlnet = controlNet.filter(_.isUsable).map(_.toUnit),
      checkpoint = Option(shared.checkpoint.get).map(_.trim).filter(_.nonEmpty),
      vae = Option(shared.vae.get).map(_.trim).filter(_.nonEmpty)
    )

  /** Form state for persistence: like [[toParams]] but keeps disabled ControlNet units. */
  def snapshot: GenerationParams =
    toParams.copy(subseed = subseed.get, subseedStrength = subseedStrength.get,
      controlnet = controlNet.map(_.toUnit), checkpoint = None, vae = None)

  /** Loads params into the form. LoRAs given as structured refs are written back into the prompt as tags.
    * With `keepImages` the current init image / mask stay (restore parameters onto the img2img inputs).
    */
  def load(p: GenerationParams, keepImages: Boolean = false, includeModel: Boolean = false,
      variationEnabledState: Boolean = true): Unit =
    val missingTags = p.loras.filterNot(l => p.prompt.contains(s"<lora:${l.name}:") || p.prompt.contains(s"<lora:${l.name}>"))
    val promptText =
      if missingTags.isEmpty then p.prompt
      else (p.prompt +: missingTags.map(l => LoraSyntax.tag(l.name, l.weight))).filter(_.nonEmpty).mkString(", ")
    prompt.set(promptText)
    negativePrompt.set(p.negativePrompt)
    seed.set(p.seed)
    subseed.set(p.subseed)
    subseedStrength.set(p.subseedStrength)
    variationEnabled.set(variationEnabledState)
    sampler.set(p.samplerName)
    scheduler.set(p.scheduler)
    steps.set(p.steps)
    cfgScale.set(p.cfgScale)
    width.set(p.width)
    height.set(p.height)
    batchSize.set(p.batchSize)
    batchCount.set(p.nIter)
    shared.clipSkip.set(p.clipSkip)
    if includeModel then
      p.checkpoint.foreach(shared.checkpoint.set)
      p.vae.foreach(shared.vae.set)
    val h = p.hires
    hiresEnabled.set(h.enabled)
    hiresUpscaler.set(h.upscaler)
    hiresScale.set(h.scale)
    hiresSteps.set(h.secondPassSteps)
    hiresDenoise.set(h.denoisingStrength)
    hiresResizeX.set(h.resizeX)
    hiresResizeY.set(h.resizeY)
    hiresSampler.set(h.samplerName.getOrElse(""))
    hiresScheduler.set(h.scheduler.getOrElse(""))
    hiresPrompt.set(h.prompt)
    hiresNegative.set(h.negativePrompt)
    hiresCfg.set(h.cfgScale.getOrElse(0.0))
    p.img2img.foreach { i =>
      if !keepImages then
        initImage.set(i.initImage)
        maskImage.set(i.mask.getOrElse(""))
      denoise.set(i.denoisingStrength)
      resizeMode.set(i.resizeMode)
      maskBlur.set(i.maskBlur)
      inpaintingFill.set(i.inpaintingFill)
      inpaintFullRes.set(i.inpaintFullRes)
      inpaintPadding.set(i.inpaintFullResPadding)
      maskInvert.set(i.inpaintingMaskInvert)
    }
    controlNet.zipWithIndex.foreach((u, idx) => u.load(p.controlnet.lift(idx)))

  /** Restores the fields of an A1111 infotext on top of the current form (img2img inputs and ControlNet are kept). */
  def restoreInfotext(text: String): GenerationParams =
    val base = snapshot
    val parsed = Infotext.parse(text)
    val restored = Infotext.toParams(parsed, base).copy(extra = Map.empty, controlnet = base.controlnet)
    val hasVariation = parsed.fields.contains("Variation seed") || parsed.fields.contains("Variation seed strength")
    load(restored, keepImages = true, variationEnabledState = hasVariation)
    restored

  /** Validation of the form; empty when the job can be submitted. Also updates [[problems]]. */
  def validate(fileExists: String => Boolean = p => Files.isRegularFile(Paths.get(p))): List[String] =
    val init = Option(initImage.get).map(_.trim).getOrElse("")
    val modeProblems = List(
      Option.when(mode.usesInitImage && init.isEmpty)("Choose an init image (drop, paste or open a file)"),
      Option.when(mode.usesInitImage && init.nonEmpty && !fileExists(init))(s"Init image not found: $init"),
      Option.when(mode == Mode.Inpaint && !hasMask.get)("Paint a mask over the area to inpaint"),
      Option.when(controlNet.exists(u => u.enabled.get && !u.isUsable))("A ControlNet unit is enabled without a model"),
      Option.when(controlNet.exists(u => u.isUsable && u.image.get.isEmpty && !mode.usesInitImage))("A ControlNet unit needs a control image"),
      Option.when(controlNet.exists(u => u.isUsable && u.guidanceStart.get > u.guidanceEnd.get))("ControlNet start must be <= end")
    ).flatten
    val all = toParams.validate ++ modeProblems
    problems.setAll(all*)
    all
