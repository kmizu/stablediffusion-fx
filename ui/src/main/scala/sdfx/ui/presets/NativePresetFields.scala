package sdfx.ui.presets

import javafx.beans.property.*
import sdfx.core.plugins.api.{ChoiceSource, Control}
import sdfx.core.presets.*
import sdfx.ui.vm.*

import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.StandardCharsets
import java.nio.charset.{CharacterCodingException, CodingErrorAction}
import java.security.MessageDigest
import scala.util.Try
import scala.util.control.NonFatal

/** The editable native form, read directly from its properties. No job, service or plugin is run here. */
final class NativePresetFields(vm: GenerationVM, plugins: PluginsVM, catalog: CatalogVM,
                               variationExpanded: BooleanProperty,
                               namedState: Boolean = false,
                               batchFields: Option[(BooleanProperty, StringProperty, StringProperty)] = None):
  private case class Binding(
    key: FieldKey,
    label: String,
    kind: PresetFieldKind,
    defaultValue: PresetAtom,
    choices: () => Vector[PresetAtom],
    min: Option[BigDecimal],
    max: Option[BigDecimal],
    read: () => PresetAtom,
    set: PresetAtom => Unit,
    multipleOf: Option[Long] = None
  ):
    def spec: PresetFieldSpec = PresetFieldSpec(key, label, kind, defaultValue, choices(), min, max)

  private def key(parts: String*): FieldKey = FieldKey(parts.toVector)
  private val limits = PresetLimits()

  /** Reject oversized raw identities before percent encoding can expand them. The remaining
    * render is at most a small multiple of the 512-byte key limit.
    */
  private def serializableKey(parts: String*): Option[FieldKey] =
    if parts.isEmpty || parts.exists(s => s == null || s.isEmpty) then return None
    var rawUnits = parts.size - 1
    var rawBytes = parts.size - 1
    var i = 0
    while i < parts.size do
      val segment = parts(i)
      rawUnits += segment.length
      if rawUnits > limits.maxKeyBytes then return None
      try
        val encoded = StandardCharsets.UTF_8.newEncoder()
          .onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT)
          .encode(CharBuffer.wrap(segment))
        rawBytes += encoded.remaining()
      catch case _: CharacterCodingException => return None
      if rawBytes > limits.maxKeyBytes then return None
      i += 1
    val candidate = key(parts*)
    FieldKey.parse(candidate.render).toOption.filter(_ == candidate)

  private def boundedPath(parts: String*): String =
    serializableKey(parts*).map(_.render).getOrElse("field")

  /** Script selection stores the plugin ID as a scalar value, not as a plugin field path. */
  private def serializableScriptId(id: String): Boolean =
    if id == null || id.isEmpty || id.length > limits.maxKeyBytes then false
    else
      try
        StandardCharsets.UTF_8.newEncoder()
          .onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT)
          .encode(CharBuffer.wrap(id)).remaining() <= limits.maxKeyBytes
      catch case _: CharacterCodingException => false
  private def n(v: Int): PresetAtom = PresetAtom.Integer(v.toLong)
  private def d(v: Double): PresetAtom =
    if java.lang.Double.isFinite(v) then PresetAtom.Decimal(BigDecimal.decimal(v)) else t(v.toString)
  private def t(v: String): PresetAtom = PresetAtom.Text(Option(v).getOrElse(""))
  private def choice(xs: => Seq[String]): () => Vector[PresetAtom] =
    () => xs.iterator.filter(_ != null).toVector.distinct.map(PresetAtom.Text.apply)
  private def list(xs: javafx.collections.ObservableList[String]): Vector[String] =
    xs.toArray(Array.empty[String]).toVector

  private val fresh = new GenerationVM(vm.mode)
  private val built = Vector.newBuilder[Binding]
  private val problems = Vector.newBuilder[PresetIssue]
  private var registered = 0
  private var registryOverflow = false

  private def add(b: Binding): Unit =
    if !registryOverflow then
      if registered == limits.maxRegisteredFieldsPerMode then registryOverflow = true
      else
        built += b
        registered += 1
  private def text(path: FieldKey, label: String, current: StringProperty, initial: String,
                   kind: PresetFieldKind = PresetFieldKind.Text): Unit =
    add(Binding(path, label, kind, t(initial), () => Vector.empty, None, None,
      () => t(current.get), a => current.set(a.asInstanceOf[PresetAtom.Text].value)))
  private def flag(path: FieldKey, label: String, current: BooleanProperty, initial: Boolean): Unit =
    add(Binding(path, label, PresetFieldKind.Flag, PresetAtom.Flag(initial), () => Vector.empty, None, None,
      () => PresetAtom.Flag(current.get), a => current.set(a.asInstanceOf[PresetAtom.Flag].value)))
  private def integer(path: FieldKey, label: String, current: IntegerProperty, initial: Int,
                      lo: Int, hi: Int, multipleOf: Option[Long] = None): Unit =
    add(Binding(path, label, PresetFieldKind.Integer, n(initial), () => Vector.empty,
      Some(BigDecimal(lo)), Some(BigDecimal(hi)),
      () => n(current.get), a => current.set(a.asInstanceOf[PresetAtom.Integer].value.toInt), multipleOf))
  private def long(path: FieldKey, label: String, current: LongProperty, initial: Long): Unit =
    add(Binding(path, label, PresetFieldKind.Integer, PresetAtom.Integer(initial), () => Vector.empty,
      None, None,
      () => PresetAtom.Integer(current.get), a => current.set(a.asInstanceOf[PresetAtom.Integer].value)))
  private def decimal(path: FieldKey, label: String, current: DoubleProperty, initial: Double,
                      lo: Double, hi: Double): Unit =
    add(Binding(path, label, PresetFieldKind.Decimal, d(initial), () => Vector.empty,
      Some(BigDecimal.decimal(lo)), Some(BigDecimal.decimal(hi)),
      () => d(current.get), a => current.set(a.asInstanceOf[PresetAtom.Decimal].value.toDouble)))
  private def stringChoice(path: FieldKey, label: String, current: StringProperty, initial: String,
                           options: () => Vector[PresetAtom]): Unit =
    add(Binding(path, label, PresetFieldKind.Choice, t(initial), options, None, None,
      () => t(current.get), a => current.set(a.asInstanceOf[PresetAtom.Text].value)))
  private def indexChoice(path: FieldKey, label: String, current: IntegerProperty, initial: Int,
                          size: Int): Unit =
    add(Binding(path, label, PresetFieldKind.Choice, n(initial),
      () => (0 until size).iterator.map(n).toVector, None, None,
      () => n(current.get), a => current.set(a.asInstanceOf[PresetAtom.Integer].value.toInt)))

  private def generation(): Unit =
    def g(name: String) = key("generation", name)
    text(g("prompt"), "Prompt", vm.prompt, fresh.prompt.get)
    text(g("negativePrompt"), "Negative prompt", vm.negativePrompt, fresh.negativePrompt.get)
    stringChoice(g("sampler"), "Sampling method", vm.sampler, fresh.sampler.get, choice(list(catalog.samplers)))
    stringChoice(g("scheduler"), "Scheduler", vm.scheduler, fresh.scheduler.get, choice(list(catalog.schedulers)))
    integer(g("steps"), "Sampling steps", vm.steps, fresh.steps.get, 1, 150)
    decimal(g("cfgScale"), "CFG Scale", vm.cfgScale, fresh.cfgScale.get, 1, 30)
    integer(g("width"), "Width", vm.width, fresh.width.get, 64, 4096, Some(8))
    integer(g("height"), "Height", vm.height, fresh.height.get, 64, 4096, Some(8))
    integer(g("batchCount"), "Batch count", vm.batchCount, fresh.batchCount.get, 1, 100)
    integer(g("batchSize"), "Batch size", vm.batchSize, fresh.batchSize.get, 1, 16)
    long(g("seed"), "Seed", vm.seed, fresh.seed.get)
    long(g("subseed"), "Variation seed", vm.subseed, fresh.subseed.get)
    decimal(g("subseedStrength"), "Variation strength", vm.subseedStrength, fresh.subseedStrength.get, 0, 1)
    // Source Extra starts false. Legacy native forms deliberately start true.
    flag(g("variationEnabled"), "Extra (Enable variation)", vm.variationEnabled, false)
    if vm.mode == Mode.Txt2Img then
      flag(g("hiresEnabled"), "Hires. fix", vm.hiresEnabled, fresh.hiresEnabled.get)
      stringChoice(g("hiresUpscaler"), "Upscaler", vm.hiresUpscaler, fresh.hiresUpscaler.get,
        choice(list(catalog.hiresUpscalers)))
      decimal(g("hiresScale"), "Upscale by", vm.hiresScale, fresh.hiresScale.get, 1, 5)
      integer(g("hiresSteps"), "Hires steps", vm.hiresSteps, fresh.hiresSteps.get, 0, 150)
      decimal(g("denoisingStrength"), "Denoising strength", vm.hiresDenoise, fresh.hiresDenoise.get, 0, 1)
      integer(g("hiresResizeX"), "Resize width to", vm.hiresResizeX, fresh.hiresResizeX.get, 0, 4096, Some(8))
      integer(g("hiresResizeY"), "Resize height to", vm.hiresResizeY, fresh.hiresResizeY.get, 0, 4096, Some(8))
      stringChoice(g("hiresSampler"), "Hires sampling method", vm.hiresSampler, fresh.hiresSampler.get,
        choice("" +: list(catalog.samplers)))
      stringChoice(g("hiresScheduler"), "Hires scheduler", vm.hiresScheduler, fresh.hiresScheduler.get,
        choice("" +: list(catalog.schedulers)))
      text(g("hiresPrompt"), "Hires prompt", vm.hiresPrompt, fresh.hiresPrompt.get)
      text(g("hiresNegative"), "Hires negative prompt", vm.hiresNegative, fresh.hiresNegative.get)
      decimal(g("hiresCfg"), "Hires CFG", vm.hiresCfg, fresh.hiresCfg.get, 0, 30)
    else
      decimal(g("denoisingStrength"), "Denoising strength", vm.denoise, fresh.denoise.get, 0, 1)
      indexChoice(g("resizeMode"), "Resize mode", vm.resizeMode, fresh.resizeMode.get, GenerationVM.ResizeModes.size)
      if vm.mode == Mode.Inpaint then
        integer(g("maskBlur"), "Mask blur", vm.maskBlur, fresh.maskBlur.get, 0, 64)
        indexChoice(g("inpaintingFill"), "Masked content", vm.inpaintingFill,
          fresh.inpaintingFill.get, GenerationVM.MaskedContent.size)
        flag(g("maskInvert"), "Mask mode", vm.maskInvert, fresh.maskInvert.get)
        flag(g("inpaintFullRes"), "Inpaint area", vm.inpaintFullRes, fresh.inpaintFullRes.get)
        integer(g("inpaintPadding"), "Only masked padding, pixels", vm.inpaintPadding,
          fresh.inpaintPadding.get, 0, 256)
    flag(key("ui", "variationExpanded"), "Variation section expanded", variationExpanded, false)

  private def controlNet(): Unit =
    vm.controlNet.zipWithIndex.foreach { (u, slot) =>
      val initial = fresh.controlNet(slot)
      def c(name: String) = key("controlnet", slot.toString, name)
      flag(c("enabled"), s"ControlNet ${slot + 1} enabled", u.enabled, initial.enabled.get)
      stringChoice(c("model"), s"ControlNet ${slot + 1} model", u.model, initial.model.get,
        choice(Seq("", "None") ++ list(catalog.choices(ChoiceSource.ControlNets))))
      text(c("image"), s"ControlNet ${slot + 1} image", u.image, initial.image.get, PresetFieldKind.PathText)
      stringChoice(c("preprocessor"), s"ControlNet ${slot + 1} preprocessor", u.preprocessor,
        initial.preprocessor.get, choice(Seq("none") ++ list(catalog.choices(ChoiceSource.Preprocessors))))
      decimal(c("weight"), s"ControlNet ${slot + 1} weight", u.weight, initial.weight.get, 0, 2)
      decimal(c("guidanceStart"), s"ControlNet ${slot + 1} start", u.guidanceStart, initial.guidanceStart.get, 0, 1)
      decimal(c("guidanceEnd"), s"ControlNet ${slot + 1} end", u.guidanceEnd, initial.guidanceEnd.get, 0, 1)
    }

  private def pluginControls(): Unit =
    val all = plugins.scriptForms ++ plugins.alwaysOnForms
    val duplicateIds = all.groupBy(_.plugin.id).collect { case (id, forms) if forms.size > 1 => id }.toSet
    duplicateIds.toVector.sorted.foreach(id =>
      problems += PresetIssue("duplicate-plugin-id", boundedPath("plugin", id), "Plugin ID is not unique"))
    val scriptIds = plugins.scriptForms.map(_.plugin.id).filterNot(duplicateIds).distinct.filter { id =>
      val valid = serializableScriptId(id)
      if !valid then problems += PresetIssue("unavailable-script-id", "script", "Script ID exceeds the native 512-byte ID limit")
      valid
    }
    stringChoice(key("script", "selected"), "Script", plugins.selectedScript, "", choice("" +: scriptIds))
    all.filterNot(f => duplicateIds(f.plugin.id)).foreach { form =>
      val controls = form.plugin.controls
      val duplicateControls = controls.groupBy(_.key).collect {
        case (id, same) if same.size > 1 => id
      }.toSet
      duplicateControls.toVector.sorted.foreach(id =>
        problems += PresetIssue("duplicate-control-key", boundedPath("plugin", form.plugin.id, id),
          "Control key is not unique in its plugin"))
      controls.filter(c => c.hasValue && !duplicateControls(c.key)).foreach { c =>
        if !registryOverflow then serializableKey("plugin", form.plugin.id, c.key) match
          case None => problems += PresetIssue("unavailable-field-key", "plugin",
            s"Plugin/control identity exceeds the canonical ${limits.maxKeyBytes}-byte field key limit")
          case Some(path) => form.prop(c.key) match
            case None => problems += PresetIssue("missing-control-property", path.render,
              "Control has no editable property")
            case Some(prop) =>
              val opts = c match
                case ch: Control.Choice => choice(ch.options)
                case dc: Control.DynamicChoice =>
                  () => ((if namedState then Seq("") else Seq.empty) ++
                    PluginValuesVM.options(dc, s => list(catalog.choices(s)))).map(PresetAtom.Text.apply).toVector.distinct
                case _ => () => Vector.empty[PresetAtom]
              val kind = c match
                case _: Control.Check => PresetFieldKind.Flag
                case _: Control.Slider => PresetFieldKind.Decimal
                case _: Control.IntField => PresetFieldKind.Integer
                case _: Control.Choice | _: Control.DynamicChoice => PresetFieldKind.Choice
                case _: Control.ImageFile => PresetFieldKind.PathText
                case _ => PresetFieldKind.Text
              val min = c match
                case s: Control.Slider => Some(BigDecimal.decimal(s.min))
                case i: Control.IntField => Some(BigDecimal(i.min))
                case _ => None
              val max = c match
                case s: Control.Slider => Some(BigDecimal.decimal(s.max))
                case i: Control.IntField => Some(BigDecimal(i.max))
                case _ => None
              def parse(raw: String): PresetAtom = c match
                case _: Control.Check => raw.trim.toLowerCase(java.util.Locale.ROOT) match
                  case "true" => PresetAtom.Flag(true)
                  case "false" => PresetAtom.Flag(false)
                  case _ => t(raw)
                case _: Control.Slider => Try(BigDecimal(raw.trim)).toOption.map(PresetAtom.Decimal.apply).getOrElse(t(raw))
                case _: Control.IntField => raw.trim.toLongOption.map(PresetAtom.Integer.apply).getOrElse(t(raw))
                case _ => t(raw)
              val default = parse(c.default)
              add(Binding(path, c.label, kind, default, opts, min, max,
                () => parse(Option(prop.get).getOrElse("")),
                a => prop.set(a match
                  case PresetAtom.Text(v) => v
                  case PresetAtom.Flag(v) => v.toString
                  case PresetAtom.Integer(v) => v.toString
                  case PresetAtom.Decimal(v) => v.bigDecimal.stripTrailingZeros.toPlainString)))
      }
    }

  if namedState then
    stringChoice(key("model", "checkpoint"), "Checkpoint", vm.shared.checkpoint, "",
      choice("" +: catalog.snapshot.get.checkpoints.map(_.name)))
    stringChoice(key("model", "vae"), "VAE", vm.shared.vae, "Automatic", choice(list(catalog.vaes)))
    integer(key("model", "clipSkip"), "Clip skip", vm.shared.clipSkip, 1, 1, 12)
    add(Binding(key("generation", "eta"), "Eta (blank = sampler default)", PresetFieldKind.Text,
      t(""), () => Vector.empty, None, None,
      () => t(vm.eta.get.map(_.toString).getOrElse("")), a => {
        val raw = a.asInstanceOf[PresetAtom.Text].value.trim
        val parsed = Option.when(raw.nonEmpty)(raw.toDouble)
        require(parsed.forall(n => java.lang.Double.isFinite(n) && n >= 0), "Eta must be finite and nonnegative")
        vm.eta.set(parsed)
      }))
    batchFields.foreach { (enabled, input, output) =>
      flag(key("batch", "enabled"), "WD batch enabled", enabled, false)
      text(key("batch", "input"), "WD batch input folder", input, "")
      text(key("batch", "output"), "WD batch output folder name", output, "")
    }

  generation()
  controlNet()
  pluginControls()
  private val bindings = if registryOverflow then Vector.empty else built.result()
  private val byKey = bindings.map(b => b.key -> b).toMap
  private val limitIssue = PresetIssue("registered-field-limit", "fields",
    s"Mode has more than ${limits.maxRegisteredFieldsPerMode} registered preset fields; presets are unavailable")
  private val bindingIssues = problems.result() ++ Option.when(registryOverflow)(limitIssue)

  private def digest(parts: Seq[String]): String =
    val md = MessageDigest.getInstance("SHA-256")
    parts.foreach { part =>
      val bytes = part.getBytes(StandardCharsets.UTF_8)
      md.update(ByteBuffer.allocate(4).putInt(bytes.length).array())
      md.update(bytes)
    }
    md.digest().map(b => f"${b & 0xff}%02x").mkString

  private def atomText(a: PresetAtom): String = a match
    case PresetAtom.Text(v) => "T" + v
    case PresetAtom.Integer(v) => "I" + v.toString
    case PresetAtom.Decimal(v) => "D" + v.bigDecimal.stripTrailingZeros.toPlainString
    case PresetAtom.Flag(v) => "F" + v.toString

  def schema: PresetSchema =
    val specs = bindings.map(_.spec)
    val revision = digest(Vector(vm.mode.toString) ++ specs.flatMap(s =>
      Vector(s.key.render, s.label, s.kind.toString, atomText(s.defaultValue),
        s.min.map(_.toString).getOrElse(""), s.max.map(_.toString).getOrElse("")) ++
        s.choices.map(atomText)) ++ bindingIssues.flatMap(i => Vector(i.code, i.path, i.message)))
    val mode = vm.mode match
      case Mode.Txt2Img => PresetMode.Txt2Img
      case Mode.Img2Img => PresetMode.Img2Img
      case Mode.Inpaint => PresetMode.Inpaint
    PresetSchema(mode, revision, specs, bindingIssues)

  /** Named states also retain prompt LoRA references; surface missing names before any form writes. */
  def namedStateResourceIssues(saved: Map[FieldKey, PresetAtom]): Vector[PresetIssue] =
    if !namedState then Vector.empty
    else
      def resourceName(raw: String): String =
        raw.replace('\\', '/').split('/').lastOption.getOrElse(raw).replaceFirst("\\.[^.]+$", "").toLowerCase(java.util.Locale.ROOT)
      val available = catalog.snapshot.get.loras.flatMap(e => Seq(resourceName(e.name), resourceName(e.path))).toSet
      Vector("prompt", "negativePrompt", "hiresPrompt", "hiresNegative").flatMap { field =>
        saved.get(key("generation", field)).toVector.flatMap {
          case PresetAtom.Text(prompt) => sdfx.core.model.LoraSyntax.extract(prompt).loras.filterNot(l => available(resourceName(l.name))).map { l =>
            PresetIssue("missing-lora", s"generation.$field", s"LoRA is absent from the current catalog: ${l.name}")
          }
          case _ => Vector.empty
        }
      }

  def capture(): FieldSnapshot =
    val values = bindings.map(b => b.key -> b.read()).toMap
    val revision = digest(values.toVector.sortBy(_._1.render).flatMap((k, v) => Vector(k.render, atomText(v))))
    FieldSnapshot(values, revision)

  private def validated(b: Binding, value: PresetAtom): Either[PresetIssue, PresetAtom] =
    val normalized: Option[PresetAtom] = (b.kind, value) match
      case (PresetFieldKind.Text | PresetFieldKind.PathText, a: PresetAtom.Text) => Some(a)
      case (PresetFieldKind.Flag, a: PresetAtom.Flag) => Some(a)
      case (PresetFieldKind.Integer, a: PresetAtom.Integer) => Some(a)
      case (PresetFieldKind.Integer, PresetAtom.Decimal(v)) if v.isWhole && v.isValidLong =>
        Some(PresetAtom.Integer(v.toLong))
      case (PresetFieldKind.Decimal, a: PresetAtom.Decimal) => Some(a)
      case (PresetFieldKind.Decimal, PresetAtom.Integer(v)) => Some(PresetAtom.Decimal(BigDecimal(v)))
      case (PresetFieldKind.Choice, a) if b.choices().contains(a) => Some(a)
      case _ => None
    normalized match
      case None => Left(PresetIssue("invalid-value", b.key.render, s"Invalid value or unavailable choice for ${b.label}"))
      case Some(a) =>
        val number = a match
          case PresetAtom.Integer(v) => Some(BigDecimal(v))
          case PresetAtom.Decimal(v) => Some(v)
          case _ => None
        val inRange = number.forall(v => b.min.forall(v >= _) && b.max.forall(v <= _))
        val aligned = number.forall(v => b.multipleOf.forall(m => (v % BigDecimal(m)) == BigDecimal(0)))
        val finiteDouble = a match
          case PresetAtom.Decimal(v) => java.lang.Double.isFinite(v.toDouble)
          case _ => true
        if !inRange || !aligned || !finiteDouble then
          Left(PresetIssue("out-of-range", b.key.render, s"Value outside native bounds for ${b.label}"))
        else Right(a)

  def write(changes: Map[FieldKey, PresetAtom]): Either[PresetError, Unit] =
    if registryOverflow then return Left(PresetError(Vector(limitIssue)))
    val checked = changes.toVector.sortBy((k, _) => boundedPath(k.segments*)).map { (k, a) =>
      serializableKey(k.segments*) match
        case None => Left(PresetIssue("unavailable-field", "field", "Invalid or oversized native field key"))
        case Some(_) => byKey.get(k) match
          case None => Left(PresetIssue("unavailable-field", boundedPath(k.segments*), "No unambiguous native field binding"))
          case Some(b) => validated(b, a).map(b -> _)
    }
    val errors = checked.collect { case Left(issue) => issue }
    if errors.nonEmpty then Left(PresetError(errors))
    else
      val prepared = checked.collect { case Right(pair) => pair }
      val before = prepared.map((b, _) => b.key -> b.read()).toMap
      var touched = Vector.empty[(Binding, PresetAtom)]
      try
        prepared.foreach { (b, a) =>
          touched = touched :+ (b -> before(b.key))
          b.set(a)
        }
        Right(())
      catch
        case NonFatal(e) =>
          val rollback = touched.reverse.flatMap { (b, old) =>
            try
              b.set(old)
              None
            catch case NonFatal(re) =>
              Some(PresetIssue("rollback-failed", b.key.render,
                Option(re.getMessage).getOrElse(re.getClass.getSimpleName)))
          }
          Left(PresetError(Vector(PresetIssue("write-failed", touched.lastOption.map(_._1.key.render).getOrElse("fields"),
            Option(e.getMessage).getOrElse(e.getClass.getSimpleName))) ++ rollback))
