package sdfx.core.config

import sdfx.core.json.SnakePickle
import sdfx.core.json.SnakePickle.{ReadWriter => RW, *}

import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import java.nio.charset.{CodingErrorAction, StandardCharsets}
import java.nio.ByteBuffer
import java.io.{ByteArrayOutputStream, InputStream}
import java.security.MessageDigest
import scala.util.control.NonFatal

/** Directories searched for each model kind. All entries are absolute or relative to the app home. */
case class ModelDirs(
    checkpoints: Seq[String] = Seq("models/checkpoints"),
    vae: Seq[String] = Seq("models/vae"),
    lora: Seq[String] = Seq("models/lora"),
    controlnet: Seq[String] = Seq("models/controlnet"),
    embeddings: Seq[String] = Seq("models/embeddings"),
    upscalers: Seq[String] = Seq("models/upscalers"),
    faceRestore: Seq[String] = Seq("models/face_restore"),
    svd: Seq[String] = Seq("models/svd"),
    zero123: Seq[String] = Seq("models/zero123"),
    photomaker: Seq[String] = Seq("models/photomaker")
) derives RW:
  /** Adds the folders of an A1111 / reForge / Forge install (read-only use of its models). */
  def withA1111(root: String): ModelDirs =
    def add(cur: Seq[String], sub: String) =
      val p = Paths.get(root, sub).toString
      if cur.contains(p) then cur else cur :+ p
    copy(
      checkpoints = add(checkpoints, "models/Stable-diffusion"),
      vae = add(vae, "models/VAE"),
      lora = add(add(lora, "models/Lora"), "models/LyCORIS"),
      controlnet = add(controlnet, "models/ControlNet"),
      embeddings = add(embeddings, "embeddings"),
      upscalers = add(add(add(add(add(upscalers, "models/ESRGAN"), "models/RealESRGAN"), "models/SwinIR"), "models/ScuNET"), "models/DAT"),
      faceRestore = add(add(faceRestore, "models/GFPGAN"), "models/Codeformer"),
      svd = add(svd, "models/svd"),
      zero123 = add(zero123, "models/z123"),
      photomaker = add(photomaker, "models/photomaker")
    )

case class HiresFixTweaksPreferences(outputDir: String = "", showCfg: Boolean = true,
    showBatchSeed: Boolean = true, showPromptModes: Boolean = true, showRemoveNetworks: Boolean = true,
    showStyles: Boolean = true, promptModeRadio: Boolean = true, checkpointLabels: String = "Default",
    saveTemplate: Boolean = true, restoreTemplate: Boolean = true) derives RW

case class Settings(
    modelDirs: ModelDirs = ModelDirs(),
    outputDir: String = "outputs",
    pythonExe: Option[String] = None,
    dtype: String = "fp16",
    vramMode: String = "gpu",
    vaeTiling: Boolean = false,
    previewEvery: Int = 2,
    previewMethod: String = "approx",
    lastModel: Option[String] = None,
    lastVae: Option[String] = None,
    mcpHttpEnabled: Boolean = false,
    mcpHttpPort: Int = 7861,
    mcpToken: Option[String] = None,
    disabledPlugins: Seq[String] = Nil,
    pluginValues: Map[String, Map[String, String]] = Map.empty,
    hfOffline: Boolean = false,
    theme: String = "dark",
    allowCustomCode: Boolean = false,
    systemInfoMetadataFields: Seq[String] = Nil,
    comparisonSendButtons: String = "Off",
    comparisonTxtButton: String = "Off",
    comparisonMouseSend: Boolean = false,
    comparisonMouseAuto: Boolean = false,
    hardwareInfoEnabled: Boolean = false,
    hardwareInfoRemark: String = "",
    apiPayloadIncludeImageContents: Boolean = false,
    nsfwCardBlurDefault: String = "Blur",
    modelPreviewOverrides: Map[String, String] = Map.empty,
    modelPreviewFolders: Map[String, Seq[String]] = Map.empty,
    wdV3Model: Option[String] = None,
    wdV3Tags: Option[String] = None,
    tagAutocompleteCsv: Option[String] = None,
    hiresFixTweaks: HiresFixTweaksPreferences = HiresFixTweaksPreferences()
) derives RW

case class SettingsRecovery(reason: String, originalSha256: Option[String])
case class SettingsLoadResult(settings: Settings, recovery: Option[SettingsRecovery])
enum SettingsSaveIntent:
  case Explicit, Recover

object Settings:
  private val MaxInput = 16 * 1024 * 1024
  private val MaxUnits = 1000000L
  private val MaxEncoded = 4L * 1024 * 1024

  private final class PreviewBudget:
    private var units = 0L
    private var bytes = 64L
    def add(value: String, max: Int): Unit =
      require(value != null && value.length <= max, "preview configuration string exceeds limit or is null")
      units += value.length
      bytes += ujson.write(value).getBytes(StandardCharsets.UTF_8).length + 16L
      require(units <= MaxUnits && bytes <= MaxEncoded, "preview configuration aggregate limit exceeded")

  /** Admission before producing a second JSON representation of the preview maps. */
  def validatePreviewConfiguration(s: Settings): Either[String, Unit] =
    try
      val budget = new PreviewBudget
      require(s.modelPreviewOverrides != null && s.modelPreviewFolders != null, "preview map is null")
      require(s.modelPreviewOverrides.size <= 10000 && s.modelPreviewFolders.size <= 64, "preview map entry limit exceeded")
      s.modelPreviewOverrides.foreach { (k, v) => budget.add(k, 8192); budget.add(v, 8192) }
      var roots = 0
      s.modelPreviewFolders.foreach { (kind, paths) =>
        budget.add(kind, 128)
        require(paths != null, "preview folder list is null")
        paths.foreach { p =>
          roots += 1
          require(roots <= 256, "preview folder count exceeded")
          budget.add(p, 8192)
        }
      }
      Right(())
    catch case NonFatal(e) => Left(Option(e.getMessage).getOrElse("invalid preview configuration"))

  def load(file: Path): Settings = loadResult(file).settings

  def loadResult(file: Path): SettingsLoadResult =
    if Files.notExists(file) then return SettingsLoadResult(Settings(), None)
    loadResultFromStream(() => Files.newInputStream(file))

  private[core] def loadResultFromStream(open: () => InputStream): SettingsLoadResult =
    val digest = MessageDigest.getInstance("SHA-256")
    var hash: Option[String] = None
    try
      val buffer = new Array[Byte](8192)
      val out = new ByteArrayOutputStream()
      val in = open()
      var total = 0
      try
        var eof = false
        while total < MaxInput && !eof do
          val n = in.read(buffer, 0, math.min(buffer.length, MaxInput - total))
          if n < 0 then eof = true
          else if n == 0 then
            val one = in.read()
            if one < 0 then eof = true
            else
              digest.update(one.toByte)
              out.write(one)
              total += 1
          else
            digest.update(buffer, 0, n)
            out.write(buffer, 0, n)
            total += n
        if !eof && in.read() != -1 then throw new IllegalArgumentException("settings input exceeds 16 MiB")
      finally in.close()
      hash = Some(digest.digest().map(b => f"${b & 0xff}%02x").mkString)
      val decoder = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
      val preflight = SettingsPreviewPreflight(decoder.decode(ByteBuffer.wrap(out.toByteArray)).toString)
      val json = ujson.read(preflight.json)
      if !json.isInstanceOf[ujson.Obj] then throw new IllegalArgumentException("settings document must be an object")
      val obj = json.obj
      var problems = Vector.empty[String]
      preflight.problem.foreach(p => problems :+= s"invalid preview configuration: $p")
      obj.get("nsfw_card_blur_default").foreach {
        case ujson.Str(v) if Set("Blur", "Hide", "Show").contains(v) => ()
        case _ => obj("nsfw_card_blur_default") = ujson.Str("Hide"); problems :+= "invalid nsfw_card_blur_default"
      }
      val decoded = Option(SnakePickle.read[Settings](json)).getOrElse(throw new IllegalArgumentException("null settings"))
      val settings = decoded.copy(modelPreviewOverrides = preflight.overrides, modelPreviewFolders = preflight.folders)
      SettingsLoadResult(settings, Option.when(problems.nonEmpty)(SettingsRecovery(problems.mkString("; "), hash)))
    catch
      case NonFatal(e) =>
        SettingsLoadResult(Settings(nsfwCardBlurDefault = "Hide"), Some(SettingsRecovery(s"settings unreadable: ${e.getMessage}", hash)))

  def save(file: Path, s: Settings): Unit =
    validatePreviewConfiguration(s).fold(reason => throw new IllegalArgumentException(reason), _ => ())
    AtomicFiles.writeString(file, SnakePickle.write(s, indent = 2))

object AtomicFiles:
  def writeString(file: Path, content: String): Unit =
    val abs = file.toAbsolutePath
    Files.createDirectories(abs.getParent)
    val tmp = Files.createTempFile(abs.getParent, ".tmp-", ".part")
    try
      Files.writeString(tmp, content)
      Files.move(tmp, abs, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    finally Files.deleteIfExists(tmp)
