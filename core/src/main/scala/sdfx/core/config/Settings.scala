package sdfx.core.config

import sdfx.core.json.SnakePickle
import sdfx.core.json.SnakePickle.{ReadWriter => RW, *}

import java.nio.file.{Files, Path, Paths, StandardCopyOption}

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
    allowCustomCode: Boolean = false
) derives RW

object Settings:
  def load(file: Path): Settings =
    if !Files.exists(file) then Settings()
    else
      // upickle reads a JSON `null` document as a null reference: treat it like any other unusable file.
      try Option(SnakePickle.read[Settings](Files.readString(file))).getOrElse(Settings())
      catch
        case e: Exception =>
          System.err.println(s"[sdfx] settings unreadable, using defaults: ${e.getMessage}")
          Settings()

  def save(file: Path, s: Settings): Unit = AtomicFiles.writeString(file, SnakePickle.write(s, indent = 2))

object AtomicFiles:
  def writeString(file: Path, content: String): Unit =
    val abs = file.toAbsolutePath
    Files.createDirectories(abs.getParent)
    val tmp = Files.createTempFile(abs.getParent, ".tmp-", ".part")
    try
      Files.writeString(tmp, content)
      Files.move(tmp, abs, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    finally Files.deleteIfExists(tmp)
