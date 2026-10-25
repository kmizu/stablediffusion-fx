package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.protocol.{EngineError, Protocol}
import sdfx.plugins.imageops.ImageOps

import java.awt.image.BufferedImage
import java.awt.Color
import java.nio.file.{Files, Path}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayBuffer

/** Deterministic PluginContext: "generates" solid colour images derived from the params (img2img may echo the init
  * image) and records every call. Interrupts are simulated like production: after `interruptAfter` successful
  * generations the next `generate` (or `upscale`) throws `EngineError(interrupted)` and only then does
  * `isInterrupted` become true (mid-generation interrupt).
  */
class FakeContext(val interruptAfter: Int = Int.MaxValue, val echoInit: Boolean = false) extends PluginContext:
  val dir: Path = Files.createTempDirectory("sdfx-plugins-test")
  private val gen = ArrayBuffer[GenerationParams]()
  def generated: Seq[GenerationParams] = gen.toSeq
  private val sv = ArrayBuffer[(String, BufferedImage, GenerationParams, String)]()
  def saved: Seq[(String, BufferedImage, GenerationParams, String)] = sv.toSeq
  /** Snapshot of each img2img call's init image and mask (plugins may delete their temp files afterwards). */
  val inputs = ArrayBuffer[(BufferedImage, Option[BufferedImage])]()
  /** Every output file path ever returned by generate. */
  val outputPaths = ArrayBuffer[Path]()
  val upscales = ArrayBuffer[(Path, String, Double)]()
  val logs = ArrayBuffer[String]()
  private var n = 0
  private var nextRandom = 1000L

  def colorOf(p: GenerationParams, seed: Long): Int =
    (seed, p.steps, p.cfgScale, p.prompt, p.negativePrompt, p.checkpoint, p.samplerName).hashCode & 0xffffff

  private def write(img: BufferedImage, sub: String): Path =
    val d = dir.resolve(sub)
    Files.createDirectories(d)
    n += 1
    val f = d.resolve(f"$n%05d.png")
    ImageIO.write(img, "png", f.toFile)
    f

  def solid(w: Int, h: Int, rgb: Int): BufferedImage =
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = img.createGraphics()
    g.setColor(new Color(rgb))
    g.fillRect(0, 0, w, h)
    g.dispose()
    img

  private var interrupted = false
  var interruptedCalls = 0

  private def interruptNow(): Nothing =
    interrupted = true
    interruptedCalls += 1
    throw EngineError(Protocol.ErrorCodes.Interrupted, "interrupted")

  /** Transient generations (not user outputs): recorded separately, files go to `transient/`. */
  private val trans = ArrayBuffer[GenerationParams]()
  def transientCalls: Seq[GenerationParams] = trans.toSeq
  val transientPaths = ArrayBuffer[Path]()

  /** Generations of any kind (regular + transient); the interrupt counter uses this. */
  def attempted: Int = gen.size + trans.size

  def generate(p: GenerationParams): Seq[OutputImage] =
    if interrupted || attempted >= interruptAfter then interruptNow()
    gen += p
    produce(p, if p.isImg2Img then "img2img" else "txt2img", outputPaths)

  override def generateTransient(p: GenerationParams): Seq[OutputImage] =
    if interrupted || attempted >= interruptAfter then interruptNow()
    trans += p
    produce(p, "transient", transientPaths)

  private def produce(p: GenerationParams, sub: String, record: ArrayBuffer[Path]): Seq[OutputImage] =
    p.img2img.foreach(ii => inputs += ((ImageIO.read(new java.io.File(ii.initImage)), ii.mask.map(m => ImageIO.read(new java.io.File(m))))))
    val base = if p.seed < 0 then { nextRandom += 1; nextRandom } else p.seed
    (0 until p.imagesPerJob).map { i =>
      val seed = base + i
      val img = p.img2img match
        case Some(ii) if echoInit =>
          val src = ImageIO.read(new java.io.File(ii.initImage))
          if src.getWidth == p.width && src.getHeight == p.height then src else ImageOps.resize(src, p.width, p.height)
        case _ => solid(p.width, p.height, colorOf(p, seed))
      val path = write(img, sub)
      record += path
      OutputImage(path, p, s"${p.prompt}\nSeed: $seed", seed)
    }

  def upscale(image: Path, upscaler: String, scale: Double): Path =
    if interrupted then interruptNow()
    upscales += ((image, upscaler, scale))
    val src = ImageIO.read(image.toFile)
    write(ImageOps.resize(src, math.round(src.getWidth * scale).toInt, math.round(src.getHeight * scale).toInt), "upscaled")

  def saveImage(img: BufferedImage, kind: String, params: GenerationParams, infotext: String): OutputImage =
    sv += ((kind, img, params, infotext))
    OutputImage(write(img, kind), params, infotext, params.seed)

  def tempFile(suffix: String): Path = Files.createTempFile(dir, "tmp", suffix)

  def choices(source: ChoiceSource): Seq[String] = source match
    case ChoiceSource.Samplers    => Seq("Euler a", "Euler", "DPM++ 2M", "DPM++ 2M SDE")
    case ChoiceSource.Schedulers  => Seq("Automatic", "Karras", "Exponential")
    case ChoiceSource.Checkpoints => Seq("modelA.safetensors", "modelB.safetensors")
    case ChoiceSource.Upscalers   => Seq("None", "Lanczos", "R-ESRGAN 4x+")
    case _                        => Nil

  /** Recorded non-generation engine calls (kind, params) and fake but well-formed replies. */
  val engineCalls = ArrayBuffer[(String, ujson.Obj)]()

  override def engineCall(kind: String, params: ujson.Obj): ujson.Value =
    if interrupted then interruptNow()
    engineCalls += ((kind, params))
    def src = ImageIO.read(new java.io.File(params("image_path").str))
    kind match
      case "restore_faces" | "novel_view" => ujson.Obj("path" -> write(src, "engine").toString)
      case "caption"   => ujson.Obj("caption" -> "a fake caption")
      case "run_code"  => ujson.Obj("stdout" -> "hello", "result" -> "42", "images" -> ujson.Arr(write(solid(8, 8, 0xff0000), "engine").toString))
      case "img2vid" =>
        val prev = write(src, "engine")
        val gif = prev.resolveSibling("clip.gif")
        Files.writeString(gif, "GIF89a")
        ujson.Obj("path" -> gif.toString, "frames" -> 4, "preview" -> prev.toString)
      case other => throw new UnsupportedOperationException(other)

  override def modelPath(source: ChoiceSource, name: String): Option[String] =
    Option.when(name.nonEmpty && name != "missing")(s"/models/${source.toString}/$name")

  def log(message: String): Unit = logs += message
  def status(text: String, done: Int, total: Int): Unit = ()
  def isInterrupted: Boolean = interrupted

  /** Writes a solid init image and returns its path. */
  def initImage(w: Int, h: Int, rgb: Int = 0x336699): String = write(solid(w, h, rgb), "init").toString

  def gradientInit(w: Int, h: Int): String =
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for y <- 0 until h; x <- 0 until w do img.setRGB(x, y, ((x * 255 / w) << 16) | ((y * 255 / h) << 8) | 0x40)
    write(img, "init").toString
