package sdfx.mcp

import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.{AppServices, EngineLaunch, PythonEnv}
import sdfx.plugins.StandardPlugins

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.ConcurrentLinkedQueue
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** AppServices over the stdlib-only fake engine (copied from the engine module's test resources). */
object TestEnv:
  lazy val script: Path = Paths.get(getClass.getResource("/fake_engine.py").toURI)

  lazy val python: Seq[String] =
    val probe = AppHome(Files.createTempDirectory("sdfx-pyprobe-"))
    PythonEnv.locate(probe, Settings()).getOrElse(sys.error("no python found for tests"))

  def launch(): EngineLaunch = EngineLaunch(python :+ script.toString, script.getParent, Map.empty, 30.seconds)

  def tempHome(): AppHome =
    val root = Files.createTempDirectory("sdfx-mcp-home-")
    def touch(rel: String): Unit =
      val p = root.resolve(rel)
      Files.createDirectories(p.getParent)
      Files.write(p, Array[Byte](1, 2, 3))
    touch("models/checkpoints/foo.safetensors")
    touch("models/checkpoints/sdxl/Bar_XL.safetensors")
    touch("models/lora/My Lora.safetensors")
    touch("models/vae/vae-ft-mse.safetensors")
    Files.createDirectories(root.resolve("config"))
    Files.writeString(root.resolve("config/styles.csv"), "name,prompt,negative_prompt\r\nPretty,\"{prompt}, masterpiece\",lowres\r\n")
    AppHome(root)

  def services(home: AppHome): AppServices =
    lazy val s: AppServices = new AppServices(home, Settings(lastModel = Some("foo")), dryRun = true,
      builtins = StandardPlugins.all(() => s.registry.xyzAxes), launchOverride = Some(() => launch()), persistSettings = false)
    s

  def deleteTree(p: Path): Unit =
    try
      val st = Files.walk(p)
      try st.sorted(java.util.Comparator.reverseOrder()).forEach(f => Files.deleteIfExists(f))
      finally st.close()
    catch case _: Exception => ()

  def eventually(timeout: FiniteDuration = 20.seconds)(cond: => Boolean): Boolean =
    val end = System.nanoTime() + timeout.toNanos
    var ok = cond
    while !ok && System.nanoTime() < end do
      Thread.sleep(25)
      ok = cond
    ok

  /** A tiny valid PNG (base64). */
  def pngBase64(w: Int = 16, h: Int = 16): String =
    val img = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB)
    val bos = new java.io.ByteArrayOutputStream()
    javax.imageio.ImageIO.write(img, "png", bos)
    java.util.Base64.getEncoder.encodeToString(bos.toByteArray)

  // ---- JSON-RPC helpers -------------------------------------------------------------------------------------

  private val ids = new java.util.concurrent.atomic.AtomicInteger(0)

  def request(method: String, params: ujson.Value = ujson.Obj(), id: ujson.Value = ujson.Null): ujson.Obj =
    ujson.Obj("jsonrpc" -> "2.0", "id" -> (if id == ujson.Null then ujson.Num(ids.incrementAndGet()) else id), "method" -> method, "params" -> params)

  def callReq(tool: String, args: ujson.Value, id: ujson.Value = ujson.Null, progressToken: Option[ujson.Value] = None): ujson.Obj =
    val p = ujson.Obj("name" -> tool, "arguments" -> args)
    progressToken.foreach(t => p("_meta") = ujson.Obj("progressToken" -> t))
    request("tools/call", p, id)

  final class Recorder extends (ujson.Value => Unit):
    val items = new ConcurrentLinkedQueue[ujson.Value]()
    def apply(v: ujson.Value): Unit = items.add(v)
    def all: Seq[ujson.Value] = items.asScala.toSeq

  /** Minimal JSON Schema check (type, required, properties, items, additionalProperties); returns problems. */
  def validate(v: ujson.Value, schema: ujson.Value, path: String = "$"): List[String] =
    val s = schema.obj
    val types = s.get("type").toSeq.flatMap {
      case ujson.Str(t) => Seq(t)
      case ujson.Arr(ts) => ts.toSeq.map(_.str)
      case _ => Nil
    }
    def matches(t: String): Boolean = t match
      case "object"  => v.isInstanceOf[ujson.Obj]
      case "array"   => v.isInstanceOf[ujson.Arr]
      case "string"  => v.isInstanceOf[ujson.Str]
      case "boolean" => v.isInstanceOf[ujson.Bool]
      case "number"  => v.isInstanceOf[ujson.Num]
      case "integer" => v match { case ujson.Num(n) => n.isWhole; case _ => false }
      case _         => true
    if types.nonEmpty && !types.exists(matches) then List(s"$path: expected ${s("type").render}, got $v")
    else
      v match
        case o: ujson.Obj =>
          val props: Map[String, ujson.Value] = s.get("properties").map(_.obj.toMap).getOrElse(Map.empty)
          val missing = s.get("required").toList.flatMap(_.arr.map(_.str)).filterNot(o.value.contains).map(k => s"$path: missing $k")
          val extra =
            if s.get("additionalProperties").contains(ujson.False) then o.value.keys.filterNot(props.contains).map(k => s"$path: unexpected $k").toList
            else Nil
          val additional = s.get("additionalProperties").toList.flatMap {
            case ps: ujson.Obj => o.value.toList.filterNot((k, _) => props.contains(k)).flatMap((k, x) => validate(x, ps, s"$path.$k"))
            case _ => Nil
          }
          missing ++ extra ++ additional ++ o.value.toList.flatMap((k, x) => props.get(k).toList.flatMap(ps => validate(x, ps, s"$path.$k")))
        case ujson.Arr(items) => s.get("items").toList.flatMap(is => items.toList.zipWithIndex.flatMap((x, i) => validate(x, is, s"$path[$i]")))
        case _                => Nil

  def textOf(result: ujson.Value): String =
    result("content").arr.filter(_("type").str == "text").map(_("text").str).mkString("\n")
