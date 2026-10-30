package sdfx.engine

import sdfx.core.png.{ImageScrambleSecret, PngMetadata, ScrambledPng}
import sdfx.core.model.GenerationParams
import scala.util.{Failure, Success, Try}
import scala.concurrent.{ExecutionContext, Future}
import java.nio.file.{Files, Path}

/** Scoped bridge for path-only worker APIs. The temporary PNG contains plaintext pixels. */
private[engine] object ScrambledInput:
  val LegacyWarning = "Legacy image password is unverified; decoded pixels may be incorrect."
  private def warnLegacy(decoded: ScrambledPng.Decoded, warn: String => Unit): Unit =
    if decoded.scrambled && decoded.passwordVerified.isEmpty then warn(LegacyWarning)

  def withPlain[A](source: Path, tempDir: Path, secret: Option[ImageScrambleSecret], warn: String => Unit = _ => ())(use: Path => A): A =
    val text = PngMetadata.readTexts(source)
    if !text.contains("Encrypt") then use(source)
    else
      val decoded = ScrambledPng.read(source, secret.map(_.use(identity)))
      warnLegacy(decoded, warn)
      Files.createDirectories(tempDir)
      val temp = Files.createTempFile(tempDir, "sdfx-unscrambled-", ".png")
      try
        PngMetadata.write(decoded.image, temp, text.toSeq.filterNot((key, _) => key == "Encrypt" || key == "EncryptPwdSha"))
        use(temp)
      finally Files.deleteIfExists(temp)

  /** Resolve the documented generation-image fields while preserving all other request values. */
  def withParams[A](params: GenerationParams, tempDir: Path, secret: Option[ImageScrambleSecret], warn: String => Unit = _ => ())(use: GenerationParams => A): A =
    val scalarHooks = Map("inpaint_difference" -> "base_path", "reference" -> "image", "revision" -> "image",
      "ip_adapter" -> "image", "controllllite" -> "image")
    def field(hook: String, key: String): Option[ujson.Value] =
      params.hooks.get(hook).collect { case o: ujson.Obj => o.value.get(key) }.flatten
    val images = params.img2img.toSeq.flatMap(i => Seq(Some(i.initImage), i.mask).flatten) ++
      params.controlnet.filter(_.enabled).flatMap(_.image) ++ scalarHooks.toSeq.flatMap((hook, key) => field(hook, key).collect { case ujson.Str(v) => v }) ++
      field("photomaker", "images").collect { case a: ujson.Arr => a.value.collect { case ujson.Str(v) => v }.toSeq }.getOrElse(Nil)
    val candidates = images.filter(_.nonEmpty).distinct.filter(s => Try(Files.isRegularFile(Path.of(s))).getOrElse(false))
    def loop(remaining: List[String], replacements: Map[String, String]): A = remaining match
      case source :: rest =>
        withPlain(Path.of(source), tempDir, secret, warn)(resolved => loop(rest, replacements.updated(source, resolved.toString)))
      case Nil =>
        def remap(s: String): String = replacements.getOrElse(s, s)
        val input = params.img2img.map(i => i.copy(initImage = remap(i.initImage), mask = i.mask.map(remap)))
        val control = params.controlnet.map(c => if c.enabled then c.copy(image = c.image.map(remap)) else c)
        val hooks = params.hooks.map { (name, value) =>
          val updated = value match
            case o: ujson.Obj if scalarHooks.contains(name) =>
              val key = scalarHooks(name)
              ujson.Obj.from(o.value.toSeq.map { (k, v) =>
                if k == key then k -> (v match { case ujson.Str(s) => ujson.Str(remap(s)); case _ => v }) else k -> v
              })
            case o: ujson.Obj if name == "photomaker" =>
              ujson.Obj.from(o.value.toSeq.map { (k, v) =>
                if k == "images" then k -> (v match
                  case a: ujson.Arr => ujson.Arr.from(a.value.map { case ujson.Str(s) => ujson.Str(remap(s)); case other => other })
                  case _ => v) else k -> v
              })
            case other => other
          name -> updated
        }
        use(params.copy(img2img = input, controlnet = control, hooks = hooks))
    loop(candidates.toList, Map.empty)

  /** Resolve named top-level image fields for a path-based non-generation worker request. */
  def withObject[A](params: ujson.Obj, keys: Seq[String], tempDir: Path, secret: Option[ImageScrambleSecret], warn: String => Unit = _ => ())(use: ujson.Obj => A): A =
    def loop(left: List[String], current: ujson.Obj): A = left match
      case Nil => use(current)
      case key :: tail =>
        current.value.get(key) match
          case Some(ujson.Str(value)) if value.nonEmpty && Try(Files.isRegularFile(Path.of(value))).getOrElse(false) =>
            withPlain(Path.of(value), tempDir, secret, warn) { resolved =>
              val next = ujson.Obj.from(current.value.toSeq.map { (k, v) => if k == key then k -> ujson.Str(resolved.toString) else k -> v })
              loop(tail, next)
            }
          case _ => loop(tail, current)
    loop(keys.toList, params)

  /** Asynchronous worker request: input files remain owned until its future completes. */
  def withObjectAsync[A](params: ujson.Obj, keys: Seq[String], tempDir: Path, secret: Option[ImageScrambleSecret], warn: String => Unit = _ => ())
      (use: ujson.Obj => Future[A])(using ec: ExecutionContext): Future[A] =
    Future {
      val owned = scala.collection.mutable.ArrayBuffer.empty[Path]
      def cleanup(): Unit =
        var first: Option[Throwable] = None
        owned.reverseIterator.foreach { path =>
          try Files.deleteIfExists(path)
          catch case e: Throwable => first match
            case None => first = Some(e)
            case Some(previous) => previous.addSuppressed(e)
        }
        first.foreach(throw _)
      try
        val current = ujson.Obj.from(params.value.toSeq)
        keys.foreach { key =>
          current.value.get(key) match
            case Some(ujson.Str(value)) if value.nonEmpty && Try(Files.isRegularFile(Path.of(value))).getOrElse(false) =>
              val source = Path.of(value)
              val text = PngMetadata.readTexts(source)
              if text.contains("Encrypt") then
                val decoded = ScrambledPng.read(source, secret.map(_.use(identity)))
                warnLegacy(decoded, warn)
                Files.createDirectories(tempDir)
                val temp = Files.createTempFile(tempDir, "sdfx-unscrambled-", ".png")
                owned += temp
                PngMetadata.write(decoded.image, temp, text.toSeq.filterNot((k, _) => k == "Encrypt" || k == "EncryptPwdSha"))
                current(key) = temp.toString
            case _ => ()
        }
        (current, () => cleanup())
      catch
        case e: Throwable =>
          try cleanup() catch case close: Throwable => e.addSuppressed(close)
          throw e
    }.flatMap { (resolved, cleanup) =>
      Future.fromTry(Try(use(resolved))).flatten.transform { result =>
        Try(cleanup()) match
          case Success(_) => result
          case Failure(close) => result match
            case Failure(error) => error.addSuppressed(close); Failure(error)
            case Success(_) => Failure(close)
      }
    }
