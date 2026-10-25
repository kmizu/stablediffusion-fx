package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.plugins.api.ChoiceSource
import sdfx.core.protocol.*

import java.nio.file.Paths
import java.util.concurrent.CopyOnWriteArrayList
import scala.concurrent.duration.*
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.util.control.NonFatal

/** Everything the engine reported about available models. */
final case class CatalogSnapshot(
    checkpoints: Seq[ModelEntry] = Nil,
    vaes: Seq[FileEntry] = Nil,
    loras: Seq[FileEntry] = Nil,
    controlnets: Seq[FileEntry] = Nil,
    embeddings: Seq[FileEntry] = Nil,
    upscalers: Seq[FileEntry] = Nil,
    samplers: SamplerList = SamplerList(Nil, Nil),
    extraFiles: Map[String, Seq[FileEntry]] = Map.empty // face_restore, svd, zero123, photomaker
)

object ModelCatalog:
  /** Engine file kinds beyond the core ones, with the Settings.modelDirs field that lists their folders. */
  val ExtraKinds: Seq[(String, sdfx.core.config.ModelDirs => Seq[String])] = Seq(
    "face_restore" -> (_.faceRestore), "svd" -> (_.svd), "zero123" -> (_.zero123), "photomaker" -> (_.photomaker))

/** Cached model / sampler lists from the engine plus name -> path resolution. */
final class ModelCatalog(client: EngineClient, home: AppHome, settings: () => Settings):
  @volatile private var snap = CatalogSnapshot()
  @volatile private var knownHashes = Map.empty[String, String] // checkpoint path -> hash
  private val listeners = new CopyOnWriteArrayList[CatalogSnapshot => Unit]()

  def snapshot: CatalogSnapshot = snap
  def checkpoints: Seq[ModelEntry] = snap.checkpoints
  def vaes: Seq[FileEntry] = snap.vaes
  def loras: Seq[FileEntry] = snap.loras
  def controlnets: Seq[FileEntry] = snap.controlnets
  def embeddings: Seq[FileEntry] = snap.embeddings
  def upscalerFiles: Seq[FileEntry] = snap.upscalers
  def samplers: SamplerList = snap.samplers
  def files(kind: String): Seq[FileEntry] = snap.extraFiles.getOrElse(kind, Nil)

  def addListener(l: CatalogSnapshot => Unit): () => Unit =
    listeners.add(l)
    () => { listeners.remove(l); () }

  /** Replaces the cached lists (used by refresh and tests). */
  def set(s: CatalogSnapshot): Unit =
    snap = s
    listeners.forEach(l =>
      try l(s)
      catch case NonFatal(_) => ()
    )

  /** Remembers a checkpoint hash (from load_model) so lookups by hash prefix work. */
  def rememberHash(m: LoadedModel): Unit =
    if m.hash.nonEmpty then knownHashes = knownHashes.updated(m.path, m.hash)

  def hashOf(path: String): Option[String] = knownHashes.get(path)

  private def abs(dirs: Seq[String]): Seq[String] = dirs.map(d => home.resolve(d).toAbsolutePath.normalize().toString)

  @volatile private var errors: Seq[String] = Nil

  /** Problems from the last refresh (one entry per list that could not be fetched). */
  def lastErrors: Seq[String] = errors

  /** Fetches every list from the engine. A list that fails keeps its previous contents (recorded in
    * [[lastErrors]]); when the checkpoint list itself fails nothing is published and the future fails.
    */
  def refreshAsync(): Future[CatalogSnapshot] =
    given ExecutionContext = ExecutionContext.global
    val d = settings().modelDirs
    val old = snap
    val errs = new java.util.concurrent.ConcurrentLinkedQueue[String]()
    def keep[T](f: Future[T], previous: T, what: String): Future[T] = f.recover { case NonFatal(e) =>
      errs.add(s"$what: ${e.getMessage}")
      previous
    }
    val extraF = Future.sequence(ModelCatalog.ExtraKinds.map { (kind, dirs) =>
      keep(client.listFiles(kind, abs(dirs(d))).map(fs => kind -> fs.sortBy(_.name.toLowerCase)),
        kind -> old.extraFiles.getOrElse(kind, Nil), kind)
    })
    val ckF = client.listModels(abs(d.checkpoints))
    val fs = (
      keep(client.listFiles("vae", abs(d.vae)), old.vaes, "vae"),
      keep(client.listFiles("lora", abs(d.lora)), old.loras, "lora"),
      keep(client.listFiles("controlnet", abs(d.controlnet)), old.controlnets, "controlnet"),
      keep(client.listFiles("embeddings", abs(d.embeddings)), old.embeddings, "embeddings"),
      keep(client.listFiles("upscalers", abs(d.upscalers)), old.upscalers, "upscalers"),
      keep(client.listSamplers(), old.samplers, "samplers")
    )
    val all = for
      v <- fs._1
      l <- fs._2
      cn <- fs._3
      e <- fs._4
      u <- fs._5
      s <- fs._6
      x <- extraF
      c <- ckF.recoverWith { case NonFatal(e) =>
        errs.add(s"checkpoints: ${e.getMessage}")
        Future.failed(e)
      }
    yield
      val ns = CatalogSnapshot(c.sortBy(_.name.toLowerCase), v.sortBy(_.name.toLowerCase), l.sortBy(_.name.toLowerCase),
        cn.sortBy(_.name.toLowerCase), e.sortBy(_.name.toLowerCase), u.sortBy(_.name.toLowerCase), s, x.toMap)
      set(ns)
      ns
    all.andThen { case _ =>
      import scala.jdk.CollectionConverters.*
      errors = errs.asScala.toList
      errors.foreach(e => System.err.println(s"[sdfx] listing $e"))
    }

  def refresh(timeout: FiniteDuration = 2.minutes): CatalogSnapshot = Await.result(refreshAsync(), timeout)

  // ---- resolution ---------------------------------------------------------------------------------------

  private def stem(s: String): String =
    val n = Paths.get(s.replace('\\', '/')).getFileName.toString
    val i = n.lastIndexOf('.')
    if i > 0 then n.substring(0, i) else n

  private def norm(s: String): String = s.replace('\\', '/').trim

  /** Generic matcher: exact name, exact path, name without extension, file stem; then case-insensitive. */
  private def find[T](items: Seq[T], q: String, name: T => String, path: T => String): Option[T] =
    val query = norm(q)
    if query.isEmpty then None
    else
      val exact: Seq[T => Boolean] = Seq(
        t => norm(name(t)) == query,
        t => norm(path(t)) == query,
        t => stem(name(t)) == query,
        t => stem(path(t)) == query,
        t => stem(name(t)) == stem(query)
      )
      val ci: Seq[T => Boolean] = Seq(
        t => norm(name(t)).equalsIgnoreCase(query),
        t => norm(path(t)).equalsIgnoreCase(query),
        t => stem(name(t)).equalsIgnoreCase(query),
        t => stem(path(t)).equalsIgnoreCase(query),
        t => stem(path(t)).equalsIgnoreCase(stem(query))
      )
      (exact ++ ci).iterator.map(p => items.find(p)).collectFirst { case Some(t) => t }

  private val TitleRe = """^(.*?)\s*\[([0-9a-fA-F]{6,64})\]$""".r

  /** Checkpoint by name, A1111 title (`name [hash]`), path, stem or hash prefix. */
  def findCheckpoint(q: String): Option[ModelEntry] =
    val query = q.trim
    def byHash(h: String) =
      val hl = h.toLowerCase
      checkpoints.find(c => knownHashes.get(c.path).exists(k => k.toLowerCase.startsWith(hl) || hl.startsWith(k.toLowerCase)))
    find(checkpoints, query, _.name, _.path)
      .orElse(query match
        case TitleRe(n, h) => find(checkpoints, n, _.name, _.path).orElse(byHash(h))
        case _             => None
      )
      .orElse(Option.when(query.matches("[0-9a-fA-F]{6,64}"))(query).flatMap(byHash))

  def findVae(q: String): Option[FileEntry] = find(vaes, q, _.name, _.path)
  def findLora(q: String): Option[FileEntry] = find(loras, q, _.name, _.path)
  def findControlNet(q: String): Option[FileEntry] =
    find(controlnets, q, _.name, _.path).orElse(q match
      case TitleRe(n, _) => find(controlnets, n, _.name, _.path)
      case _             => None
    )
  def findEmbedding(q: String): Option[FileEntry] = find(embeddings, q, _.name, _.path)
  def findUpscaler(q: String): Option[FileEntry] = find(upscalerFiles, q, _.name, _.path)

  /** LoRA name -> path for every name that resolves. */
  def loraPaths(names: Seq[String]): Map[String, String] = names.flatMap(n => findLora(n).map(n -> _.path)).toMap

  val latentUpscalers: Seq[String] = Seq("Latent", "Latent (antialiased)", "Latent (bicubic)", "Latent (nearest-exact)")

  def upscalerNames: Seq[String] = (samplers.upscalers ++ upscalerFiles.map(_.name)).distinct

  def choices(src: ChoiceSource): Seq[String] = src match
    case ChoiceSource.Samplers       => samplers.samplers.filter(_.supported).map(_.name)
    case ChoiceSource.Schedulers     => samplers.schedulers
    case ChoiceSource.Checkpoints    => checkpoints.map(_.name)
    case ChoiceSource.Vaes           => vaes.map(_.name)
    case ChoiceSource.Loras          => loras.map(_.name)
    case ChoiceSource.Upscalers      => upscalerNames.filterNot(_.startsWith("Latent"))
    case ChoiceSource.HiresUpscalers => (latentUpscalers.filter(l => samplers.upscalers.isEmpty || samplers.upscalers.contains(l) || l == "Latent") ++ upscalerNames).distinct
    case ChoiceSource.Preprocessors  => samplers.preprocessors
    case ChoiceSource.ControlNets    => controlnets.map(_.name)
    case ChoiceSource.FaceRestorers  => files("face_restore").map(_.name)
    case ChoiceSource.SvdModels      => files("svd").map(_.name)
    case ChoiceSource.Zero123Models  => files("zero123").map(_.name)
    case ChoiceSource.PhotoMakerModels => files("photomaker").map(_.name)

  /** Path of a choice listed by [[choices]] for file-backed sources. */
  def pathOf(src: ChoiceSource, name: String): Option[String] =
    val entries: Seq[FileEntry] = src match
      case ChoiceSource.ControlNets      => controlnets
      case ChoiceSource.Vaes             => vaes
      case ChoiceSource.Loras            => loras
      case ChoiceSource.Upscalers | ChoiceSource.HiresUpscalers => upscalerFiles
      case ChoiceSource.FaceRestorers    => files("face_restore")
      case ChoiceSource.SvdModels        => files("svd")
      case ChoiceSource.Zero123Models    => files("zero123")
      case ChoiceSource.PhotoMakerModels => files("photomaker")
      case ChoiceSource.Checkpoints      => checkpoints.map(m => FileEntry(m.name, m.path))
      case _                             => Nil
    find(entries, name, _.name, _.path).map(_.path)
