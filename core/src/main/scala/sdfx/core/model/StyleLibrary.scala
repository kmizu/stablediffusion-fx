package sdfx.core.model

import java.nio.file.{Files, Path, StandardCopyOption}
import java.util.UUID
import scala.jdk.CollectionConverters.*
import scala.util.Using

class StyleConflict(message: String) extends IllegalStateException(message)

/** Extended-style identity is a UUID, not its display name or checkpoint/VAE tuple. */
case class ExtendedStyle(id: String, name: String, checkpoint: String, vae: String, prompt: String,
    negativePrompt: String, width: Int, height: Int):
  def effectiveVae: String = ExtendedStyle.normalizeVae(vae)
  def restore(base: GenerationParams): GenerationParams = base.copy(prompt = prompt, negativePrompt = negativePrompt,
    width = width, height = height, checkpoint = Option(checkpoint).filter(_.nonEmpty), vae = Some(effectiveVae))

object ExtendedStyle:
  def normalizeVae(value: String): String = Option(value).filter(_.trim.nonEmpty).getOrElse("Automatic")

/** AppHome-scoped style storage. Never writes an import/source file; all replacements compare byte versions. */
final class StyleLibrary(val master: Path, clock: () => Long = () => System.currentTimeMillis()):
  private val directory = master.toAbsolutePath.normalize().getParent
  val extendedFile: Path = directory.resolve("extended-styles.csv")
  val backupDirectory: Path = directory.resolve("style-backups")
  private var pending = false
  private var lastBackup = clock()
  private var sequence = 0L
  private val extendedHeader = Vector("id", "name", "sd_model_checkpoint", "sd_vae", "prompt", "negative_prompt", "width", "height")
  private def categoryFiles: Vector[Path] =
    val folder = directory.resolve("style-files")
    if !Files.isDirectory(folder) then Vector.empty
    else Using.resource(Files.list(folder))(_.iterator().asScala.filter(p => Files.isRegularFile(p) && p.toString.endsWith(".csv")).toVector)
  private def diskVersion(path: Path): String = StyleDocument.fingerprint(if Files.exists(path) then Files.readAllBytes(path) else Array.emptyByteArray, Files.exists(path))
  private def check(doc: StyleDocument): Unit =
    if diskVersion(doc.path) != doc.version then throw new StyleConflict(s"${doc.path.getFileName} changed outside this editor. Reload before saving; your external edits were not overwritten.")
  private def atomic(path: Path, bytes: Array[Byte], expected: String): Unit =
    require(!Files.isSymbolicLink(path), "Edit the actual styles file instead of replacing a symbolic link")
    Files.createDirectories(path.toAbsolutePath.getParent)
    val temp = Files.createTempFile(path.toAbsolutePath.getParent, ".styles-", ".tmp")
    try
      Files.write(temp, bytes)
      if diskVersion(path) != expected then throw new StyleConflict(s"${path.getFileName} changed while saving; reload before saving")
      Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
    finally Files.deleteIfExists(temp)
  private def owned(path: Path): Boolean =
    val p = path.toAbsolutePath.normalize()
    p == master.toAbsolutePath.normalize() || p == extendedFile || p.getParent == directory.resolve("style-files")
  /** Stage every destination before changing any. Roll back our own published replacements on I/O failure;
    * a concurrently changed external file is never overwritten by rollback either.
    */
  private def transaction(changes: Seq[(Path, Option[Array[Byte]], String)]): Unit =
    val prepared = scala.collection.mutable.ArrayBuffer[(Path, Option[Path], Option[Array[Byte]], String)]()
    val applied = scala.collection.mutable.ArrayBuffer[(Path, Option[Array[Byte]], String)]()
    try
      changes.foreach { (path, bytes, expected) =>
        require(owned(path) && !Files.isSymbolicLink(path), "Style writes must stay in this library; import sources are read-only")
        if diskVersion(path) != expected then throw new StyleConflict(s"${path.getFileName} changed outside this editor; reload before saving")
        val old = Option.when(Files.exists(path))(Files.readAllBytes(path))
        val temp = bytes.map { b =>
          Files.createDirectories(path.toAbsolutePath.getParent)
          val p = Files.createTempFile(path.toAbsolutePath.getParent, ".styles-", ".tmp")
          prepared += ((path, Some(p), old, expected))
          Files.write(p, b)
          p
        }
        if temp.isEmpty then prepared += ((path, None, old, expected))
      }
      changes.foreach((p, _, v) => if diskVersion(p) != v then throw new StyleConflict(s"${p.getFileName} changed while saving"))
      prepared.zip(changes).foreach { case ((path, temp, old, expected), (_, bytes, _)) =>
        if diskVersion(path) != expected then throw new StyleConflict(s"${path.getFileName} changed while saving")
        temp match
          case Some(p) => Files.move(p, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
          case None => Files.deleteIfExists(path)
        applied += ((path, old, StyleDocument.fingerprint(bytes.getOrElse(Array.emptyByteArray), bytes.isDefined)))
      }
    catch
      case e: Throwable =>
        applied.reverse.foreach { (path, old, written) =>
          if diskVersion(path) == written then old match
            case Some(bytes) => atomic(path, bytes, written)
            case None => Files.deleteIfExists(path)
        }
        throw e
    finally prepared.flatMap(_._2).foreach(Files.deleteIfExists)
  def load(): StyleDocument = StyleDocument.read(master).copy(mirrorVersions = categoryFiles.map(p => p -> diskVersion(p)).toMap)
  def loadExtended(): StyleDocument =
    val doc = StyleDocument.read(extendedFile)
    if !Files.exists(extendedFile) then doc.copy(header = extendedHeader) else doc
  def save(doc: StyleDocument): StyleDocument = StyleLibrary.lock(directory).synchronized {
    require(owned(doc.path), "Use Import / Restore to read external styles; their source files are never overwritten")
    check(doc)
    val bytes = doc.bytes
    if diskVersion(doc.path) != StyleDocument.fingerprint(bytes) then
      if doc.path.toAbsolutePath.normalize() == master.toAbsolutePath.normalize() then
        val old = StyleDocument.read(master)
        val oldGroups = groups(old)
        val nextGroups = groups(doc)
        val mirrors = (oldGroups.keySet ++ nextGroups.keySet).toSeq.sorted.filter(n => oldGroups.get(n) != nextGroups.get(n)).map { name =>
          val path = categoryFile(name)
          val mirror = StyleDocument.read(path)
          val expected = doc.mirrorVersions.getOrElse(path, StyleDocument.fingerprint(Array.emptyByteArray, false))
          if diskVersion(path) != expected then throw new StyleConflict(s"Category '$name' changed outside this editor; reload before saving")
          if Files.exists(path) && basic(mirror) != oldGroups.getOrElse(name, Vector.empty).map(basicFields) then
            throw new StyleConflict(s"Category '$name' has external edits. Open that category and save it explicitly before editing its master rows.")
          val built = mirrorFrom(mirror, nextGroups.getOrElse(name, Vector.empty))
          (path, Option.when(hasContent(built))(built.bytes), expected)
        }
        transaction(mirrors :+ (doc.path, Some(bytes), doc.version))
        pending = true
      else atomic(doc.path, bytes, doc.version)
    doc.copy(version = StyleDocument.fingerprint(bytes), mirrorVersions = categoryFiles.map(p => p -> diskVersion(p)).toMap)
  }
  private def basicFields(fields: Map[String, String]): Map[String, String] =
    Seq("name", "prompt", "negative_prompt", "notes").map(k => k -> fields.getOrElse(k, "")).toMap
  private def populated(doc: StyleDocument): Vector[StyleEntry] = doc.entries.filter(_.cells.exists(_.nonEmpty))
  private def hasContent(doc: StyleDocument): Boolean = populated(doc).nonEmpty
  private def normalized(doc: StyleDocument, entry: StyleEntry): Map[String, String] =
    val values = doc.fields(entry)
    val base = if doc.header.exists(_.trim.equalsIgnoreCase("prompt")) then values else values.removed("text")
    base.updated("prompt", doc.value(entry, "prompt"))
  private def basic(doc: StyleDocument): Vector[Map[String, String]] = populated(doc).map(e => basicFields(normalized(doc, e)))
  private def groups(doc: StyleDocument): Map[String, Vector[Map[String, String]]] =
    doc.entries.filter(e => doc.value(e, "name").contains("::")).map { e =>
      val parts = doc.value(e, "name").split("::", 2)
      parts(0) -> normalized(doc, e).updated("name", parts(1))
    }.groupMap(_._1)(_._2)
  private def mirrorFrom(doc: StyleDocument, fields: Vector[Map[String, String]]): StyleDocument =
    var next = doc
    val used = scala.collection.mutable.Set[String]()
    val ids = fields.map { f =>
      next.entries.find(e => !used(e.id) && next.value(e, "name") == f("name")) match
        case Some(row) =>
          val changes = f.filter((k, v) => next.value(row, k) != v)
          if changes.nonEmpty then next = next.edit(row.id, changes, uniqueNames = false)
          used += row.id
          row.id
        case None =>
          next = next.append(f, uniqueNames = false)
          val id = next.entries.last.id
          used += id
          id
    }
    next.copy(entries = ids.map(id => next.entries.find(_.id == id).get))
  def categoryFile(name: String): Path =
    require(name.nonEmpty && name == name.trim && name != "." && name != ".." && name.matches("[\\p{L}\\p{N} _.-]+"), "Category names cannot contain paths, colons or surrounding spaces")
    directory.resolve("style-files").resolve(name + ".csv")
  def categories(): Vector[String] =
    val files = categoryFiles.map(_.getFileName.toString.stripSuffix(".csv"))
    (files ++ load().promptStyles.flatMap(s => Option.when(s.name.contains("::"))(s.name.split("::", 2)(0)))).distinct.sorted
  def loadCategory(name: String): StyleDocument =
    val path = categoryFile(name)
    val root = load()
    val existing = StyleDocument.read(path)
    val doc = if Files.exists(path) then existing else root.entries.filter(e => root.value(e, "name").startsWith(name + "::")).foldLeft(existing) { (d, e) =>
      d.append(normalized(root, e).updated("name", root.value(e, "name").stripPrefix(name + "::")), uniqueNames = false)
    }
    doc.copy(masterVersion = Some(root.version), mirrorVersions = root.mirrorVersions)
  def createCategory(name: String): StyleDocument = StyleLibrary.lock(directory).synchronized {
    val doc = loadCategory(name)
    if Files.exists(doc.path) then doc else save(doc)
  }
  def pruneEmptyCategories(): Unit = StyleLibrary.lock(directory).synchronized {
    val active = groups(load()).keySet
    val empty = categoryFiles.map(StyleDocument.read).filter(d => !hasContent(d) && !active(d.path.getFileName.toString.stripSuffix(".csv")))
    transaction(empty.map(d => (d.path, None, d.version)))
  }
  /** Preserve a rejected/uncommitted draft separately; never bypass conflict checks on the active library. */
  def preserveDraft(doc: StyleDocument): Path = StyleLibrary.lock(directory).synchronized {
    val out = directory.resolve("style-drafts").resolve(s"${clock()}-${UUID.randomUUID()}.csv")
    atomic(out, doc.bytes, StyleDocument.fingerprint(Array.emptyByteArray, false))
    out
  }
  def saveCategory(name: String, doc: StyleDocument): StyleDocument = StyleLibrary.lock(directory).synchronized {
    require(doc.path.toAbsolutePath.normalize() == categoryFile(name).toAbsolutePath.normalize(), "Wrong category destination")
    check(doc)
    val root = load()
    if doc.masterVersion.exists(_ != root.version) then throw new StyleConflict("The master styles file changed while this category was open. Reload before saving.")
    var merged = root.delete(root.entries.filter(e => root.value(e, "name").startsWith(name + "::")).map(_.id))
    populated(doc).foreach { e =>
      merged = merged.append(normalized(doc, e).updated("name", s"$name::${doc.value(e, "name")}"), uniqueNames = false)
    }
    val categoryBytes = Option.when(hasContent(doc))(doc.bytes)
    val masterBytes = merged.bytes
    val categoryVersion = StyleDocument.fingerprint(categoryBytes.getOrElse(Array.emptyByteArray), categoryBytes.isDefined)
    val masterChanged = StyleDocument.fingerprint(masterBytes) != root.version
    val changes = Seq((doc.path, categoryBytes, doc.version), (master, Some(masterBytes), root.version)).filter {
      case (path, bytes, expected) => StyleDocument.fingerprint(bytes.getOrElse(Array.emptyByteArray), bytes.isDefined) != expected
    }
    transaction(changes)
    pending ||= masterChanged
    doc.copy(version = categoryVersion,
      masterVersion = Some(StyleDocument.fingerprint(masterBytes)), mirrorVersions = categoryFiles.map(p => p -> diskVersion(p)).toMap)
  }
  def moveToCategory(snapshot: StyleDocument, ids: Seq[String], name: String): Unit = StyleLibrary.lock(directory).synchronized {
    check(snapshot)
    // Publish an edited/reordered category draft first. Otherwise its new ordinal positions could refer to
    // different identities in the old master snapshot. saveCategory retains these runtime row IDs.
    val root = if snapshot.path.toAbsolutePath.normalize() != master.toAbsolutePath.normalize() then
      saveCategory(snapshot.path.getFileName.toString.stripSuffix(".csv"), snapshot)
    else snapshot
    categoryFile(name)
    val rootDoc = if root.path.toAbsolutePath.normalize() == master.toAbsolutePath.normalize() then root else
      val fresh = load()
      if root.masterVersion.exists(_ != fresh.version) then throw new StyleConflict("Master changed while this category was open; reload before moving")
      fresh
    val oldPrefix = if root.path.toAbsolutePath.normalize() == master.toAbsolutePath.normalize() then "" else root.path.getFileName.toString.stripSuffix(".csv") + "::"
    val selected = ids.map { id =>
      val row = root.entries.find(_.id == id).getOrElse(throw new IllegalArgumentException("Selected style no longer exists"))
      row -> (oldPrefix + root.value(row, "name"))
    }.filterNot((_, full) => full.startsWith(name + "::"))
    val removed = if oldPrefix.isEmpty then selected.map(_._1.id) else
      val categoryRows = populated(root)
      val masterRows = rootDoc.entries.filter(e => rootDoc.value(e, "name").startsWith(oldPrefix))
      selected.flatMap { (row, _) => masterRows.lift(categoryRows.indexWhere(_.id == row.id)).map(_.id) }
    val moved = selected.foldLeft(rootDoc.delete(removed)) { case (doc, (entry, _)) =>
      doc.append(normalized(root, entry).updated("name", name + "::" + root.value(entry, "name").split("::", 2).last))
    }
    save(moved)
  }
  def createExtended(name: String, p: GenerationParams): ExtendedStyle = StyleLibrary.lock(directory).synchronized {
    require(name.trim.nonEmpty, "Enter a style name")
    val id = UUID.randomUUID().toString
    val doc = loadExtended().append(Map("id" -> id, "name" -> name.trim, "sd_model_checkpoint" -> p.checkpoint.getOrElse(""),
      "sd_vae" -> p.vae.getOrElse("Automatic"), "prompt" -> p.prompt, "negative_prompt" -> p.negativePrompt,
      "width" -> p.width.toString, "height" -> p.height.toString), uniqueNames = false)
    save(doc)
    extended().find(_.id == id).get
  }
  def extended(target: Option[(String, String)] = None): Vector[ExtendedStyle] =
    val d = loadExtended()
    d.entries.filter(e => d.value(e, "name").nonEmpty).map { e =>
      def v(k: String) = d.value(e, k)
      ExtendedStyle(v("id"), v("name"), v("sd_model_checkpoint"), v("sd_vae"), v("prompt"), v("negative_prompt"),
        v("width").toInt, v("height").toInt)
    }.filter(s => target.forall((m, v) => s.checkpoint == m && s.effectiveVae == ExtendedStyle.normalizeVae(v)))
  def deleteExtended(id: String): Unit =
    val doc = loadExtended()
    save(doc.delete(doc.entries.filter(e => doc.value(e, "id") == id).map(_.id)))
  def backups(): Vector[Path] =
    if !Files.isDirectory(backupDirectory) then Vector.empty
    else Using.resource(Files.list(backupDirectory))(_.iterator().asScala.filter(p => Files.isRegularFile(p) && (p.toString.endsWith(".csv") || p.toString.endsWith(".aes"))).toVector.sortBy(_.getFileName.toString).reverse)
  def backupIfDue(password: Option[String] = None): Option[Path] = StyleLibrary.lock(directory).synchronized {
    if pending && clock() - lastBackup >= 600000 then Some(backupNow(password)) else None
  }
  def backupNow(password: Option[String] = None): Path = StyleLibrary.lock(directory).synchronized {
    require(Files.isRegularFile(master), "No master styles file to back up")
    val plain = Files.readAllBytes(master)
    StyleDocument.parse(master, plain) // Never publish a corrupted CSV as a valid backup.
    sequence += 1
    val suffix = if password.isDefined then ".csv.aes" else ".csv"
    val out = backupDirectory.resolve(f"${clock()}%016d-$sequence%06d$suffix")
    val bytes = password.map(p => AesCryptV2.encrypt(plain, p)).getOrElse(plain)
    atomic(out, bytes, StyleDocument.fingerprint(Array.emptyByteArray, false))
    backups().drop(24).foreach(Files.deleteIfExists)
    pending = false
    lastBackup = clock()
    out
  }
  def restore(source: Path, password: Option[String] = None): StyleDocument = StyleLibrary.lock(directory).synchronized {
    val current = load()
    val raw = Files.readAllBytes(source)
    val plain = if source.toString.endsWith(".aes") then AesCryptV2.decrypt(raw, password.getOrElse(throw new IllegalArgumentException("Enter the backup password"))) else raw
    val restored = StyleDocument.parse(master, plain)
    if Files.isRegularFile(master) then backupNow()
    val restoredGroups = groups(restored)
    val names = (categories().toSet ++ restoredGroups.keySet).toSeq.sorted
    val mirrors = names.map { name =>
      val path = categoryFile(name)
      val old = StyleDocument.read(path)
      val rebuilt = mirrorFrom(old, restoredGroups.getOrElse(name, Vector.empty))
      (path, Option.when(hasContent(rebuilt))(rebuilt.bytes), current.mirrorVersions.getOrElse(path, StyleDocument.fingerprint(Array.emptyByteArray, false)))
    }
    transaction(mirrors :+ (master, Some(plain), current.version))
    pending = StyleDocument.fingerprint(plain) != current.version
    restored.copy(mirrorVersions = categoryFiles.map(p => p -> diskVersion(p)).toMap)
  }

object StyleLibrary:
  private val locks = new java.util.concurrent.ConcurrentHashMap[Path, Object]()
  private[model] def lock(path: Path): Object = locks.computeIfAbsent(path.toAbsolutePath.normalize(), _ => new Object())
