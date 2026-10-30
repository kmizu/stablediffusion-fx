package sdfx.plugins.prompt

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import sdfx.plugins.Support
import scala.util.Random

final case class StylePileSelection(groups: Map[String, Seq[(String, java.net.URL)]],
    inspirationLists: Map[String, Seq[String]], warnings: Seq[String] = Nil):
  def choices(group: String): Seq[String] = groups.getOrElse(group, Nil).map(_._1)
  def preview(group: String, name: String): Option[java.net.URL] = groups.getOrElse(group, Nil).find(_._1 == name).map(_._2)
  def inspiration(group: String): Seq[String] = inspirationLists.getOrElse(group, Nil)

/** Catalog data is adapted from pinned GPL-3.0 StylePile material; see bundled NOTICE.txt. */
object StylePileCatalog:
  private lazy val data: ujson.Value =
    val stream = Option(getClass.getResourceAsStream("/stylepile/catalog.json")).getOrElse(
      throw new IllegalStateException("Bundled StylePile catalog missing"))
    try ujson.read(new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8))
    finally stream.close()
  def literal(name: String): ujson.Value = data("literals")(name)
  lazy val bundled: StylePileSelection = StylePileSelection(
    data("choices").obj.toMap.map { (group, entries) => group -> entries.arr.toSeq.map { entry =>
      val url = Option(getClass.getResource("/stylepile/" + entry("preview").str)).getOrElse(
        throw new IllegalStateException("Bundled StylePile preview missing: " + entry("preview").str))
      entry("name").str -> url
    } }, data("inspiration").obj.toMap.map((group, entries) => group -> entries.arr.toSeq.map(_.str)))
  def choices(group: String): Seq[String] = bundled.choices(group)
  def inspiration(group: String): Seq[String] = bundled.inspiration(group)
  def preview(group: String, name: String): Option[java.net.URL] = bundled.preview(group, name)

  /** One immutable choice snapshot; originals are never edited by user customization. */
  def withOverrides(root: java.nio.file.Path): StylePileSelection =
    import java.nio.file.{Files, LinkOption}
    import scala.jdk.CollectionConverters.*
    import scala.util.control.NonFatal
    if !Files.exists(root) then return bundled
    val warnings = scala.collection.mutable.ArrayBuffer[String]()
    def warn(message: String): Unit = if warnings.size < 100 then warnings += message
    if !Files.isDirectory(root) then return bundled.copy(warnings = Seq("StylePile catalog path is not a folder: " + root))
    val realRoot = root.toRealPath()
    def text(path: java.nio.file.Path): Seq[String] =
      require(Files.size(path) <= 65536, "catalog text exceeds 64 KiB")
      require(path.toRealPath().startsWith(realRoot), "catalog text escapes custom folder")
      Files.readAllLines(path, java.nio.charset.StandardCharsets.UTF_8).asScala.toSeq.map(_.trim).filter(_.nonEmpty)
    val groups = bundled.groups.map { (group, original) =>
      val folder = root.resolve(group)
      var result = original
      try
        if Files.exists(folder) then
          require(Files.isDirectory(folder) && folder.toRealPath().startsWith(realRoot), "group escapes custom folder or is not a directory")
          val stream = Files.list(folder)
          val paths = try stream.limit(4097).iterator().asScala.toVector finally stream.close()
          require(paths.size <= 4096, "group has more than 4096 files")
          val hidden = if Files.exists(folder.resolve("disabled.txt")) then text(folder.resolve("disabled.txt")).filterNot(_.startsWith("#")).toSet else Set.empty[String]
          val candidates = paths.filter { path =>
            val lower = path.getFileName.toString.toLowerCase(java.util.Locale.ROOT)
            Seq(".jpg", ".jpeg", ".png", ".gif").exists(lower.endsWith)
          }.groupBy(path => path.getFileName.toString.replaceFirst("\\.[^.]+$", ""))
          val custom = candidates.toSeq.sortBy(_._1).flatMap { (name, files) =>
            try
              require(name.nonEmpty && name != "Random" && name != "Not set" && !name.exists(_.isControl), "reserved or invalid choice name")
              require(files.size == 1, "ambiguous filename")
              val path = files.head
              require(Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) && path.toRealPath().startsWith(realRoot), "preview is not a local file in the catalog")
              require(Files.size(path) <= 32L * 1024 * 1024, "preview exceeds 32 MiB")
              Some(name -> path.toRealPath().toUri.toURL)
            catch
              case NonFatal(error) =>
                warn(s"StylePile $group/$name: ${error.getMessage}")
                None
          }
          val replacement = custom.toMap
          val originalNames = original.map(_._1).toSet
          result = (original.map((name, url) => name -> replacement.getOrElse(name, url)) ++ custom.filterNot(entry => originalNames(entry._1)))
            .filterNot(entry => hidden(entry._1))
      catch case NonFatal(error) => warn(s"StylePile $group: ${error.getMessage}")
      group -> result
    }
    val inspiration = bundled.inspirationLists.map { (group, original) =>
      var result = original
      try
        val path = root.resolve("Inspiration").resolve(group + ".txt")
        if Files.exists(path) then
          val values = text(path)
          require(values.nonEmpty, "inspiration list is empty")
          result = values
      catch case NonFatal(error) => warn(s"StylePile inspiration $group: ${error.getMessage}")
      group -> result
    }
    StylePileSelection(groups, inspiration, warnings.toSeq)

/** Pure job expansion, independent of JavaFX and image generation. */
object StylePile:
  private def lines(text: String): Vector[String] = text.linesIterator.map(_.trim).toVector
  private def clean(text: String): String = text.replaceAll("\\s+", " ").replaceAll("\\s*,\\s*", ",")
    .replaceAll(",+", ",").stripPrefix(",").stripSuffix(",").trim

  def expand(p: GenerationParams, v: Values, catalog: StylePileSelection = StylePileCatalog.bundled): Seq[GenerationParams] =
    require(p.prompt.nonEmpty, "StylePile requires a prompt or variable template")
    val strengths = (Seq("type", "direction", "mood", "color") ++
      (1 to 3).flatMap(i => Seq(s"artist$i", s"movement$i"))).map { key =>
      val value = v.double(key + "_strength", 1.3)
      require(value.isFinite && value >= 0 && value <= 2, s"StylePile invalid strength: $key")
      key -> value
    }.toMap
    val sequential = lines(v.str("sequential"))
    val subsequential = lines(v.str("subsequential")) match
      case empty if empty.isEmpty => Vector("")
      case other => other
    val count = if v.bool("override_count", true) && v.str("sequential").nonEmpty then sequential.size else p.nIter
    val total = count.toLong * p.batchSize * subsequential.size
    require(count > 0 && p.batchSize > 0 && total > 0 && total <= 10000, "StylePile sequence must contain 1 to 10000 images")
    val seed = Support.fixSeed(p.seed)
    val rng = new Random(seed)
    def pick(values: Seq[String]): String = if values.isEmpty then "" else values(rng.nextInt(values.size))
    def select(key: String, pool: Seq[String], fallback: String = "Not set"): String = v.str(key, fallback) match
      case "Not set" | "" => ""
      case "Random" =>
        require(pool.nonEmpty, s"StylePile has no available choices for $key")
        pick(pool)
      case value => value
    def weighted(text: String, key: String): String = if text.isEmpty then "" else s"($text:${Values.fmt(strengths(key))})"
    def dictionary(field: String, key: String): String =
      StylePileCatalog.literal(field).obj.getOrElse(key, throw new IllegalArgumentException(s"Unknown StylePile $field: $key")).str
    def noise(text: String): String =
      val alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      def token(length: Int): String = (0 until length).map(_ => alphabet(rng.nextInt(alphabet.length))).mkString
      var result = text
      for marker <- Seq("xs", "XS", "s", "S", "m", "M", "l", "L", "xl", "XL") do
        if result.contains(s"[$marker]") then
          val replacement = marker.toLowerCase match
            case "xs" => (100000 + rng.nextInt(899999)).toString
            case "s" => token(5)
            case "m" => token(10)
            case "l" => token(5) + " " + token(5)
            case "xl" => token(10) + " " + token(10)
          result = result.replace(s"[$marker]", replacement)
      result
    (0 until count).flatMap { x =>
      val artists = (1 to 3).map(i => weighted(select(s"artist$i", catalog.choices("Artists")) match
        case "" => ""
        case artist => "by Artist " + artist, s"artist$i")).filter(_.nonEmpty)
      val movements = (1 to 3).map(i => weighted(select(s"movement$i", catalog.choices("Art Movements")), s"movement$i")).filter(_.nonEmpty)
      (0 until p.batchSize).flatMap { copy =>
        val imageType = select("image_type", StylePileCatalog.literal("ResultNames").arr.toSeq.map(_.str))
        val concept = select("concept", StylePileCatalog.literal("ElementConcept").arr.toSeq.map(_.str))
        val front = if imageType.isEmpty then "" else weighted((concept + " " + dictionary("ResultTypeBefore", imageType)).trim, "type") + " of "
        val direction = weighted(select("direction", catalog.choices("Directions")), "direction")
        val mood = weighted(select("mood", catalog.choices("Moods")), "mood")
        val color = weighted(select("color", catalog.choices("Colors")), "color")
        var subject = p.prompt.replace("[X]", if sequential.isEmpty then "" else sequential(x % sequential.size))
        for marker <- Seq("A", "B", "C") do
          val replacement = pick(lines(v.str("random_" + marker.toLowerCase)))
          subject = subject.replace(s"[$marker]", replacement)
          if marker == "A" then subject = subject.replace("[R]", replacement)
        val preset = v.str("preset", "None")
        val positive = front + (Seq(direction, mood, subject) ++ artists ++
          Seq(if imageType.isEmpty then "" else dictionary("ResultTypePositives", imageType)) ++ movements ++
          Seq(color, dictionary("Preset", preset))).filter(_.nonEmpty).mkString(",")
        val negative = clean(Seq(p.negativePrompt, if imageType.isEmpty then "" else dictionary("ResultTypeNegatives", imageType),
          StylePileCatalog.literal("AlwaysBad").str, dictionary("PresetNegatives", preset)).filter(_.nonEmpty).mkString(","))
        subsequential.zipWithIndex.map { (sub, z) =>
          val offset = if v.bool("increase_seed", true) then copy.toLong * subsequential.size + z else 0L
          p.copy(prompt = clean(noise(positive.replace("[Y]", sub))), negativePrompt = negative,
            seed = seed + offset, batchSize = 1, nIter = 1,
            extra = p.extra ++ Map("StylePile template" -> p.prompt, "StylePile" -> ujson.Obj(
              "revision" -> "85b549a10ff6c5b1604e980cd2107253b088e832", "sequence" -> x, "copy" -> copy,
              "subsequence" -> z, "settings" -> ujson.Obj.from(v.map.toSeq.sortBy(_._1).map((key, value) => key -> ujson.Str(value)))).render()))
        }
      }
    }


object StylePileTools:
  def weight(text: String, value: Double): String =
    require(text.nonEmpty, "Enter text to weight")
    require(value.isFinite && value >= 0 && value <= 2, "Influence must be between zero and two")
    s"($text:${Values.fmt(value)})"
  def morph(from: String, to: String, position: Double): String =
    require(from.nonEmpty && to.nonEmpty, "Enter both prompt parts")
    require(position.isFinite && position >= 0 && position <= 1, "Switch position must be between zero and one")
    s"[$from:$to:${Values.fmt(position)}]"
  def bounce(from: String, to: String): String =
    require(from.nonEmpty && to.nonEmpty, "Enter both prompt parts")
    s"[$from|$to]"
  def inspire(seed: Long, catalog: StylePileSelection = StylePileCatalog.bundled): String =
    val rng = new Random(seed)
    Seq("Subjects", "Actions", "Locations").map { group =>
      val choices = catalog.inspiration(group)
      choices(rng.nextInt(choices.size))
    }.mkString(",") + ","
