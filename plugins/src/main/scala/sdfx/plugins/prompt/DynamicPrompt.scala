package sdfx.plugins.prompt

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*
import scala.util.Random
import scala.util.Using

/** The Dynamic Prompts / Wildcards template language (sd-dynamic-prompts, stable-diffusion-webui-wildcards):
  *
  *  - `{a|b|c}` one of the options; `{0.5::a|2::b}` weighted options
  *  - `{2$$a|b|c}` two distinct options joined with ", "; `{1-3$$ and $$a|b|c}` 1 to 3 joined with " and "
  *  - `__name__` a random line of `<wildcard dir>/name.txt` (sub folders with `/`, `*` globs, e.g. `__colors/` + `*__`);
  *    wildcard lines may contain templates themselves (resolved recursively)
  *  - `\{`, `\}`, `\|` are literal characters
  */
object DynamicPrompt:
  sealed trait Node
  final case class Lit(text: String) extends Node
  final case class Variant(options: Vector[(Double, Vector[Node])], min: Int, max: Int, sep: String) extends Node
  final case class Wildcard(name: String) extends Node

  final class Wildcards(dirs: Seq[Path]):
    private val cache = scala.collection.mutable.Map[String, Vector[String]]()

    /** All wildcard names (relative paths without .txt) for UI listing. */
    def names: Vector[String] =
      dirs.filter(Files.isDirectory(_)).flatMap { d =>
        Using.resource(Files.walk(d)) { s =>
          s.iterator().asScala.filter(p => p.getFileName.toString.endsWith(".txt"))
            .map(p => d.relativize(p).toString.replace('\\', '/').stripSuffix(".txt")).toVector
        }
      }.distinct.sorted.toVector

    def lines(name: String): Vector[String] = cache.getOrElseUpdate(name, load(name))

    private def load(name: String): Vector[String] =
      val pattern = name.replace('\\', '/')
      val matched =
        if pattern.contains("*") then
          val re = ("^" + java.util.regex.Pattern.quote(pattern).replace("*", "\\E.*\\Q") + "$").r
          names.filter(n => re.matches(n))
        else Vector(pattern)
      matched.flatMap { n =>
        dirs.map(_.resolve(n + ".txt")).find(Files.isRegularFile(_)).toVector.flatMap { f =>
          Files.readAllLines(f).asScala.map(_.trim).filter(l => l.nonEmpty && !l.startsWith("#")).toVector
        }
      }

  // --------------------------------------------------------------------------------------------- parsing
  def parse(text: String): Vector[Node] =
    val (nodes, rest) = parseSeq(text, 0, top = true)
    if rest < text.length then nodes :+ Lit(text.substring(rest)) else nodes

  private def parseSeq(s: String, start: Int, top: Boolean): (Vector[Node], Int) =
    val out = Vector.newBuilder[Node]
    val lit = new StringBuilder
    def flush(): Unit = if lit.nonEmpty then { out += Lit(lit.toString); lit.clear() }
    var i = start
    var done = false
    while i < s.length && !done do
      val c = s.charAt(i)
      if c == '\\' && i + 1 < s.length && "{}|_".contains(s.charAt(i + 1)) then
        lit += s.charAt(i + 1); i += 2
      else if c == '{' then
        flush()
        val (v, next) = parseVariant(s, i + 1)
        out += v; i = next
      else if !top && (c == '|' || c == '}') then done = true
      else if c == '_' && s.startsWith("__", i) then
        val end = s.indexOf("__", i + 2)
        if end > i + 2 && !s.substring(i + 2, end).exists(ch => ch.isWhitespace || "{}|".contains(ch)) then
          flush(); out += Wildcard(s.substring(i + 2, end)); i = end + 2
        else { lit += c; i += 1 }
      else { lit += c; i += 1 }
    flush()
    (out.result(), i)

  private val CountRe = """^\s*(\d+)?\s*(?:-\s*(\d+)?)?\s*\$\$""".r

  private def parseVariant(s: String, start: Int): (Node, Int) =
    var i = start
    var min = 1
    var max = 1
    var sep = ", "
    CountRe.findPrefixMatchOf(s.substring(i)).foreach { m =>
      val a = Option(m.group(1)).map(_.toInt)
      val dash = m.matched.contains("-")
      val b = Option(m.group(2)).map(_.toInt)
      min = a.getOrElse(1)
      max = if dash then b.getOrElse(Int.MaxValue) else a.getOrElse(1)
      i += m.end
      val sepEnd = s.indexOf("$$", i)
      val close = s.indexOf("}", i)
      val bar = s.indexOf("|", i)
      if sepEnd >= 0 && (close < 0 || sepEnd < close) && (bar < 0 || sepEnd < bar) then
        sep = s.substring(i, sepEnd); i = sepEnd + 2
    }
    val opts = Vector.newBuilder[(Double, Vector[Node])]
    var closed = false
    while !closed && i <= s.length do
      val (nodes, next) = parseSeq(s, i, top = false)
      val (w, ns) = weightOf(nodes)
      opts += ((w, ns))
      if next >= s.length then { closed = true; i = next }
      else if s.charAt(next) == '|' then i = next + 1
      else { closed = true; i = next + 1 }
    val o = opts.result()
    (Variant(o, math.max(0, math.min(min, o.size)), math.max(0, math.min(max, o.size)), sep), i)

  private val WeightRe = """^\s*(\d+(?:\.\d+)?)\s*::""".r

  private def weightOf(nodes: Vector[Node]): (Double, Vector[Node]) = nodes.headOption match
    case Some(Lit(t)) =>
      WeightRe.findPrefixMatchOf(t) match
        case Some(m) => (m.group(1).toDouble, Lit(t.substring(m.end)) +: nodes.tail)
        case None    => (1.0, nodes)
    case _ => (1.0, nodes)

  // ------------------------------------------------------------------------------------------- expansion
  private val MaxDepth = 12

  def random(text: String, rng: Random, wc: Wildcards): String = randomNodes(parse(text), rng, wc, 0)

  private def randomNodes(nodes: Vector[Node], rng: Random, wc: Wildcards, depth: Int): String =
    nodes.map {
      case Lit(t) => t
      case Wildcard(n) =>
        val ls = wc.lines(n)
        if ls.isEmpty || depth >= MaxDepth then s"__${n}__" else random(ls(rng.nextInt(ls.size)), rng, wc, depth + 1)
      case Variant(opts, min, max, sep) =>
        if opts.isEmpty then ""
        else
          val k = if max <= min then min else min + rng.nextInt(math.min(max, opts.size) - min + 1)
          pickWeighted(opts, k, rng).map(o => randomNodes(o._2, rng, wc, depth)).mkString(sep)
    }.mkString

  private def random(text: String, rng: Random, wc: Wildcards, depth: Int): String = randomNodes(parse(text), rng, wc, depth)

  private def pickWeighted(opts: Vector[(Double, Vector[Node])], k: Int, rng: Random): Vector[(Double, Vector[Node])] =
    var pool = opts
    val out = Vector.newBuilder[(Double, Vector[Node])]
    for _ <- 0 until math.min(k, opts.size) do
      val total = pool.map(_._1).sum
      var r = rng.nextDouble() * total
      val idx = pool.indexWhere { o => r -= o._1; r <= 0 } match
        case -1 => pool.size - 1
        case x  => x
      out += pool(idx)
      pool = pool.patch(idx, Nil, 1)
    out.result()

  /** Every combination, in order, at most `limit` prompts (sd-dynamic-prompts combinatorial generator). */
  def combinatorial(text: String, wc: Wildcards, limit: Int): Vector[String] =
    combos(parse(text), wc, limit, 0).take(limit)

  private def combos(nodes: Vector[Node], wc: Wildcards, limit: Int, depth: Int): Vector[String] =
    nodes.foldLeft(Vector("")) { (acc, n) =>
      val parts: Vector[String] = n match
        case Lit(t) => Vector(t)
        case Wildcard(name) =>
          val ls = wc.lines(name)
          if ls.isEmpty || depth >= MaxDepth then Vector(s"__${name}__")
          else ls.flatMap(l => combos(parse(l), wc, limit, depth + 1))
        case Variant(opts, min, max, sep) =>
          val expanded = opts.map(o => combos(o._2, wc, limit, depth))
          (min to math.min(max, opts.size)).toVector.flatMap { k =>
            if k == 0 then Vector("")
            else
              expanded.indices.combinations(k).toVector.flatMap { idxs =>
                idxs.foldLeft(Vector(Vector.empty[String])) { (a, i) => for x <- a; y <- expanded(i) yield x :+ y }
                  .map(_.mkString(sep))
              }
          }
      (for a <- acc; p <- parts yield a + p).take(limit)
    }

  /** Number of combinations without materialising them beyond `limit`. */
  def count(text: String, wc: Wildcards, limit: Int): Int = combinatorial(text, wc, limit).size
