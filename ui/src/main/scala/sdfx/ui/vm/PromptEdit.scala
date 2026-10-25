package sdfx.ui.vm

import java.math.MathContext

/** Pure prompt text transformations used by the prompt editors. */
object PromptEdit:
  /** Result of an edit: new text and the selection to restore. */
  case class Edit(text: String, selStart: Int, selEnd: Int)

  // A1111 defaults: keyedit_delimiters + whitespace delimiters (Tab, CR, LF).
  private val Delimiters: Set[Char] = ".,\\/!?%^*;:{}=`~() \t\r\n".toSet
  val AttentionDelta = 0.1

  /** prompt-bracket-checker: a warning for unbalanced (), [] or {} (escaped \\( etc. are ignored), "" if fine. */
  def bracketIssues(text: String): String =
    val pairs = Seq('(' -> ')', '[' -> ']', '{' -> '}')
    var escaped = false
    val counts = scala.collection.mutable.Map[Char, Int]().withDefaultValue(0)
    text.foreach { c =>
      if escaped then escaped = false
      else if c == '\\' then escaped = true
      else counts(c) += 1
    }
    pairs.collect { case (o, cl) if counts(o) != counts(cl) =>
      val (more, n) = if counts(o) > counts(cl) then (o, counts(o) - counts(cl)) else (cl, counts(cl) - counts(o))
      s"$n extra '$more'"
    }.mkString("Unbalanced brackets: ", ", ", "") match
      case "Unbalanced brackets: " => ""
      case s                       => s
  val ExtraDelta = 0.05

  /** A1111 `edit-attention.js` (Ctrl+Up / Ctrl+Down). `delta` is +1 / -1 steps.
    * Without a selection the enclosing `<...>`, `(...)`, `[...]` block or the word under the caret is used.
    * Returns the unchanged text and selection when there is nothing to adjust.
    */
  def adjustWeight(text: String, selStart0: Int, selEnd0: Int, delta: Int): Edit =
    val unchanged = Edit(text, selStart0, selEnd0)
    var s = math.max(0, math.min(selStart0, selEnd0)).min(text.length)
    var e = math.max(selStart0, selEnd0).min(text.length)
    var t = text
    def at(i: Int): Char = if i >= 0 && i < t.length then t.charAt(i) else '\u0000'

    def selectBlock(open: Char, close: Char): Boolean =
      if s != e then false
      else
        val before = t.substring(0, s)
        val bo = before.lastIndexOf(open)
        if bo == -1 then false
        else
          val bc = before.lastIndexOf(close)
          if bc != -1 && bc > bo then false
          else
            val after = t.substring(s)
            val ac = after.indexOf(close)
            if ac == -1 then false
            else
              val ao = after.indexOf(open)
              if ao != -1 && ao < ac then false
              else
                val content = t.substring(bo + 1, s + ac)
                if content.matches("(?s).*:-?[\\d.]+") then
                  val lastColon = content.lastIndexOf(':')
                  s = bo + 1
                  e = s + lastColon
                else
                  s = bo + 1
                  e = s + content.length
                true

    def selectWord(): Unit =
      if s == e then
        while s > 0 && !Delimiters.contains(at(s - 1)) do s -= 1
        while e < t.length && !Delimiters.contains(at(e)) do e += 1
        while at(s) == ' ' && s < e do s += 1
        while e > s && at(e - 1) == ' ' do e -= 1

    if !selectBlock('<', '>') && !selectBlock('(', ')') && !selectBlock('[', ']') then selectWord()

    var close = ')'
    var step = AttentionDelta
    val startCh = if s > 0 then at(s - 1) else '\u0000'
    val endCh = at(e)
    if startCh == '<' then
      close = '>'
      step = ExtraDelta
    else if (startCh == '(' && endCh == ')') || (startCh == '[' && endCh == ']') then
      // old style (((emphasis))) -> (emphasis:1.3)
      var n = 0
      while at(s - n - 1) == startCh && at(e + n) == endCh do n += 1
      val w0 = if startCh == '[' then math.pow(1 / 1.1, n) else math.pow(1.1, n)
      val w = math.round(w0 / AttentionDelta) * AttentionDelta
      t = t.substring(0, s - n) + "(" + t.substring(s, e) + ":" + fmt(w) + ")" + t.substring(e + n)
      s -= n - 1
      e -= n - 1
    else if startCh != '(' then
      while e > s && at(e - 1) == ' ' do e -= 1
      if s == e then return unchanged
      t = t.substring(0, s) + "(" + t.substring(s, e) + ":1.0)" + t.substring(e)
      s += 1
      e += 1

    if at(e) != ':' then return unchanged
    val rest = t.substring(e + 1)
    val closeIdx = rest.indexOf(close)
    if closeIdx < 0 then return unchanged
    val weightLength = closeIdx + 1
    val parsed = rest.substring(0, closeIdx).trim.toDoubleOption
    parsed match
      case None => unchanged
      case Some(w0) =>
        val w = round12(w0 + (if delta >= 0 then step else -step) * math.max(1, math.abs(delta)))
        if close == ')' && w == 1.0 then
          val endParen = t.substring(e).indexOf(')')
          t = t.substring(0, s - 1) + t.substring(s, e) + t.substring(e + endParen + 1)
          Edit(t, s - 1, e - 1)
        else
          t = t.substring(0, e + 1) + fmt(w) + t.substring(e + weightLength)
          Edit(t, s, e)

  private def round12(d: Double): Double = BigDecimal(d).round(new MathContext(12)).toDouble

  /** JS style number: integers keep one decimal ("1.0"), otherwise the shortest representation. */
  def fmt(d: Double): String =
    val r = round12(d)
    if r == math.rint(r) then s"${r.toLong}.0"
    else BigDecimal(r).bigDecimal.stripTrailingZeros.toPlainString

  private val TagRe = """<[^>]*>""".r
  private val TokenRe = """[\p{L}]+|\p{N}|[^\s\p{L}\p{N}]""".r

  /** Rough CLIP token estimate (words, digits and punctuation; extra-network tags excluded). */
  def approxTokens(text: String): Int =
    TokenRe.findAllIn(TagRe.replaceAllIn(text, " ")).size

  /** A1111 style counter: "tokens/limit" where the limit grows in chunks of 75. */
  def tokenCounter(text: String): String =
    val n = approxTokens(text)
    val limit = math.max(1, (n + 74) / 75) * 75
    s"$n/$limit"

  /** Inserts `insert` at the caret, adding a ", " separator when the caret follows other text. */
  def insertAt(text: String, caret: Int, insert: String): Edit =
    val c = caret.max(0).min(text.length)
    val before = text.substring(0, c)
    val after = text.substring(c)
    val sep =
      if before.trim.isEmpty then ""
      else if before.trim.endsWith(",") then (if before.endsWith(" ") then "" else " ")
      else if before.last.isWhitespace then ""
      else ", "
    val t = before + sep + insert + after
    val pos = c + sep.length + insert.length
    Edit(t, pos, pos)
