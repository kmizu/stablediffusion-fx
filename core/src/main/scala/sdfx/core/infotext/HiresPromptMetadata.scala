package sdfx.core.infotext

import sdfx.core.model.HiresPromptModes
import scala.util.Try

/** Original template, distinct from the resolved prompts written in ordinary PNG fields. */
case class HiresPromptTemplate(mode: String, prompt: String, removeNetworks: Boolean = false)

case class HiresPromptMetadata(positive: Option[HiresPromptTemplate], negative: Option[HiresPromptTemplate],
    appendSeparator: String = "{newline}", prependSeparator: String = "{newline}", marker: String = "@"):
  /** Source-compatible restore guard. A mismatch on one side does not discard the other side. */
  def matching(base: String, negativeBase: String, resolved: String, negativeResolved: String): HiresPromptMetadata =
    def matches(t: HiresPromptTemplate, first: String, second: String): Boolean =
      Try {
        val separator = if t.mode == "Prepend" then prependSeparator else appendSeparator
        val (b, h) = HiresPromptModes.transform(first.strip, t.prompt.strip, t.mode,
          t.removeNetworks, separator, marker)
        b.strip == first.strip && h.strip == second.strip
      }.getOrElse(false)
    copy(positive = positive.filter(matches(_, base, resolved)),
      negative = negative.filter(matches(_, negativeBase, negativeResolved)))

  def encode: String =
    val o = ujson.Obj()
    def add(key: String, t: HiresPromptTemplate): Unit =
      val a = ujson.Arr(t.mode, t.prompt)
      if key == "h" && t.removeNetworks then a.value += ujson.Bool(true)
      o(key) = a
    positive.foreach(add("h", _))
    negative.foreach(add("n", _))
    if appendSeparator != "{newline}" then o("a") = appendSeparator
    if prependSeparator != "{newline}" then o("p") = prependSeparator
    if marker != "@" then o("c") = marker
    HiresPromptMetadata.swapQuotes(o.render())

object HiresPromptMetadata:
  /** Capture only prompt-related inputs before composition; never include transient seed or runtime state. */
  def capture(p: sdfx.core.model.GenerationParams, config: ujson.Obj): ujson.Obj =
    def text(key: String, default: String) = config.value.get(key).map(_.str).getOrElse(default)
    val positiveMode = text("prompt_mode", "Default")
    val negativeMode = text("negative_mode", "Default")
    val remove = config.value.get("remove_networks").contains(ujson.Bool(true))
    val metadata = HiresPromptMetadata(
      Option.when(positiveMode != "Default" || remove)(HiresPromptTemplate(positiveMode, p.hires.prompt.strip, remove)),
      Option.when(negativeMode != "Default")(HiresPromptTemplate(negativeMode, p.hires.negativePrompt.strip)),
      text("append_separator", text("separator", "{newline}")),
      text("prepend_separator", text("separator", "{newline}")), text("marker", "@"))
    val keys = Set("prompt_mode", "negative_mode", "remove_networks", "append_separator",
      "prepend_separator", "separator", "marker", "styles_enabled", "styles", "empty_prompt", "empty_negative_prompt")
    ujson.Obj("version" -> 1, "prompt" -> p.prompt, "negative_prompt" -> p.negativePrompt,
      "hires_prompt" -> p.hires.prompt, "hires_negative_prompt" -> p.hires.negativePrompt,
      "styles" -> ujson.Arr.from(p.styles),
      "config" -> ujson.read(ujson.Obj.from(config.value.toSeq.filter((k, _) => keys(k))).render()),
      "mode" -> (if metadata.positive.nonEmpty || metadata.negative.nonEmpty then metadata.encode else ""))

  private[infotext] def swapQuotes(s: String): String =
    s.map { case '\'' => '"'; case '"' => '\''; case c => c }

  /** Bounded, atomic decode; list metadata has no unambiguous single-image template. */
  def parse(text: String): Option[HiresPromptMetadata] =
    if text.length > 65536 then None
    else
      def decode(s: String): Option[HiresPromptMetadata] = Try {
        val o = ujson.read(s).obj
        def template(key: String): Option[HiresPromptTemplate] = o.get(key).map { v =>
          val a = v.arr
          require(a.size == 2 || (key == "h" && a.size == 3))
          val mode = a(0).str
          require(HiresPromptModes.modes.contains(mode))
          HiresPromptTemplate(mode, a(1).str, a.size == 3 && a(2).bool)
        }
        val positive = template("h")
        val negative = template("n")
        require(positive.nonEmpty || negative.nonEmpty)
        def field(key: String, default: String) = o.get(key).map(_.str).getOrElse(default)
        val append = field("a", "{newline}")
        val prepend = field("p", "{newline}")
        val marker = field("c", "@")
        HiresPromptModes.expandSeparator(append)
        HiresPromptModes.expandSeparator(prepend)
        require(marker.codePointCount(0, marker.length) == 1 && !marker.matches("(?U)[\\s\\w]"))
        HiresPromptMetadata(positive, negative, append, prepend, marker)
      }.toOption
      decode(text).orElse(decode(swapQuotes(text)))
