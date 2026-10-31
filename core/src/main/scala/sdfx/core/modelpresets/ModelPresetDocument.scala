package sdfx.core.modelpresets

import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.immutable.VectorMap

/** Independent representation of the public Model Preset Manager exchange document. */
final case class ModelPresetDocument(url: String, defaultPreset: String,
    triggerWords: Vector[String], presets: VectorMap[String, String]):
  import ModelPresetDocument.*
  def validate(): ModelPresetDocument =
    require(url.length <= 4096, "Model URL is too long")
    require(presets.nonEmpty && presets.size <= 512, "Use between 1 and 512 presets")
    require(presets.contains(defaultPreset), "Default preset is missing")
    presets.foreach { (name, text) => validName(name); require(text.length <= 262144, "Preset text is too long") }
    require(triggerWords.size <= 1024 && triggerWords.forall(w => w.nonEmpty && w.length <= 4096 && !w.contains('\u0000')), "Invalid trigger words")
    require(json.getBytes(UTF_8).length <= MaxBytes, "Preset document is too large")
    this
  def json: String = ujson.Obj("url" -> url, "default_preset" -> defaultPreset,
    "trigger_words" -> ujson.Arr.from(triggerWords), "presets" -> ujson.Obj.from(presets.iterator.map((k, v) => k -> ujson.Str(v)))).render(2)
  def share: String = Marker + "\n" + json
  def save(name: String, text: String): ModelPresetDocument =
    validName(name)
    copy(presets = presets.updated(name, text)).validate()
  def setDefault(name: String): ModelPresetDocument =
    require(presets.contains(name), "Preset does not exist")
    copy(defaultPreset = name).validate()
  def rename(from: String, to: String): ModelPresetDocument =
    validName(to)
    require(presets.contains(from), "Preset does not exist")
    require(from == to || !presets.contains(to), "A preset with that name already exists")
    copy(defaultPreset = if defaultPreset == from then to else defaultPreset,
      presets = VectorMap.from(presets.iterator.map((k, v) => (if k == from then to else k) -> v))).validate()
  def delete(name: String): ModelPresetDocument =
    require(presets.contains(name), "Preset does not exist")
    val remaining = presets.removed(name)
    val next = if remaining.isEmpty then VectorMap("default" -> "") else remaining
    copy(presets = next, defaultPreset = if next.contains(defaultPreset) then defaultPreset else next.head._1).validate()

object ModelPresetDocument:
  val MaxBytes = 4 * 1024 * 1024
  val Marker = "###ModelPresets###"
  val empty = ModelPresetDocument("", "default", Vector.empty, VectorMap("default" -> ""))
  def validName(name: String): Unit =
    require(name.nonEmpty && name.length <= 256 && !name.exists(Character.isISOControl), "Preset name must have 1–256 printable characters")
  def parseBytes(bytes: Array[Byte]): ModelPresetDocument =
    val checked = sdfx.core.presets.PresetJson.readSource(bytes, sdfx.core.presets.PresetLimits(maxBytes = MaxBytes, maxDepth = 8))
    require(checked.isRight, "Invalid, duplicate-key or excessive preset JSON")
    parseValidated(new String(bytes, UTF_8))
  def parse(text: String): ModelPresetDocument = parseBytes(text.getBytes(UTF_8))
  private def parseValidated(text: String): ModelPresetDocument =
    try
      val obj = ujson.read(text).obj
      require(Set("url", "default_preset", "trigger_words", "presets").subsetOf(obj.keySet.toSet), "Missing preset document fields")
      ModelPresetDocument(obj("url").str, obj("default_preset").str,
        obj("trigger_words").arr.toVector.map(_.str), VectorMap.from(obj("presets").obj.iterator.map((k, v) => k -> v.str))).validate()
    catch
      case e: IllegalArgumentException => throw e
      case scala.util.control.NonFatal(e) => throw new IllegalArgumentException("Invalid model preset JSON", e)
  def parseShared(text: String): ModelPresetDocument =
    require(text.getBytes(UTF_8).length <= MaxBytes, "Shared preset text is too large")
    val marker = text.indexOf(Marker)
    if marker < 0 then return parse(text)
    val start = text.indexOf('{', marker + Marker.length)
    require(start >= 0, "Sharing marker has no JSON document")
    var depth = 0
    var quoted = false
    var escaped = false
    var i = start
    while i < text.length do
      val ch = text.charAt(i)
      if quoted then
        if escaped then escaped = false
        else if ch == '\\' then escaped = true
        else if ch == '"' then quoted = false
      else if ch == '"' then quoted = true
      else if ch == '{' then depth += 1
      else if ch == '}' then
        depth -= 1
        if depth == 0 then return parse(text.substring(start, i + 1))
      i += 1
    throw new IllegalArgumentException("Shared preset JSON is incomplete")
  private def triggerPattern(word: String): java.util.regex.Pattern =
    require(word.nonEmpty, "Trigger word is empty")
    java.util.regex.Pattern.compile("(?<![\\p{L}\\p{M}\\p{N}_])" + java.util.regex.Pattern.quote(word) + "(?![\\p{L}\\p{M}\\p{N}_])")
  def containsTrigger(text: String, word: String): Boolean = triggerPattern(word).matcher(text).find()
  def toggleTrigger(text: String, word: String, selected: Boolean): String =
    if selected then if containsTrigger(text, word) then text else word + " " + text
    else
      val matcher = triggerPattern(word).matcher(text)
      val spans = scala.collection.mutable.ArrayBuffer.empty[(Int, Int)]
      while matcher.find() do spans += ((matcher.start(), matcher.end()))
      var result = text
      def horizontal(c: Char): Boolean = c == ' ' || c == '\t'
      spans.reverseIterator.foreach { (begin, finish) =>
        var left = begin
        while left > 0 && horizontal(result(left - 1)) do left -= 1
        var right = finish
        while right < result.length && horizontal(result(right)) do right += 1
        val start = if left > 0 && result(left - 1) == ',' then left - 1 else begin
        if start == begin && right < result.length && result(right) == ',' then
          right += 1
          while right < result.length && horizontal(result(right)) do right += 1
        result = result.take(start) + result.drop(right)
      }
      result.trim
  def appendTemplate(text: String): String = text +
    (if text.isEmpty then "{Your Prompt Here}\n" else "") +
    "Negative prompt:\nSteps: 20, Sampler: Euler a, CFG scale: 7, Size: 512x512, Clip skip: 1\n"
