package sdfx.core.modelpresets

import java.net.{HttpURLConnection, URI}
import java.io.{ByteArrayOutputStream, StringReader}
import javax.swing.text.html.{HTML, HTMLEditorKit}
import javax.swing.text.html.parser.ParserDelegator

final case class FetchedModelPresets(document: ModelPresetDocument, thumbnail: Option[Array[Byte]], information: ModelPresetDocument):
  def selectDocument(overwrite: Boolean): ModelPresetDocument = if overwrite then document else information

/** Bounded data-only Civitai integration; remote descriptions are never displayed as executable HTML. */
final class CivitaiPresetClient(
    read: (URI, Int, () => Unit) => Array[Byte] = CivitaiPresetClient.readRemote):
  private val root = "https://civitai.com"
  private def get(path: String, check: () => Unit): ujson.Value =
    check()
    val data = read(URI.create(root + "/api/v1/" + path), ModelPresetDocument.MaxBytes, check)
    require(data.length <= ModelPresetDocument.MaxBytes, "Civitai response is too large")
    require(sdfx.core.presets.PresetJson.readSource(data, sdfx.core.presets.PresetLimits(maxDepth = 32)).isRight,
      "Invalid Civitai JSON")
    ujson.read(data)
  private def id(value: ujson.Value): String =
    val text = value match
      case ujson.Num(n) if n.isWhole => n.toLong.toString
      case ujson.Str(s) => s
      case _ => ""
    require(text.matches("[1-9][0-9]{0,18}"), "Invalid Civitai model ID")
    text
  def fetch(hash: String, local: ModelPresetDocument, check: () => Unit): FetchedModelPresets =
    require(hash.matches("[0-9a-fA-F]{64}"), "A full checkpoint SHA256 is required")
    check()
    var model: Option[ujson.Value] = None
    val version = if local.url.trim.isEmpty then get("model-versions/by-hash/" + hash, check)
    else
      val uri = URI.create(local.url.trim)
      require(uri.getScheme == "https" && uri.getHost == "civitai.com" && uri.getUserInfo == null &&
        (uri.getPort == -1 || uri.getPort == 443), "Automatic info retrieval requires an HTTPS civitai.com model URL")
      val versionQuery = Option(uri.getRawQuery).toVector.flatMap(_.split('&')).find(_.startsWith("modelVersionId="))
      versionQuery match
        case Some(v) => get("model-versions/" + id(ujson.Str(v.stripPrefix("modelVersionId="))), check)
        case None =>
          val parts = uri.getPath.split('/').filter(_.nonEmpty)
          require(parts.length >= 2 && Set("models", "model-versions")(parts(0)), "Use a Civitai model or model-version URL")
          if parts(0) == "model-versions" then get("model-versions/" + id(ujson.Str(parts(1))), check)
          else
            val found = get("models/" + id(ujson.Str(parts(1))), check)
            model = Some(found)
            found.obj.get("modelVersions").flatMap(_.arr.headOption)
              .getOrElse(throw new IllegalArgumentException("Civitai model has no versions"))
    val modelId = version.obj.get("modelId").orElse(model.flatMap(_.obj.get("id"))).map(id)
      .getOrElse(throw new IllegalArgumentException("Civitai response has no model ID"))
    val details = model.getOrElse(get("models/" + modelId, check))
    val words = version.obj.get("trainedWords").toVector.flatMap(_.arr.toVector).map(_.str)
    val description = details.obj.get("description").collect { case ujson.Str(s) => s }.getOrElse("")
    val plain = CivitaiPresetClient.plainHtml(description)
    val information = local.copy(url = root + "/models/" + modelId, triggerWords = words).validate()
    val document = if plain.contains(ModelPresetDocument.Marker) then ModelPresetDocument.parseShared(plain)
      else information
    val image = version.obj.get("images").flatMap(_.arr.headOption)
      .flatMap(_.obj.get("url")).collect { case ujson.Str(s) => s }
    val thumbnail = image.map { url =>
      check()
      val uri = URI.create(url)
      CivitaiPresetClient.requireRemote(uri)
      val bytes = read(uri, 16 * 1024 * 1024, check)
      require(bytes.length <= 16 * 1024 * 1024, "Thumbnail response is too large")
      bytes
    }
    check()
    FetchedModelPresets(document, thumbnail, information)

object CivitaiPresetClient:
  private[modelpresets] def requireRemote(uri: URI): Unit =
    require(uri.getScheme == "https" && Set("civitai.com", "image.civitai.com", "blobs-b2.civitai.com")(uri.getHost) &&
      uri.getUserInfo == null && (uri.getPort == -1 || uri.getPort == 443), "Untrusted Civitai download URL")
  def readRemote(uri: URI, limit: Int, check: () => Unit): Array[Byte] =
    val deadline = System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(30)
    def current(): Unit =
      check()
      if System.nanoTime() > deadline then throw new java.net.SocketTimeoutException("Civitai request exceeded 30 seconds")
    def read(url: URI, redirects: Int): Array[Byte] =
      requireRemote(url); current()
      val connection = url.toURL.openConnection().asInstanceOf[HttpURLConnection]
      connection.setInstanceFollowRedirects(false)
      connection.setConnectTimeout(5000); connection.setReadTimeout(1000)
      connection.setRequestProperty("User-Agent", "StableDiffusion-NUI-ModelPresets")
      try
        val status = connection.getResponseCode
        if Set(301, 302, 303, 307, 308)(status) then
          require(redirects < 3, "Too many Civitai redirects")
          read(url.resolve(connection.getHeaderField("Location")), redirects + 1)
        else
          require(status == 200, s"Civitai request failed: HTTP $status")
          require(connection.getContentLengthLong <= limit, "Civitai response is too large")
          val input = connection.getInputStream
          val output = new ByteArrayOutputStream()
          val buffer = new Array[Byte](32768)
          try
            var finished = false
            while !finished do
              current()
              val n = input.read(buffer)
              if n < 0 then finished = true
              else
                require(output.size().toLong + n <= limit, "Civitai response is too large")
                output.write(buffer, 0, n)
            current(); output.toByteArray
          finally input.close()
      finally connection.disconnect()
    read(uri, 0)
  def plainHtml(html: String): String =
    require(html.length <= ModelPresetDocument.MaxBytes, "Description is too large")
    // Decode entities without HTML rendering's whitespace collapse, then strip markup outside JSON strings.
    new ParserDelegator() // initializes the standard HTML entity table
    val dtd = javax.swing.text.html.parser.DTD.getDTD("html32")
    val entity = "&(#x[0-9a-fA-F]+|#[0-9]+|[A-Za-z][A-Za-z0-9]+);".r
    val decoded = entity.replaceAllIn(html, m => {
      val token = m.group(1)
      val replacement = if token.startsWith("#") then
        val value = scala.util.Try(Integer.parseInt(token.drop(if token.startsWith("#x") then 2 else 1),
          if token.startsWith("#x") then 16 else 10)).toOption.filter(Character.isValidCodePoint)
        value.map(cp => new String(Character.toChars(cp))).getOrElse(m.matched)
      else if token == "apos" then "'"
      else Option(dtd.getEntity(token)).map(e => new String(e.getData)).getOrElse(m.matched)
      java.util.regex.Matcher.quoteReplacement(replacement)
    })
    val text = new StringBuilder
    val marker = decoded.indexOf(ModelPresetDocument.Marker)
    var json = false
    var quoted = false
    var escaped = false
    var i = 0
    while i < decoded.length do
      val ch = decoded.charAt(i)
      if !quoted && ch == '<' then
        var end = i + 1
        var attributeQuote = ' '
        while end < decoded.length && (decoded.charAt(end) != '>' || attributeQuote != ' ') do
          val c = decoded.charAt(end)
          if attributeQuote != ' ' && c == attributeQuote then attributeQuote = ' '
          else if attributeQuote == ' ' && (c == '\'' || c == '"') then attributeQuote = c
          end += 1
        if end < decoded.length then
          val tag = decoded.substring(i + 1, end).trim.toLowerCase(java.util.Locale.ROOT)
          if tag.startsWith("br") || tag == "/p" || tag == "/div" then text.append('\n')
          i = end
        else text.append(ch)
      else
        text.append(ch)
        if marker >= 0 && i > marker && ch == '{' then json = true
        if json then
          if quoted then
            if escaped then escaped = false
            else if ch == '\\' then escaped = true
            else if ch == '"' then quoted = false
          else if ch == '"' then quoted = true
      i += 1
    text.toString
