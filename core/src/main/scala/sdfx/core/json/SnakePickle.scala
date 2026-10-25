package sdfx.core.json

/** upickle configuration: camelCase Scala fields <-> snake_case JSON keys, defaults always written.
  * Used for every JSON surface (engine protocol, settings, MCP) so naming matches A1111 conventions.
  */
object SnakePickle extends upickle.AttributeTagged:

  def camelToSnake(s: String): String =
    val sb = new StringBuilder(s.length + 8)
    s.foreach { c =>
      if c.isUpper then
        sb += '_'
        sb += c.toLower
      else sb += c
    }
    sb.toString

  def snakeToCamel(s: String): String =
    val parts = s.split("_", -1)
    if parts.length == 1 then s
    else parts.head + parts.tail.map(p => if p.isEmpty then "" else s"${p.head.toUpper}${p.tail}").mkString

  override def objectAttributeKeyReadMap(s: CharSequence): CharSequence = snakeToCamel(s.toString)
  override def objectAttributeKeyWriteMap(s: CharSequence): CharSequence = camelToSnake(s.toString)
  override def serializeDefaults: Boolean = true

  // Seq/Map helpers are inherited; this keeps Option as null/value (upickle 4 default).
