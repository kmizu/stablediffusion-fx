package sdfx.mcp

/** Minimal JSON-RPC 2.0 message model used by the MCP server. */
object JsonRpc:
  val Version = "2.0"

  object Codes:
    val ParseError = -32700
    val InvalidRequest = -32600
    val MethodNotFound = -32601
    val InvalidParams = -32602
    val InternalError = -32603
    /** Not part of JSON-RPC: used when a request was cancelled by the client and a reply is still required. */
    val RequestCancelled = -32800

  /** A JSON-RPC error raised while handling a request. */
  final case class RpcError(code: Int, message: String, data: Option[ujson.Value] = None) extends Exception(message)

  sealed trait Message
  final case class Request(id: ujson.Value, method: String, params: ujson.Value) extends Message
  final case class Notification(method: String, params: ujson.Value) extends Message
  final case class Response(id: ujson.Value, result: Option[ujson.Value], error: Option[ujson.Value]) extends Message

  /** Parses one (non-batch) message; Left carries an error response for invalid requests. */
  def parse(v: ujson.Value): Either[ujson.Value, Message] = v match
    case o: ujson.Obj =>
      val m = o.value
      val idOpt = m.get("id")
      val validId = idOpt.forall {
        case _: ujson.Str | _: ujson.Num => true
        case _                           => false
      }
      def bad(msg: String) = Left(error(idOpt.filter(_ => validId).getOrElse(ujson.Null), Codes.InvalidRequest, msg))
      if !m.get("jsonrpc").contains(ujson.Str(Version)) then bad("jsonrpc must be \"2.0\"")
      else
        m.get("method") match
          case Some(ujson.Str(method)) =>
            val params = m.get("params").filterNot(_ == ujson.Null).getOrElse(ujson.Obj())
            params match
              case _: ujson.Obj | _: ujson.Arr =>
                idOpt match
                  case None                => Right(Notification(method, params))
                  case Some(id) if validId => Right(Request(id, method, params))
                  case Some(_)             => bad("id must be a string or a number")
              case _ => bad("params must be an object or an array")
          case Some(_) => bad("method must be a string")
          case None =>
            if m.contains("result") || m.contains("error") then
              idOpt match
                case Some(id) => Right(Response(id, m.get("result"), m.get("error")))
                case None     => bad("response without id")
            else bad("missing method")
    case _ => Left(error(ujson.Null, Codes.InvalidRequest, "a JSON-RPC message must be an object"))

  def result(id: ujson.Value, result: ujson.Value): ujson.Value =
    ujson.Obj("jsonrpc" -> Version, "id" -> id, "result" -> result)

  def error(id: ujson.Value, code: Int, message: String, data: Option[ujson.Value] = None): ujson.Value =
    val e = ujson.Obj("code" -> code, "message" -> message)
    data.foreach(d => e("data") = d)
    ujson.Obj("jsonrpc" -> Version, "id" -> id, "error" -> e)

  def notification(method: String, params: ujson.Value): ujson.Value =
    ujson.Obj("jsonrpc" -> Version, "method" -> method, "params" -> params)

  def parseError(detail: String): ujson.Value = error(ujson.Null, Codes.ParseError, s"Parse error: $detail")

  val MaxDepth = 256

  /** Parses JSON text, rejecting nesting deeper than [[MaxDepth]] before handing it to ujson (no stack overflow). */
  def parseText(text: String): Either[String, ujson.Value] =
    var depth = 0
    var inStr = false
    var esc = false
    var i = 0
    var tooDeep = false
    while i < text.length && !tooDeep do
      val c = text.charAt(i)
      if inStr then
        if esc then esc = false
        else if c == '\\' then esc = true
        else if c == '"' then inStr = false
      else if c == '"' then inStr = true
      else if c == '[' || c == '{' then
        depth += 1
        if depth > MaxDepth then tooDeep = true
      else if c == ']' || c == '}' then depth -= 1
      i += 1
    if tooDeep then Left(s"nesting deeper than $MaxDepth")
    else
      try Right(ujson.read(text))
      catch
        case e: StackOverflowError => Left("nesting too deep")
        case scala.util.control.NonFatal(e) => Left(Option(e.getMessage).getOrElse("invalid JSON").linesIterator.nextOption().getOrElse("invalid JSON"))

  /** Canonical map key for a request id (`1` and `1.0` are the same id; strings are distinct from numbers). */
  def idKey(id: ujson.Value): String = id match
    case ujson.Num(n) if n.isWhole && math.abs(n) < 9.0e15 => s"n:${n.toLong}"
    case ujson.Num(n)                                     => s"n:$n"
    case ujson.Str(s)                                     => s"s:$s"
    case other                                            => s"x:${ujson.write(other)}"

  /** Single-line serialization (no raw newlines), as required by the stdio transport. */
  def render(v: ujson.Value): String = ujson.write(v)
