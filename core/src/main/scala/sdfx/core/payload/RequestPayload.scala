package sdfx.core.payload

/** Diagnostic identity, independent of the engine transport's request IDs. */
final case class PayloadKey(jobId: String, sequence: Int)

/** JSON strings are frozen at submission; no mutable form or ujson references escape. */
final case class SubmittedPayloadContext(
    origin: String,
    sourceMode: Option[String] = None,
    configurationJson: Option[String] = None
)

enum PayloadState(val wireName: String):
  case Prepared extends PayloadState("prepared")
  case Submitted extends PayloadState("submitted")
  case Succeeded extends PayloadState("succeeded")
  case Failed extends PayloadState("failed")
  case Cancelled extends PayloadState("cancelled")

  def isTerminal: Boolean = this == Succeeded || this == Failed || this == Cancelled

/** Memory-only documents. None means no body was dispatched/retained, never a previous job's body. */
final case class PayloadSnapshot(
    key: PayloadKey,
    requestType: String,
    bodyJson: Option[String],
    capturedAtMillis: Long,
    contextJson: Option[String] = None,
    assetsJson: Option[String] = None,
    state: PayloadState = PayloadState.Prepared,
    warnings: Vector[String] = Vector.empty,
    sourceMode: Option[String] = None
):
  /** An enclosing script's failure cannot rewrite a completed sub-generation. */
  def transition(next: PayloadState): PayloadSnapshot = if state.isTerminal then this else copy(state = next)

/** Immutable presentation data, built off the UI thread. Container values are never reserialized here. */
final case class PayloadNode(
    label: String,
    path: String,
    depth: Int,
    kind: String,
    valueJson: Option[String],
    children: Vector[PayloadNode]
):
  def isContainer: Boolean = kind == "object" || kind == "array"
  def childCount: Int = children.size

object RequestPayload:
  /** This conversion is required at the dispatch boundary; callers must not swallow its failure. */
  def freeze(value: ujson.Value): String = ujson.write(value)

  def keyPath(parent: String, key: String): String = s"$parent[${ujson.write(ujson.Str(key))}]"
  def indexPath(parent: String, index: Int): String = s"$parent[$index]"

  /** Whole-tree allocation limits, separate from incremental visible expansion limits. maxNodes also counts
    * raw object-key tokens before parsing. Text includes all labels and rendered scalar JSON; paths are separate.
    */
  final case class TreeLimits(maxNodes: Int = 100000, maxDepth: Int = 512, maxPathBytes: Int = 65536,
      maxTotalPathBytes: Int = 32 * 1024 * 1024, maxTextBytes: Int = 24 * 1024 * 1024):
    require(maxNodes > 0 && maxDepth > 0 && maxPathBytes >= 0 && maxTotalPathBytes >= 0 && maxTextBytes >= 0)

  /** The exact frozen document remains available for Copy even when its whole tree cannot be presented. */
  final class TreeUnavailable(val reason: String) extends IllegalArgumentException(s"Tree unavailable: $reason")

  /** Parse/build on a worker thread. Bound JSON before parsing, then reserve derived strings before rendering
    * or concatenating them. Keeping each parent's byte count avoids rescanning its prefix for every child.
    */
  def tree(json: String, limits: TreeLimits = TreeLimits()): PayloadNode =
    def unavailable(reason: String): Nothing = throw new TreeUnavailable(reason)

    // Measure the current ujson string spelling without building an escaped string or UTF-8 byte array.
    // ujson.write is used only after admission, preserving its short control escapes and literal Unicode.
    def textBytes(text: String, quoted: Boolean, cap: Long, reason: String): Long =
      var bytes = if quoted then 2L else 0L
      if bytes > cap then unavailable(reason)
      var index = 0
      while index < text.length do
        val char = text.charAt(index)
        val pair = Character.isHighSurrogate(char) && index + 1 < text.length && Character.isLowSurrogate(text.charAt(index + 1))
        val increment =
          if quoted && (char == '"' || char == '\\' || char == '\b' || char == '\t' || char == '\n' || char == '\f' || char == '\r') then 2
          else if quoted && char < ' ' then 6
          else if pair then 4
          else if char <= 0x7f then 1
          else if char <= 0x7ff then 2
          else if Character.isSurrogate(char) then 1 // JVM UTF-8 replacement for an unpaired surrogate.
          else 3
        bytes += increment
        if bytes > cap then unavailable(reason)
        index += (if pair then 2 else 1)
      bytes

    val parsed = PayloadJsonBudget.admit(json,
      PayloadJsonBudget.Limits(maxNodes = limits.maxNodes, maxDepth = limits.maxDepth))(ujson.read(json)) match
      case Right(value) => value
      case Left(reason) => unavailable(reason)
    if limits.maxPathBytes < 1 then unavailable("path_limit")
    if limits.maxTotalPathBytes < 1 then unavailable("total_path_limit")
    if limits.maxTextBytes < 1 then unavailable("text_limit")
    var totalPathBytes = 1L // Root '$'.
    var totalTextBytes = 1L // Root label '$'.

    def reserveLabel(label: String): Unit =
      totalTextBytes += textBytes(label, false, limits.maxTextBytes.toLong - totalTextBytes, "text_limit")

    def childPathBytes(parentBytes: Long, label: String, objectKey: Boolean): Long =
      val singleRemaining = limits.maxPathBytes.toLong - parentBytes - 2 // '[' and ']'.
      val totalRemaining = limits.maxTotalPathBytes.toLong - totalPathBytes - parentBytes - 2
      val reason = if singleRemaining <= totalRemaining then "path_limit" else "total_path_limit"
      val suffixBytes = textBytes(label, objectKey, math.min(singleRemaining, totalRemaining), reason)
      val bytes = parentBytes + 2 + suffixBytes
      totalPathBytes += bytes
      bytes

    def scalarJson(value: ujson.Value): String =
      value match
        case ujson.Str(text) =>
          totalTextBytes += textBytes(text, true, limits.maxTextBytes.toLong - totalTextBytes, "text_limit")
          freeze(value)
        case _ =>
          // Number/bool/null rendering is constant-size; no unadmitted string is serialized here.
          val scalar = freeze(value)
          totalTextBytes += textBytes(scalar, false, limits.maxTextBytes.toLong - totalTextBytes, "text_limit")
          scalar

    final case class Work(value: ujson.Value, label: String, path: String, pathBytes: Long, depth: Int, finish: Boolean)
    val pending = scala.collection.mutable.ArrayDeque(Work(parsed, "$", "$", 1L, 0, false))
    val completed = scala.collection.mutable.ArrayBuffer.empty[PayloadNode]
    while pending.nonEmpty do
      val work = pending.removeLast()
      val entries: Vector[(String, ujson.Value)] = work.value match
        case o: ujson.Obj => o.value.toVector.sortBy(_._1)
        case a: ujson.Arr => a.value.iterator.zipWithIndex.map((v, i) => i.toString -> v).toVector
        case _ => Vector.empty
      if !work.finish && entries.nonEmpty then
        pending.append(work.copy(finish = true))
        entries.reverseIterator.foreach { (label, value) =>
          val objectKey = work.value.isInstanceOf[ujson.Obj]
          val pathBytes = childPathBytes(work.pathBytes, label, objectKey)
          reserveLabel(label)
          val path = if objectKey then keyPath(work.path, label) else indexPath(work.path, label.toInt)
          pending.append(Work(value, label, path, pathBytes, work.depth + 1, false))
        }
      else
        val kind = work.value match
          case _: ujson.Obj => "object"
          case _: ujson.Arr => "array"
          case _: ujson.Str => "string"
          case _: ujson.Num => "number"
          case _: ujson.Bool => "boolean"
          case _ => "null"
        val children = if entries.isEmpty then Vector.empty else completed.takeRight(entries.size).toVector
        if entries.nonEmpty then completed.dropRightInPlace(entries.size)
        val scalar = Option.when(kind != "object" && kind != "array")(scalarJson(work.value))
        completed.append(PayloadNode(work.label, work.path, work.depth, kind, scalar, children))
    completed.head

/** A persistent DFS cursor. -1 means the node itself has not been visited; otherwise only the named child
  * is next. Retaining a wide container costs one frame, without enumerating or copying its siblings.
  */
final case class PayloadTreeFrame(node: PayloadNode, nextChildIndex: Int = -1):
  def path: String = node.path

/** Node depths are absolute and remain unchanged by continuation. Raising maxDepth explicitly resumes a
  * depth boundary. A slice processes at most maxNodes nodes and 3*maxNodes cursor frames, including unwinding.
  */
final case class PayloadTreeWork(state: PayloadTreeState, remaining: List[PayloadTreeFrame], visited: Int,
    depthLimited: Boolean, processedFrames: Long)

final case class PayloadTreeState(
    expanded: Set[String] = Set.empty,
    marked: Set[String] = Set.empty,
    selectedPath: Option[String] = None,
    initialLimitReached: Boolean = false,
    initialRemaining: List[PayloadTreeFrame] = Nil
):
  def toggle(node: PayloadNode): PayloadTreeState =
    if !node.isContainer then this
    else copy(expanded = if expanded(node.path) then expanded - node.path else expanded + node.path)

  def mark(node: PayloadNode): PayloadTreeState =
    copy(marked = if marked(node.path) then marked - node.path else marked + node.path)

  def showPath(node: PayloadNode): PayloadTreeState = copy(selectedPath = Some(node.path))

  def subtree(node: PayloadNode, expand: Boolean, maxNodes: Int = 1000, maxDepth: Int = 128): PayloadTreeWork =
    resume(List(PayloadTreeFrame(node)), expand, maxNodes, maxDepth)

  def resume(nodes: List[PayloadTreeFrame], expand: Boolean, maxNodes: Int = 1000, maxDepth: Int = 128): PayloadTreeWork =
    visit(nodes, maxNodes, maxDepth, node => expand)

  def resumeInitial(maxNodes: Int = 1000, maxDepth: Int = 128): PayloadTreeWork =
    val result = visit(initialRemaining, maxNodes, maxDepth, node => node.depth == 0 || node.childCount < 2)
    result.copy(state = result.state.copy(initialLimitReached = result.remaining.nonEmpty, initialRemaining = result.remaining))

  private[payload] def visit(nodes: List[PayloadTreeFrame], maxNodes: Int, maxDepth: Int,
      expand: PayloadNode => Boolean): PayloadTreeWork =
    require(maxNodes > 0, "node budget must be positive")
    require(maxDepth >= 0, "depth budget must be nonnegative")
    var pending = nodes
    var result = expanded
    var visited = 0
    var depthLimited = false
    var processedFrames = 0L
    while pending.nonEmpty && visited < maxNodes && processedFrames < maxNodes.toLong * 3 && !depthLimited do
      val frame = pending.head
      val node = frame.node
      processedFrames += 1
      if frame.nextChildIndex < 0 then
        visited += 1
        if node.depth > maxDepth then
          // Keep this node plus its ancestor cursors. Do not enumerate other depth-limited siblings.
          depthLimited = true
        else
          pending = pending.tail
          if node.isContainer then
            result = if expand(node) then result + node.path else result - node.path
            if node.children.nonEmpty then pending = PayloadTreeFrame(node, 0) :: pending
      else if frame.nextChildIndex >= node.childCount then
        pending = pending.tail
      else
        val child = node.children(frame.nextChildIndex)
        pending = PayloadTreeFrame(child) :: frame.copy(nextChildIndex = frame.nextChildIndex + 1) :: pending.tail
    PayloadTreeWork(copy(expanded = result), pending, visited, depthLimited, processedFrames)

object PayloadTreeState:
  def initial(root: PayloadNode, maxNodes: Int = 1000, maxDepth: Int = 128): PayloadTreeState =
    PayloadTreeState(initialRemaining = List(PayloadTreeFrame(root))).resumeInitial(maxNodes, maxDepth).state
