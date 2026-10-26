package sdfx.ui.view

import javafx.application.Platform
import javafx.scene.control.*
import javafx.scene.input.{KeyCode, KeyEvent, MouseButton, MouseEvent}
import javafx.scene.layout.{FlowPane, Priority, VBox}
import sdfx.core.payload.*
import sdfx.ui.Fx

import java.util.concurrent.{Executors, ThreadFactory}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong, AtomicReference}
import scala.collection.mutable
import scala.util.control.NonFatal

/** Pure adapters deliberately have no Toolkit or clipboard dependency. */
object PayloadJsonTree:
  val SliceLimit = 1000
  val WindowLimit = 10000
  val DepthLimit = 128
  private val LabelLimit = 512

  enum Action:
    case Toggle, ToggleSubtree, ExpandSubtree, CollapseSubtree, Mark, ShowPath
  final case class Modifiers(alt: Boolean = false, shift: Boolean = false,
      control: Boolean = false, meta: Boolean = false)
  final case class Entry(node: Option[PayloadNode], notice: String = "")
  final case class ChildSlice(nodes: Vector[PayloadNode], nextOffset: Option[Int], boundary: Option[String])

  def action(modifiers: Modifiers): Action =
    if modifiers.alt then Action.Mark
    else if modifiers.shift then Action.ShowPath
    else if modifiers.control || modifiers.meta then Action.ToggleSubtree
    else Action.Toggle

  def change(state: PayloadTreeState, node: PayloadNode, action: Action): PayloadTreeState = action match
    case Action.Mark => state.mark(node)
    case Action.ShowPath => state.showPath(node)
    case Action.Toggle => state.toggle(node)
    // Recursive changes are scheduled through subtreeSlice, never performed by a cell handler.
    case _ => state

  def subtreeSlice(state: PayloadTreeState, remaining: List[PayloadTreeFrame], expand: Boolean,
      maxDepth: Int, maxNodes: Int = SliceLimit): PayloadTreeWork =
    state.resume(remaining, expand, math.max(1, math.min(SliceLimit, maxNodes)), maxDepth)

  def children(node: PayloadNode, offset: Int, available: Int, maxDepth: Int): ChildSlice =
    val start = offset.max(0).min(node.childCount)
    if start == node.childCount then ChildSlice(Vector.empty, None, None)
    else if node.children(start).depth > maxDepth then
      ChildSlice(Vector.empty, Some(start), Some(s"Tree depth limit $maxDepth; continue in this subtree"))
    else if available <= 0 then
      ChildSlice(Vector.empty, Some(start), Some(s"Tree window limit $WindowLimit; continue in a new window"))
    else
      val end = (start.toLong + math.min(SliceLimit, available)).min(node.childCount.toLong).toInt
      ChildSlice(node.children.slice(start, end), Option.when(end < node.childCount)(end), None)

  /** Bounded by code points so a clipped emoji never ends in an unpaired surrogate. */
  private def clipped(text: String, max: Int = LabelLimit): String =
    var index = 0
    var count = 0
    while index < text.length && count < max do
      index += Character.charCount(text.codePointAt(index))
      count += 1
    if index < text.length then text.substring(0, index) + "…" else text

  def label(node: PayloadNode): String =
    val value = if node.isContainer then s"${node.kind} (${node.childCount})" else node.valueJson.getOrElse("null")
    // Reserve space for the type/count or value even when a hostile key is itself multi-MiB.
    // Each segment adds at most one ellipsis: 193 + 2 + 317 = 512 code points total.
    clipped(s"${clipped(node.label, 192)}: ${clipped(value, 316)}")

  def pathLabel(path: String): String = clipped(path)

  /** Theme lookup colors keep the same value distinctions in both app themes. */
  def typeColor(kind: String): String = kind match
    case "string" => "-sdfx-ok"
    case "number" => "-sdfx-hires"
    case "boolean" => "-sdfx-accent"
    case "null" => "-sdfx-muted"
    case "array" => "-sdfx-warn"
    case _ => "-sdfx-text"

  /** Counts all attached TreeItems, including lazy/continuation rows, not just JSON data rows. */
  final class MaterializationBudget(val limit: Int = WindowLimit):
    require(limit > 0)
    private var count = 0
    def used: Int = count
    def remaining: Int = limit - count
    def reserve(requested: Int): Int =
      val n = requested.max(0).min(SliceLimit).min(remaining)
      count += n
      n
    private[view] def release(n: Int): Unit = count = (count - n).max(0)
    def reset(): Unit = count = 0

  /** FX-only ordering and bounded publication. Explicit newer expansion input supersedes pending
    * recursive work. Iterators are resumed between pulses without copying/scanning a whole frontier.
    */
  final class ExpansionPublication(postUi: (() => Unit) => Unit):
    private var revision = 0L
    def supersede(): Long = { revision += 1; revision }
    def isCurrent(ticket: Long): Boolean = ticket == revision
    def publish(ticket: Long, paths: Iterator[String], expand: Boolean, validWindow: () => Boolean)(
        update: (String, Boolean) => Unit, observed: Int => Unit, complete: () => Unit): Unit =
      def current = isCurrent(ticket) && validWindow()
      def slice(): Unit =
        if current then
          var count = 0
          while current && count < SliceLimit && paths.hasNext do
            update(paths.next(), expand)
            count += 1
          observed(count)
          if current then
            if paths.hasNext then postUi(() => slice()) else complete()
      postUi(() => slice())

  /** Latest-document guard is shared by parse, child pages and subtree work. Neither stale errors nor
    * stale successes reach the controls. Schedulers are injected so ordinary tests need no Toolkit.
    */
  final class AsyncPresentation(schedule: (() => Unit) => Unit, deliver: (() => Unit) => Unit,
      isUiThread: () => Boolean, parser: String => PayloadNode = json => RequestPayload.tree(json)):
    private val generation = new AtomicLong()
    private val closed = new AtomicBoolean()
    private val pendingParse = new AtomicReference[() => Unit]()
    private val parseQueued = new AtomicBoolean()
    private val pendingPublication = new AtomicReference[() => Unit]()
    private val publicationQueued = new AtomicBoolean()
    def invalidate(): Unit =
      generation.incrementAndGet()
      pendingParse.set(null)
      pendingPublication.set(null)
    def dispose(): Unit = { closed.set(true); invalidate() }
    private def task[A](ticket: Long, body: () => A, success: A => Unit,
        error: Throwable => Unit, publish: (() => Unit) => Unit): () => Unit =
      def current = !closed.get() && generation.get() == ticket
      () => {
        if current then
          val result = try
            require(!isUiThread(), "Payload presentation work cannot run on the UI thread")
            Right(body())
          catch case NonFatal(e) => Left(e)
          if current then publish(() => {
            if current then result match
              case Right(value) => success(value)
              case Left(e) => error(e)
          })
      }
    def run[A](body: => A)(success: A => Unit, error: Throwable => Unit): Unit =
      if !closed.get() then schedule(task(generation.get(), () => body, success, error, deliver))

    // Drain Runnables retain only this gate. Documents/results live in replaceable slots, so rapid
    // selections do not leave a queue of old strings or immutable roots waiting for a worker/FX pulse.
    private def drainParse(): Unit =
      parseQueued.set(false)
      val work = pendingParse.getAndSet(null)
      if work != null then work()
    private def drainPublication(): Unit =
      publicationQueued.set(false)
      val work = pendingPublication.getAndSet(null)
      if work != null then work()
    private def publishLatest(work: () => Unit): Unit =
      if !closed.get() then
        pendingPublication.set(work)
        if closed.get() then pendingPublication.compareAndSet(work, null)
        else if publicationQueued.compareAndSet(false, true) then deliver(() => drainPublication())

    def showJson(json: String)(success: PayloadNode => Unit, error: Throwable => Unit): Unit =
      invalidate()
      if !closed.get() then
        val work = task(generation.get(), () => checkedParse(json), success, error, publishLatest)
        pendingParse.set(work)
        if closed.get() then pendingParse.compareAndSet(work, null)
        else if parseQueued.compareAndSet(false, true) then schedule(() => drainParse())

    private def checkedParse(json: String): PayloadNode =
      // Admission precedes any ujson allocation and bounds tokens/depth as well as UTF-8 bytes.
      // The frozen source belongs to the caller; rejected presentation never rewrites its document.
      def unavailable(reason: String): Nothing =
        val bounded = clipped(reason, 128)
        val detail = bounded match
          case "bytes_limit" => "JSON exceeds the 24 MiB tree presentation limit"
          case "nodes_limit" => "JSON exceeds the 100,000-token tree presentation limit"
          case "depth_limit" => "JSON exceeds the depth-512 tree presentation limit"
          case "path_limit" => "A JSON path exceeds the tree presentation limit"
          case "total_path_limit" => "Combined JSON paths exceed the tree presentation limit"
          case "text_limit" => "Derived JSON text exceeds the tree presentation limit"
          case "invalid_json" => "JSON could not be parsed"
          case _ => "JSON tree presentation unavailable"
        throw new IllegalArgumentException(s"$detail ($bounded)")
      PayloadJsonBudget.guard(json) match
        case Left(reason) => unavailable(reason)
        case Right(_) => ()
      // Keep structured allocation rejections distinct from malformed JSON. The parser runs once;
      // a rejected immutable tree is never rebuilt just to diagnose it. VM-fatal errors still escape.
      try parser(json)
      catch
        case e: RequestPayload.TreeUnavailable => unavailable(e.reason)
        case NonFatal(_) => unavailable("invalid_json")

/** Read-only, literal-text JSON tree. A continuation replaces the current bounded window; it never
  * increases the number of retained TreeItems. Documents, marks and exact paths remain immutable.
  * All public methods and onStateChanged run on FX. Owns no store, engine, clipboard or raw editor.
  */
final class PayloadJsonTree:
  import PayloadJsonTree.*

  private val worker = Executors.newSingleThreadExecutor((r => {
    val thread = new Thread(r, "sdfx-payload-tree")
    thread.setDaemon(true)
    thread
  }): ThreadFactory)
  private val async = new AsyncPresentation(r => worker.execute(() => r()), r => Platform.runLater(() => r()), () => Platform.isFxApplicationThread)
  private val expansionPublication = new ExpansionPublication(r => Platform.runLater(() => r()))
  private val budget = new MaterializationBudget()
  private var disposed = false
  private var document: Option[PayloadNode] = None
  private var currentState = PayloadTreeState()
  private var windowDepth = DepthLimit
  private var windowTicket = 0L
  private var recursiveActive = false
  private var syncing = false
  private var history = List.empty[(PayloadNode, Int, Int)]
  private var currentWindow: Option[(PayloadNode, Int, Int)] = None
  private var recursiveContinuation: Option[(List[PayloadTreeFrame], Boolean, Int)] = None
  private val items = mutable.HashMap.empty[String, JsonItem]
  // Retain explicit decisions across page replacement, without retaining TreeItems from old windows.
  private val explicitExpansion = mutable.HashMap.empty[String, Boolean]

  var onStateChanged: PayloadTreeState => Unit = _ => ()
  /** Native acceptance instrumentation; counts visited touched paths, including unattached paths. */
  private[ui] var onExpansionSlice: Int => Unit = _ => ()
  def state: PayloadTreeState = currentState

  private val status = W.label("No JSON selected.", "muted")
  status.setId("payload-tree-status")
  status.setWrapText(true)
  private val path = W.label("Path: select Show path or Shift-click a value.", "muted")
  path.setId("payload-tree-path")
  path.setWrapText(true)
  path.setAccessibleText("Selected JSON path")
  private val tree = new TreeView[Entry]()
  tree.setId("payload-json-tree")
  tree.setAccessibleText("Read-only JSON tree")
  tree.setFixedCellSize(28)
  tree.setShowRoot(true)
  private val back = W.button("Back", "Return to the previous tree window")(backWindow())
  back.setId("payload-tree-back")
  private val rootButton = W.button("Root", "Return to the document root")(document.foreach(n => openWindow(n, 0, DepthLimit, true)))
  rootButton.setId("payload-tree-root")
  private val toggle = W.button("Toggle", "Toggle the selected container")(selected(Action.Toggle))
  toggle.setId("payload-tree-toggle")
  private val expand = W.button("Expand subtree", "Expand up to 10,000 nodes in bounded worker slices")(selected(Action.ExpandSubtree))
  expand.setId("payload-tree-expand")
  private val collapse = W.button("Collapse subtree", "Collapse up to 10,000 nodes in bounded worker slices")(selected(Action.CollapseSubtree))
  collapse.setId("payload-tree-collapse")
  private val mark = W.button("Mark / unmark", "Toggle the selected value's mark")(selected(Action.Mark))
  mark.setId("payload-tree-mark")
  private val showPath = W.button("Show path", "Show the selected value's exact JSON path")(selected(Action.ShowPath))
  showPath.setId("payload-tree-show-path")
  private val continue = W.button("Continue", "Continue the selected boundary, or the bounded subtree operation")(continueWork())
  continue.setId("payload-tree-continue")
  private val actions = new FlowPane(6, 6, back, rootButton, toggle, expand, collapse, mark, showPath, continue)
  private val help = W.label("Click: toggle · Ctrl/Meta: subtree · Alt: mark · Shift: path. Keyboard: select a row, then use the actions above. Boundary rows continue in a new tree window.", "hint")
  help.setWrapText(true)
  val node = new VBox(7, actions, status, tree, path, help)
  node.setId("payload-json-tree-view")
  VBox.setVgrow(tree, Priority.ALWAYS)

  private class MoreItem(message: String, val proceed: () => Unit) extends TreeItem[Entry](Entry(None, message))
  private class JsonItem(val data: PayloadNode) extends TreeItem[Entry](Entry(Some(data))):
    var offset = 0
    var loading = false
    expandedProperty.addListener((_, _, expanded) => {
      if !syncing && !disposed then
        cancelRecursive(notify = true)
        currentState = currentState.copy(expanded =
          if expanded then currentState.expanded + data.path else currentState.expanded - data.path)
        explicitExpansion(data.path) = expanded
        changed()
      if expanded && !disposed && offset == 0 && !loading then load(this, 0)
    })

  tree.setCellFactory(_ => new TreeCell[Entry]():
    override def updateItem(entry: Entry, empty: Boolean): Unit =
      super.updateItem(entry, empty)
      setGraphic(null)
      setTooltip(null)
      if empty || entry == null then
        setText(null)
        setStyle("")
        setAccessibleText(null)
      else entry.node match
        case Some(n) =>
          val marked = currentState.marked(n.path)
          val text = (if marked then "● " else "") + label(n)
          setText(text)
          setAccessibleText(text)
          setStyle(s"-fx-text-fill: ${typeColor(n.kind)};" +
            (if marked then "-fx-background-color: -sdfx-accent-soft; -fx-font-weight: bold;" else ""))
        case None =>
          setText(entry.notice)
          setAccessibleText(entry.notice)
          setStyle("-fx-text-fill: -sdfx-warn;")
    addEventFilter(MouseEvent.MOUSE_PRESSED, e => {
      if e.getButton == MouseButton.PRIMARY && !isEmpty then
        tree.getSelectionModel.select(getTreeItem)
        getTreeItem match
          case more: MoreItem => more.proceed()
          case item: JsonItem => interact(item, action(Modifiers(e.isAltDown, e.isShiftDown, e.isControlDown, e.isMetaDown)))
          case _ => ()
        e.consume()
    })
  )
  tree.addEventFilter(KeyEvent.KEY_PRESSED, e => {
    if e.getCode == KeyCode.ENTER || e.getCode == KeyCode.SPACE then
      Option(tree.getSelectionModel.getSelectedItem).foreach {
        case more: MoreItem => more.proceed()
        case item: JsonItem => interact(item, action(Modifiers(e.isAltDown, e.isShiftDown, e.isControlDown, e.isMetaDown)))
        case _ => ()
      }
      e.consume()
  })
  tree.getSelectionModel.selectedItemProperty.addListener((_, _, _) => updateButtons())
  updateButtons()

  private def fxOnly(): Unit = require(Platform.isFxApplicationThread, "PayloadJsonTree must be used on FX")
  private def changed(): Unit =
    path.setText(currentState.selectedPath.map(p => "Path: " + pathLabel(p)).getOrElse("Path: select Show path or Shift-click a value."))
    // The exact path remains in state; path presentation alone is clipped.
    onStateChanged(currentState)
    updateButtons()

  def clear(): Unit =
    fxOnly()
    if !disposed then
      async.invalidate()
      expansionPublication.supersede()
      recursiveActive = false
      windowTicket += 1
      document = None
      currentState = PayloadTreeState()
      explicitExpansion.clear()
      history = Nil
      currentWindow = None
      recursiveContinuation = None
      tree.setRoot(null)
      items.clear()
      budget.reset()
      status.setText("No JSON selected.")
      changed()

  def show(root: PayloadNode): Unit =
    clear()
    if !disposed then
      document = Some(root)
      openWindow(root, 0, root.depth + DepthLimit, false)

  def showJson(frozenJson: String): Unit =
    clear()
    if !disposed then
      status.setText("Building JSON tree…")
      async.showJson(frozenJson)(root => {
        document = Some(root)
        openWindow(root, 0, root.depth + DepthLimit, false)
      }, e => status.setText("Tree unavailable: " + pathLabel(Fx.message(e))))

  def dispose(): Unit =
    fxOnly()
    if !disposed then
      clear()
      disposed = true
      onStateChanged = _ => ()
      onExpansionSlice = _ => ()
      async.dispose()
      worker.shutdownNow()
      updateButtons()

  private def updateButtons(): Unit =
    val selected = Option(tree.getSelectionModel.getSelectedItem)
    val data = selected.flatMap(v => v.getValue.node)
    toggle.setDisable(disposed || !data.exists(_.isContainer))
    expand.setDisable(disposed || !data.exists(_.isContainer))
    collapse.setDisable(disposed || !data.exists(_.isContainer))
    mark.setDisable(disposed || data.isEmpty)
    showPath.setDisable(disposed || data.isEmpty)
    continue.setDisable(disposed || (!selected.exists(_.isInstanceOf[MoreItem]) && recursiveContinuation.isEmpty))
    back.setDisable(disposed || history.isEmpty)
    rootButton.setDisable(disposed || document.isEmpty)

  private def selected(action: Action): Unit =
    Option(tree.getSelectionModel.getSelectedItem).collect { case item: JsonItem => item }.foreach(interact(_, action))

  private def interact(item: JsonItem, action: Action): Unit =
    if !disposed then action match
      case Action.ToggleSubtree => recursive(item.data, !currentState.expanded(item.data.path))
      case Action.ExpandSubtree => recursive(item.data, true)
      case Action.CollapseSubtree => recursive(item.data, false)
      case Action.Toggle =>
        if item.data.isContainer then
          explicitExpansion(item.data.path) = !item.isExpanded
          item.setExpanded(!item.isExpanded)
      case other =>
        currentState = change(currentState, item.data, other)
        tree.refresh()
        changed()

  private def itemCost(n: PayloadNode): Int = if n.children.nonEmpty then 2 else 1
  private def makeItem(n: PayloadNode): JsonItem =
    require(budget.reserve(itemCost(n)) == itemCost(n), "TreeItem allocation exceeded the live budget")
    val item = new JsonItem(n)
    items(n.path) = item
    if n.children.nonEmpty then
      item.getChildren.add(new MoreItem("Expand to load children", () => load(item, 0)))
    item

  private def openWindow(root: PayloadNode, offset: Int, depth: Int, remember: Boolean): Unit =
    if !disposed then
      val cancelled = recursiveActive || recursiveContinuation.nonEmpty
      cancelRecursive(notify = false)
      if remember then history = currentWindow.toList ::: history.take(15)
      currentWindow = Some((root, offset, depth))
      windowTicket += 1
      windowDepth = depth
      syncing = true
      try
        tree.setRoot(null)
        items.clear()
        budget.reset()
        val item = makeItem(root)
        tree.setRoot(item)
        // Window root is always open, including a focused deep subtree. No old TreeItems survive.
        currentState = currentState.copy(expanded = currentState.expanded + root.path)
        item.offset = offset
        item.setExpanded(true)
        if offset > 0 then load(item, offset)
        tree.getSelectionModel.select(item)
      finally syncing = false
      status.setText(s"Tree window: ${pathLabel(root.path)} · children from $offset · depth ≤ $depth · at most $WindowLimit rows. Boundaries are explicit." +
        (if cancelled then " Previous subtree work cancelled by navigation." else ""))
      changed()

  private def backWindow(): Unit = history match
    case (root, offset, depth) :: tail =>
      history = tail
      openWindow(root, offset, depth, false)
    case _ => ()

  private def load(item: JsonItem, offset: Int): Unit =
    if !disposed && !item.loading && items.get(item.data.path).contains(item) then
      item.loading = true
      val ticket = windowTicket
      val available = math.min(499, budget.remaining.max(1))
      val depth = windowDepth
      async.run(children(item.data, offset, available, depth))(page => {
        if ticket == windowTicket && items.get(item.data.path).contains(item) then
          item.loading = false
          // Only the last child is a continuation. Replacing it frees its counted slot.
          if !item.getChildren.isEmpty && item.getChildren.get(item.getChildren.size - 1).isInstanceOf[MoreItem] then
            item.getChildren.remove(item.getChildren.size - 1)
            budget.release(1)
          var next = offset
          var created = 0
          val iterator = page.nodes.iterator
          val expandLater = mutable.ArrayBuffer.empty[JsonItem]
          var stopped = false
          while iterator.hasNext && !stopped do
            val n = iterator.next()
            val cost = itemCost(n)
            // Reserve one continuation slot, and cap all new TreeItems (including lazy rows) per FX slice.
            if cost + 1 > budget.remaining || created + cost + 1 > SliceLimit then stopped = true
            else
              val child = makeItem(n)
              item.getChildren.add(child)
              created += cost
              next += 1
              val expanded = explicitExpansion.getOrElse(n.path,
                currentState.expanded(n.path) || (n.isContainer && n.childCount < 2))
              if expanded && n.children.nonEmpty then expandLater += child
          item.offset = next
          if next < item.data.childCount then
            val depthBoundary = page.boundary.exists(_.contains("depth"))
            val limitBoundary = stopped || budget.remaining <= 1
            val message = page.boundary.getOrElse(
              if limitBoundary then s"$WindowLimit live-row boundary; continue from child $next in a new window"
              else s"Load next children ($next of ${item.data.childCount})")
            val resume = () => {
              if depthBoundary || limitBoundary then
                openWindow(item.data, next, if depthBoundary then depth + DepthLimit else depth, true)
              else load(item, next)
            }
            require(budget.reserve(1) == 1)
            item.getChildren.add(new MoreItem(message, resume))
          syncing = true
          try expandLater.foreach { child =>
            currentState = currentState.copy(expanded = currentState.expanded + child.data.path)
            child.setExpanded(true)
          }
          finally syncing = false
          changed()
      }, e => {
        if ticket == windowTicket then
          item.loading = false
          status.setText("Tree children unavailable: " + pathLabel(Fx.message(e)))
      })

  private def continueWork(): Unit =
    Option(tree.getSelectionModel.getSelectedItem) match
      case Some(more: MoreItem) => more.proceed()
      case _ => recursiveContinuation.foreach { (frames, expand, depth) =>
        val nextDepth = if frames.headOption.exists(_.node.depth > depth) then depth + DepthLimit else depth
        startRecursive(frames, expand, nextDepth)
      }

  private def recursive(root: PayloadNode, expand: Boolean): Unit =
    startRecursive(List(PayloadTreeFrame(root)), expand, root.depth + DepthLimit)

  private def cancelRecursive(notify: Boolean): Unit =
    val hadWork = recursiveActive || recursiveContinuation.nonEmpty
    expansionPublication.supersede()
    recursiveActive = false
    recursiveContinuation = None
    if notify && hadWork then status.setText("Subtree work cancelled by newer expansion input.")

  private def startRecursive(frames: List[PayloadTreeFrame], expand: Boolean, depth: Int): Unit =
    val ticket = expansionPublication.supersede()
    val window = windowTicket
    recursiveActive = true
    recursiveContinuation = None
    def slice(pending: List[PayloadTreeFrame], total: Int): Unit =
      async.run {
        // A single empty-state traversal collects exactly this slice's visited containers, including
        // already-collapsed small nodes. Publish only that delta into live state, never a saved set.
        val work = subtreeSlice(PayloadTreeState(), pending, true, depth, (WindowLimit - total).min(SliceLimit))
        work
      }(work => {
        def valid = !disposed && window == windowTicket && expansionPublication.isCurrent(ticket)
        if !valid then onExpansionSlice(0) // actual stale worker result settled, with no state/UI update
        else
          var selectedBefore: TreeItem[Entry] = null
          var startedBatch = false
          expansionPublication.publish(ticket, work.state.expanded.iterator, expand, () => valid)(
            (p, expanded) => {
              if !startedBatch then
                selectedBefore = tree.getSelectionModel.getSelectedItem
                startedBatch = true
              currentState = currentState.copy(expanded =
                if expanded then currentState.expanded + p else currentState.expanded - p)
              explicitExpansion(p) = expanded
              // At most one attached-item lookup/update for each visited touched path. Unchanged
              // containers need no setter and cannot enqueue redundant lazy child requests.
              items.get(p).filter(_.isExpanded != expanded).foreach { item =>
                syncing = true
                try item.setExpanded(expanded) finally syncing = false
              }
            },
            count => {
              if tree.getSelectionModel.getSelectedItem == null && selectedBefore != null then
                var visible = selectedBefore
                while visible != null && tree.getRow(visible) < 0 do visible = visible.getParent
                if visible != null then tree.getSelectionModel.select(visible)
              startedBatch = false
              selectedBefore = null
              onExpansionSlice(count)
              tree.refresh()
              changed()
            },
            () => {
              val count = total + work.visited
              if work.remaining.nonEmpty && !work.depthLimited && count < WindowLimit then slice(work.remaining, count)
              else
                recursiveActive = false
                recursiveContinuation = Option.when(work.remaining.nonEmpty)((work.remaining, expand, depth))
                status.setText(if work.remaining.nonEmpty then
                  s"Subtree ${if expand then "expansion" else "collapse"}: $count nodes processed; ${if work.depthLimited then s"depth $depth" else s"$WindowLimit-node"} boundary. Continue resumes exactly here."
                  else s"Subtree ${if expand then "expanded" else "collapsed"}: $count nodes processed. Tree rows remain lazy and windowed.")
                changed()
            })
      }, e => if expansionPublication.isCurrent(ticket) then {
        recursiveActive = false
        status.setText("Subtree work unavailable: " + pathLabel(Fx.message(e)))
      })
    slice(frames, 0)
