package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.event.Event
import javafx.scene.Scene
import javafx.scene.control.{Button, Label, TreeCell, TreeItem, TreeView}
import javafx.scene.input.{MouseButton, MouseEvent}
import javafx.stage.Stage
import sdfx.core.payload.{PayloadNode, RequestPayload}
import sdfx.ui.view.{PayloadJsonTree, Theme}
import sdfx.ui.view.PayloadJsonTree.{Entry, Modifiers}

import java.nio.file.{Files, Path, Paths}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import javax.imageio.ImageIO
import scala.collection.mutable.ArrayDeque
import scala.jdk.CollectionConverters.*

/** ROOT-ONLY manual native check, intentionally not a munit suite. No AppServices, engine, Python,
  * model, system clipboard, settings, or user files are touched. Args: output-directory.
  */
object RealPayloadJsonTreeCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String)(test: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while !test && System.nanoTime() < end do Thread.sleep(20)
    require(test, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val started = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); started.countDown() })
    require(started.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var stage: Stage = null
    var view: PayloadJsonTree = null
    var primary: Throwable = null
    var checks = Vector.empty[String]
    var maxLive = 0
    var afterDispose = 0
    var expansionSlices = Vector.empty[Int]
    def checked(text: String): Unit = checks :+= text
    def tree: TreeView[Entry] = view.node.lookup("#payload-json-tree").asInstanceOf[TreeView[Entry]]
    def button(id: String): Button = view.node.lookup("#" + id).asInstanceOf[Button]
    def status: String = view.node.lookup("#payload-tree-status").asInstanceOf[Label].getText
    def selectionDiagnostic: String =
      val model = tree.getSelectionModel
      s"selectedIndex=${model.getSelectedIndex}, selectedItem=${Option(model.getSelectedItem).map(_.getValue.node.map(_.path))}, expandDisabled=${button("payload-tree-expand").isDisabled}, expanded=${view.state.expanded}, status=$status"
    def all(): Vector[TreeItem[Entry]] =
      val queue = ArrayDeque.empty[TreeItem[Entry]]
      Option(tree.getRoot).foreach(queue.append(_))
      val result = Vector.newBuilder[TreeItem[Entry]]
      var count = 0
      while queue.nonEmpty do
        val item = queue.removeHead()
        result += item
        item.getChildren.asScala.foreach(queue.append(_))
        count += 1
        require(count <= 10000, s"Attached TreeItems exceeded 10000: $count")
      maxLive = math.max(maxLive, count)
      result.result()
    def item(path: String): TreeItem[Entry] =
      all().find(_.getValue.node.exists(_.path == path)).getOrElse(throw new IllegalStateException("Missing path " + path))
    def select(path: String): Unit = fx { tree.getSelectionModel.select(item(path)) }
    def press(path: String, modifiers: Modifiers = Modifiers()): Unit =
      fx {
        val selected = item(path)
        tree.scrollTo(tree.getRow(selected))
        view.node.applyCss()
        view.node.layout()
      }
      def cell = tree.lookupAll(".tree-cell").asScala.collect {
        case c: TreeCell[?] if c.getTreeItem != null &&
            c.getTreeItem.getValue.asInstanceOf[Entry].node.exists(_.path == path) => c
      }.headOption
      eventually("visible native cell " + path) { fx { cell.nonEmpty } }
      fx {
        val event = new MouseEvent(MouseEvent.MOUSE_PRESSED, 30, 12, 30, 12, MouseButton.PRIMARY, 1,
          modifiers.shift, modifiers.control, modifiers.alt, modifiers.meta,
          true, false, false, false, false, true, null)
        Event.fireEvent(cell.get, event)
      }
    def screenshot(file: Path): Unit = fx {
      view.node.applyCss()
      view.node.layout()
      require(ImageIO.write(SwingFXUtils.fromFXImage(stage.getScene.snapshot(null), null), "png", file.toFile))
    }
    try
      fx {
        view = new PayloadJsonTree()
        view.onExpansionSlice = count => {
          require(count <= 1000, s"Recursive FX publication updated $count paths in one slice")
          expansionSlices :+= count
        }
        stage = new Stage()
        val scene = new Scene(view.node, 1280, 800)
        Theme.apply(scene, "dark")
        stage.setScene(scene)
        stage.setTitle("Payload JSON tree — renderer acceptance")
        stage.show()
        require(tree.getRoot == null)
        require(status.contains("No JSON"))
      }
      checked("empty viewer uses native controls and has no engine or clipboard dependency")
      val frozen = ujson.write(ujson.Obj(
        "a" -> ujson.Obj("one" -> ujson.Obj("leaf" -> 1), "two" -> ujson.Arr(true, false)),
        "hostile <script>猫.\"/[]🧪" -> "<b>日本語</b> & café 🧪",
        "long" -> ("猫🧪" * 5000), "null" -> ujson.Null))
      val fixture = RequestPayload.tree(frozen)
      val branch = fixture.children.head
      val nested = branch.children.head
      fx { view.showJson(frozen) }
      eventually("parsed literal tree root") { fx { tree.getRoot != null && tree.getRoot.getChildren.size > 1 } }
      fx {
        require(tree.getRoot.isExpanded)
        require(!item(branch.path).isExpanded)
        require(all().size < 20, "Collapsed branches were eagerly materialized")
        require(tree.getRoot.getChildren.asScala.flatMap(_.getValue.node.map(_.label)).toVector == fixture.children.map(_.label))
      }
      press(branch.path)
      eventually("ordinary toggle loads lazy children") { fx { item(branch.path).isExpanded && all().exists(_.getValue.node.exists(_.path == nested.path)) } }
      fx { require(item(nested.path).isExpanded, "Small container did not initially expand") }
      press(branch.path, Modifiers(alt = true, shift = true))
      fx { require(view.state.marked(branch.path)); require(view.state.selectedPath.isEmpty) }
      press(branch.path, Modifiers(shift = true))
      fx { require(view.state.selectedPath.contains(branch.path)) }
      press(branch.path, Modifiers(meta = true))
      eventually("Meta recursive collapse") { fx { status.contains("Subtree collapsed") } }
      fx { require(!view.state.expanded(branch.path)); require(!view.state.expanded(nested.path)) }
      press(branch.path, Modifiers(control = true))
      eventually("Ctrl recursive expansion") { fx { status.contains("Subtree expanded") } }
      fx { require(view.state.expanded(branch.path)); require(view.state.expanded(nested.path)) }
      checked("native ordinary / Alt / Shift / Ctrl / Meta handlers and lazy small-node initial policy")
      select(branch.path)
      fx {
        println("TREE before accessible collapse: " + selectionDiagnostic)
        button("payload-tree-collapse").fire()
        println("TREE after collapse fire: " + selectionDiagnostic)
      }
      eventually("accessible collapse") { fx { status.contains("Subtree collapsed") } }
      fx {
        println("TREE before accessible expand: " + selectionDiagnostic)
        require(Option(tree.getSelectionModel.getSelectedItem).exists(_.getValue.node.exists(_.path == branch.path)),
          "Recursive collapse discarded the selected row and disabled accessible actions")
        button("payload-tree-expand").fire()
        println("TREE after expand fire: " + selectionDiagnostic)
      }
      eventually("accessible expansion") { fx { status.contains("Subtree expanded") } }
      fx {
        button("payload-tree-mark").fire()
        require(!view.state.marked(branch.path))
        button("payload-tree-show-path").fire()
        require(view.state.selectedPath.contains(branch.path))
        button("payload-tree-toggle").fire()
        require(!item(branch.path).isExpanded)
      }
      checked("keyboard-accessible Toggle / Expand subtree / Collapse subtree / Mark / Show path buttons")
      val hostile = fixture.children.find(_.label.startsWith("hostile")).get
      press(hostile.path, Modifiers(shift = true))
      fx {
        require(view.state.selectedPath.contains(hostile.path), "Escaped path was altered")
        require(view.node.lookup("#payload-tree-path").asInstanceOf[Label].getText.contains("[\"hostile"))
      }
      screenshot(output.resolve("payload-tree-literal-1280x800.png"))
      checked("literal HTML-looking Japanese/emoji text, escaped path and bounded long scalar cell")

      val siblings = RequestPayload.tree("""{"A":{"one":{"leaf":1},"two":[true,false]},"B":{"leaf":true}}""")
      val a = siblings.children.head
      val b = siblings.children.last
      fx { view.show(siblings) }
      eventually("sibling B initially expanded") { fx { tree.getRoot != null && all().exists(i => i.getValue.node.exists(_.path == b.path) && i.isExpanded) } }
      fx {
        expansionSlices = Vector.empty
        tree.getSelectionModel.select(item(b.path))
        button("payload-tree-mark").fire()
        button("payload-tree-show-path").fire()
        tree.getSelectionModel.select(item(a.path))
        button("payload-tree-expand").fire()
        // This is deliberately one FX block: no worker publication can interleave before B's input.
        tree.getSelectionModel.select(item(b.path))
        button("payload-tree-toggle").fire()
        require(!item(b.path).isExpanded)
      }
      // Zero updates is reported when the cancelled worker result actually reaches FX. Waiting for
      // that event proves B after A settles; two arbitrary FX barriers would not prove worker order.
      eventually("cancelled A worker actually settles") { fx { status.contains("cancelled") && expansionSlices.contains(0) } }
      fx {
        require(!item(b.path).isExpanded && !view.state.expanded(b.path), "Pending A recursion reverted newer B toggle")
        require(view.state.marked(b.path) && view.state.selectedPath.contains(b.path))
      }
      checked("same-FX A-expand then newer B-toggle cancels old recursive publication and preserves mark/path")

      // Large tree presentation is independent from the complete immutable model. Count every attached
      // TreeItem, including markers, after each actual Continue button. No hidden 10000-row accumulation.
      val wide = RequestPayload.tree((0 until 12000).mkString("[", ",", "]"))
      fx { view.show(wide) }
      eventually("wide first page") { fx { tree.getRoot != null && tree.getRoot.getChildren.size > 100 } }
      select("$[0]")
      fx { button("payload-tree-mark").fire(); button("payload-tree-show-path").fire() }
      var pages = 0
      var replaced = false
      var maxIndex = 0
      while maxIndex < 11500 && pages < 40 do
        val before = fx {
          val root = tree.getRoot
          val rows = all()
          maxIndex = rows.flatMap(_.getValue.node.filter(_.depth == 1).map(_.label.toInt)).maxOption.getOrElse(0)
          val more = root.getChildren.asScala.last
          require(more.getValue.node.isEmpty, "Wide fixture unexpectedly has no continuation")
          tree.getSelectionModel.select(more)
          val before = (root, root.getChildren.size, more.getValue.notice)
          button("payload-tree-continue").fire()
          before
        }
        eventually("wide continuation " + pages) { fx {
          val root = tree.getRoot
          val last = root.getChildren.asScala.lastOption
          val changed = (root ne before._1) || root.getChildren.size != before._2 || last.exists(_.getValue.notice != before._3)
          if root ne before._1 then replaced = true
          all()
          changed && last.exists(_.getValue.notice != "Expand to load children")
        } }
        pages += 1
      require(maxIndex >= 11500 && replaced, "Continuation never crossed the 10000-row replacement boundary")
      fx {
        require(view.state.marked("$[0]"))
        require(view.state.selectedPath.contains("$[0]"))
        require(!button("payload-tree-back").isDisabled)
        button("payload-tree-back").fire()
        button("payload-tree-root").fire()
      }
      eventually("Root restores first wide page") { fx { tree.getRoot.getChildren.asScala.exists(_.getValue.node.exists(_.path == "$[0]")) } }
      fx { require(view.state.marked("$[0]")); require(view.state.selectedPath.contains("$[0]")); all() }
      checked(s"12000-leaf tree: $pages native continuations, replacement windows/Root/Back, preserved mark/path; live rows <=10000")

      // Materialize a genuinely large attached window before both a one-node and a wide recursive
      // action. A Root reset to a 499-child page would miss the reviewed whole-window FX scan.
      val mixed = RequestPayload.tree(ujson.write(ujson.Obj("A_small" -> ujson.Obj(),
        "B_bulk" -> ujson.Arr.from(0 until 12000))))
      val small = mixed.children.head
      val bulk = mixed.children.last
      val firstBulkPath = bulk.children.head.path
      fx { view.show(mixed) }
      eventually("mixed root children") { fx { tree.getRoot != null && all().exists(_.getValue.node.exists(_.path == bulk.path)) } }
      select(bulk.path)
      fx { button("payload-tree-toggle").fire() }
      eventually("mixed bulk first page") { fx { item(bulk.path).getChildren.size > 100 } }
      var bulkPages = 0
      while fx { all().size < 9000 } && bulkPages < 20 do
        val before = fx {
          val before = all().size
          val more = item(bulk.path).getChildren.asScala.last
          require(more.getValue.node.isEmpty)
          tree.getSelectionModel.select(more)
          button("payload-tree-continue").fire()
          before
        }
        eventually("large mixed window page " + bulkPages) { fx { all().size > before } }
        bulkPages += 1
      fx {
        require(all().size >= 9000)
        tree.getSelectionModel.select(item(firstBulkPath))
        button("payload-tree-mark").fire()
        button("payload-tree-show-path").fire()
        expansionSlices = Vector.empty
        tree.getSelectionModel.select(item(small.path))
        button("payload-tree-expand").fire()
      }
      eventually("one-node recursive action on9000-row window") { fx { status.contains("Subtree expanded: 1 nodes processed") } }
      fx {
        require(expansionSlices == Vector(1), "Small subtree scanned unrelated attached rows")
        require(all().size >= 9000)
      }
      checked("one-node recursive action on>=9000 attached rows updates exactly one path")
      fx {
        expansionSlices = Vector.empty
        tree.getSelectionModel.select(item(bulk.path))
        button("payload-tree-expand").fire()
      }
      eventually("native 10000-operation recursive boundary") { fx { status.contains("10000-node") } }
      fx {
        require(expansionSlices.nonEmpty && expansionSlices.forall(_ <= 1000))
        require(status.contains("10000 nodes processed"))
        require(view.state.marked(firstBulkPath) && view.state.selectedPath.contains(firstBulkPath))
        all()
        tree.getSelectionModel.select(item(bulk.path))
        button("payload-tree-continue").fire()
      }
      eventually("native recursive Continue finishes exact remaining2001") { fx { status.contains("Subtree expanded: 2001 nodes processed") } }
      fx {
        require(expansionSlices.forall(_ <= 1000))
        require(view.state.marked(firstBulkPath) && view.state.selectedPath.contains(firstBulkPath))
        all()
      }
      checked("native 12001-node recursive operation:10000 boundary+exact2001 Continue, every FX publication <=1000, marks/path retained")

      val deep = RequestPayload.tree(("[" * 140) + "null" + ("]" * 140))
      fx { view.show(deep) }
      eventually("visible depth128 boundary") { fx { all().exists(_.getValue.notice.contains("depth limit 128")) } }
      fx {
        val boundary = all().find(_.getValue.notice.contains("depth limit 128")).get
        tree.getSelectionModel.select(boundary)
        button("payload-tree-continue").fire()
      }
      eventually("deep subtree continuation reaches scalar") { fx { all().exists(_.getValue.node.exists(_.depth == 140)) } }
      fx { require(tree.getRoot.getValue.node.exists(_.depth == 128)); require(!button("payload-tree-back").isDisabled); all() }
      checked("depth140 tree stops visibly at128 and resumes in a focused bounded subtree")

      fx {
        for i <- 0 until 50 do view.showJson(s"{\"latest\":$i}")
      }
      eventually("latest selection publication") { fx { tree.getRoot != null && tree.getRoot.getChildren.asScala.exists(_.getValue.node.exists(_.valueJson.contains("49"))) } }
      fx {
        view.onStateChanged = _ => afterDispose += 1
        view.showJson("[1,2,3]")
        view.dispose()
        afterDispose = 0
      }
      // Pump two FX barriers so queued publications execute their disposed guards.
      fx { () }; fx { require(tree.getRoot == null); require(afterDispose == 0) }
      checked("latest-selection burst and dispose suppress late native state publication")
      // Restore a second independent widget for the final screenshot; disposed widgets remain inert.
      fx {
        view = new PayloadJsonTree()
        stage.getScene.setRoot(view.node)
        view.show(fixture)
      }
      eventually("final screenshot root") { fx { tree.getRoot.getChildren.size > 1 } }
      screenshot(output.resolve("payload-tree-1280x800.png"))
      val report = ujson.Obj("success" -> true, "checks" -> ujson.Arr.from(checks),
        "max_recursive_fx_updates" -> expansionSlices.maxOption.getOrElse(0),
        "max_live_tree_items" -> maxLive, "tree_item_limit" -> 10000, "viewport_width" -> 1280,
        "viewport_height" -> 800, "engine_started" -> false, "clipboard_accessed" -> false,
        "screenshot" -> output.resolve("payload-tree-1280x800.png").toString)
      Files.writeString(output.resolve("payload-tree-report.json"), ujson.write(report, indent = 2))
      println("Real payload JSON tree check passed: " + output.resolve("payload-tree-report.json"))
    catch
      case e: Throwable =>
        primary = e
        if stage != null then
          try screenshot(output.resolve("payload-tree-failure.png")) catch case _: Throwable => ()
        throw e
    finally
      NativeAcceptanceSafety.cleanup(primary)(
        () => { if view != null then fx { view.dispose() } },
        () => { if stage != null then fx { stage.close() } },
        () => Platform.exit())
