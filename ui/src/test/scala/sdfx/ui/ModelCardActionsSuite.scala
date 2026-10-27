package sdfx.ui

import javafx.beans.property.SimpleLongProperty
import sdfx.core.modelcards.{CardId, ModelCardRecord}
import sdfx.core.protocol.{EngineHello, FileEntry, ModelEntry}
import sdfx.engine.{CatalogSnapshot, EngineState}
import sdfx.ui.view.UiContext
import sdfx.ui.vm.*

import java.nio.file.Path
import scala.collection.mutable
import scala.concurrent.{Await, ExecutionContext, Future, Promise}
import scala.concurrent.duration.*

/** Headless contract tests: these exercise injected state, not a native MainWindow or engine load. */
class ModelCardActionsSuite extends munit.FunSuite:
  private given ExecutionContext = ExecutionContext.parasitic
  private def path(s: String): String = Path.of(s).toAbsolutePath.normalize.toString
  private val checkpoint = CardId("checkpoint", path("models/checkpoints/赤 model.safetensors"))
  private val checkpointEntry = ModelEntry("Literal display [x]", checkpoint.modelPath, "checkpoint")
  private val ready = EngineState.Ready(EngineHello(1, "test"))
  private def record(id: CardId, name: String) = ModelCardRecord(id, name, None, None, None)

  private final class GateFixture(windowsPaths: Boolean = false):
    var snapshot = CardGateSnapshot(ready, true, true, false, CatalogSnapshot(checkpoints = Seq(checkpointEntry)))
    val validations = mutable.Queue.empty[(CardId, Promise[Either[String, Unit]])]
    val fx = mutable.Queue.empty[() => Unit]
    val starts = mutable.ArrayBuffer.empty[String]
    val gate = new CheckpointCardGate(
      () => snapshot,
      id => { val p = Promise[Either[String, Unit]](); validations.enqueue((id, p)); p.future },
      run => fx.enqueue(run),
      name => { starts += name; 42L }, windowsPaths
    )
    def complete(result: Either[String, Unit]): Unit =
      validations.dequeue()._2.success(result)
      while fx.nonEmpty do fx.dequeue()()

  test("only exact Ready, idle queue, no load or extended style admits") {
    val f = new GateFixture
    assert(f.gate.state(checkpoint).allowed)
    val forbidden = Seq(
      f.snapshot.copy(engine = EngineState.Busy),
      f.snapshot.copy(engine = EngineState.Stopped),
      f.snapshot.copy(engine = EngineState.Starting),
      f.snapshot.copy(engine = EngineState.Crashed("failure")),
      f.snapshot.copy(queueIdle = false),
      f.snapshot.copy(loadsEmpty = false), // covers ordinary checkpoint and VAE loads
      f.snapshot.copy(extendedStyleApplying = true),
      f.snapshot.copy(catalog = CatalogSnapshot())
    )
    assert(EngineState.Busy.isUsable)
    forbidden.foreach { s =>
      f.snapshot = s
      val state = f.gate.state(checkpoint)
      assert(!state.allowed, s.toString)
      assert(state.reason.exists(_.nonEmpty))
    }
    assertEquals(f.starts.toSeq, Seq.empty)
    assertEquals(f.validations.size, 0)
  }

  test("validation reserves one token and rechecks state and catalog before dispatch") {
    val f = new GateFixture
    val pending = f.gate.admit(checkpoint)
    assertEquals(f.validations.head._1, checkpoint)
    assert(!f.gate.state(checkpoint).allowed)
    assert(Await.result(f.gate.admit(checkpoint), 1.second).isLeft)
    f.snapshot = f.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry.copy(path = path("other.safetensors")))))
    f.complete(Right(()))
    assert(Await.result(pending, 1.second).isLeft)
    assertEquals(f.starts.toSeq, Seq.empty)
    f.snapshot = f.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry)))
    assert(f.gate.state(checkpoint).allowed)
    val pending2 = f.gate.admit(checkpoint)
    f.snapshot = f.snapshot.copy(queueIdle = false)
    f.complete(Right(()))
    assert(Await.result(pending2, 1.second).isLeft)
    assertEquals(f.starts.toSeq, Seq.empty)
  }

  test("revision changes when the owner publishes state changes and pending transitions") {
    val f = new GateFixture
    val before = f.gate.revision.get
    f.snapshot = f.snapshot.copy(loadsEmpty = false)
    f.gate.stateChanged()
    assert(f.gate.revision.get > before)
    assert(!f.gate.state(checkpoint).allowed)
    f.snapshot = f.snapshot.copy(loadsEmpty = true)
    f.gate.stateChanged()
    val idleRevision = f.gate.revision.get
    val pending = f.gate.admit(checkpoint)
    assert(f.gate.revision.get > idleRevision)
    val reservedRevision = f.gate.revision.get
    f.complete(Left("missing"))
    assert(Await.result(pending, 1.second).isLeft)
    assert(f.gate.revision.get > reservedRevision)
  }

  test("ambiguous checkpoint names or paths cannot dispatch by name") {
    val f = new GateFixture
    val duplicateName = checkpointEntry.copy(path = path("different/model.safetensors"))
    val duplicatePath = checkpointEntry.copy(name = "different name")
    Seq(duplicateName, duplicatePath).foreach { duplicate =>
      f.snapshot = f.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry, duplicate)))
      assert(!f.gate.state(checkpoint).allowed)
      assert(Await.result(f.gate.admit(checkpoint), 1.second).isLeft)
      assertEquals(f.validations.size, 0)
      assertEquals(f.starts.toSeq, Seq.empty)
    }
  }

  test("validation failure and disposal release or invalidate reservation") {
    val f = new GateFixture
    val failed = f.gate.admit(checkpoint)
    f.complete(Left("file disappeared"))
    assert(Await.result(failed, 1.second).isLeft)
    assert(f.gate.state(checkpoint).allowed)
    val disposed = f.gate.admit(checkpoint)
    f.gate.dispose()
    f.complete(Right(()))
    assert(Await.result(disposed, 1.second).isLeft)
    assertEquals(f.starts.toSeq, Seq.empty)
  }

  test("exceptional validation releases its reservation") {
    val f = new GateFixture
    val pending = f.gate.admit(checkpoint)
    f.validations.dequeue()._2.failure(new IllegalStateException("validation failed"))
    while f.fx.nonEmpty do f.fx.dequeue()()
    assert(Await.result(pending, 1.second).isLeft)
    assert(f.gate.state(checkpoint).allowed)
    assertEquals(f.starts.toSeq, Seq.empty)
  }

  test("Right is dispatch admission, retains literal name and path, and does not mean loaded") {
    val f = new GateFixture
    val pending = f.gate.admit(checkpoint)
    assert(!pending.isCompleted)
    f.validations.dequeue()._2.success(Right(()))
    assert(!pending.isCompleted) // dispatch waits for the owner thread
    assertEquals(f.starts.toSeq, Seq.empty)
    f.fx.dequeue()()
    assertEquals(Await.result(pending, 1.second), Right(CheckpointAdmission(42L, checkpoint.modelPath)))
    assertEquals(f.starts.toSeq, Seq(checkpointEntry.name))
    // Load completion and failure feedback belong to MainWindow and are native acceptance gates.
  }

  private final class ActionsFixture(delayed: Boolean = false, windowsPaths: Boolean = false):
    val catalog = new CatalogVM
    val ctx = new UiContext(null, catalog, null)
    val revision = new SimpleLongProperty(0L)
    var selected = Vector.empty[CardId]
    val validations = mutable.Queue.empty[(CardId, Promise[Either[String, Unit]])]
    val fx = mutable.Queue.empty[() => Unit]
    val validate: CardId => Future[Either[String, Unit]] = id =>
      if delayed then
        val p = Promise[Either[String, Unit]]()
        validations.enqueue(id -> p)
        p.future
      else Future.successful(Right(()))
    val actions = new ModelCardActions(ctx, _ => CheckpointCardAdmission(true, None, revision.get),
      id => { selected :+= id; Future.successful(Right(CheckpointAdmission(7L, id.modelPath))) }, revision, validate,
      run => if delayed then fx.enqueue(run) else run(), windowsPaths)
    def setCatalog(s: CatalogSnapshot): Unit = catalog.snapshot.set(s)
    def complete(result: Either[String, Unit]): Unit =
      validations.dequeue()._2.success(result)
      while fx.nonEmpty do fx.dequeue()()
    def result(f: Future[Either[String, Unit]]): Either[String, Unit] = Await.result(f, 1.second)

  test("checkpoint selection delegates the unchanged identity and returns admission") {
    val f = new ActionsFixture
    val card = record(checkpoint, checkpointEntry.name)
    f.setCatalog(CatalogSnapshot(checkpoints = Seq(checkpointEntry)))
    assert(f.actions.state(card).canSelectCheckpoint)
    assertEquals(Await.result(f.actions.select(card), 1.second), Right(CheckpointAdmission(7L, checkpoint.modelPath)))
    assertEquals(f.selected, Vector(checkpoint))
  }

  test("captured positive and negative targets preserve each actual mode and insertion syntax") {
    val f = new ActionsFixture
    val lora = CardId("lora", path("models/lora/my detail.safetensors"))
    val embedding = CardId("embedding", path("models/embeddings/soft light.pt"))
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry("my detail.safetensors", lora.modelPath)),
      embeddings = Seq(FileEntry("soft light.pt", embedding.modelPath))))
    for mode <- Mode.values; negative <- Seq(false, true) do
      var text = "a cat"
      var caret = 5
      var calls = 0
      val target = PromptInsertionTarget(mode.key, negative, text, caret, () => (text, caret), token => {
        calls += 1
        val edit = PromptEdit.insertAt(text, caret, token)
        text = edit.text
        caret = edit.selEnd
      })
      f.actions.setTarget(Some(target))
      assert(f.actions.state(record(lora, "my detail.safetensors")).canInsert)
      assertEquals(f.result(f.actions.insert(record(lora, "my detail.safetensors"))), Right(()))
      assertEquals(text, "a cat, <lora:my detail:1.0>")
      assertEquals(calls, 1)
      f.actions.setTarget(Some(PromptInsertionTarget(mode.key, negative, text, caret, () => (text, caret), token => {
        calls += 1
        text = PromptEdit.insertAt(text, caret, token).text
      })))
      assertEquals(f.result(f.actions.insert(record(embedding, "soft light.pt"))), Right(()))
      assert(text.endsWith("soft light"))
      assertEquals(calls, 2)
  }

  test("stale text or caret and duplicate token paths never overwrite") {
    val f = new ActionsFixture
    val id = CardId("lora", path("a/duplicate.safetensors"))
    val other = path("b/duplicate.safetensors")
    var text = "draft"
    var caret = 5
    var calls = 0
    def target = PromptInsertionTarget("inpaint", true, text, caret, () => (text, caret), _ => calls += 1)
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry("duplicate.safetensors", id.modelPath))))
    f.actions.setTarget(Some(target))
    text = "draft changed"
    assert(!f.actions.state(record(id, "duplicate.safetensors")).canInsert)
    assert(f.result(f.actions.insert(record(id, "duplicate.safetensors"))).isLeft)
    f.actions.setTarget(Some(target))
    caret = 1
    assert(f.result(f.actions.insert(record(id, "duplicate.safetensors"))).isLeft)
    f.actions.setTarget(Some(target))
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry("duplicate.safetensors", id.modelPath), FileEntry("duplicate.safetensors", other))))
    assert(!f.actions.state(record(id, "duplicate.safetensors")).canInsert)
    assert(f.result(f.actions.insert(record(id, "duplicate.safetensors"))).isLeft)
    assertEquals(calls, 0)
    assertEquals(text, "draft changed")
  }

  test("embedding duplicates and records absent from current catalog are unavailable") {
    val f = new ActionsFixture
    val id = CardId("embedding", path("a/glow.pt"))
    val card = record(id, "glow.pt")
    var insertions = 0
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => insertions += 1)))
    f.setCatalog(CatalogSnapshot(embeddings = Seq(FileEntry("glow.pt", id.modelPath))))
    assert(f.actions.state(card).canInsert)
    f.setCatalog(CatalogSnapshot(embeddings = Seq(FileEntry("glow.pt", id.modelPath), FileEntry("glow.pt", path("b/glow.pt")))))
    assert(!f.actions.state(card).canInsert)
    assert(f.result(f.actions.insert(card)).isLeft)
    f.setCatalog(CatalogSnapshot(embeddings = Seq(FileEntry("glow.pt", path("b/glow.pt")))))
    assert(f.result(f.actions.insert(card)).isLeft)
    assertEquals(insertions, 0)
  }

  test("built-in non-action kinds and arbitrary generic kinds stay inspect-only") {
    val f = new ActionsFixture
    val inspectOnly = Seq("vae", "controlnet", "upscaler", "face_restore", "svd", "zero123", "photomaker", "hypernetwork", "custom mixed Case")
    inspectOnly.foreach { kind =>
      val card = record(CardId(kind, path(s"models/$kind/file.bin")), "file.bin")
      val state = f.actions.state(card)
      assert(!state.canSelectCheckpoint, kind)
      assert(!state.canInsert, kind)
      assert(f.result(f.actions.insert(card)).isLeft)
      assert(Await.result(f.actions.select(card), 1.second).isLeft)
    }
    assertEquals(f.selected, Vector.empty)
  }

  test("insertion waits for metadata and owner dispatch before success") {
    val f = new ActionsFixture(delayed = true)
    val id = CardId("lora", path("models/色.safetensors"))
    val card = record(id, "色.safetensors")
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry(card.name, id.modelPath))))
    var inserted = Vector.empty[String]
    f.actions.setTarget(Some(PromptInsertionTarget("img2img", true, "x", 1, () => ("x", 1), s => inserted :+= s)))
    val result = f.actions.insert(card)
    assertEquals(f.validations.head._1, id)
    assert(!result.isCompleted)
    assertEquals(inserted, Vector.empty)
    f.validations.dequeue()._2.success(Right(()))
    assert(!result.isCompleted)
    assertEquals(inserted, Vector.empty)
    f.fx.dequeue()()
    assertEquals(f.result(result), Right(()))
    assertEquals(inserted, Vector("<lora:色:1.0>"))
  }

  test("action uses the literal engine entry name even when it differs from the local filename") {
    val f = new ActionsFixture
    val id = CardId("lora", path("models/lora/internal-file.safetensors"))
    val card = record(id, "Display token.safetensors")
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry(card.name, id.modelPath))))
    var inserted = Vector.empty[String]
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), s => inserted :+= s)))
    assertEquals(f.result(f.actions.insert(card)), Right(()))
    assertEquals(inserted, Vector("<lora:Display token:1.0>"))
  }

  test("missing model and metadata failure never insert") {
    for kind <- Seq("lora", "embedding") do
      val f = new ActionsFixture(delayed = true)
      val id = CardId(kind, path(s"models/$kind/test.safetensors"))
      val card = record(id, "test.safetensors")
      val file = FileEntry(card.name, id.modelPath)
      f.setCatalog(if kind == "lora" then CatalogSnapshot(loras = Seq(file)) else CatalogSnapshot(embeddings = Seq(file)))
      var count = 0
      f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => count += 1)))
      val pending = f.actions.insert(card)
      f.complete(Left("model disappeared"))
      assert(f.result(pending).isLeft)
      assertEquals(count, 0)
  }

  test("exceptional insertion validation reports failure without writing") {
    val f = new ActionsFixture(delayed = true)
    val id = CardId("embedding", path("models/embeddings/token.pt"))
    val card = record(id, "token.pt")
    f.setCatalog(CatalogSnapshot(embeddings = Seq(FileEntry(card.name, id.modelPath))))
    var count = 0
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => count += 1)))
    val pending = f.actions.insert(card)
    f.validations.dequeue()._2.failure(new IllegalStateException("metadata failed"))
    while f.fx.nonEmpty do f.fx.dequeue()()
    assert(f.result(pending).isLeft)
    assertEquals(count, 0)
  }

  test("disposing actions invalidates pending insertion and late completion") {
    val f = new ActionsFixture(delayed = true)
    val id = CardId("lora", path("models/lora/token.safetensors"))
    val card = record(id, "token.safetensors")
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry(card.name, id.modelPath))))
    var count = 0
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => count += 1)))
    val pending = f.actions.insert(card)
    f.actions.dispose()
    f.complete(Right(()))
    assert(f.result(pending).isLeft)
    assertEquals(count, 0)
    assert(f.result(f.actions.insert(card)).isLeft)
  }

  test("insertion rechecks captured target, current text and caret, and catalog after validation") {
    for change <- Seq("text", "caret", "target", "removed", "renamed", "duplicate") do
      val f = new ActionsFixture(delayed = true)
      val id = CardId("lora", path("models/lora/focus.safetensors"))
      val card = record(id, "focus.safetensors")
      val entry = FileEntry(card.name, id.modelPath)
      f.setCatalog(CatalogSnapshot(loras = Seq(entry)))
      var text = "draft"
      var caret = 5
      var calls = 0
      def target = PromptInsertionTarget("inpaint", true, text, caret, () => (text, caret), _ => calls += 1)
      f.actions.setTarget(Some(target))
      val pending = f.actions.insert(card)
      change match
        case "text" => text = "new draft"
        case "caret" => caret = 1
        case "target" => f.actions.setTarget(Some(target.copy(modeKey = "txt2img", negative = false)))
        case "removed" => f.setCatalog(CatalogSnapshot())
        case "renamed" => f.setCatalog(CatalogSnapshot(loras = Seq(entry.copy(name = "new-name.safetensors"))))
        case "duplicate" => f.setCatalog(CatalogSnapshot(loras = Seq(entry, entry.copy(path = path("other/focus.safetensors")))))
      f.complete(Right(()))
      assert(f.result(pending).isLeft, change)
      assertEquals(calls, 0, change)
  }

  test("same path rename invalidates displayed checkpoint and insertion records") {
    val f = new ActionsFixture
    val old = record(checkpoint, checkpointEntry.name)
    f.setCatalog(CatalogSnapshot(checkpoints = Seq(checkpointEntry.copy(name = "new display"))))
    assert(!f.actions.state(old).canSelectCheckpoint)
    assert(Await.result(f.actions.select(old), 1.second).isLeft)
    assertEquals(f.selected, Vector.empty)
    val id = CardId("embedding", path("models/embeddings/token.pt"))
    val oldEmbedding = record(id, "old display")
    f.setCatalog(CatalogSnapshot(embeddings = Seq(FileEntry("new display", id.modelPath))))
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => fail("unexpected insertion"))))
    assert(!f.actions.state(oldEmbedding).canInsert)
    assert(f.result(f.actions.insert(oldEmbedding)).isLeft)
  }

  test("checkpoint same-path rename during metadata validation is not dispatched") {
    val f = new GateFixture
    val pending = f.gate.admit(checkpoint)
    f.snapshot = f.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry.copy(name = "renamed"))))
    f.complete(Right(()))
    assert(Await.result(pending, 1.second).isLeft)
    assertEquals(f.starts.toSeq, Seq.empty)
  }

  test("Windows path case alias matches a single record and duplicates are ambiguous") {
    val f = new ActionsFixture(windowsPaths = true)
    val id = CardId("lora", path("models/Lora/Detail.safetensors"))
    val alias = id.modelPath.toUpperCase(java.util.Locale.ROOT)
    val card = record(id, "Detail.safetensors")
    val entry = FileEntry(card.name, alias)
    f.setCatalog(CatalogSnapshot(loras = Seq(entry)))
    var inserted = Vector.empty[String]
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), s => inserted :+= s)))
    assert(f.actions.state(card).canInsert)
    assertEquals(f.result(f.actions.insert(card)), Right(()))
    assertEquals(inserted, Vector("<lora:Detail:1.0>"))
    f.setCatalog(CatalogSnapshot(loras = Seq(entry, entry.copy(path = id.modelPath))))
    assert(!f.actions.state(card).canInsert)
    assert(f.result(f.actions.insert(card)).isLeft)
    val gate = new GateFixture(windowsPaths = true)
    gate.snapshot = gate.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry.copy(path = checkpoint.modelPath.toUpperCase(java.util.Locale.ROOT)))))
    assert(gate.gate.state(checkpoint).allowed)
    gate.snapshot = gate.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry, checkpointEntry.copy(path = checkpoint.modelPath.toUpperCase(java.util.Locale.ROOT)))))
    assert(!gate.gate.state(checkpoint).allowed)
  }

  test("case-sensitive identity does not merge differently cased paths") {
    val f = new ActionsFixture(windowsPaths = false)
    val id = CardId("lora", path("models/lora/Detail.safetensors"))
    val card = record(id, "Detail.safetensors")
    f.setCatalog(CatalogSnapshot(loras = Seq(FileEntry(card.name, id.modelPath.toUpperCase(java.util.Locale.ROOT)))))
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => fail("unexpected insertion"))))
    assert(!f.actions.state(card).canInsert)
    assert(f.result(f.actions.insert(card)).isLeft)
  }

  test("Windows case-only catalog path change during validation keeps the same identity") {
    val f = new ActionsFixture(delayed = true, windowsPaths = true)
    val id = CardId("lora", path("models/lora/Detail.safetensors"))
    val card = record(id, "Detail.safetensors")
    val entry = FileEntry(card.name, id.modelPath)
    f.setCatalog(CatalogSnapshot(loras = Seq(entry)))
    var calls = 0
    f.actions.setTarget(Some(PromptInsertionTarget("txt2img", false, "", 0, () => ("", 0), _ => calls += 1)))
    val pending = f.actions.insert(card)
    f.setCatalog(CatalogSnapshot(loras = Seq(entry.copy(path = id.modelPath.toUpperCase(java.util.Locale.ROOT)))))
    f.complete(Right(()))
    assertEquals(f.result(pending), Right(()))
    assertEquals(calls, 1)
    val gate = new GateFixture(windowsPaths = true)
    val admitted = gate.gate.admit(checkpoint)
    gate.snapshot = gate.snapshot.copy(catalog = CatalogSnapshot(checkpoints = Seq(checkpointEntry.copy(path = checkpoint.modelPath.toUpperCase(java.util.Locale.ROOT)))))
    gate.complete(Right(()))
    assertEquals(Await.result(admitted, 1.second), Right(CheckpointAdmission(42L, checkpoint.modelPath)))
  }

  test("post-terminal observation waits for real idle, coalesces and stops on disposal") {
    var idle = false
    var changes = 0
    var scheduled = Vector.empty[(Long, () => Unit)]
    var next = 0L
    val cancelled = mutable.Set.empty[Long]
    val observer = new CardIdleObserver(() => idle, run => {
      next += 1
      val id = next
      scheduled :+= id -> run
      () => { cancelled += id; () }
    }, () => changes += 1)
    def pulse(): Unit =
      val (id, run) = scheduled.head
      scheduled = scheduled.tail
      if !cancelled(id) then run()
    observer.terminal()
    observer.terminal()
    assertEquals(scheduled.size, 1)
    pulse() // terminal was delivered while queue still has running job
    assertEquals(changes, 0)
    assertEquals(scheduled.size, 1)
    idle = true
    pulse()
    assertEquals(changes, 1)
    assertEquals(scheduled.size, 0)
    idle = false
    observer.terminal()
    val queuedBeforeDispose = scheduled.head._2
    observer.dispose()
    queuedBeforeDispose() // late delivery after cancellation must not reschedule or publish
    assertEquals(changes, 1)
    assertEquals(scheduled.size, 1)
    pulse()
    assertEquals(scheduled.size, 0)
    observer.terminal()
    assertEquals(scheduled.size, 0)
  }
