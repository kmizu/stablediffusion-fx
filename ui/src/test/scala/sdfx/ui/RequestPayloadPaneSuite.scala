package sdfx.ui

import java.nio.charset.StandardCharsets.UTF_8
import scala.collection.mutable.ArrayBuffer
import sdfx.core.payload.*
import sdfx.engine.RequestPayloadStore
import sdfx.ui.view.{PayloadDocument, RequestPayloadPresentation}

/** Pure presentation contract: this suite must never start JavaFX or touch the system clipboard.
  * Actual controls, FX scheduling and NativeAcceptanceSafety belong to root's manual native helper.
  */
class RequestPayloadPaneSuite extends munit.FunSuite:
  private def capture(job: String, sequence: Int = 1, mode: String = "txt2img", body: Option[String] = Some("{\"seed\":42}"),
      state: PayloadState = PayloadState.Prepared, context: Option[String] = Some("{\"origin\":\"ui\"}"),
      assets: Option[String] = Some("{\"assets\":[]}"), warnings: Vector[String] = Vector.empty): PayloadSnapshot =
    PayloadSnapshot(PayloadKey(job, sequence), if mode == "txt2img" then "txt2img" else "img2img", body,
      1000L + sequence, context, assets, state, warnings, Some(mode))

  private final class Harness(limits: RequestPayloadStore.Limits = RequestPayloadStore.Limits()):
    val store = new RequestPayloadStore(limits)
    val copied = ArrayBuffer.empty[String]
    var now = 10000L
    var writerFailure: Option[RuntimeException] = None
    val model = new RequestPayloadPresentation(store, (text: String) => {
      writerFailure.foreach(error => throw error)
      copied += text
      ()
    }, () => now)

  test("empty mode has no selected body and opening/refreshing never copies") {
    val h = new Harness
    h.model.openMode("txt2img")
    h.model.refresh()
    assertEquals(h.model.snapshots, Vector.empty)
    assertEquals(h.model.selectedSnapshot, None)
    assertEquals(h.model.selectedJson, None)
    assert(!h.model.copyEnabled)
    assert(h.model.statusText.toLowerCase.contains("no payload"))
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector.empty)
  }

  test("identical bodies in different jobs and sub-generations retain distinct history identities") {
    val h = new Harness
    h.store.put(capture("first"))
    h.store.put(capture("second"))
    h.store.put(capture("second", 2))
    h.model.openMode("txt2img")
    assertEquals(h.model.snapshots.map(_.key), Vector(PayloadKey("first", 1), PayloadKey("second", 1), PayloadKey("second", 2)))
    assertEquals(h.model.selectedSnapshot.map(_.key), Some(PayloadKey("second", 2)))
    h.model.select(PayloadKey("second", 1))
    assertEquals(h.model.selectedSnapshot.map(_.key), Some(PayloadKey("second", 1)))
    assertEquals(h.model.selectedJson, Some("{\"seed\":42}"))
    assertEquals(h.copied.toVector, Vector.empty)
  }

  test("history follows the latest capture until the user chooses an older sub-generation") {
    val h = new Harness
    h.store.put(capture("script", 1))
    h.model.openMode("txt2img")
    h.store.put(capture("script", 2))
    h.model.refresh()
    assertEquals(h.model.selectedSnapshot.map(_.key), Some(PayloadKey("script", 2)))
    h.model.select(PayloadKey("script", 1))
    h.store.put(capture("script", 3))
    h.model.refresh()
    assertEquals(h.model.selectedSnapshot.map(_.key), Some(PayloadKey("script", 1)))
    assertEquals(h.model.snapshots.size, 3)
  }

  test("opening a mode selects its own latest capture and Inpaint remains distinct from img2img") {
    val h = new Harness
    h.store.put(capture("text", mode = "txt2img"))
    h.store.put(capture("image", mode = "img2img"))
    h.store.put(capture("paint", mode = "inpaint"))
    h.model.openMode("img2img")
    assertEquals(h.model.snapshots.map(_.key.jobId), Vector("image"))
    h.model.openMode("inpaint")
    assertEquals(h.model.snapshots.map(_.key.jobId), Vector("paint"))
    assertEquals(h.model.selectedSnapshot.map(_.requestType), Some("img2img"))
    h.model.openMode("txt2img")
    assertEquals(h.model.selectedSnapshot.map(_.key.jobId), Some("text"))
  }

  test("a late old-mode completion cannot replace the active mode selection") {
    val h = new Harness
    h.store.put(capture("text"))
    h.store.put(capture("paint", mode = "inpaint"))
    h.model.openMode("inpaint")
    h.store.transition(PayloadKey("text", 1), PayloadState.Succeeded)
    h.model.refresh()
    assertEquals(h.model.selectedSnapshot.map(_.key.jobId), Some("paint"))
    assertEquals(h.model.snapshots.map(_.key.jobId), Vector("paint"))
    h.model.select(PayloadKey("text", 1))
    assertEquals(h.model.selectedSnapshot.map(_.key.jobId), Some("paint"))
  }

  test("refresh reads the current store rather than preserving an earlier view of lifecycle") {
    val h = new Harness
    h.store.put(capture("running"))
    h.model.openMode("txt2img")
    h.store.transition(PayloadKey("running", 1), PayloadState.Submitted)
    h.store.transition(PayloadKey("running", 1), PayloadState.Failed)
    h.model.refresh()
    assertEquals(h.model.selectedSnapshot.map(_.state), Some(PayloadState.Failed))
    assert(h.model.statusText.toLowerCase.contains("failed"))
    assertEquals(h.model.selectedJson, Some("{\"seed\":42}"))
  }

  test("a successful sub-generation remains succeeded when a later request fails or cancels") {
    for terminal <- Vector(PayloadState.Failed, PayloadState.Cancelled) do
      val h = new Harness
      h.store.put(capture("script", 1, state = PayloadState.Succeeded))
      h.store.put(capture("script", 2))
      h.store.finishJob("script", terminal)
      h.model.openMode("txt2img")
      assertEquals(h.model.selectedSnapshot.map(_.state), Some(terminal))
      h.model.select(PayloadKey("script", 1))
      assertEquals(h.model.selectedSnapshot.map(_.state), Some(PayloadState.Succeeded))
  }

  test("a cancelled undispatched job cannot borrow the previous job's exact body") {
    val h = new Harness
    h.store.put(capture("completed", state = PayloadState.Succeeded))
    h.store.put(capture("queued", body = None, state = PayloadState.Cancelled,
      warnings = Vector("Cancelled before dispatch; request body unavailable")))
    h.model.openMode("txt2img")
    assertEquals(h.model.selectedSnapshot.map(_.key.jobId), Some("queued"))
    assertEquals(h.model.selectedJson, None)
    assert(!h.model.copyEnabled)
    assert(h.model.statusText.toLowerCase.contains("cancelled"))
    assert(h.model.statusText.contains("Cancelled before dispatch"))
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector.empty)
  }

  test("an oversized unavailable body stays absent while retained context remains independently copyable") {
    val h = new Harness(RequestPayloadStore.Limits(maxBodyBytes = 4))
    h.store.put(capture("oversized"))
    h.model.openMode("txt2img")
    assertEquals(h.model.selectedJson, None)
    assert(h.model.statusText.toLowerCase.contains("unavailable"))
    assert(!h.model.copyEnabled)
    h.model.chooseDocument(PayloadDocument.Context)
    assertEquals(h.model.selectedJson, Some("{\"origin\":\"ui\"}"))
    assert(h.model.copyEnabled)
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector("{\"origin\":\"ui\"}"))
  }

  test("explicit body context and image copies preserve their exact independent strings") {
    val h = new Harness
    val body = "{\n  \"z\" : 1, \"prompt\" : \"<script>猫🦊</script>\", \"a\" : null\n}"
    val context = "{ \"submitted\" : {\"script\":\"literal\"} }"
    val assets = "{\"assets\":[{\"status\":\"omitted_source_init_images_policy\"}]}"
    h.store.put(capture("literal", body = Some(body), context = Some(context), assets = Some(assets)))
    h.model.openMode("txt2img")
    assertEquals(h.copied.toVector, Vector.empty)
    h.model.copySelected()
    h.model.chooseDocument(PayloadDocument.Context)
    h.model.copySelected()
    h.model.chooseDocument(PayloadDocument.ImageContents)
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector(body, context, assets))
  }

  test("missing image diagnostics cannot silently copy the body selected previously") {
    val h = new Harness
    h.store.put(capture("no-images", assets = None))
    h.model.openMode("txt2img")
    h.model.chooseDocument(PayloadDocument.ImageContents)
    assertEquals(h.model.selectedJson, None)
    assert(!h.model.copyEnabled)
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector.empty)
  }

  test("copy success is visible for exactly three seconds and is reset by another explicit click") {
    val h = new Harness
    h.store.put(capture("copy"))
    h.model.openMode("txt2img")
    assertEquals(h.model.copyFeedback, None)
    h.model.copySelected()
    assertEquals(h.model.copyFeedback, Some("Copied"))
    h.now += 2999
    assertEquals(h.model.copyFeedback, Some("Copied"))
    h.now += 1
    assertEquals(h.model.copyFeedback, None)
    h.model.copySelected()
    assertEquals(h.model.copyFeedback, Some("Copied"))
    assertEquals(h.copied.size, 2)
  }

  test("clipboard writer failures show a bounded failure message and do not claim success") {
    val h = new Harness
    h.store.put(capture("copy-error"))
    h.model.openMode("txt2img")
    h.writerFailure = Some(new IllegalStateException("external clipboard unavailable"))
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector.empty)
    assert(h.model.copyFeedback.exists(_.toLowerCase.contains("failed")))
    h.now += 3000
    assertEquals(h.model.copyFeedback, None)
  }

  test("preview is a strict 64KiB UTF-8 prefix while explicit Copy retains the entire long Unicode body") {
    val h = new Harness
    val body = "{\"prompt\":\"" + ("猫🦊<b>" * 9000) + "\"}"
    val preview = RequestPayloadPresentation.rawPreview(body)
    assert(preview.clipped)
    assertEquals(preview.totalBytes, body.getBytes(UTF_8).length.toLong)
    assert(preview.text.getBytes(UTF_8).length <= 65536)
    assert(body.startsWith(preview.text))
    assert(!Character.isHighSurrogate(preview.text.last))
    assert(preview.notice.contains("Copy"))
    assert(preview.notice.contains(preview.totalBytes.toString))
    h.store.put(capture("large", body = Some(body)))
    h.model.openMode("txt2img")
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector(body))
  }

  test("preview byte boundaries preserve complete Unicode codepoints rather than splitting a surrogate pair") {
    val body = "a" * 65535 + "🦊" + "tail"
    val preview = RequestPayloadPresentation.rawPreview(body)
    assertEquals(preview.text, "a" * 65535)
    assertEquals(preview.totalBytes, 65543L)
    assert(preview.clipped)
    val exact = RequestPayloadPresentation.rawPreview("猫🦊")
    assertEquals(exact.text, "猫🦊")
    assertEquals(exact.totalBytes, 7L)
    assert(!exact.clipped)
  }

  test("selection revision invalidates stale body work on document mode and lifecycle document changes") {
    val h = new Harness
    h.store.put(capture("text"))
    h.store.put(capture("image", mode = "img2img"))
    h.model.openMode("txt2img")
    val bodyWork = h.model.selectionRevision
    assert(h.model.acceptsRevision(bodyWork))
    h.model.chooseDocument(PayloadDocument.Context)
    assert(!h.model.acceptsRevision(bodyWork))
    val contextWork = h.model.selectionRevision
    h.store.transition(PayloadKey("text", 1), PayloadState.Succeeded, Some("{\"model\":\"reported\"}"))
    h.model.refresh()
    assert(!h.model.acceptsRevision(contextWork))
    val modeWork = h.model.selectionRevision
    h.model.openMode("img2img")
    assert(!h.model.acceptsRevision(modeWork))
  }

  test("evicted selected history becomes unavailable and cannot copy another job without selection") {
    val h = new Harness(RequestPayloadStore.Limits(maxCaptures = 2))
    h.store.put(capture("old"))
    h.store.put(capture("new"))
    h.model.openMode("txt2img")
    h.model.select(PayloadKey("old", 1))
    val work = h.model.selectionRevision
    h.store.put(capture("newest"))
    h.model.refresh()
    assertEquals(h.model.snapshots.map(_.key.jobId), Vector("new", "newest"))
    assertEquals(h.model.selectedSnapshot, None)
    assertEquals(h.model.selectedJson, None)
    assert(!h.model.copyEnabled)
    assert(!h.model.acceptsRevision(work))
    assert(h.model.statusText.toLowerCase.contains("unavailable"))
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector.empty)
  }

  test("copy rechecks retention when eviction occurs before the scheduled view refresh") {
    val h = new Harness(RequestPayloadStore.Limits(maxCaptures = 1))
    h.store.put(capture("old"))
    h.model.openMode("txt2img")
    h.store.put(capture("new"))
    h.model.copySelected()
    assertEquals(h.copied.toVector, Vector.empty)
  }

  test("disposing invalidates pending work drops presentation references and prevents late copying") {
    val h = new Harness
    h.store.put(capture("before-close"))
    h.model.openMode("txt2img")
    val work = h.model.selectionRevision
    h.model.dispose()
    h.store.put(capture("after-close"))
    h.model.refresh()
    h.model.openMode("txt2img")
    h.model.copySelected()
    assert(!h.model.acceptsRevision(work))
    assertEquals(h.model.snapshots, Vector.empty)
    assertEquals(h.model.selectedSnapshot, None)
    assertEquals(h.model.selectedJson, None)
    assertEquals(h.copied.toVector, Vector.empty)
    assertEquals(h.store.snapshots.size, 2)
  }

  test("All captures exposes unscoped direct and MCP entries without assigning a fabricated mode") {
    val h = new Harness
    h.store.put(capture("native"))
    h.store.put(capture("direct").copy(sourceMode = None, contextJson = Some("{\"origin\":\"direct\"}")))
    h.store.put(capture("mcp").copy(sourceMode = None, contextJson = Some("{\"origin\":\"mcp\"}")))
    h.model.openMode("txt2img")
    assertEquals(h.model.historyEntries.map(_.key.jobId), Vector("native"))
    h.model.openAllCaptures()
    assertEquals(h.model.historyEntries.map(_.key.jobId), Vector("native", "direct", "mcp"))
    assertEquals(h.model.historyEntries.last.sourceMode, None)
    assert(h.model.historyEntries.last.label.contains("source mode unavailable"))
    assertEquals(h.model.selectedSnapshot.map(_.key.jobId), Some("mcp"))
    assert(h.model.statusText.contains("source mode unavailable"))
  }

  test("history entries keep bounded labels and identity but no retained JSON documents") {
    val h = new Harness
    val huge = "x" * 100000
    h.store.put(capture("long-id", body = Some("{\"text\":\"" + huge + "\"}")))
    h.model.openAllCaptures()
    val entries = h.model.historyEntries
    assertEquals(entries.map(_.key), Vector(PayloadKey("long-id", 1)))
    assert(entries.head.label.length <= 160)
    assert(!entries.head.productIterator.exists {
      case s: String => s.contains(huge)
      case _ => false
    })
    assertEquals(h.model.selectedJson, Some("{\"text\":\"" + huge + "\"}"))
  }

  test("store rejection count remains visible even when no entry could be retained") {
    val h = new Harness(RequestPayloadStore.Limits(maxEntryBytes = 1))
    h.store.put(capture("unretainable"))
    h.model.openAllCaptures()
    assertEquals(h.store.rejectedCaptures, 1L)
    assert(h.model.rejectedCaptureText.contains("1"))
    assert(h.model.statusText.toLowerCase.contains("no payload"))
  }

  test("a backward wall clock does not leave copy feedback visible indefinitely") {
    val h = new Harness
    h.store.put(capture("copy"))
    h.model.openMode("txt2img")
    h.model.copySelected()
    assertEquals(h.model.copyFeedback, Some("Copied"))
    h.now -= 1
    assertEquals(h.model.copyFeedback, None)
    h.now += 1
    assertEquals(h.model.copyFeedback, None)
  }

  test("status keeps a bounded display when retained identity fields are unusually long") {
    val h = new Harness
    val longId = "j" * 100000
    val longType = "t" * 100000
    h.store.put(capture(longId).copy(requestType = longType))
    h.model.openAllCaptures()
    assertEquals(h.model.selectedSnapshot.map(_.key.jobId), Some(longId))
    assert(h.model.statusText.length < 1000)
    assert(h.model.historyEntries.head.label.length <= 160)
  }

  test("many retained warnings keep status short while every diagnostic remains inspectable") {
    val h = new Harness
    val warnings = Vector.tabulate(32)(i => f"warning-$i%02d " + ("x" * 990))
    h.store.put(capture("warning-heavy", warnings = warnings))
    h.model.openMode("txt2img")
    assert(h.model.statusText.length < 1000)
    assert(h.model.statusText.contains("32 warnings"))
    assertEquals(h.model.warningDetails, warnings)
    assertEquals(h.model.selectedJson, Some("{\"seed\":42}"))
  }
