package sdfx.ui

import java.nio.file.{Files, Path}
import javafx.scene.image.WritableImage
import scala.collection.mutable
import sdfx.core.modelcards.*
import sdfx.engine.CardPreviewSession
import sdfx.ui.view.*

/** Exercises the very state controller delegated to by ModelPreviewCard, never a fake copy.
  * Visual snapshots are renderer inputs, not evidence of actual GaussianBlur/native focus behavior.
  */
class ModelPreviewCardStateSuite extends munit.FunSuite:
  import PreviewLoaderFixtures.*

  private final class Pending(val ref: PreviewReference, val deliver: PreviewImageResult => Unit) extends PreviewTicket:
    var releases = 0
    def cancel(): Unit = { releases += 1 }
    def finish(): WritableImage =
      val image = new WritableImage(2, 1)
      image.getPixelWriter.setArgb(0, 0, 0xffa02060)
      deliver(PreviewImageResult.Ready(image))
      image

  private final class Harness(initial: String = "Blur", shared: Option[CardPreviewSession] = None):
    val dir = Files.createTempDirectory("sdfx-preview-state-")
    val session = shared.getOrElse(new CardPreviewSession(initial))
    val pending = mutable.ArrayBuffer.empty[Pending]
    val rendered = mutable.ArrayBuffer.empty[PreviewCardVisual]
    val state = new ModelPreviewCardState(session,
      (ref, edge, complete) =>
        assertEquals(edge, 112)
        val work = new Pending(ref, complete)
        pending += work
        work,
      visual => { rendered += visual; () })
    def record(name: String = "NSFW-preview.png", kind: String = "checkpoint"): ModelCardRecord =
      val path = dir.resolve(name)
      writeImage(path)
      val ref = new ModelPreviewResolver(LocalPreviewFiles.scoped(new LocalPlatform))
        .validateExplicit(path).fold(fail(_), identity)
      ModelCardRecord(CardId(kind, dir.resolve(s"$kind-model.safetensors").toString), "Readable model identity",
        None, Some(ref), None)
    def last: PreviewCardVisual = rendered.last
    def close(): Unit =
      state.dispose()
      val walk = Files.walk(dir)
      try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(path => Files.deleteIfExists(path))
      finally walk.close()

  private def withHarness(initial: String = "Blur")(body: Harness => Unit): Unit =
    val h = new Harness(initial)
    try body(h)
    finally h.close()

  test("hide_does_not_request_and_hover_focus_or_hold_never_reveal") {
    withHarness("Hide") { h =>
      val record = h.record()
      h.state.bind(record, 1L)
      h.state.setHover(true)
      h.state.setRevealFocused(true)
      h.state.revealKey("SPACE", true)
      h.state.revealKey("ENTER", true)
      assertEquals(h.pending.size, 0)
      assertEquals(h.last.presentation, PreviewPresentation.Hidden)
      assertEquals(h.last.image, None)
      assert(!h.last.previewVisible && !h.last.previewManaged)
      assert(!h.last.revealControlVisible)
      assertEquals(h.last.record.map(_.id), Some(record.id), "Hide must retain readable identity/actions")
    }
  }

  test("policy_before_pixels_initial_blur_applies_20_without_an_unblurred_attachment") {
    withHarness() { h =>
      h.state.bind(h.record(), 3L)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      assertEquals(h.last.targetBlurRadius, 20.0)
      assertEquals(h.last.image, None)
      val image = h.pending.head.finish()
      val attached = h.rendered.filter(_.image.nonEmpty)
      assert(attached.nonEmpty)
      assert(attached.forall(_.presentation == PreviewPresentation.Blurred))
      assert(attached.forall(_.targetBlurRadius == 20.0))
      assert(attached.forall(_.transition.isEmpty), "New pixels must start concealed, not animate from zero")
      assert(h.last.image.exists(_ eq image))
    }
  }

  test("hover_reveal_and_leave_request_150ms_exact_spline_in_both_directions") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L); h.pending.head.finish()
      h.state.setHover(true)
      assertEquals(h.last.presentation, PreviewPresentation.Normal)
      assertEquals(h.last.targetBlurRadius, 0.0)
      val reveal = h.last.transition.get
      assertEquals(reveal.durationMillis, 150.0)
      assertEquals(reveal.spline, (0.79, 0.14, 0.15, 0.86))
      h.state.setHover(false)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      assertEquals(h.last.targetBlurRadius, 20.0)
      assertEquals(h.last.transition.map(_.durationMillis), Some(150.0))
      assertEquals(h.last.transition.map(_.spline), Some((0.79, 0.14, 0.15, 0.86)))
    }
  }

  test("keyboard_focus_alone_and_unrelated_keys_never_reveal_intentional_hold_does") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L); h.pending.head.finish()
      h.state.setRevealFocused(true)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      h.state.revealKey("A", true)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      for key <- Vector("SPACE", "ENTER") do
        h.state.revealKey(key, true)
        assertEquals(h.last.presentation, PreviewPresentation.Normal)
        h.state.revealKey(key, false)
        assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      h.state.setRevealFocused(false)
      h.state.revealKey("SPACE", true)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
    }
  }

  test("reveal_reset_on_focus_loss_window_deactivation_and_escape_requires_a_fresh_press") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L); h.pending.head.finish()
      val dismissals: Vector[() => Unit] = Vector(
        () => h.state.setRevealFocused(false),
        () => h.state.setWindowActive(false),
        () => h.state.escape())
      dismissals.foreach { dismiss =>
        h.state.setWindowActive(true)
        h.state.setRevealFocused(true)
        h.state.revealKey("SPACE", true)
        assertEquals(h.last.presentation, PreviewPresentation.Normal)
        dismiss()
        assertEquals(h.last.presentation, PreviewPresentation.Blurred)
        h.state.setWindowActive(true)
        h.state.setRevealFocused(true)
        assertEquals(h.last.presentation, PreviewPresentation.Blurred, "Focus restoration must not resume a stale hold")
        h.state.revealKey("SPACE", false)
      }
    }
  }

  for key <- Vector("SPACE", "ENTER") do
    test(s"recycling_requires_release_of_old_${key.toLowerCase}_hold_before_new_record_reveal") {
      withHarness() { h =>
        h.state.bind(h.record("NSFW-a.png"), 1L)
        h.pending.last.finish()
        h.state.setRevealFocused(true)
        h.state.revealKey(key, true)
        assertEquals(h.last.presentation, PreviewPresentation.Normal)
        h.state.clear()
        h.state.bind(h.record("NSFW-b.png", kind = "lora"), 2L)
        h.pending.last.finish()
        assertEquals(h.last.presentation, PreviewPresentation.Blurred)
        // Native pressed routing refreshes focus for the recycled, still-focused control.
        h.state.setRevealFocused(true)
        h.state.revealKey(key, true)
        assertEquals(h.last.presentation, PreviewPresentation.Blurred,
          s"$key auto-repeat from the old record must not reveal the new record")
        h.state.revealKey(key, false)
        assertEquals(h.last.presentation, PreviewPresentation.Blurred)
        h.state.revealKey(key, true)
        assertEquals(h.last.presentation, PreviewPresentation.Normal,
          s"A fresh deliberate $key press after release must reveal the new record")
        h.state.revealKey(key, false)
        assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      }
    }

  test("hide_from_revealed_releases_immediately_with_no_fade_or_late_original_flash") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L)
      h.pending.head.finish()
      h.state.setHover(true)
      val old = h.pending.head
      h.session.setTemporary(CardPreviewMode.Hide)
      val hideStart = h.rendered.size - 1
      assertEquals(old.releases, 1)
      assertEquals(h.last.image, None)
      assertEquals(h.last.transition, None)
      assert(!h.last.previewVisible && !h.last.previewManaged)
      old.finish() // A hostile/late callback after cancellation must still be ignored.
      assert(h.rendered.drop(hideStart).forall(v => v.image.isEmpty && v.presentation == PreviewPresentation.Hidden))
      h.session.setTemporary(CardPreviewMode.Blur)
      assertEquals(h.pending.size, 2)
      h.pending.last.finish()
      assertEquals(h.last.presentation, PreviewPresentation.Blurred, "Returning from Hide resets old hover/hold")
      assertEquals(h.last.targetBlurRadius, 20.0)
    }
  }

  test("window_deactivation_and_escape_end_hover_reveal_without_resuming_on_activation") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L); h.pending.head.finish()
      h.state.setHover(true)
      h.state.setWindowActive(false)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      h.state.setWindowActive(true)
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      h.state.setHover(false)
      h.state.setHover(true)
      assertEquals(h.last.presentation, PreviewPresentation.Normal)
      h.state.escape()
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
    }
  }

  test("unavailable_preview_keeps_card_identity_and_reason_without_pixels_or_new_request_loop") {
    withHarness() { h =>
      val record = h.record()
      h.state.bind(record, 1L)
      h.pending.head.deliver(PreviewImageResult.Unavailable("Preview workers busy"))
      assertEquals(h.last.record.map(_.id), Some(record.id))
      assertEquals(h.last.status, Some("Preview workers busy"))
      assertEquals(h.last.image, None)
      assertEquals(h.pending.size, 1)
    }
  }

  test("capacity_release_retry_only_restarts_visible_admission_refusal_and_never_hidden_or_dropped_results") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L)
      h.pending.head.deliver(PreviewImageResult.Unavailable(PreviewImageLoader.visibleCapacityReason))
      assertEquals(h.pending.size, 1)
      h.state.retryVisibleCapacity()
      assertEquals(h.pending.size, 2)
      assertEquals(h.pending.head.releases, 1)
      h.pending.last.deliver(PreviewImageResult.Unavailable("Preview delivery busy; refresh to retry"))
      h.state.retryVisibleCapacity()
      assertEquals(h.pending.size, 2, "Mailbox drops require refresh, not a release retry loop")
      h.state.bind(h.record(), 2L)
      h.pending.last.deliver(PreviewImageResult.Unavailable(PreviewImageLoader.visibleCapacityReason))
      h.session.setTemporary(CardPreviewMode.Hide)
      val count = h.pending.size
      h.state.retryVisibleCapacity()
      assertEquals(h.pending.size, count)
      assertEquals(h.last.image, None)
    }
  }

  test("shared_session_change_before_late_pixels_is_rechecked_by_both_cards") {
    val shared = new CardPreviewSession("Show")
    val a = new Harness(shared = Some(shared))
    val b = new Harness(shared = Some(shared))
    try
      a.state.bind(a.record(), 1L); b.state.bind(b.record(kind = "hypernetwork"), 9L)
      shared.setTemporary(CardPreviewMode.Blur)
      a.pending.head.finish(); b.pending.head.finish()
      assertEquals(a.last.presentation, PreviewPresentation.Blurred)
      assertEquals(b.last.presentation, PreviewPresentation.Blurred)
      shared.setTemporary(CardPreviewMode.Hide)
      assert(a.last.image.isEmpty && b.last.image.isEmpty)
      assertEquals(a.pending.head.releases, 1)
      assertEquals(b.pending.head.releases, 1)
    finally
      a.close(); b.close()
  }

  test("cancelled_recycled_completion_checks_record_revision_token_and_source_identity") {
    withHarness() { h =>
      val original = h.record()
      val safe = h.record("safe-preview.png")
      h.state.bind(original, 10L)
      val old = h.pending.head
      h.state.bind(safe, 11L)
      assertEquals(old.releases, 1)
      old.finish()
      assertEquals(h.last.image, None)
      val current = h.pending.last.finish()
      assert(h.last.image.exists(_ eq current))
      assertEquals(h.last.presentation, PreviewPresentation.Normal)
      h.state.bind(safe, 12L) // Same record, new scan revision must invalidate the earlier token.
      val fresh = h.pending.last
      h.pending(h.pending.size - 2).finish()
      assertEquals(h.last.image, None)
      fresh.finish()
      val changedSource = safe.copy(preview = original.preview)
      h.state.bind(changedSource, 12L) // Same id/revision, changed selected preview must also invalidate.
      fresh.finish()
      assertEquals(h.last.image, None)
      h.pending.last.finish()
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      assert(h.pending.dropRight(1).forall(_.releases == 1))
    }
  }

  test("clear_and_dispose_cancel_once_remove_pixels_reset_reveal_and_unsubscribe") {
    withHarness() { h =>
      h.state.bind(h.record(), 1L); h.pending.head.finish()
      h.state.setHover(true)
      h.state.clear(); h.state.clear()
      assertEquals(h.pending.head.releases, 1)
      assertEquals(h.last.image, None)
      assertEquals(h.last.record, None)
      assertEquals(h.last.transition, None)
      h.state.bind(h.record(), 2L); h.pending.last.finish()
      assertEquals(h.last.presentation, PreviewPresentation.Blurred)
      h.state.dispose(); h.state.dispose()
      val renderCount = h.rendered.size
      h.session.setTemporary(CardPreviewMode.Show)
      h.pending.last.finish()
      assertEquals(h.rendered.size, renderCount)
      assert(h.pending.forall(_.releases == 1))
    }
  }

  test("safe_source_and_missing_preview_do_not_inherit_nsfw_model_name_policy") {
    withHarness("Hide") { h =>
      val safe = h.record("safe.png").copy(name = "nsfw label",
        id = CardId("hypernetwork", h.dir.resolve("nsfw-model.pt").toString))
      h.state.bind(safe, 1L)
      assertEquals(h.pending.size, 1)
      h.pending.head.finish()
      assertEquals(h.last.presentation, PreviewPresentation.Normal)
      assert(!h.last.revealControlVisible)
      h.state.bind(safe.copy(preview = None), 2L)
      assertEquals(h.pending.size, 1)
      assertEquals(h.last.image, None)
      assertEquals(h.last.presentation, PreviewPresentation.Normal)
    }
  }
