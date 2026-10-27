package sdfx.ui

import java.awt.image.BufferedImage
import java.nio.channels.SeekableByteChannel
import java.nio.file.{Files, Path}
import java.util.concurrent.{CountDownLatch, Executors, ExecutorService, TimeUnit}
import java.util.concurrent.atomic.{AtomicInteger, AtomicReference}
import javax.imageio.ImageIO
import scala.collection.mutable
import sdfx.core.modelcards.*
import sdfx.ui.view.*

/** Tests run the real owner/admission/cache/handoff, replacing only codec blocking and FX scheduling.
  * No toolkit startup: actual native ImageView/effect/rendering remains Task7.
  */
class PreviewImageLoaderSuite extends munit.FunSuite:
  import PreviewLoaderFixtures.*

  private def withHarness(body: Harness => Unit): Unit =
    val h = new Harness
    try body(h)
    finally h.close()

  test("blocked_decoder_repeated_close_reopen_retains_the_same_two_workers_and_source_buffers") {
    withHarness { h =>
      val entered = new CountDownLatch(2)
      val releaseFirst = new CountDownLatch(1)
      val releaseSecond = new CountDownLatch(1)
      val starts = new AtomicInteger
      h.beforeOpen = snapshot =>
        val n = starts.incrementAndGet()
        if n <= 2 then
          val digest = java.security.MessageDigest.getInstance("SHA-256")
          digest.update(snapshot.bytes, 0, snapshot.length)
          val before = digest.digest().toVector
          entered.countDown()
          awaitIgnoringInterrupt(if n == 1 then releaseFirst else releaseSecond)
          digest.update(snapshot.bytes, 0, snapshot.length)
          assertEquals(digest.digest().toVector, before, "A cancelled active worker still owns immutable source bytes")
      h.cleanupGates ++= Seq(releaseFirst, releaseSecond)
      val old = h.owner.acquire()
      old.loader.request(h.ref("first.png"), 112, h.collect)
      old.loader.request(h.ref("second.png"), 112, h.collect)
      await(entered)
      val sources = h.allocations.liveOf("source")
      assertEquals(sources.size, 2)
      assert(sources.values.sum <= 64L * MiB)
      assertReturnsPromptly(old.close())
      var latest: PreviewLoaderLease = old
      for generation <- 1 to 20 do
        latest = h.owner.acquire()
        val reference = h.ref(s"reopen-$generation.png")
        assertReturnsPromptly(latest.loader.request(reference, 112, h.collect))
        assertEquals(starts.get(), 2, "Closing must not treat interrupt as codec completion")
        assertEquals(h.allocations.liveOf("source"), sources)
        assertEquals(h.executorCreations.get(), 1)
        assert(h.threadCreations.get() <= 2)
        assertEquals(h.owner.metrics.activeWorkers, 2)
        if generation < 20 then assertReturnsPromptly(latest.close())
      releaseFirst.countDown()
      h.until("one current-generation request starts after an actual exit") { starts.get() == 3 }
      assertEquals(starts.get(), 3)
      releaseSecond.countDown()
      h.until("all source reservations leave real workers") { h.allocations.liveOf("source").isEmpty }
      h.fx.drain()
      assertEquals(h.readyCount, 1, "Abandoned-generation pixels must never be delivered")
      latest.close()
      h.until("all ownership released") { h.allocations.live.isEmpty }
      h.allocations.assertReleasedOnce()
      assertEquals(h.executorCreations.get(), 1)
    }
  }

  test("queue_and_visible_consumer_caps_reject_without_allocating_and_cancel_releases_capacity") {
    withHarness { h =>
      val gate = new CountDownLatch(1)
      val entered = new CountDownLatch(2)
      h.cleanupGates += gate
      h.beforeOpen = _ => { entered.countDown(); awaitIgnoringInterrupt(gate) }
      val lease = h.owner.acquire()
      val tickets = (0 until 34).map(i => lease.loader.request(h.ref(s"q-$i.png"), 112, h.collect))
      await(entered)
      assert(h.owner.metrics.queued <= 32)
      assert(h.owner.metrics.visibleConsumers <= 32)
      assertEquals(h.allocations.liveOf("source").size, 2)
      assert(h.owner.metrics.rejected > 0)
      h.fx.drain()
      assert(h.unavailable.nonEmpty, "Capacity refusal needs a visible reason")
      tickets.drop(2).foreach(_.cancel())
      assertEquals(h.owner.metrics.queued, 0)
      assertEquals(h.owner.metrics.visibleConsumers, 2)
      val replacement = lease.loader.request(h.ref("after-cancel.png"), 112, h.collect)
      assertEquals(h.owner.metrics.queued, 1)
      replacement.cancel()
      gate.countDown()
      lease.close()
      h.until("workers finish") { h.allocations.live.isEmpty }
      h.allocations.assertReleasedOnce()
    }
  }

  test("a_new_request_cannot_join_an_abandoned_same_key_operation_that_is_still_inside_a_codec") {
    withHarness { h =>
      val gate = new CountDownLatch(1)
      val entered = new CountDownLatch(1)
      val opened = new AtomicInteger
      h.cleanupGates += gate
      h.beforeOpen = _ =>
        if opened.incrementAndGet() == 1 then
          entered.countDown()
          awaitIgnoringInterrupt(gate)
      val lease = h.owner.acquire()
      val reference = h.ref("same-key-cancel.png")
      val old = lease.loader.request(reference, 112, h.collect)
      await(entered)
      old.cancel()
      val replacement = lease.loader.request(reference, 112, h.collect)
      h.until("fresh same-key operation starts independently") { opened.get() == 2 && h.fx.pending > 0 }
      h.fx.drain()
      assertEquals(h.readyCount, 1)
      gate.countDown()
      h.until("abandoned original exits") { h.owner.metrics.activeWorkers == 0 }
      h.fx.drain()
      assertEquals(h.readyCount, 1)
      replacement.cancel(); lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("blocked_fx_bounded_mailbox_and_one_coalesced_drain_own_pixels_until_drop_or_delivery") {
    withHarness { h =>
      val lease = h.owner.acquire()
      val tickets = (0 until 24).map(i => lease.loader.request(h.ref(s"mail-$i.png", 512, 512), 512, h.collect))
      h.until("worker queue exhausted") { h.owner.metrics.queued == 0 && h.owner.metrics.activeWorkers == 0 }
      val m = h.owner.metrics
      assert(m.mailboxItems <= 8)
      assert(m.mailboxBytes <= 8L * MiB)
      assertEquals(h.fx.pending, 1, "Workers must not post one runLater closure per completion")
      assertEquals(m.pendingFxDrains, 1)
      assert(h.allocations.peakPixelBytes <= 32L * MiB)
      assert(h.allocations.peakPixelCount <= 128)
      assertEquals(h.readyCount, 0)
      tickets.take(23).foreach(_.cancel())
      h.fx.drain()
      assert(h.readyCount <= 1, "Only the one remaining valid consumer may attach")
      assert(h.unavailable.exists(_.contains("delivery busy")), "The retained overflow consumer needs a visible retry reason")
      tickets.last.cancel()
      lease.close()
      assertEquals(h.allocations.live, Map.empty[Long, (String, Long)])
      h.allocations.assertReleasedOnce()
    }
  }

  test("each_coalesced_fx_turn_delivers_at_most_eight_ready_or_overflow_callbacks_total") {
    withHarness { h =>
      val lease = h.owner.acquire()
      (0 until 24).foreach(i => lease.loader.request(h.ref(s"turn-$i.png", 512, 512), 512, h.collect))
      h.until("saturated completion queue") { h.owner.metrics.queued == 0 && h.owner.metrics.activeWorkers == 0 }
      assertEquals(h.fx.pending, 1)
      var turns = 0
      while h.fx.pending > 0 && turns < 10 do
        val before = h.results.size
        h.fx.runOne()
        val callbacks = h.results.size - before
        assert(callbacks > 0 && callbacks <= 8, s"One FX turn delivered $callbacks callbacks")
        assert(h.fx.pending <= 1, "Continuation must remain coalesced")
        turns += 1
      assert(turns >= 3 && turns < 10)
      assertEquals(h.results.size, 24)
      assert(h.unavailable.exists(_.contains("delivery busy")))
      lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("capacity_listener_reports_actual_full_consumer_release_not_blocked_worker_cancellation") {
    withHarness { h =>
      val gate = new CountDownLatch(1)
      val entered = new CountDownLatch(2)
      h.cleanupGates += gate
      h.beforeOpen = _ => { entered.countDown(); awaitIgnoringInterrupt(gate) }
      val lease = h.owner.acquire()
      val notifications = new AtomicInteger
      val remove = lease.loader.addCapacityListener(() => { notifications.incrementAndGet(); () })
      val tickets = (0 until 32).map(i => lease.loader.request(h.ref(s"capacity-$i.png"), 112, h.collect))
      await(entered)
      lease.loader.request(h.ref("over-capacity.png"), 112, h.collect)
      assert(h.unavailable.exists(_.contains("visible capacity")))
      assertEquals(notifications.get(), 0)
      tickets.last.cancel() // A queued consumer genuinely frees the full visible-admission budget.
      h.fx.runOne()
      assertEquals(notifications.get(), 1)
      tickets.head.cancel() // Interrupt/cancel does not free this still-blocked source/worker.
      h.fx.drain()
      assertEquals(notifications.get(), 1)
      assertEquals(h.owner.metrics.activeWorkers, 2)
      assertEquals(h.allocations.liveOf("source").size, 2)
      remove(); remove()
      gate.countDown()
      h.until("capacity fixture finishes") { h.owner.metrics.queued == 0 && h.owner.metrics.activeWorkers == 0 }
      h.fx.drain()
      assertEquals(notifications.get(), 1, "Removed browser listener must never be called")
      lease.close()
    }
  }

  test("capacity_listener_registry_is_bounded_to_32_and_removal_releases_a_registration") {
    withHarness { h =>
      val lease = h.owner.acquire()
      val removals = (0 until 32).map(_ => lease.loader.addCapacityListener(() => ()))
      intercept[IllegalStateException] { lease.loader.addCapacityListener(() => ()) }
      removals.head()
      val replacement = lease.loader.addCapacityListener(() => ())
      replacement()
      removals.foreach(_())
      lease.close()
    }
  }

  test("cache_pins_conversion_and_mailbox_share_one_budget_and_pinned_pixels_are_not_evicted") {
    withHarness { h =>
      val lease = h.owner.acquire()
      val pinned = (0 until 24).map { i =>
        val ticket = lease.loader.request(h.ref(s"pin-$i.png", 512, 512), 512, h.collect)
        h.until("completion queued") { h.fx.pending > 0 }
        h.fx.drain()
        ticket
      }
      assertEquals(h.readyCount, 24)
      val pinnedImages = h.results.collect { case PreviewImageResult.Ready(image) => image }.toVector
      for i <- 0 until 40 do
        val t = lease.loader.request(h.ref(s"cache-$i.png", 512, 512), 512, h.collect)
        h.until("completion queued") { h.fx.pending > 0 }
        h.fx.drain()
        t.cancel()
      assert(h.allocations.peakPixelBytes <= 32L * MiB)
      assert(h.allocations.peakPixelCount <= 128)
      assert(h.owner.metrics.cachedItems <= 128)
      pinnedImages.foreach(image => assertEquals(image.getPixelReader.getArgb(0, 0), 0xff336699))
      assertEquals(h.owner.metrics.visibleConsumers, 24)
      pinned.foreach(_.cancel())
      lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("small_cached_thumbnails_enforce_128_allocation_limit_independently_of_byte_limit") {
    withHarness { h =>
      val lease = h.owner.acquire()
      for i <- 0 until 160 do
        val ticket = lease.loader.request(h.ref(s"small-$i.png", 8, 8), 8, h.collect)
        h.until("small thumbnail completion") { h.fx.pending > 0 }
        h.fx.drain()
        ticket.cancel()
      assert(h.allocations.peakPixelCount <= 128)
      assert(h.allocations.pixelAllocationCount > 128, "Fixture must exercise eviction and new allocations")
      assert(h.allocations.peakPixelBytes < MiB, "This fixture isolates allocation count from the byte cap")
      lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("shared_leases_validate_each_source_read_and_keep_the_other_window_pinned_after_one_close") {
    withHarness { h =>
      val opened = new AtomicInteger
      h.beforeOpen = _ => { opened.incrementAndGet(); () }
      val first = h.owner.acquire()
      val second = h.owner.acquire()
      val reference = h.ref("shared.png")
      val firstTicket = first.loader.request(reference, 112, h.collect)
      h.until("first shared image") { h.fx.pending > 0 }; h.fx.drain()
      h.until("first worker exits") { h.owner.metrics.activeWorkers == 0 }
      val secondTicket = second.loader.request(reference, 112, h.collect)
      h.until("second shared image") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(h.snapshotReads.get(), 2)
      assertEquals(opened.get(), 1, "Revalidated identical source should reuse its successful thumbnail")
      val images = h.results.collect { case PreviewImageResult.Ready(image) => image }.toVector
      assert(images(0) eq images(1))
      firstTicket.cancel(); first.close()
      assertEquals(h.owner.metrics.visibleConsumers, 1)
      assert(h.allocations.live.values.exists(_._1 == "fx"))
      assertEquals(images(1).getPixelReader.getArgb(0, 0), 0xff336699)
      secondTicket.cancel(); second.close()
      h.until("shared source release") { h.allocations.live.isEmpty }
      h.allocations.assertReleasedOnce()
    }
  }

  test("decoded_and_fx_conversion_copies_hold_overlapping_reservations_until_transfer_finishes") {
    withHarness { h =>
      val lease = h.owner.acquire()
      val ticket = lease.loader.request(h.ref("conversion.png", 512, 512), 512, h.collect)
      h.until("conversion completion") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(h.readyCount, 1)
      assert(h.allocations.peakPixelBytes >= 2L * MiB,
        "The decoded 512x512 ARGB raster and its FX copy must both be reserved during conversion")
      assert(h.allocations.peakPixelCount >= 2)
      ticket.cancel(); lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("oversized_header_no_decode_checks_side_and_pixel_product_before_reading_frame") {
    for (fixtureWidth, fixtureHeight) <- Vector((16385, 1), (1, 16385), (8001, 4000), (0, 1), (1, -1)) do
      withHarness { h =>
        val reads = new AtomicInteger
        val closes = new AtomicInteger
        h.readerOverride = Some(_ => new PreviewImageReader:
          def width: Int = fixtureWidth
          def height: Int = fixtureHeight
          def format: String = "png"
          def read(frameIndex: Int, subsampling: Int): BufferedImage =
            reads.incrementAndGet()
            new BufferedImage(1, 1, BufferedImage.TYPE_INT_ARGB)
          def close(): Unit = { closes.incrementAndGet(); () }
        )
        val lease = h.owner.acquire()
        lease.loader.request(h.ref("invalid-header.png"), 512, h.collect)
        h.until("invalid-header result") { h.fx.pending > 0 }
        h.fx.drain()
        assertEquals(reads.get(), 0, s"Header $fixtureWidth x $fixtureHeight must be rejected before raster allocation")
        assertEquals(closes.get(), 1)
        assertEquals(h.readyCount, 0)
        assert(h.unavailable.nonEmpty)
        assertEquals(h.allocations.pixelAllocationCount, 0)
        lease.close()
      }
  }

  test("first_frame_gif_and_subsampling_preserve_aspect_and_never_exceed_512") {
    withHarness { h =>
      val frames = mutable.ArrayBuffer.empty[(Int, Int)]
      h.readerOverride = Some(_ => new PreviewImageReader:
        def width = 1600
        def height = 800
        def format = "gif"
        def read(frameIndex: Int, subsampling: Int): BufferedImage =
          frames.synchronized { frames += ((frameIndex, subsampling)) }
          val image = new BufferedImage((1600 + subsampling - 1) / subsampling,
            (800 + subsampling - 1) / subsampling, BufferedImage.TYPE_INT_ARGB)
          image.setRGB(0, 0, if frameIndex == 0 then 0xffff0000 else 0xff00ff00)
          image
        def close(): Unit = ()
      )
      val lease = h.owner.acquire()
      lease.loader.request(h.ref("animated.gif"), 4096, h.collect)
      h.until("GIF complete") { h.fx.pending > 0 }
      h.fx.drain()
      val image = h.results.collectFirst { case PreviewImageResult.Ready(image) => image }.get
      assertEquals(frames.map(_._1).toVector, Vector(0))
      assert(frames.head._2 >= 3, "Source is subsampled before reading a raster")
      assert(image.getWidth <= 512 && image.getHeight <= 512)
      assert(math.abs(image.getWidth / image.getHeight - 2.0) < 0.02)
      lease.close()
    }
  }

  test("real_gif_decodes_first_frame_only_and_keeps_selected_source_bytes") {
    withHarness { h =>
      val path = h.dir.resolve("two-frames.gif")
      val writer = ImageIO.getImageWritersByFormatName("gif").next()
      val output = ImageIO.createImageOutputStream(path.toFile)
      try
        writer.setOutput(output)
        writer.prepareWriteSequence(null)
        for color <- Vector(0xffff0000, 0xff00ff00) do
          val frame = new BufferedImage(4, 2, BufferedImage.TYPE_INT_RGB)
          for x <- 0 until 4; y <- 0 until 2 do frame.setRGB(x, y, color)
          writer.writeToSequence(new javax.imageio.IIOImage(frame, null, null), null)
        writer.endWriteSequence()
      finally
        writer.dispose()
        output.close()
      val original = Files.readAllBytes(path).toVector
      val lease = h.owner.acquire()
      lease.loader.request(h.resolve(path), 4, h.collect)
      h.until("real GIF result") { h.fx.pending > 0 }
      h.fx.drain()
      val image = h.results.collectFirst { case PreviewImageResult.Ready(image) => image }.get
      assertEquals((image.getWidth, image.getHeight), (4.0, 2.0))
      assertEquals(image.getPixelReader.getArgb(0, 0), 0xffff0000)
      assertEquals(Files.readAllBytes(path).toVector, original)
      lease.close()
    }
  }

  test("valid_32MP_header_is_subsampled_and_real_decoder_limits_long_side_preserving_aspect") {
    withHarness { h =>
      val lease = h.owner.acquire()
      lease.loader.request(h.ref("wide.png", 1600, 800), 512, h.collect)
      h.until("real image downsample") { h.fx.pending > 0 }; h.fx.drain()
      val image = h.results.collectFirst { case PreviewImageResult.Ready(image) => image }.get
      assert(image.getWidth > 0 && image.getWidth <= 512)
      assert(image.getHeight > 0 && image.getHeight <= 512)
      assert(math.abs(image.getWidth / image.getHeight - 2.0) < 0.02)
      assertEquals(image.getPixelReader.getArgb(0, 0), 0xff336699)
      val frames = mutable.ArrayBuffer.empty[(Int, Int)]
      h.readerOverride = Some(_ => new PreviewImageReader:
        def width = 8000
        def height = 4000
        def format = "png"
        def read(frameIndex: Int, subsampling: Int): BufferedImage =
          frames += ((frameIndex, subsampling))
          new BufferedImage((8000 + subsampling - 1) / subsampling,
            (4000 + subsampling - 1) / subsampling, BufferedImage.TYPE_INT_ARGB)
        def close(): Unit = ()
      )
      lease.loader.request(h.ref("32mp.png"), 512, h.collect)
      h.until("exact cap image") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(h.readyCount, 2)
      assertEquals(frames.head._1, 0)
      assert(frames.head._2 >= 15)
      lease.close()
    }
  }

  test("same_stamp_digest_is_revalidated_before_cache_hit_and_original_source_is_unchanged") {
    withHarness { h =>
      val ref = h.ref("same-stamp.bmp", 16, 8, 0xff336699)
      val initial = Files.readAllBytes(ref.localPath)
      val stamp = Files.getLastModifiedTime(ref.localPath)
      val lease = h.owner.acquire()
      def terminal(label: String, outcome: AtomicReference[Option[PreviewImageResult]]): javafx.scene.image.Image =
        h.until(label) {
          if h.fx.pending > 0 then h.fx.runOne()
          outcome.get().nonEmpty
        }
        outcome.get().get match
          case PreviewImageResult.Ready(image) => image
          case PreviewImageResult.Unavailable(reason) => fail(s"$label unexpectedly unavailable: $reason")
      val firstOutcome = new AtomicReference[Option[PreviewImageResult]](None)
      val first = lease.loader.request(ref, 112, result => { h.collect(result); firstOutcome.set(Some(result)) })
      assertEquals(terminal("first exact request", firstOutcome).getPixelReader.getArgb(0, 0), 0xff336699)
      first.cancel()
      assertEquals(Files.readAllBytes(ref.localPath).toVector, initial.toVector)
      // ImageIO.write(File) deletes its output first. Encode elsewhere, then overwrite the SAME file.
      val encoded = h.dir.resolve("replacement-encoded.bmp")
      writeImage(encoded, 16, 8, 0xffaa1100)
      val replacement = Files.readAllBytes(encoded)
      assertEquals(replacement.length, initial.length)
      Files.write(ref.localPath, replacement, java.nio.file.StandardOpenOption.WRITE,
        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING)
      assertEquals(Files.size(ref.localPath), initial.length.toLong)
      Files.setLastModifiedTime(ref.localPath, stamp)
      val validatedAfter = h.resolve(ref.localPath)
      assertEquals(validatedAfter.identity, ref.identity,
        "The fixture must preserve the FULL validated identity, not only size and mtime")
      val initialDigest = java.security.MessageDigest.getInstance("SHA-256").digest(initial).toVector
      val replacementDigest = java.security.MessageDigest.getInstance("SHA-256").digest(replacement).toVector
      assert(initialDigest != replacementDigest, "The fixture must change content while preserving identity")
      val secondOutcome = new AtomicReference[Option[PreviewImageResult]](None)
      val second = lease.loader.request(ref, 112, result => { h.collect(result); secondOutcome.set(Some(result)) })
      assertEquals(terminal("second exact request with same identity and new digest", secondOutcome)
        .getPixelReader.getArgb(0, 0), 0xffaa1100)
      val pixels = h.results.collect { case PreviewImageResult.Ready(image) => image.getPixelReader.getArgb(0, 0) }
      assertEquals(pixels.toVector, Vector(0xff336699, 0xffaa1100))
      assertEquals(h.snapshotReads.get(), 2, "A cache hit cannot bypass bounded source revalidation")
      assertEquals(Files.readAllBytes(ref.localPath).toVector, replacement.toVector)
      assertEquals(h.resolve(ref.localPath).identity, ref.identity)
      second.cancel(); lease.close()
    }
  }

  test("changed_identity_rejects_a_cached_thumbnail_and_cancelled_completion_releases_once") {
    withHarness { h =>
      val ref = h.ref("identity.png")
      val lease = h.owner.acquire()
      val first = lease.loader.request(ref, 112, h.collect)
      h.until("initial delivery") { h.fx.pending > 0 }; h.fx.drain(); first.cancel()
      h.snapshotFailure = Some("preview identity changed")
      lease.loader.request(ref, 112, h.collect)
      h.until("identity rejection") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(h.readyCount, 1)
      assert(h.unavailable.exists(_.contains("identity")))
      h.snapshotFailure = None
      val cancelled = lease.loader.request(h.ref("cancelled.png"), 112, h.collect)
      h.until("undelivered pixels") { h.fx.pending > 0 }
      cancelled.cancel(); cancelled.cancel()
      h.fx.drain()
      assertEquals(h.readyCount, 1)
      lease.close(); lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("source_above_32MiB_is_unavailable_before_codec_and_exact_caller_buffer_is_used") {
    withHarness { h =>
      val ref = h.ref("large.png")
      val channel = java.nio.channels.FileChannel.open(ref.localPath, java.nio.file.StandardOpenOption.WRITE)
      try
        channel.position(32L * MiB)
        channel.write(java.nio.ByteBuffer.wrap(Array[Byte](0)))
      finally channel.close()
      // Early resolver refusal is legal; an already-bound reference still must not reach the codec.
      val fresh = new ModelPreviewResolver(LocalPreviewFiles.scoped(new LocalPlatform))
        .validateExplicit(ref.localPath).getOrElse(ref)
      val opens = new AtomicInteger
      h.beforeOpen = _ => { opens.incrementAndGet(); () }
      val lease = h.owner.acquire()
      lease.loader.request(fresh, 112, h.collect)
      h.until("source refusal") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(opens.get(), 0)
      assertEquals(h.readyCount, 0)
      assert(h.unavailable.nonEmpty)
      lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("corrupt_source_and_codec_failure_keep_identity_available_and_release_every_allocation") {
    withHarness { h =>
      val path = h.dir.resolve("corrupt.png")
      Files.write(path, Array[Byte](1, 2, 3, 4))
      val lease = h.owner.acquire()
      lease.loader.request(h.resolve(path), 112, h.collect)
      h.until("corrupt source result") { h.fx.pending > 0 }; h.fx.drain()
      val closed = new AtomicInteger
      h.readerOverride = Some(_ => new PreviewImageReader:
        def width = 16
        def height = 8
        def format = "png"
        def read(frameIndex: Int, subsampling: Int): BufferedImage = throw new java.io.IOException("fixture codec failure")
        def close(): Unit = { closed.incrementAndGet(); () }
      )
      lease.loader.request(h.ref("codec-error.png"), 112, h.collect)
      h.until("codec failure result") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(h.readyCount, 0)
      assertEquals(h.unavailable.size, 2)
      assert(h.unavailable.forall(_.nonEmpty))
      assertEquals(closed.get(), 1)
      lease.close()
      h.allocations.assertReleasedOnce()
    }
  }

  test("reader_close_failure_after_pixels_exist_releases_the_undelivered_image") {
    withHarness { h =>
      h.readerOverride = Some(_ => new PreviewImageReader:
        def width = 16
        def height = 8
        def format = "png"
        def read(frameIndex: Int, subsampling: Int): BufferedImage =
          new BufferedImage(16, 8, BufferedImage.TYPE_INT_ARGB)
        def close(): Unit = throw new java.io.IOException("fixture close failed")
      )
      val lease = h.owner.acquire()
      lease.loader.request(h.ref("close-error.png"), 112, h.collect)
      h.until("reader close error") { h.fx.pending > 0 }; h.fx.drain()
      assertEquals(h.readyCount, 0)
      assert(h.unavailable.exists(_.contains("close failed")))
      lease.close()
      h.until("close error source exit") { h.allocations.live.isEmpty }
      h.allocations.assertReleasedOnce()
    }
  }

  test("shutdown_cannot_create_a_replacement_executor_while_old_codecs_ignore_interrupts") {
    withHarness { h =>
      val gate = new CountDownLatch(1)
      val entered = new CountDownLatch(2)
      h.cleanupGates += gate
      h.beforeOpen = _ => { entered.countDown(); awaitIgnoringInterrupt(gate) }
      val lease = h.owner.acquire()
      (0 until 2).foreach(i => lease.loader.request(h.ref(s"shutdown-$i.png"), 112, h.collect))
      await(entered)
      assertReturnsPromptly(h.owner.shutdownProcess())
      val later = h.owner.acquire()
      later.loader.request(h.ref("post-shutdown.png"), 112, h.collect)
      h.fx.drain()
      assert(h.unavailable.nonEmpty)
      assertEquals(h.executorCreations.get(), 1)
      assertEquals(h.allocations.liveOf("source").size, 2)
      gate.countDown(); later.close(); lease.close()
      h.until("shutdown real completion") { h.allocations.live.isEmpty }
      h.allocations.assertReleasedOnce()
    }
  }

private[ui] object PreviewLoaderFixtures:
  val MiB = 1024L * 1024L
  private val timeoutSeconds = 10L
  def await(latch: CountDownLatch): Unit =
    assert(latch.await(timeoutSeconds, TimeUnit.SECONDS), "Finite fixture latch timed out")
  def awaitIgnoringInterrupt(latch: CountDownLatch): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSeconds)
    var released = false
    while !released && System.nanoTime() < end do
      try released = latch.await(math.max(1L, end - System.nanoTime()), TimeUnit.NANOSECONDS)
      catch case _: InterruptedException => ()
    assert(released, "Blocked codec fixture timed out; cleanup must always release it")
  def assertReturnsPromptly[A](action: => A): A =
    val caller = Executors.newSingleThreadExecutor((runnable: Runnable) =>
      val thread = new Thread(runnable, "preview-caller-fixture")
      thread.setDaemon(true)
      thread)
    try caller.submit(new java.util.concurrent.Callable[A] { def call(): A = action }).get(2, TimeUnit.SECONDS)
    finally caller.shutdownNow()

  final class LocalPlatform extends LocalPreviewFiles.Platform:
    private val delegate = LocalPreviewFiles.systemPlatform
    def windows = delegate.windows
    def driveType(root: Path): Either[String, String] = Right("Fixed")
    def stamp(path: Path) = delegate.stamp(path)
    def realPath(path: Path) = delegate.realPath(path)
    def openNoFollow(path: Path): Either[String, SeekableByteChannel] = delegate.openNoFollow(path)

  def writeImage(path: Path, width: Int = 16, height: Int = 8, color: Int = 0xff336699): Unit =
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val graphics = image.createGraphics()
    try
      graphics.setColor(new java.awt.Color(color, true))
      graphics.fillRect(0, 0, width, height)
    finally graphics.dispose()
    val format = if path.toString.endsWith(".bmp") then "bmp" else "png"
    assert(ImageIO.write(image, format, path.toFile))

  final class FxQueue:
    private val work = mutable.Queue.empty[() => Unit]
    def submit(task: () => Unit): Unit = synchronized { work.enqueue(task) }
    def pending: Int = synchronized { work.size }
    def runOne(): Unit =
      val task = synchronized { work.dequeue() }
      task()
    def drain(): Unit =
      var remaining = 100
      while pending > 0 && remaining > 0 do
        runOne()
        remaining -= 1
      assert(remaining > 0, "Drain must not recursively spin on capacity refusal")

  /** Records real allocation lifecycle events, independently of owner metrics. IDs cannot be reused. */
  final class AllocationLedger:
    private val allocated = mutable.Map.empty[Long, (String, Long)]
    private val released = mutable.Map.empty[Long, Int]
    private var peakBytes = 0L
    private var peakCount = 0
    private val failures = mutable.ArrayBuffer.empty[String]
    def record(e: PreviewAllocationEvent): Unit = synchronized {
      try recordChecked(e)
      catch
        case error: AssertionError =>
          failures += error.toString
          throw error
    }
    private def recordChecked(e: PreviewAllocationEvent): Unit =
      if e.acquired then
        assert(!allocated.contains(e.id), s"Allocation id ${e.id} reused")
        assert(e.bytes > 0)
        allocated(e.id) = ((e.kind, e.bytes))
      else
        assert(allocated.contains(e.id), s"Release without reservation ${e.id}")
        released(e.id) = released.getOrElse(e.id, 0) + 1
        assert(released(e.id) == 1, s"Double release ${e.id}")
      val pixels = live.values.filter(_._1 != "source").toVector
      peakBytes = math.max(peakBytes, pixels.map(_._2).sum)
      peakCount = math.max(peakCount, pixels.size)
      assert(pixels.map(_._2).sum <= 32L * MiB, "Physical pixel reservations exceed 32MiB")
      assert(pixels.size <= 128, "Physical pixel reservations exceed 128")
      val sources = liveOf("source")
      assert(sources.size <= 2 && sources.values.sum <= 64L * MiB)
    def live: Map[Long, (String, Long)] = synchronized { allocated.toMap -- released.keySet }
    def liveOf(kind: String): Map[Long, Long] = live.collect { case (id, (`kind`, bytes)) => id -> bytes }
    def peakPixelBytes: Long = synchronized { peakBytes }
    def peakPixelCount: Int = synchronized { peakCount }
    def pixelAllocationCount: Int = synchronized { allocated.values.count(_._1 != "source") }
    def assertReleasedOnce(): Unit = synchronized {
      assert(failures.isEmpty, failures.mkString("; "))
      assert(live.isEmpty, s"Leaked allocations: $live")
      assert(allocated.keySet == released.keySet)
      assert(released.values.forall(_ == 1))
    }

  final class Harness:
    val dir = Files.createTempDirectory("sdfx-preview-loader-")
    val fx = new FxQueue
    val allocations = new AllocationLedger
    val cleanupGates = mutable.ArrayBuffer.empty[CountDownLatch]
    val executorCreations = new AtomicInteger
    val threadCreations = new AtomicInteger
    val snapshotReads = new AtomicInteger
    val results = mutable.ArrayBuffer.empty[PreviewImageResult]
    private val workerFailures = mutable.ArrayBuffer.empty[Throwable]
    @volatile var beforeOpen: PreviewSourceSnapshot => Unit = _ => ()
    @volatile var readerOverride: Option[PreviewSourceSnapshot => PreviewImageReader] = None
    @volatile var snapshotFailure: Option[String] = None
    private val executors = mutable.ArrayBuffer.empty[ExecutorService]
    val owner = new PreviewDecodeOwner(PreviewDecodeDependencies(
      readSnapshot = (ref, buffer, cancelled) =>
        snapshotReads.incrementAndGet()
        snapshotFailure match
          case Some(reason) => Left(reason)
          case None =>
            val result = LocalPreviewFiles.scoped(new LocalPlatform).readSnapshot(ref, buffer, cancelled)
            result.foreach(snapshot => assert(snapshot.bytes eq buffer, "Do not clone an admitted source buffer"))
            result,
      openReader = snapshot =>
        try beforeOpen(snapshot)
        catch case failure: Throwable =>
          workerFailures.synchronized { workerFailures += failure }
          throw failure
        readerOverride.fold(PreviewImageCodec.open(snapshot))(factory => Right(factory(snapshot))),
      scheduleFx = fx.submit,
      executorFactory = () =>
        executorCreations.incrementAndGet()
        val executor = Executors.newFixedThreadPool(2, (runnable: Runnable) =>
          val thread = new Thread(runnable, s"preview-test-${threadCreations.incrementAndGet()}")
          thread.setDaemon(true)
          thread)
        executors += executor
        executor,
      allocationEvent = allocations.record
    ))
    def resolve(path: Path): PreviewReference =
      new ModelPreviewResolver(LocalPreviewFiles.scoped(new LocalPlatform)).validateExplicit(path)
        .fold(reason => throw new AssertionError(reason), identity)
    def ref(name: String, width: Int = 16, height: Int = 8, color: Int = 0xff336699): PreviewReference =
      val path = dir.resolve(name)
      writeImage(path, width, height, color)
      resolve(path)
    val collect: PreviewImageResult => Unit = result => results.synchronized { results += result; () }
    def readyCount: Int = results.synchronized { results.count { case PreviewImageResult.Ready(_) => true; case _ => false } }
    def unavailable: Vector[String] = results.synchronized {
      results.collect { case PreviewImageResult.Unavailable(reason) => reason }.toVector
    }
    def until(clue: String)(condition: => Boolean): Unit =
      val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutSeconds)
      while !condition && System.nanoTime() < deadline do java.util.concurrent.locks.LockSupport.parkNanos(1000000L)
      assert(condition, s"Timed out waiting for $clue")
    def close(): Unit =
      cleanupGates.foreach(_.countDown())
      var error: Option[Throwable] = None
      def attempt(action: => Unit): Unit =
        try action
        catch case failure: Throwable =>
          error match
            case Some(first) => first.addSuppressed(failure)
            case None => error = Some(failure)
      attempt(owner.shutdownProcess())
      executors.foreach(e => attempt(assert(e.awaitTermination(timeoutSeconds, TimeUnit.SECONDS), "Worker fixture did not terminate")))
      attempt(fx.drain())
      attempt(allocations.assertReleasedOnce())
      attempt(assert(workerFailures.synchronized { workerFailures.isEmpty },
        workerFailures.synchronized { workerFailures.mkString("Worker fixture failures: ", "; ", "") }))
      attempt {
        val walk = Files.walk(dir)
        try walk.sorted(java.util.Comparator.reverseOrder[Path]()).forEach(path => Files.deleteIfExists(path))
        finally walk.close()
      }
      error.foreach(throw _)
