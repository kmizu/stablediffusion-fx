package sdfx.core

import java.nio.ByteBuffer
import java.nio.channels.SeekableByteChannel
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import sdfx.core.modelcards.*

/** Real temporary-file IO; only the external drive classification is deterministic here.
  * This does not replace ROOT's actual Windows mapped-drive/junction release gate.
  */
private[core] class PreviewTestPlatform extends LocalPreviewFiles.Platform:
  protected val native = LocalPreviewFiles.systemPlatform
  def windows: Boolean = native.windows
  def driveType(root: Path): Either[String, String] = Right("Fixed")
  def stamp(path: Path): Either[String, LocalPreviewFiles.PathStamp] = native.stamp(path)
  def realPath(path: Path): Either[String, Path] = native.realPath(path)
  def openNoFollow(path: Path): Either[String, SeekableByteChannel] = native.openNoFollow(path)

class LocalPreviewFilesSuite extends munit.FunSuite with TempDir:
  private val cap = 32 * 1024 * 1024
  private def write(dir: Path, name: String = "preview.png", text: String = "abc"): Path =
    Files.write(dir.resolve(name), text.getBytes(UTF_8))
  private def ref(path: Path, platform: LocalPreviewFiles.Platform = new PreviewTestPlatform): PreviewReference =
    new ModelPreviewResolver(LocalPreviewFiles.scoped(platform)).validateExplicit(path).fold(fail(_), identity)
  private def read(reference: PreviewReference, platform: LocalPreviewFiles.Platform = new PreviewTestPlatform,
      buffer: Array[Byte] = new Array[Byte](64)) =
    LocalPreviewFiles.scoped(platform).readSnapshot(reference, buffer, () => false)

  test("windows_path_classes_reject_implicit_drives_namespaces_and_uri_inputs") {
    val rejected = Vector("C:relative.png", "C:", "\\root-relative.png", "/root-relative.png",
      "\\\\server\\share\\a.png", "//server/share/a.png", "\\\\?\\C:\\a.png",
      "\\\\.\\C:\\a.png", "\\\\?\\UNC\\server\\share\\a.png", "file:///C:/a.png",
      "https://example.invalid/a.png", "", "C:\\folder\\a.png:stream", "C:\\a\u0000.png")
    rejected.foreach { path =>
      assert(LocalPreviewFiles.validateWindowsPath(path, allowRelative = false).isLeft, path)
      assert(LocalPreviewFiles.validateWindowsPath(path, allowRelative = true).isLeft, path)
    }
    Vector("C:\\folder\\a.png", "D:/folder/a.png", "c:\\日本語\\a.png").foreach { path =>
      assert(LocalPreviewFiles.validateWindowsPath(path, allowRelative = false).isRight, path)
    }
    assert(LocalPreviewFiles.validateWindowsPath("models/local", allowRelative = true).isRight)
    assert(LocalPreviewFiles.validateWindowsPath("models/local", allowRelative = false).isLeft)
  }

  tmp.test("mapped_drive_rejected_before_any_filesystem_probe") { dir =>
    val path = write(dir)
    for answer <- Vector(Right("Network"), Right("Unknown"), Right("NoRootDirectory"), Left("probe unavailable")) do
      var traversed = false
      val platform = new PreviewTestPlatform:
        override def windows = true
        override def driveType(root: Path) = answer
        override def stamp(p: Path) = { traversed = true; super.stamp(p) }
        override def realPath(p: Path) = { traversed = true; super.realPath(p) }
      assert(LocalPreviewFiles.scoped(platform).validate(path, None).isLeft)
      assert(!traversed, "Rejected drive must not reach realPath or attribute traversal")
  }

  tmp.test("fixed_removable_and_ram_drives_allow_local_identity_validation") { dir =>
    val path = write(dir)
    for kind <- Vector("Fixed", "Removable", "Ram") do
      val platform = new PreviewTestPlatform:
        override def windows = true
        override def driveType(root: Path) = Right(kind)
      val id = LocalPreviewFiles.scoped(platform).validate(path, None).fold(fail(_), identity)
      assertEquals(id.realPath, path.toRealPath())
      assertEquals(id.size, 3L)
      assert(id.fileKey.nonEmpty)
      assert(id.ancestorKeys.exists(_.path == dir.toRealPath()))
      assert(id.ancestorKeys.forall(_.fileKey.nonEmpty))
  }

  tmp.test("actual_native_identity_is_stable_across_rename_and_distinguishes_same_stamp_replacement") { dir =>
    val path = write(dir)
    val original = LocalPreviewFiles.scoped().validateModel(path, directoryModel = false).fold(fail(_), identity)
    val repeated = LocalPreviewFiles.scoped().validateModel(path, directoryModel = false).fold(fail(_), identity)
    assert(original.fileKey.nonEmpty)
    assertEquals(repeated.fileKey, original.fileKey)
    val originalTime = Files.getLastModifiedTime(path)
    // Keep the old object alive under another name, so the filesystem cannot recycle its file ID.
    val retained = Files.move(path, dir.resolve("retained-original.png"))
    Files.writeString(path, "xyz")
    Files.setLastModifiedTime(path, originalTime)
    val replacement = LocalPreviewFiles.scoped().validateModel(path, directoryModel = false).fold(fail(_), identity)
    val renamed = LocalPreviewFiles.scoped().validateModel(retained, directoryModel = false).fold(fail(_), identity)
    assertEquals(replacement.size, original.size)
    assertEquals(replacement.modifiedMillis, original.modifiedMillis)
    assertEquals(renamed.fileKey, original.fileKey)
    assertNotEquals(replacement.fileKey, original.fileKey)
    assertEquals(Files.readString(retained), "abc")
    assertEquals(Files.readString(path), "xyz")
  }

  tmp.test("drive_probe_is_shared_across_ancestors_within_one_validation") { dir =>
    val nested = Files.createDirectories(dir.resolve("one").resolve("two").resolve("three"))
    val path = write(nested)
    var probes = Vector.empty[Path]
    val platform = new PreviewTestPlatform:
      override def windows = true
      override def driveType(root: Path) = { probes :+= root; Right("Fixed") }
    assert(LocalPreviewFiles.scoped(platform).validate(path, None).isRight)
    assertEquals(probes, Vector(path.getRoot))
  }

  tmp.test("model_root_resolution_is_relative_to_app_home_and_preserves_selected_spelling") { dir =>
    val scope = LocalPreviewFiles.scoped(new PreviewTestPlatform)
    assertEquals(scope.resolveRoot("Models/../MiXeD", dir), Right(dir.resolve("MiXeD")))
    assertEquals(scope.resolveRoot(dir.resolve("Original").toString, dir), Right(dir.resolve("Original")))
    val windows = LocalPreviewFiles.scoped(new PreviewTestPlatform:
      override def windows = true)
    Vector("C:models", "\\models", "\\\\server\\share", "file:///C:/models").foreach { raw =>
      assert(windows.resolveRoot(raw, dir).isLeft, raw)
    }
  }

  tmp.test("model_validation_is_metadata_only_for_large_weights_and_directory_models") { dir =>
    val model = write(dir, "model.safetensors")
    val channel = java.nio.channels.FileChannel.open(model, java.nio.file.StandardOpenOption.WRITE)
    try
      channel.position(cap.toLong + 1L)
      channel.write(ByteBuffer.wrap(Array[Byte](0)))
    finally channel.close()
    val platform = new PreviewTestPlatform:
      override def openNoFollow(p: Path) = throw new AssertionError("Model weights must not be opened")
    val scope = LocalPreviewFiles.scoped(platform)
    val id = scope.validateModel(model, directoryModel = false).fold(fail(_), identity)
    assertEquals(id.size, cap.toLong + 2L)
    assert(scope.validateModel(dir, directoryModel = true).isRight)
    assert(scope.validateModel(model, directoryModel = true).isLeft)
    assert(scope.validateModel(dir, directoryModel = false).isLeft)
    assert(scope.validate(dir, None).isLeft, "Preview validation must still reject a directory")
  }

  tmp.test("unknown_identity_unavailable_for_file_and_each_ancestor") { dir =>
    val path = write(dir)
    assert(LocalPreviewFiles.scoped(new PreviewTestPlatform).validate(path, None).isRight,
      "Native control must validate before injecting missing identity")
    for missing <- Vector(path, dir, dir.getRoot) do
      var reached = false
      val platform = new PreviewTestPlatform:
        override def stamp(p: Path) = super.stamp(p).map { s =>
          if p == missing then
            reached = true
            s.copy(fileKey = None)
          else s
        }
      assertEquals(LocalPreviewFiles.scoped(platform).validate(path, None), Left("Filesystem identity is unavailable"))
      assert(reached, s"Missing-key fixture was not reached: $missing")
    var deniedReached = false
    val denied = new PreviewTestPlatform:
      override def stamp(p: Path) =
        if p == dir then
          deniedReached = true
          Left("ancestor unavailable")
        else super.stamp(p)
    assertEquals(LocalPreviewFiles.scoped(denied).validate(path, None), Left("ancestor unavailable"))
    assert(deniedReached, "Denied-ancestor fixture was not reached")
  }

  tmp.test("automatic_parent_escape_is_rejected_and_explicit_location_is_allowed") { dir =>
    val approved = Files.createDirectory(dir.resolve("approved"))
    val outside = write(dir)
    assert(LocalPreviewFiles.scoped(new PreviewTestPlatform).validate(outside, Some(approved.toRealPath())).isLeft)
    assert(LocalPreviewFiles.scoped(new PreviewTestPlatform).validate(outside, None).isRight)
  }

  tmp.test("snapshot_uses_caller_buffer_exact_length_digest_and_one_no_follow_open") { dir =>
    val path = write(dir)
    val reference = ref(path)
    var opens = Vector.empty[Path]
    var closes = 0
    val platform = new PreviewTestPlatform:
      override def openNoFollow(p: Path) =
        opens :+= p
        super.openNoFollow(p).map(channel => new WrappedChannel(channel, onClose = () => closes += 1))
    val buffer = Array.fill[Byte](64)(42)
    val snapshot = read(reference, platform, buffer).fold(fail(_), identity)
    assert(snapshot.bytes eq buffer, "No trimmed copy or second encoded buffer")
    assertEquals(snapshot.length, 3)
    assertEquals(snapshot.bytes.take(snapshot.length).toVector, Vector[Byte](97, 98, 99))
    assertEquals(snapshot.sha256, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assertEquals(snapshot.identity, reference.identity)
    assertEquals(opens, Vector(reference.identity.realPath))
    assertEquals(closes, 1)
    assertEquals(Files.readString(path), "abc")
    assertEquals(buffer.drop(3).toVector, Vector.fill[Byte](61)(42))
  }

  tmp.test("snapshot_boundaries_accept_exact_cap_reject_larger_and_undersized_buffer") { dir =>
    val path = write(dir)
    val reference = ref(path)
    assert(read(reference, buffer = new Array[Byte](2)).isLeft)
    assert(read(reference, buffer = new Array[Byte](cap + 1)).isLeft)
    val channel = java.nio.channels.FileChannel.open(path, java.nio.file.StandardOpenOption.WRITE)
    try
      channel.position(cap - 1L)
      channel.write(ByteBuffer.wrap(Array[Byte](0)))
    finally channel.close()
    val exactRef = ref(path)
    val buffer = new Array[Byte](cap)
    assertEquals(read(exactRef, buffer = buffer).fold(fail(_), identity).length, cap)
    val grow = java.nio.channels.FileChannel.open(path, java.nio.file.StandardOpenOption.WRITE)
    try
      grow.position(cap.toLong)
      grow.write(ByteBuffer.wrap(Array[Byte](0)))
    finally grow.close()
    var opened = false
    val noOpen = new PreviewTestPlatform:
      override def openNoFollow(p: Path) = { opened = true; super.openNoFollow(p) }
    val large = LocalPreviewFiles.scoped(noOpen).validate(path, None)
    // A resolver may reject the oversized file early; otherwise the snapshot must reject before opening it.
    large.foreach { _ => assert(read(ref(path), noOpen, buffer).isLeft) }
    assert(!opened)
    assertEquals(Files.size(path), cap.toLong + 1L)
  }

  tmp.test("cancel_before_open_never_reads_source") { dir =>
    val reference = ref(write(dir))
    var opened = false
    val platform = new PreviewTestPlatform:
      override def openNoFollow(p: Path) = { opened = true; super.openNoFollow(p) }
    assert(LocalPreviewFiles.scoped(platform).readSnapshot(reference, new Array[Byte](64), () => true).isLeft)
    assert(!opened)
  }

  tmp.test("cancel_during_read_closes_source_and_returns_no_snapshot") { dir =>
    val reference = ref(write(dir))
    var cancelled = false
    var closed = false
    val platform = new PreviewTestPlatform:
      override def openNoFollow(p: Path) = super.openNoFollow(p).map(channel =>
        new WrappedChannel(channel, afterRead = () => cancelled = true, onClose = () => closed = true))
    val result = LocalPreviewFiles.scoped(platform).readSnapshot(reference, new Array[Byte](64), () => cancelled)
    assert(result.isLeft)
    assert(closed)
  }

  tmp.test("read_failure_closes_source_and_returns_visible_problem") { dir =>
    val reference = ref(write(dir))
    var closed = false
    val platform = new PreviewTestPlatform:
      override def openNoFollow(p: Path) = super.openNoFollow(p).map(channel =>
        new WrappedChannel(channel, onClose = () => closed = true):
          override def read(dst: ByteBuffer): Int = throw new java.io.IOException("fixture read failure"))
    assert(read(reference, platform).isLeft)
    assert(closed)
  }

  tmp.test("shortened_file_during_open_is_not_delivered_as_success") { dir =>
    val path = write(dir)
    val reference = ref(path)
    var closed = false
    val platform = new PreviewTestPlatform:
      override def openNoFollow(p: Path) = super.openNoFollow(p).map { channel =>
        Files.writeString(path, "a")
        new WrappedChannel(channel, onClose = () => closed = true)
      }
    assert(read(reference, platform).isLeft)
    assert(closed)
  }

  tmp.test("changed_content_with_identical_size_and_mtime_gets_new_digest") { dir =>
    val path = write(dir)
    val reference = ref(path)
    val stamp = Files.getLastModifiedTime(path)
    val first = read(reference).fold(fail(_), identity)
    Files.writeString(path, "xyz")
    Files.setLastModifiedTime(path, stamp)
    val second = read(reference).fold(fail(_), identity)
    assertEquals(second.identity.fileKey, first.identity.fileKey)
    assertEquals(second.identity.modifiedMillis, first.identity.modifiedMillis)
    assertNotEquals(second.sha256, first.sha256)
    assertEquals(new String(second.bytes, 0, second.length, UTF_8), "xyz")
    assertEquals(new String(first.bytes, 0, first.length, UTF_8), "abc")
  }

  tmp.test("same_stamp_filekey_replacement_is_unavailable_before_open") { dir =>
    val path = write(dir)
    val reference = ref(path)
    var opened = false
    val platform = new PreviewTestPlatform:
      override def stamp(p: Path) = super.stamp(p).map(s => if p == path then s.copy(fileKey = Some("replacement")) else s)
      override def openNoFollow(p: Path) = { opened = true; super.openNoFollow(p) }
    assert(read(reference, platform).isLeft)
    assert(!opened)
  }

  tmp.test("ancestor_junction_retarget_injected_during_open_rejects_same_stamp_source") { dir =>
    val path = write(dir)
    val reference = ref(path)
    var changed = false
    var closed = false
    val platform = new PreviewTestPlatform:
      override def stamp(p: Path) = super.stamp(p).map { s =>
        if changed && p == dir then s.copy(fileKey = Some("new-ancestor"), reparseTarget = Some(dir.resolve("other"))) else s
      }
      override def openNoFollow(p: Path) = super.openNoFollow(p).map { channel =>
        changed = true
        new WrappedChannel(channel, onClose = () => closed = true)
      }
    assert(read(reference, platform).isLeft)
    assert(closed)
  }

  tmp.test("selected_chain_changed_after_read_is_unavailable_and_channel_closes") { dir =>
    val path = write(dir)
    val reference = ref(path)
    var changed = false
    var closed = false
    val platform = new PreviewTestPlatform:
      override def stamp(p: Path) = super.stamp(p).map(s =>
        if changed && p == path then s.copy(fileKey = Some("retargeted-after-read")) else s)
      override def openNoFollow(p: Path) = super.openNoFollow(p).map(channel =>
        new WrappedChannel(channel, afterRead = () => changed = true, onClose = () => closed = true))
    assert(read(reference, platform).isLeft)
    assert(closed)
  }

  tmp.test("explicit_local_link_injected_preserves_selected_uri_and_captures_target") { dir =>
    val selected = write(dir, "chosen-NSFW.png", "selected")
    val target = write(dir, "safe-target.png", "abc")
    val platform = new PreviewTestPlatform:
      override def realPath(p: Path) = if p == selected then super.realPath(target) else super.realPath(p)
      override def stamp(p: Path) = super.stamp(p).map(s =>
        if p == selected then s.copy(regularFile = false, reparseTarget = Some(target)) else s)
    val reference = ref(selected, platform)
    assertEquals(reference.source, selected.toUri.toASCIIString)
    assertEquals(reference.identity.realPath, target.toRealPath())
    assert(reference.identity.ancestorKeys.exists(a => a.path == selected && a.reparseTarget.contains(target)))
    val snapshot = read(reference, platform).fold(fail(_), identity)
    assertEquals(new String(snapshot.bytes, 0, snapshot.length, UTF_8), "abc")
    assert(CardPreviewPolicy.matches(Some(reference.source)))
    assertEquals(Files.readString(selected), "selected")
  }

  tmp.test("same_stamp_link_retarget_injected_cannot_reuse_old_reference") { dir =>
    val selected = write(dir, "chosen.png")
    val targetA = write(dir, "a.png", "aaa")
    val targetB = write(dir, "b.png", "bbb")
    Files.setLastModifiedTime(targetB, Files.getLastModifiedTime(targetA))
    var current = targetA
    val platform = new PreviewTestPlatform:
      override def realPath(p: Path) = if p == selected then super.realPath(current) else super.realPath(p)
      override def stamp(p: Path) = super.stamp(p).map(s =>
        if p == selected then s.copy(regularFile = false, reparseTarget = Some(current)) else s)
    val reference = ref(selected, platform)
    current = targetB
    assert(read(reference, platform).isLeft)
    assertEquals(Files.readString(targetA), "aaa")
    assertEquals(Files.readString(targetB), "bbb")
  }

  tmp.test("fresh_snapshot_revalidates_drive_type_instead_of_reusing_resolver_cache") { dir =>
    val path = write(dir)
    var drive = "Fixed"
    val platform = new PreviewTestPlatform:
      override def windows = true
      override def driveType(root: Path) = Right(drive)
    val scope = LocalPreviewFiles.scoped(platform)
    val reference = new ModelPreviewResolver(scope).validateExplicit(path).fold(fail(_), identity)
    drive = "Network"
    assert(scope.readSnapshot(reference, new Array[Byte](64), () => false).isLeft)
  }

  private class WrappedChannel(delegate: SeekableByteChannel,
      afterRead: () => Unit = () => (), onClose: () => Unit = () => ()) extends SeekableByteChannel:
    def read(dst: ByteBuffer): Int =
      val result = delegate.read(dst)
      afterRead()
      result
    def write(src: ByteBuffer): Int = throw new AssertionError("Preview source must never be written")
    def position(): Long = delegate.position()
    def position(value: Long): SeekableByteChannel = { delegate.position(value); this }
    def size(): Long = delegate.size()
    def truncate(size: Long): SeekableByteChannel = throw new AssertionError("Preview source must never be truncated")
    def isOpen(): Boolean = delegate.isOpen()
    def close(): Unit = try delegate.close() finally onClose()
