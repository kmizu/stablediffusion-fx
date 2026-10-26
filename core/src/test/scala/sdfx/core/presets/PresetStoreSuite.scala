package sdfx.core.presets

import java.io.IOException
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}
import java.security.MessageDigest
import java.util.UUID
import sdfx.core.TempDir

/** File-system assertions deliberately use independent byte fixtures, not a second store. */
class PresetStoreSuite extends munit.FunSuite with TempDir:
  private val limits = PresetLimits()

  private def record(name: String, id: String): PresetRecord =
    PresetRecord(UUID.fromString(id), name, Map.empty, None, Set.empty, Vector.empty, Vector.empty)

  private def collection(id: String, presets: PresetRecord*): PresetCollection =
    PresetCollection(id, Set.empty, Vector.empty, presets.toVector, Vector.empty)

  private def library(names: String*): PresetLibrary =
    PresetLibrary(1, Vector(collection("config-txt2img", names.zipWithIndex.map { (name, i) =>
      record(name, f"00000000-0000-0000-0000-${i + 1}%012d")
    }*)), Vector.empty)

  private def bytes(path: Path): Vector[Byte] = Files.readAllBytes(path).toVector

  private def sha256(value: Array[Byte]): String =
    MessageDigest.getInstance("SHA-256").digest(value).map(b => f"${b & 0xff}%02x").mkString

  private def saved(store: PresetStore, expected: PresetRevision, next: PresetLibrary): LoadedPresets =
    store.save(expected, next).fold(e => fail(s"save failed: $e"), identity)

  tmp.test("first_load_is_empty_without_seeding_or_a_revision") { dir =>
    val store = new PresetStore(dir, limits)
    val loaded = store.load().fold(e => fail(s"load failed: $e"), identity)
    assertEquals(loaded.revision, PresetRevision(None))
    assertEquals(loaded.library.collections.flatMap(_.presets), Vector.empty)
    assert(!Files.exists(store.libraryPath))
    assert(!Files.exists(dir.resolve("library-v1.json.bak")))
  }

  tmp.test("stale_revision_preserves_external_edit") { dir =>
    val store = new PresetStore(dir, limits)
    val a = saved(store, PresetRevision(None), library("A"))
    val external = "{ \"external\": 9007199254740993 }\n".getBytes(UTF_8)
    Files.write(store.libraryPath, external)
    assert(store.save(a.revision, library("stale candidate")).isLeft)
    assertEquals(bytes(store.libraryPath), external.toVector)
  }

  tmp.test("corrupt_file_never_seeded_or_overwritten") { dir =>
    val store = new PresetStore(dir, limits)
    val a = saved(store, PresetRevision(None), library("A"))
    val aBytes = bytes(store.libraryPath)
    val b = saved(store, a.revision, library("B"))
    val corrupt = "{\"collections\":[".getBytes(UTF_8)
    Files.write(store.libraryPath, corrupt)
    assert(store.load().isLeft)
    assert(store.save(b.revision, library("C")).isLeft)
    assertEquals(bytes(store.libraryPath), corrupt.toVector)
    assertEquals(bytes(dir.resolve("library-v1.json.bak")), aBytes)
  }

  tmp.test("oversize_existing_library_is_rejected_without_replacement") { dir =>
    val store = new PresetStore(dir, limits)
    Files.write(store.libraryPath, Array.fill[Byte](4194305)(' '.toByte))
    assert(store.load().isLeft)
    assert(store.save(PresetRevision(None), library("A")).isLeft)
    assertEquals(Files.size(store.libraryPath), 4194305L)
  }

  tmp.test("missing_after_load_is_conflict") { dir =>
    val store = new PresetStore(dir, limits)
    val a = saved(store, PresetRevision(None), library("A"))
    Files.delete(store.libraryPath)
    assert(store.load().isLeft)
    assert(store.save(a.revision, library("B")).isLeft)
    assert(!Files.exists(store.libraryPath))
  }

  tmp.test("delete_last_is_empty_library") { dir =>
    val store = new PresetStore(dir, limits)
    val a = saved(store, PresetRevision(None), library("A"))
    val empty = saved(store, a.revision, library())
    assertEquals(empty.library.collections.flatMap(_.presets), Vector.empty)
    assertEquals(store.load().toOption.get.library.collections.flatMap(_.presets), Vector.empty)
    assert(Files.exists(store.libraryPath))
  }

  tmp.test("same_name_other_collection_survives") { dir =>
    val first = record("Same", "00000000-0000-0000-0000-000000000001")
    val second = record("Same", "00000000-0000-0000-0000-000000000002")
    val doc = PresetLibrary(1, Vector(collection("config-txt2img", first),
      collection("config-img2img", second)), Vector.empty)
    val store = new PresetStore(dir, limits)
    saved(store, PresetRevision(None), doc)
    val got = store.load().toOption.get.library
    assertEquals(got.collections.map(_.presets.head.name), Vector("Same", "Same"))
    assertEquals(got.collections.map(_.presets.head.id), Vector(first.id, second.id))
  }

  tmp.test("names_are_data_not_paths") { dir =>
    val store = new PresetStore(dir, limits)
    saved(store, PresetRevision(None), library("subdir/../outside"))
    assertEquals(store.load().toOption.get.library.collections.head.presets.head.name, "subdir/../outside")
    assertEquals(store.libraryPath, dir.resolve("library-v1.json"))
    assert(!Files.exists(dir.resolve("outside")))
  }

  tmp.test("restore_preserves_rejected_bytes") { dir =>
    val store = new PresetStore(dir, limits)
    val a = saved(store, PresetRevision(None), library("A"))
    val aBytes = bytes(store.libraryPath)
    saved(store, a.revision, library("B"))
    val corrupt = "{invalid original bytes 9007199254740993}".getBytes(UTF_8)
    Files.write(store.libraryPath, corrupt)
    val current = PresetRevision(Some(sha256(corrupt)))
    val restored = store.restoreBackup(current).fold(e => fail(s"restore failed: $e"), identity)
    assertEquals(bytes(store.libraryPath), aBytes)
    assertEquals(restored.revision.sha256, Some(sha256(aBytes.toArray)))
    val rejected = Files.list(dir)
    val files = try rejected.toArray.map(_.asInstanceOf[Path]).filter(
      p => p.getFileName.toString.contains(".rejected-") && p.getFileName.toString.endsWith(".json")).toVector
    finally rejected.close()
    assertEquals(files.size, 1)
    assertEquals(bytes(files.head), corrupt.toVector)
  }

  tmp.test("stale_restore_revision_preserves_later_external_edit") { dir =>
    val store = new PresetStore(dir, limits)
    val a = saved(store, PresetRevision(None), library("A"))
    saved(store, a.revision, library("B"))
    val firstCorrupt = "{first broken}".getBytes(UTF_8)
    Files.write(store.libraryPath, firstCorrupt)
    val previewRevision = PresetRevision(Some(sha256(firstCorrupt)))
    val laterExternal = "{second broken, user edited}".getBytes(UTF_8)
    Files.write(store.libraryPath, laterExternal)
    assert(store.restoreBackup(previewRevision).isLeft)
    assertEquals(bytes(store.libraryPath), laterExternal.toVector)
  }

  tmp.test("restore_aborts_if_rejected_bytes_cannot_be_preserved") { dir =>
    val normal = new PresetStore(dir, limits)
    val a = saved(normal, PresetRevision(None), library("A"))
    val aBytes = bytes(normal.libraryPath)
    saved(normal, a.revision, library("B"))
    val corrupt = "{broken and must not disappear}".getBytes(UTF_8)
    Files.write(normal.libraryPath, corrupt)
    val failing = new PresetStore(dir, limits, new PresetFileOps:
      def writeForced(path: Path, value: Array[Byte]): Unit =
        if value.toVector == corrupt.toVector then throw new IOException("injected rejected-copy failure")
        PresetFileOps.nio.writeForced(path, value)
      def atomicReplace(from: Path, to: Path): Unit = PresetFileOps.nio.atomicReplace(from, to)
    )
    assert(failing.restoreBackup(PresetRevision(Some(sha256(corrupt)))).isLeft)
    assertEquals(bytes(normal.libraryPath), corrupt.toVector)
    assertEquals(bytes(dir.resolve("library-v1.json.bak")), aBytes)
  }

  tmp.test("failed_atomic_move_preserves_original_and_never_backs_up_failed_candidate") { dir =>
    val normal = new PresetStore(dir, limits)
    val a = saved(normal, PresetRevision(None), library("A"))
    val aBytes = bytes(normal.libraryPath)
    val b = saved(normal, a.revision, library("B"))
    val bBytes = bytes(normal.libraryPath)
    val failing = new PresetStore(dir, limits, new PresetFileOps:
      def writeForced(path: Path, value: Array[Byte]): Unit = PresetFileOps.nio.writeForced(path, value)
      def atomicReplace(from: Path, to: Path): Unit =
        if to == normal.libraryPath then throw new IOException("injected target atomic move failure")
        PresetFileOps.nio.atomicReplace(from, to)
    )
    assert(failing.save(b.revision, library("C")).isLeft)
    assertEquals(bytes(normal.libraryPath), bBytes)
    val backup = bytes(dir.resolve("library-v1.json.bak"))
    // A backup may advance from A to the good pre-write B before the C move fails.
    assert(backup == aBytes || backup == bBytes)
  }

  tmp.test("failed_temp_write_preserves_original_and_backup") { dir =>
    val normal = new PresetStore(dir, limits)
    val a = saved(normal, PresetRevision(None), library("A"))
    val aBytes = bytes(normal.libraryPath)
    val b = saved(normal, a.revision, library("B"))
    val bBytes = bytes(normal.libraryPath)
    val backupBytes = bytes(dir.resolve("library-v1.json.bak"))
    val failing = new PresetStore(dir, limits, new PresetFileOps:
      def writeForced(path: Path, value: Array[Byte]): Unit = throw new IOException("injected forced write failure")
      def atomicReplace(from: Path, to: Path): Unit = PresetFileOps.nio.atomicReplace(from, to)
    )
    assert(failing.save(b.revision, library("C")).isLeft)
    assertEquals(bytes(normal.libraryPath), bBytes)
    assertEquals(bytes(dir.resolve("library-v1.json.bak")), backupBytes)
    assertEquals(backupBytes, aBytes)
  }

  tmp.test("failed_backup_atomic_move_preserves_original_and_old_backup") { dir =>
    val normal = new PresetStore(dir, limits)
    val a = saved(normal, PresetRevision(None), library("A"))
    val aBytes = bytes(normal.libraryPath)
    val b = saved(normal, a.revision, library("B"))
    val bBytes = bytes(normal.libraryPath)
    val failing = new PresetStore(dir, limits, new PresetFileOps:
      def writeForced(path: Path, value: Array[Byte]): Unit = PresetFileOps.nio.writeForced(path, value)
      def atomicReplace(from: Path, to: Path): Unit =
        if to == dir.resolve("library-v1.json.bak") then throw new IOException("injected backup atomic move failure")
        PresetFileOps.nio.atomicReplace(from, to)
    )
    assert(failing.save(b.revision, library("C")).isLeft)
    assertEquals(bytes(normal.libraryPath), bBytes)
    assertEquals(bytes(dir.resolve("library-v1.json.bak")), aBytes)
  }

  tmp.test("symlink_or_reparse_target_refused_when_host_supports_links") { dir =>
    val outside = dir.resolve("linked-target.json")
    Files.write(outside, PresetJson.writeNative(library("A"), limits).toOption.get)
    val store = new PresetStore(dir, limits)
    try
      val created = try
        Files.createSymbolicLink(store.libraryPath, outside)
        true
      catch case _: UnsupportedOperationException | _: java.nio.file.FileSystemException | _: SecurityException => false
      assume(created, "host cannot create a symbolic link; junction/reparse coverage needs a capable Windows host")
      val original = bytes(outside)
      assert(store.load().isLeft)
      assert(store.save(PresetRevision(None), library("A")).isLeft)
      assertEquals(bytes(outside), original)
      Files.delete(store.libraryPath)
      val lockTarget = dir.resolve("linked-lock-target")
      Files.writeString(lockTarget, "lock target")
      val lockPath = dir.resolve(".library-v1.lock")
      Files.deleteIfExists(lockPath)
      Files.createSymbolicLink(lockPath, lockTarget)
      assert(store.save(PresetRevision(None), library("A")).isLeft)
      assertEquals(Files.readString(lockTarget), "lock target")
      Files.delete(dir.resolve(".library-v1.lock"))
      val first = saved(store, PresetRevision(None), library("A"))
      val second = saved(store, first.revision, library("B"))
      val originalTarget = bytes(outside)
      val backupPath = dir.resolve("library-v1.json.bak")
      Files.delete(backupPath)
      Files.createSymbolicLink(backupPath, outside)
      assert(store.save(second.revision, library("C")).isLeft)
      assertEquals(bytes(outside), originalTarget)
    finally
      Files.deleteIfExists(store.libraryPath)
      Files.deleteIfExists(dir.resolve(".library-v1.lock"))
      Files.deleteIfExists(dir.resolve("library-v1.json.bak"))
  }
