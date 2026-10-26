package sdfx.core

import sdfx.core.model.*
import java.nio.file.Files

class StyleLibrarySuite extends munit.FunSuite with TempDir:
  tmp.test("atomic save rejects stale snapshots and preserves an external editor's exact bytes") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    val doc = library.load().append(Map("name" -> "A", "prompt" -> "one"))
    val saved = library.save(doc)
    Files.writeString(saved.path, "name,prompt,negative_prompt\nExternal,manual,n\n")
    intercept[StyleConflict](library.save(saved.edit(saved.entries.head.id, Map("prompt" -> "stale"))))
    assertEquals(Files.readString(saved.path), "name,prompt,negative_prompt\nExternal,manual,n\n")
    assertEquals(library.load().promptStyles.head.name, "External")
  }

  tmp.test("category moves merge prefixed names into live master, remove empty files and retain notes") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    val master = library.save(library.load().append(Map("name" -> "A", "prompt" -> "p", "notes" -> "note")))
    library.moveToCategory(master, Seq(master.entries.head.id), "Artists")
    assertEquals(Styles.load(library.master).map(_.name), Vector("Artists::A"))
    val category = library.loadCategory("Artists")
    assertEquals(category.value(category.entries.head, "notes"), "note")
    library.saveCategory("Artists", category.delete(category.entries.map(_.id)))
    assert(!Files.exists(library.categoryFile("Artists")))
    assertEquals(Styles.load(library.master), Vector.empty)
    intercept[IllegalArgumentException](library.categoryFile("../escape"))
  }

  tmp.test("extended presets have stable UUID identities, allow identical names, and restore only documented settings") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    val p = GenerationParams(prompt = "cat", negativePrompt = "bad", width = 640, height = 384,
      checkpoint = Some("Model A"), vae = Some("VAE A"), steps = 7, cfgScale = 4)
    val a = library.createExtended("Same", p)
    val b = library.createExtended("Same", p)
    assert(a.id != b.id)
    assertEquals(library.extended().map(_.id), Vector(a.id, b.id))
    assertEquals(library.extended(Some("Model A" -> "VAE A")).size, 2)
    assertEquals(library.extended(Some("Other" -> "VAE A")).size, 0)
    val restored = a.restore(GenerationParams(steps = 33, cfgScale = 9, seed = 99))
    assertEquals((restored.prompt, restored.negativePrompt, restored.width, restored.height), ("cat", "bad", 640, 384))
    assertEquals((restored.steps, restored.cfgScale, restored.seed), (33, 9.0, 99L))
    assertEquals(a.copy(vae = "").restore(p).vae, Some("Automatic"))
    assertEquals(a.copy(vae = "  ").restore(p).vae, Some("Automatic"))
    val blankVaeDoc = library.loadExtended()
    val blankRow = blankVaeDoc.entries.find(e => blankVaeDoc.value(e, "id") == a.id).get
    library.save(blankVaeDoc.edit(blankRow.id, Map("sd_vae" -> ""), uniqueNames = false))
    assertEquals(library.extended(Some("Model A" -> "Automatic")).map(_.id), Vector(a.id))
    assertEquals(library.extended(Some("Model A" -> "")).map(_.id), Vector(a.id))
    library.deleteExtended(a.id)
    assertEquals(library.extended().map(_.id), Vector(b.id))
  }

  tmp.test("pending-change backups wait ten minutes, keep24, and failed encrypted restore leaves master intact") { dir =>
    var clock = 0L
    val library = new StyleLibrary(dir.resolve("styles.csv"), () => clock)
    val saved = library.save(library.load().append(Map("name" -> "A", "prompt" -> "日本語")))
    assertEquals(library.backupIfDue(), None)
    clock = 600000L
    val first = library.backupIfDue().get
    assertEquals(Files.readAllBytes(first).toSeq, Files.readAllBytes(saved.path).toSeq)
    assertEquals(library.backupIfDue(), None)
    val encrypted = library.backupNow(Some("password"))
    val before = Files.readAllBytes(saved.path).toSeq
    intercept[IllegalArgumentException](library.restore(encrypted, Some("wrong")))
    assertEquals(Files.readAllBytes(saved.path).toSeq, before)
    (1 to 26).foreach { _ => clock += 600000; library.backupNow() }
    assertEquals(library.backups().size, 24)
    library.restore(library.backups().head)
    assertEquals(Styles.load(saved.path).head.prompt, "日本語")
  }

  tmp.test("moving between category mirrors removes the old entry instead of allowing later resurrection") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    library.saveCategory("Artists", library.loadCategory("Artists").append(Map("name" -> "A", "prompt" -> "old")))
    val master = library.load()
    library.moveToCategory(master, master.entries.map(_.id), "Other")
    assertEquals(Styles.load(library.master).map(_.name), Vector("Other::A"))
    assertEquals(library.loadCategory("Artists").promptStyles, Vector.empty)
    assert(!Files.exists(library.categoryFile("Artists")))
    assertEquals(library.loadCategory("Other").promptStyles.map(_.name), Vector("A"))
  }

  tmp.test("restoring a master backup synchronizes category mirrors before any later category edit") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    library.saveCategory("Artists", library.loadCategory("Artists").append(Map("name" -> "A", "prompt" -> "old")))
    val backup = library.backupNow()
    val current = library.loadCategory("Artists")
    library.saveCategory("Artists", current.edit(current.entries.head.id, Map("prompt" -> "new")))
    library.restore(backup)
    val restored = library.loadCategory("Artists")
    assertEquals(restored.promptStyles.head.prompt, "old")
    library.saveCategory("Artists", restored)
    assertEquals(Styles.load(library.master).head.prompt, "old")
  }

  tmp.test("moving an unsaved reordered category draft moves the selected identity, not its old ordinal neighbour") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    library.saveCategory("Artists", library.loadCategory("Artists").append(Map("name" -> "A", "prompt" -> "alpha"))
      .append(Map("name" -> "B", "prompt" -> "beta")))
    val category = library.loadCategory("Artists")
    val draft = category.reorder(category.entries.reverse.map(_.id))
    val selected = draft.entries.find(e => draft.value(e, "name") == "B").get
    library.moveToCategory(draft, Seq(selected.id), "Other")
    assertEquals(Styles.load(library.master).map(_.name).toSet, Set("Artists::A", "Other::B"))
    assertEquals(library.loadCategory("Artists").promptStyles.map(_.name), Vector("A"))
    assertEquals(library.loadCategory("Other").promptStyles.map(_.prompt), Vector("beta"))
  }

  tmp.test("category moves preserve a foreign text column when canonical prompt also exists") { dir =>
    val file = dir.resolve("styles.csv")
    Files.writeString(file, "name,prompt,text,negative_prompt\nA,actual prompt,foreign payload,n\n")
    val library = new StyleLibrary(file)
    val original = library.load()
    library.moveToCategory(original, Seq(original.entries.head.id), "Artists")
    val master = library.load()
    val category = library.loadCategory("Artists")
    assertEquals(master.value(master.entries.head, "text"), "foreign payload")
    assertEquals(category.value(category.entries.head, "text"), "foreign payload")
    assertEquals(master.promptStyles.head.prompt, "actual prompt")
  }

  tmp.test("an unnamed category row with prompt or notes is retained instead of being mistaken for an empty file") { dir =>
    val library = new StyleLibrary(dir.resolve("styles.csv"))
    library.saveCategory("Drafts", library.loadCategory("Drafts").append(Map("name" -> "", "prompt" -> "keep this draft", "notes" -> "not empty")))
    assert(Files.exists(library.categoryFile("Drafts")))
    val stored = library.loadCategory("Drafts")
    assertEquals(stored.value(stored.entries.head, "prompt"), "keep this draft")
    library.pruneEmptyCategories()
    assert(Files.exists(library.categoryFile("Drafts")))
    val backup = library.backupNow()
    library.restore(backup)
    assertEquals(library.loadCategory("Drafts").value(library.loadCategory("Drafts").entries.head, "prompt"), "keep this draft")
  }
