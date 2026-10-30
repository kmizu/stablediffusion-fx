package sdfx.plugins

import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import sdfx.plugins.prompt.{StylePile, StylePileCatalog}

class StylePileSuite extends munit.FunSuite:
  test("pinned catalog preserves complete source choices and modifiers"):
    assertEquals(StylePileCatalog.choices("Artists").size, 300)
    assertEquals(StylePileCatalog.choices("Art Movements").size, 127)
    assertEquals(StylePileCatalog.choices("Moods").size, 25)
    assertEquals(StylePileCatalog.literal("ResultNames").arr.size, 6)
    assertEquals(StylePileCatalog.literal("Preset").obj.size, 10)

  test("X then batch copy then Y ordering and seed reset match documented source"):
    val p = GenerationParams(prompt = "[X] [Y]", seed = 10, batchSize = 2, nIter = 7)
    val out = StylePile.expand(p, Values(Map("sequential" -> "cat\ndog", "subsequential" -> "red\nblue")))
    assertEquals(out.map(_.prompt), Seq("cat red", "cat blue", "cat red", "cat blue", "dog red", "dog blue", "dog red", "dog blue"))
    assertEquals(out.map(_.seed), Seq(10L, 11L, 12L, 13L, 10L, 11L, 12L, 13L))
    assert(out.forall(p => p.batchSize == 1 && p.nIter == 1))

  test("random substitutions are reproducible, replace all same markers, and do not touch negatives"):
    val p = GenerationParams(prompt = "[X] [A] [A] [B] [C]", negativePrompt = "[A]", seed = 7)
    val v = Values(Map("sequential" -> "[A]", "random_a" -> "cat\ndog", "random_b" -> "blue", "random_c" -> "hat"))
    val a = StylePile.expand(p, v)
    assertEquals(a, StylePile.expand(p, v))
    val words = a.head.prompt.split(" ")
    assertEquals(words.take(3).distinct.length, 1)
    assertEquals(a.head.negativePrompt, "[A],watermark,signature")

  test("image-type modifiers preserve negative composition and all weighted slots"):
    val v = Values(Map("image_type" -> "Vector Art", "artist1" -> "Artist One", "artist2" -> "Artist Two",
      "artist3" -> "Artist Three", "movement1" -> "Cubism", "movement2" -> "Impressionism", "movement3" -> "Surrealism"))
    val out = StylePile.expand(GenerationParams(prompt = "teapot", negativePrompt = "blur", seed = 1), v).head
    assert(out.prompt.startsWith("(Vector image:1.3) of teapot"))
    Seq("Artist One", "Artist Two", "Artist Three", "Cubism", "Impressionism", "Surrealism").foreach(s => assert(out.prompt.contains(s)))
    assertEquals(out.negativePrompt, "blur,(Watermark:1.5),(Text:1.3),watermark,signature")

  test("oversized sequence and invalid strength fail instead of truncating"):
    intercept[IllegalArgumentException](StylePile.expand(GenerationParams(prompt = "x", batchSize = 100, nIter = 1000), Values.empty))
    intercept[IllegalArgumentException](StylePile.expand(GenerationParams(prompt = "x"), Values(Map("type_strength" -> "NaN"))))


  test("all bundled source material matches pinned SHA256 provenance"):
    def bytes(name: String): Array[Byte] =
      val input = Option(getClass.getResourceAsStream("/stylepile/" + name)).getOrElse(fail(s"Missing resource: $name"))
      try input.readAllBytes() finally input.close()
    val manifest = ujson.read(new String(bytes("manifest.json"), java.nio.charset.StandardCharsets.UTF_8))
    val files = manifest("files").arr.filter(_("path").str != "scripts/StylePile.py")
    assertEquals(files.size, 517)
    files.foreach { entry =>
      val raw = bytes(entry("path").str)
      val actual = java.security.MessageDigest.getInstance("SHA-256").digest(raw).map(b => f"${b & 0xff}%02x").mkString
      assertEquals(actual, entry("sha256").str, entry("path").str)
    }


  test("noise placeholders keep source lengths and same-marker sharing"):
    val out = StylePile.expand(GenerationParams(prompt = "[xs] [s] [s] [m] [l] [xl]", seed = 7), Values.empty).head.prompt.split(" ")
    assertEquals(out.map(_.length).toSeq, Seq(6, 5, 5, 10, 5, 5, 10, 10))
    assert(out.head.matches("[0-9]{6}"))
    assertEquals(out(1), out(2))
    assert(out.forall(_.matches("[0-9A-Z]+")))

  test("count override off cycles X, preserves blank Y, and fixed seed stays fixed"):
    val out = StylePile.expand(GenerationParams(prompt = "[X] [Y]", seed = 9, nIter = 3), Values(Map(
      "sequential" -> "cat\ndog", "subsequential" -> "red\n\nblue", "override_count" -> "false", "increase_seed" -> "false")))
    assertEquals(out.map(_.prompt), Seq("cat red", "cat", "cat blue", "dog red", "dog", "dog blue", "cat red", "cat", "cat blue"))
    assert(out.forall(_.seed == 9))


  test("native script exposes complete visual choices and preserves editable artist slots"):
    import sdfx.core.plugins.api.{Control, PluginKind, Tab}
    val plugin = new StylePilePlugin
    assertEquals(plugin.kind, PluginKind.Script)
    assertEquals(plugin.tabs, Set(Tab.Txt2Img, Tab.Img2Img))
    assert(StandardPlugins.withOwnAxes.exists(_.id == "stylepile"))
    val visual = plugin.controls.collect { case choice: Control.Choice if choice.previewResources.nonEmpty => choice }
    assertEquals(visual.size, 9)
    val artist = visual.find(_.key == "artist1").get
    assertEquals(artist.options.size, 302)
    assertEquals(artist.previewResources.size, 300)
    assert(artist.allowCustom)
    assert(plugin.controls.map(_.key).distinct.size == plugin.controls.size)


  test("helper tools insert weighted, morph and alternating syntax without losing text"):
    import sdfx.plugins.prompt.StylePileTools
    assertEquals(StylePileTools.weight("red scarf", 1.3), "(red scarf:1.3)")
    assertEquals(StylePileTools.morph("cat", "dog", .5), "[cat:dog:0.5]")
    assertEquals(StylePileTools.bounce("cat", "dog"), "[cat|dog]")
    intercept[IllegalArgumentException](StylePileTools.morph("cat", "dog", Double.NaN))
    intercept[IllegalArgumentException](StylePileTools.weight("cat", 3))

  test("inspiration uses all three original lists and is reproducible for a fixed seed"):
    import sdfx.plugins.prompt.StylePileTools
    val result = StylePileTools.inspire(7)
    assertEquals(result, StylePileTools.inspire(7))
    val parts = result.stripSuffix(",").split(",")
    assertEquals(parts.length, 3)
    Seq("Subjects", "Actions", "Locations").zip(parts).foreach((group, part) =>
      assert(StylePileCatalog.inspiration(group).contains(part)))


  test("user catalog adds, overrides and hides choices without modifying bundled data"):
    import java.nio.file.Files
    val root = Files.createTempDirectory("stylepile-overrides")
    val artists = Files.createDirectories(root.resolve("Artists"))
    val original = StylePileCatalog.choices("Artists")
    Files.write(artists.resolve("My Artist.png"), Array[Byte](1, 2, 3))
    Files.write(artists.resolve(original.head + ".jpg"), Array[Byte](4, 5, 6))
    Files.writeString(artists.resolve("disabled.txt"), original(1) + "\n")
    val snapshot = StylePileCatalog.withOverrides(root)
    assert(snapshot.choices("Artists").contains("My Artist"))
    assert(!snapshot.choices("Artists").contains(original(1)))
    assertEquals(java.nio.file.Paths.get(snapshot.preview("Artists", original.head).get.toURI).toRealPath(), artists.resolve(original.head + ".jpg").toRealPath())
    assertEquals(StylePileCatalog.choices("Artists"), original)
    Files.write(artists.resolve("Later.png"), Array[Byte](0))
    assert(!snapshot.choices("Artists").contains("Later"), "queued catalog choice snapshot must not drift")

  test("ambiguous and reserved custom names are omitted with visible warnings"):
    import java.nio.file.Files
    val root = Files.createTempDirectory("stylepile-invalid")
    val moods = Files.createDirectories(root.resolve("Moods"))
    Seq("Duplicate.jpg", "Duplicate.png", "Random.png").foreach(name => Files.write(moods.resolve(name), Array[Byte](0)))
    val snapshot = StylePileCatalog.withOverrides(root)
    assert(!snapshot.choices("Moods").contains("Duplicate"))
    assert(!snapshot.choices("Moods").contains("Random"))
    assert(snapshot.warnings.size >= 2)

  test("random selection uses the same user catalog choices as the native form"):
    import java.nio.file.Files
    val root = Files.createTempDirectory("stylepile-custom-random")
    val moods = Files.createDirectories(root.resolve("Moods"))
    Files.writeString(moods.resolve("disabled.txt"), StylePileCatalog.choices("Moods").mkString("\n"))
    Files.write(moods.resolve("My Mood.png"), Array[Byte](0))
    val snapshot = StylePileCatalog.withOverrides(root)
    val result = StylePile.expand(GenerationParams(prompt = "teapot", seed = 7), Values(Map("mood" -> "Random")), snapshot)
    assert(result.head.prompt.contains("(My Mood:1.3)"))


  test("custom inspiration lists feed the same native suggestion helper"):
    import java.nio.file.Files
    import sdfx.plugins.prompt.StylePileTools
    val root = Files.createTempDirectory("stylepile-inspiration")
    val folder = Files.createDirectories(root.resolve("Inspiration"))
    Seq("Subjects" -> "teapot", "Actions" -> "floating", "Locations" -> "in space").foreach { (name, text) =>
      Files.writeString(folder.resolve(name + ".txt"), text)
    }
    assertEquals(StylePileTools.inspire(7, StylePileCatalog.withOverrides(root)), "teapot,floating,in space,")

  test("empty user-filtered random pool fails explicitly"):
    import java.nio.file.Files
    val root = Files.createTempDirectory("stylepile-empty-pool")
    val folder = Files.createDirectories(root.resolve("Moods"))
    Files.writeString(folder.resolve("disabled.txt"), StylePileCatalog.choices("Moods").mkString("\n"))
    intercept[IllegalArgumentException](StylePile.expand(GenerationParams(prompt = "teapot", seed = 7),
      Values(Map("mood" -> "Random")), StylePileCatalog.withOverrides(root)))


  test("painting negative modifiers do not merge into the user's last word"):
    val result = StylePile.expand(GenerationParams(prompt = "teapot", negativePrompt = "blur", seed = 7),
      Values(Map("image_type" -> "Painting"))).head
    assertEquals(result.negativePrompt, "blur,Low quality,Bad composition,Faded,(Photo:1.5),(Frame:1.3),watermark,signature")
