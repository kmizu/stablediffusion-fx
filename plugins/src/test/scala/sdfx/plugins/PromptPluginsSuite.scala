package sdfx.plugins

import sdfx.core.image.Gif
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.plugins.prompt.DynamicPrompt

import java.nio.file.Files
import scala.util.Random

class PromptPluginsSuite extends munit.FunSuite:
  test("video scripts reject plaintext GIF before generating frames when session disallows it") {
    for plugin <- Seq(new SeedTravelPlugin, new ShiftAttentionPlugin, new SvdPlugin) do
      val ctx = new FakeContext():
        override def plaintextVideoAllowed: Boolean = false
      val error = intercept[IllegalStateException](plugin.run(GenerationParams(), Values.defaults(plugin.controls), ctx))
      assert(error.getMessage.contains("plaintext video"))
      assert(ctx.generated.isEmpty && ctx.engineCalls.isEmpty)
  }
  test("dynamic prompt hash injection is opt-in and includes negative wildcard sources"):
    val dir = Files.createTempDirectory("wildcard-injection")
    Files.writeString(dir.resolve("positive.txt"), "teapot\n")
    Files.writeString(dir.resolve("negative.txt"), "blurry\n")
    val plugin = new DynamicPromptsPlugin
    val values = defaults(plugin, "wildcard_dirs" -> dir.toString)
    val params = GenerationParams(prompt = "__positive__", negativePrompt = "__negative__", seed = 7)
    val off = plugin.expand(params, values)
    val on = plugin.expand(params, values, trackSources = true)
    assertEquals(on.head.prompt, off.head.prompt)
    assertEquals(on.head.negativePrompt, off.head.negativePrompt)
    assert(!off.head.extra.contains("Hashes"))
    assertEquals(ujson.read(on.head.extra("Hashes")).obj.keySet.toSet,
      Set("wildcards_file:positive", "wildcards_file:negative"))
  test("wildcard provenance records consumed bytes and survives cached reads"):
    val dir = Files.createTempDirectory("wildcard-provenance")
    val bytes = "red\nblue\n".getBytes(java.nio.charset.StandardCharsets.UTF_8)
    Files.write(dir.resolve("colors.txt"), bytes)
    val wc = new DynamicPrompt.Wildcards(Seq(dir), trackSources = true)
    assertEquals(wc.lines("colors"), Vector("red", "blue"))
    val hash = java.security.MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 0xff}%02x").mkString.take(10)
    assertEquals(wc.sourceHashes, Map("wildcards_file:colors" -> hash))
    wc.clearTrace()
    assert(wc.sourceHashes.isEmpty)
    Files.writeString(dir.resolve("colors.txt"), "changed\n")
    assertEquals(wc.lines("colors"), Vector("red", "blue"))
    assertEquals(wc.sourceHashes, Map("wildcards_file:colors" -> hash))
  private def defaults(p: Plugin, extra: (String, String)*) = Values(Values.defaults(p.controls).map ++ extra.toMap)

  private lazy val wcDir =
    val d = Files.createTempDirectory("wildcards")
    Files.writeString(d.resolve("colors.txt"), "# comment\nred\nblue\n\ngreen\n")
    Files.createDirectories(d.resolve("animals"))
    Files.writeString(d.resolve("animals/cats.txt"), "tabby cat\n{black|white} cat\n")
    Files.writeString(d.resolve("animals/dogs.txt"), "shiba\n")
    Files.writeString(d.resolve("nested.txt"), "__colors__ hat\n")
    d
  private def wc = new DynamicPrompt.Wildcards(Seq(wcDir))

  test("variants, counts, separators, weights and escapes") {
    val rnd = new Random(1)
    assert(Set("a cat", "a dog").contains(DynamicPrompt.random("a {cat|dog}", rnd, wc)))
    val two = DynamicPrompt.random("{2$$x|y|z}", new Random(3), wc)
    assertEquals(two.split(", ").length, 2)
    assertEquals(two.split(", ").distinct.length, 2)
    val sep = DynamicPrompt.random("{2$$ and $$x|y}", new Random(3), wc)
    assert(sep == "x and y" || sep == "y and x", sep)
    val heavy = (0 until 200).map(i => DynamicPrompt.random("{100::a|0.01::b}", new Random(i), wc))
    assert(heavy.count(_ == "a") > 190)
    assertEquals(DynamicPrompt.random("""literal \{x\} and \| bar""", rnd, wc), "literal {x} and | bar")
    assertEquals(DynamicPrompt.random("no templates here", rnd, wc), "no templates here")
    val range = (0 until 50).map(i => DynamicPrompt.random("{1-3$$a|b|c}", new Random(i), wc).split(", ").length).toSet
    assert(range.subsetOf(Set(1, 2, 3)) && range.size > 1)
  }

  test("wildcards: files, comments, sub folders, globs, nesting, missing") {
    assert(Set("red", "blue", "green").contains(DynamicPrompt.random("__colors__", new Random(0), wc)))
    val nested = DynamicPrompt.random("__nested__", new Random(0), wc)
    assert(nested.endsWith(" hat") && !nested.contains("__"), nested)
    val glob = (0 until 30).map(i => DynamicPrompt.random("__animals/*__", new Random(i), wc)).toSet
    assert(glob.contains("shiba") && glob.exists(_.endsWith("cat")), glob.toString)
    assertEquals(DynamicPrompt.random("__missing__", new Random(0), wc), "__missing__")
    assertEquals(wc.names, Vector("animals/cats", "animals/dogs", "colors", "nested"))
  }

  test("combinatorial enumerates every combination in order, with a limit") {
    assertEquals(DynamicPrompt.combinatorial("{a|b} {1|2}", wc, 100), Vector("a 1", "a 2", "b 1", "b 2"))
    assertEquals(DynamicPrompt.combinatorial("__colors__", wc, 100), Vector("red", "blue", "green"))
    assertEquals(DynamicPrompt.combinatorial("{2$$x|y|z}", wc, 100), Vector("x, y", "x, z", "y, z"))
    assertEquals(DynamicPrompt.combinatorial("{a|b|c} {a|b|c}", wc, 4).size, 4)
  }

  test("dynamic prompts script: one generation per image, own prompt and seed each") {
    val ctx = new FakeContext()
    val p = new DynamicPromptsPlugin
    val v = defaults(p, "wildcard_dirs" -> wcDir.toString)
    val out = p.run(GenerationParams(prompt = "a __colors__ {cat|dog}", seed = 10, batchSize = 2, nIter = 2), v, ctx)
    assertEquals(out.images.size, 4)
    assertEquals(ctx.generated.map(_.seed), Seq(10L, 11L, 12L, 13L))
    assert(ctx.generated.forall(g => g.batchSize == 1 && !g.prompt.contains("{") && !g.prompt.contains("__")))
    val comb = p.expand(GenerationParams(prompt = "{a|b} __colors__", seed = 5), defaults(p, "mode" -> "combinatorial",
      "wildcard_dirs" -> wcDir.toString, "fixed_seed" -> "true"))
    assertEquals(comb.map(_.prompt), Seq("a red", "a blue", "a green", "b red", "b blue", "b green"))
    assert(comb.forall(_.seed == 5))
  }

  test("prompt format, filter, negative weight, NovelAI conversion") {
    assertEquals(PromptText.format("  a ,, b,  c , a ,(  x  ) ", dedupe = true), "a, b, c, (x)")
    val f = new PromptFilterPlugin
    val vf = defaults(f, "banned" -> "ugly", "replace" -> "girl => woman\nfoo=>bar")
    assertEquals(f.filter("girl, ugly face, foo, girl", vf), "woman, bar")
    val n = new NegativePromptWeightPlugin
    assertEquals(n.process(GenerationParams(negativePrompt = "lowres, bad"), defaults(n, "weight" -> "1.5")).negativePrompt,
      "(lowres, bad:1.5)")
    assertEquals(n.process(GenerationParams(negativePrompt = "x"), defaults(n)).negativePrompt, "x")
    assertEquals(PromptText.fromNovelAI("{cat}, [dog], {{big}}"), "(cat:1.05), (dog:0.952), (big:1.103)")
    assertEquals(PromptText.fromNovelAI("plain { open"), "plain { open")
  }

  test("randomize is reproducible per seed and stays in its ranges") {
    val r = new RandomizePlugin
    val v = defaults(r, "enabled" -> "true", "samplers" -> "A, B", "steps" -> "10-12", "cfg" -> "3-4", "sizes" -> "512x768")
    val a = r.process(GenerationParams(seed = 9), v)
    assertEquals(a, r.process(GenerationParams(seed = 9), v))
    assert(Set("A", "B").contains(a.samplerName))
    assert(a.steps >= 10 && a.steps <= 12 && a.cfgScale >= 3 && a.cfgScale <= 4)
    assertEquals((a.width, a.height), (512, 768))
  }

  test("style variables substitute saved styles") {
    val s = new StyleVariablesPlugin(() => Seq(sdfx.core.model.PromptStyle("Cine", "cinematic, film grain", "cartoon")))
    val p = s.process(GenerationParams(prompt = "a cat, {style:cine}", negativePrompt = "$Cine"), Values.empty)
    assertEquals(p.prompt, "a cat, cinematic, film grain")
    assertEquals(p.negativePrompt, "cartoon")
    assertEquals(s.substitute("{style:unknown}", false), "{style:unknown}")
  }

  test("test my prompt, scramble, tweak weights produce labelled grids") {
    val ctx = new FakeContext()
    val t = new TestMyPromptPlugin
    val out = t.run(GenerationParams(prompt = "a, b, c", seed = 1), defaults(t), ctx)
    assertEquals(ctx.generated.map(_.prompt), Seq("a, b, c", "b, c", "a, c", "a, b"))
    assertEquals(out.grids.size, 1)
    val sc = new ScramblePromptsPlugin
    val v = defaults(sc, "count" -> "3", "keep_first" -> "1", "drop" -> "0")
    assert(sc.variation("first, b, c, d", v, new Random(1)).startsWith("first, "))
    assertEquals(sc.variation("first, b, c, d", v, new Random(1)).split(", ").toSet, Set("first", "b", "c", "d"))
    val tw = new TweakWeightsPlugin
    val w = tw.variation("cat, dog, <lora:x:1>", defaults(tw, "pattern" -> "cat", "min" -> "1.2", "max" -> "1.2"), new Random(0))
    assertEquals(w, "(cat:1.2), dog, <lora:x:1>")
  }

  test("lora queue and batch checkpoints") {
    val ctx = new FakeContext()
    val lq = new LoraQueuePlugin
    lq.run(GenerationParams(prompt = "a cat", seed = 3), defaults(lq, "loras" -> "detail\nstyle_x", "weight" -> "0.7"), ctx)
    assertEquals(ctx.generated.map(_.prompt), Seq("a cat <lora:detail:0.7>", "a cat <lora:style_x:0.7>"))
    val bc = new BatchCheckpointsPromptsPlugin
    val items = bc.parse("modelA | {prompt}, masterpiece\nmodelB", GenerationParams(prompt = "cat"))
    assertEquals(items.map(i => (i.params.checkpoint, i.params.prompt)), Seq((Some("modelA"), "cat, masterpiece"), (Some("modelB"), "cat")))
  }

  test("seed travel slerps toward the next seed and writes a GIF") {
    val ctx = new FakeContext()
    val st = new SeedTravelPlugin
    val frames = st.frames(GenerationParams(seed = 1), defaults(st, "seeds" -> "5", "steps" -> "4"))
    assertEquals(frames.map(f => (f.params.seed, f.params.subseedStrength)).take(4), Seq((1L, 0.0), (1L, 0.25), (1L, 0.5), (1L, 0.75)))
    assertEquals(frames.last.params.seed, 5L)
    st.run(GenerationParams(seed = 1, width = 64, height = 64), defaults(st, "seeds" -> "5", "steps" -> "2"), ctx)
    val gif = ctx.logs.find(_.startsWith("Seed travel GIF: ")).map(l => java.nio.file.Paths.get(l.stripPrefix("Seed travel GIF: ")))
    assert(gif.isDefined)
    assertEquals(Gif.frameCount(gif.get), 3)
  }

  test("shift attention sweeps the weight range") {
    val sa = new ShiftAttentionPlugin
    assertEquals(sa.promptAt("a (cat:0.5~1.5), (dog:1~0)", 0.5), "a (cat:1.0), (dog:0.5)")
    val ctx = new FakeContext()
    sa.run(GenerationParams(prompt = "(cat:0~1)", seed = 2, width = 32, height = 32), defaults(sa, "frames" -> "3"), ctx)
    assertEquals(ctx.generated.map(_.prompt), Seq("(cat:0.0)", "(cat:0.5)", "(cat:1.0)"))
  }

  test("one button prompt generates varied, reproducible prompts") {
    val ob = new OneButtonPromptPlugin
    val v = defaults(ob)
    assertEquals(ob.generate(new Random(4), v), ob.generate(new Random(4), v))
    assert((0 until 20).map(i => ob.generate(new Random(i), v)).toSet.size > 15)
    assert(ob.generate(new Random(1), defaults(ob, "prefix" -> "PFX")).startsWith("PFX, "))
  }
