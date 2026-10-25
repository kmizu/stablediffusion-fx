package sdfx.plugins

import sdfx.core.infotext.Infotext
import sdfx.core.model.{GenerationParams, LoraRef}
import sdfx.core.plugins.api.*

class XyzGridSuite extends munit.FunSuite:
  val base = GenerationParams(prompt = "a cat, in a hat, at night", seed = 42, width = 64, height = 64)
  def vals(kv: (String, String)*) = Values(Values.defaults(new XyzGridPlugin().controls).map ++ kv)

  test("job count and value application for X and Y") {
    val ctx = FakeContext()
    val out = new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "10-30 (+10)", "y_type" -> "CFG Scale", "y_values" -> "5, 7"), ctx)
    assertEquals(ctx.generated.size, 6)
    assertEquals(ctx.generated.map(p => (p.steps, p.cfgScale)).toSet, Set(10, 20, 30).flatMap(s => Set(5.0, 7.0).map(c => (s, c))))
    assertEquals(out.grids.size, 1)
    assertEquals(out.images, Nil) // sub images off by default
    assert(ctx.generated.forall(_.seed == 42))
  }

  test("grid dimensions include the legend; without legend it is exactly cells") {
    val ctx = FakeContext()
    new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "1-3", "y_type" -> "Seed", "y_values" -> "1,2"), ctx)
    val g = ctx.saved.head._2
    assert(g.getWidth > 3 * 64 && g.getHeight > 2 * 64, s"${g.getWidth}x${g.getHeight}")
    val ctx2 = FakeContext()
    new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "1-3", "y_type" -> "Seed", "y_values" -> "1,2", "draw_legend" -> "false"), ctx2)
    val g2 = ctx2.saved.head._2
    assertEquals((g2.getWidth, g2.getHeight), (192, 128))
    val ctx3 = FakeContext()
    new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "1-3", "draw_legend" -> "false", "margin" -> "8"), ctx3)
    assertEquals(ctx3.saved.head._2.getWidth, 3 * (64 + 8))
  }

  test("checkpoint axis is iterated outermost even on X") {
    val ctx = FakeContext()
    new XyzGridPlugin().run(
      base,
      vals("x_type" -> "Checkpoint name", "x_values" -> "A, B", "y_type" -> "Steps", "y_values" -> "1-3", "z_type" -> "CFG Scale", "z_values" -> "4,5"),
      ctx
    )
    val ckpts = ctx.generated.map(_.checkpoint.get)
    assertEquals(ckpts, Seq.fill(6)("A") ++ Seq.fill(6)("B"))
    // only one model switch
    assertEquals(ckpts.sliding(2).count(s => s(0) != s(1)), 1)
  }

  test("default order is Z outermost, then Y, then X") {
    assertEquals(
      XyzGridPlugin.order((0, 0, 0), (2, 2, 2)).take(3),
      Seq(XyzGridPlugin.Cell(0, 0, 0), XyzGridPlugin.Cell(1, 0, 0), XyzGridPlugin.Cell(0, 1, 0))
    )
    assertEquals(XyzGridPlugin.order((0, 3, 0), (2, 2, 2)).take(2), Seq(XyzGridPlugin.Cell(0, 0, 0), XyzGridPlugin.Cell(1, 0, 0)))
    assertEquals(XyzGridPlugin.order((0, 3, 0), (2, 2, 2)).map(_.y), Seq.fill(4)(0) ++ Seq.fill(4)(1))
  }

  test("Z axis yields combined grid plus optional sub grids; infotext names the axes") {
    val ctx = FakeContext()
    val out = new XyzGridPlugin().run(
      base,
      vals("x_type" -> "Steps", "x_values" -> "1,2", "z_type" -> "Sampler", "z_values" -> "Euler, Euler a", "include_sub_grids" -> "true", "include_sub_images" -> "true"),
      ctx
    )
    assertEquals(out.grids.size, 3)
    assertEquals(out.images.size, 4)
    assert(ctx.saved.forall(_._1 == "grids"))
    val info = Infotext.parse(ctx.saved.head._4)
    assertEquals(info.get("Script"), Some("X/Y/Z plot"))
    assertEquals(info.get("X Type"), Some("Steps"))
    assertEquals(info.get("X Values"), Some("1, 2"))
    assertEquals(info.get("Z Type"), Some("Sampler"))
    assertEquals(info.get("Y Type"), None)
    val combined = ctx.saved.head._2
    assert(combined.getWidth > ctx.saved(1)._2.getWidth * 2 - 1)
  }

  test("prompt S/R replaces the first value in prompt and negative prompt") {
    val ctx = FakeContext()
    new XyzGridPlugin().run(base.copy(negativePrompt = "ugly cat"), vals("x_type" -> "Prompt S/R", "x_values" -> "cat, dog, \"big, red fox\""), ctx)
    assertEquals(ctx.generated.map(_.prompt), Seq("a cat, in a hat, at night", "a dog, in a hat, at night", "a big, red fox, in a hat, at night"))
    assertEquals(ctx.generated.map(_.negativePrompt).last, "ugly big, red fox")
  }

  test("prompt S/R fails when the search term is missing") {
    intercept[IllegalArgumentException](new XyzGridPlugin().run(base, vals("x_type" -> "Prompt S/R", "x_values" -> "zebra, horse"), FakeContext()))
  }

  test("prompt order permutes tokens") {
    val ctx = FakeContext()
    new XyzGridPlugin().run(base, vals("x_type" -> "Prompt order", "x_values" -> "cat, hat, night"), ctx)
    assertEquals(ctx.generated.size, 6)
    assert(ctx.generated.map(_.prompt).contains("a night, in a hat, at cat"))
    assertEquals(XyzGridPlugin.applyOrder("a cat, in a hat, at night", Seq("hat", "cat", "night")), "a hat, in a cat, at night")
  }

  test("seed -1 is fixed per cell unless 'keep -1' is set") {
    val ctx = FakeContext()
    new XyzGridPlugin().run(base.copy(seed = -1), vals("x_type" -> "Steps", "x_values" -> "1,2"), ctx)
    assert(ctx.generated.forall(_.seed >= 0))
    assertEquals(ctx.generated.map(_.seed).distinct.size, 1)
    val ctx2 = FakeContext()
    new XyzGridPlugin().run(base, vals("x_type" -> "Seed", "x_values" -> "-1, 5", "keep_random_seed" -> "true"), ctx2)
    assertEquals(ctx2.generated.map(_.seed), Seq(-1L, 5L))
    val ctx3 = FakeContext()
    new XyzGridPlugin().run(base, vals("x_type" -> "Seed", "x_values" -> "-1, 5"), ctx3)
    assert(ctx3.generated.head.seed >= 0)
  }

  test("size, LoRA weight, VAE, denoising, hires axes") {
    val x = new XyzGridPlugin()
    def ap(n: String, v: String, p: GenerationParams = base) = x.axes.find(_.name == n).get.apply(p, v)
    assertEquals(ap("Size", "768x512").width, 768)
    val lp = base.copy(prompt = "girl <lora:foo:0.8> <lora:bar:1>", loras = Seq(LoraRef("foo", 0.8)))
    val l = ap("LoRA weight", "foo:0.3", lp)
    assertEquals(l.prompt, "girl <lora:foo:0.3> <lora:bar:1>")
    assertEquals(l.loras, Seq(LoraRef("foo", 0.3)))
    assertEquals(ap("LoRA weight", "baz:0.5").prompt, "a cat, in a hat, at night <lora:baz:0.5>")
    assertEquals(ap("VAE", "Automatic").vae, None)
    assertEquals(ap("VAE", "x.vae.pt").vae, Some("x.vae.pt"))
    assertEquals(ap("Denoising", "0.3").hires.denoisingStrength, 0.3)
    assertEquals(ap("Hires upscale", "1.5").hires.scale, 1.5)
    assertEquals(ap("Hires steps", "12").hires.secondPassSteps, 12)
    assertEquals(ap("Clip skip", "2").clipSkip, 2)
    intercept[IllegalArgumentException](x.run(base, vals("x_type" -> "Size", "x_values" -> "big"), FakeContext()))
  }

  test("extra axes from other plugins are offered and applied through process") {
    val all = StandardPlugins.withOwnAxes
    val xyz = all.collectFirst { case x: XyzGridPlugin => x }.get
    assert(xyz.axes.exists(_.name == "FreeU B1"))
    val choice = xyz.controls.collectFirst { case c: Control.Choice if c.key == "x_type" => c }.get
    assert(choice.options.contains("DynThres Mimic Scale"))
    val ctx = FakeContext()
    xyz.run(base, vals("x_type" -> "FreeU B1", "x_values" -> "1.1, 1.2"), ctx)
    val freeu = new FreeUPlugin
    val processed = ctx.generated.map(p => freeu.process(p, Values.defaults(freeu.controls).updated("enabled", "true")))
    assertEquals(processed.map(_.hooks("freeu")("b1").num), Seq(1.1, 1.2))
    assertEquals(processed.map(_.hooks("freeu")("b2").num), Seq(1.02, 1.02))
  }

  test("interrupt (engine throws mid-cell) stops the plot and still saves the partial grid") {
    val ctx = FakeContext(interruptAfter = 2)
    val out = new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "1-5", "draw_legend" -> "false", "include_sub_images" -> "true"), ctx)
    assertEquals(ctx.generated.size, 2)
    assertEquals(ctx.interruptedCalls, 1) // exactly one throwing generate, then the loop stopped
    assertEquals(out.grids.size, 1)
    assertEquals(out.images.size, 2)
    assertEquals(ctx.saved.size, 1)
    assertEquals(ctx.saved.head._2.getWidth, 5 * 64) // full layout, missing cells left empty
  }

  test("interrupt before the first cell returns nothing without throwing") {
    val ctx = FakeContext(interruptAfter = 0)
    val out = new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "1-3"), ctx)
    assertEquals(out, JobOutput(Nil))
  }

  test("non-interrupt engine errors still propagate") {
    val ctx = new FakeContext:
      override def generate(p: GenerationParams) = throw sdfx.core.protocol.EngineError("oom", "out of memory")
    intercept[sdfx.core.protocol.EngineError](new XyzGridPlugin().run(base, vals("x_type" -> "Steps", "x_values" -> "1-3"), ctx))
  }

  test("batch cells use the first image in the grid; the rest are sub images (A1111)") {
    val ctx = FakeContext()
    val out = new XyzGridPlugin().run(
      base.copy(batchSize = 4),
      vals("x_type" -> "Steps", "x_values" -> "1, 2", "draw_legend" -> "false", "include_sub_images" -> "true"),
      ctx
    )
    val g = ctx.saved.head._2
    assertEquals((g.getWidth, g.getHeight), (128, 64))
    assertEquals(out.images.size, 8)
    val first = javax.imageio.ImageIO.read(out.images.head.path.toFile)
    assertEquals(g.getRGB(10, 10), first.getRGB(10, 10))
  }
