package sdfx.plugins

class ModelToolsSuite extends munit.FunSuite:
  test("output file names cannot leave the model folder"):
    for raw <- Seq("../../evil", "a\\b:c", "C:/x/y", "a*b?<c>|d") do
      val f = ModelTools.fileName(raw, "x")
      assert(!f.exists(c => "\\/:*?\"<>|".contains(c)), f)
      assert(f.endsWith(".safetensors") && !f.startsWith("."), f)
    assertEquals(ModelTools.fileName("", "A_x_B_0.5"), "A_x_B_0.5.safetensors")
    assertEquals(ModelTools.fileName("merged.safetensors", "x"), "merged.safetensors")
    assertEquals(ModelTools.fileName("...", "fb"), "fb.safetensors")
    assertEquals(ModelTools.stem("D:/m/foo.v2.safetensors"), "foo.v2")

  test("block presets and the model tools are registered"):
    assert(BlockPresets.names.contains("GRAD_V"))
    val ids = StandardPlugins.withOwnAxes.map(_.id).toSet
    assert(ids("model_merge") && ids("model_converter"))
