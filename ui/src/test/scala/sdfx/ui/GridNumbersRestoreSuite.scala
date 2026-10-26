package sdfx.ui

import sdfx.core.image.GridNumbers
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.Values
import sdfx.plugins.GridNumbersPlugin
import sdfx.ui.vm.{GenerationVM, JobBuilder, Mode, PluginsVM}

/** The same form view-model restoration used by MainWindow PNG restore and Send to img2img. */
class GridNumbersRestoreSuite extends munit.FunSuite:
  private def plugins(mode: Mode) = new PluginsVM(mode, Nil, Seq(new GridNumbersPlugin), _ => Values.empty)

  for mode <- Seq(Mode.Txt2Img, Mode.Img2Img); numbers <- Seq(false, true); xyz <- Seq(false, true) do
    test(s"$mode PNG restore replaces both grid flags ($numbers, $xyz), including explicit false, in the next native job") {
      val forms = plugins(mode)
      val form = forms.form("grid_numbers").get
      form.load(Values(Map("enabled" -> "false", "image_numbers" -> (!numbers).toString, "xyz_info" -> (!xyz).toString)))
      val vm = new GenerationVM(mode)
      val text = s"cat\nSteps: 6, Sampler: Euler, CFG scale: 7, Seed: 3, Size: 512x512, Grid image numbers: $numbers, Grid XYZ info: $xyz"
      vm.restoreInfotext(text)
      forms.restoreInfotext(text)
      assert(form.isActive)
      assertEquals(form.values.bool("image_numbers"), numbers)
      assertEquals(form.values.bool("xyz_info"), xyz)
      assert(vm.toParams.extra.isEmpty) // Configuration must come from the restored widgets, not hidden VM extras.
      val job = JobBuilder.build(vm, forms)
      assertEquals(job.alwaysOn.map(_._1.id), Seq("grid_numbers"))
      val restored = job.alwaysOn.foldLeft(job.params)((p, entry) => entry._1.process(p, entry._2))
      assertEquals(restored.extra(GridNumbers.NumbersKey), if numbers then "True" else "False")
      assertEquals(restored.extra(GridNumbers.XyzKey), if xyz then "True" else "False")
    }

  test("unrelated or malformed infotext does not overwrite current grid form values") {
    val forms = plugins(Mode.Txt2Img)
    val form = forms.form("grid_numbers").get
    form.load(Values(Map("enabled" -> "true", "image_numbers" -> "false", "xyz_info" -> "true")))
    val before = form.values
    forms.restoreInfotext("cat\nSteps: 6, Seed: 3")
    assertEquals(form.values, before)
    forms.restoreInfotext("cat\nSteps: 6, Seed: 3, Grid image numbers: nonsense, Grid XYZ info: unknown")
    assertEquals(form.values, before)
  }

  test("a single recognized grid flag resets the missing counterpart to its native default") {
    val forms = plugins(Mode.Txt2Img)
    val form = forms.form("grid_numbers").get
    form.load(Values(Map("enabled" -> "false", "image_numbers" -> "false", "xyz_info" -> "true")))
    forms.restoreInfotext("cat\nSteps: 6, Seed: 3, Grid image numbers: FALSE")
    assertEquals(form.values.map, Map("enabled" -> "true", "image_numbers" -> "false", "xyz_info" -> "false"))
    forms.restoreInfotext("cat\nSteps: 6, Seed: 3, Grid XYZ info: TRUE")
    assertEquals(form.values.map, Map("enabled" -> "true", "image_numbers" -> "true", "xyz_info" -> "true"))
  }
