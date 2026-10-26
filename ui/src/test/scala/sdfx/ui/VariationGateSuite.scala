package sdfx.ui

import sdfx.core.model.*
import sdfx.core.payload.RequestPayload
import sdfx.core.protocol.Json
import sdfx.ui.vm.*

class VariationGateSuite extends munit.FunSuite:

  private def png(fields: String): String =
    s"a house\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512${if fields.isEmpty then "" else ", " + fields}"

  test("png_variation_metadata_enables_gate") {
    Seq("Variation seed: 99", "Variation seed strength: 0.4", "Variation seed: 99, Variation seed strength: 0.4").foreach { fields =>
      Mode.values.foreach { mode =>
        val vm = new GenerationVM(mode)
        vm.variationEnabled.set(false)
        vm.restoreInfotext(png(fields))
        assert(vm.variationEnabled.get, s"${mode.key}: $fields")
        assertEquals(vm.subseed.get, if fields.contains("Variation seed: 99") then 99L else -1L)
        assertEquals(vm.subseedStrength.get, if fields.contains("Variation seed strength") then 0.4 else 0.0)
      }
    }
  }

  test("png_without_variation_restores_existing_zero_values") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      vm.restoreInfotext(png(""))
      assert(!vm.variationEnabled.get)
      assertEquals(vm.subseed.get, -1L)
      assertEquals(vm.subseedStrength.get, 0.0)
      assertEquals(vm.toParams.subseedStrength, 0.0)
    }
  }

  test("gate_only_change_does_not_zero_editable_strength") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      vm.variationEnabled.set(false)
      assertEquals(vm.subseed.get, 99L)
      assertEquals(vm.subseedStrength.get, 0.4)
      assertEquals(vm.snapshot.subseedStrength, 0.4)
    }
  }

  test("strength_only_change_does_not_change_gate") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.variationEnabled.set(false)
      vm.subseedStrength.set(0.4)
      assert(!vm.variationEnabled.get)
      assertEquals(vm.toParams.subseedStrength, 0.0)
      vm.variationEnabled.set(true)
      vm.subseedStrength.set(0.2)
      assert(vm.variationEnabled.get)
      assertEquals(vm.toParams.subseedStrength, 0.2)
    }
  }

  test("queued_job_and_payload_do_not_follow_later_gate") {
    Mode.values.foreach { mode =>
      val vm = new GenerationVM(mode)
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      vm.variationEnabled.set(false)
      val plugins = new PluginsVM(mode, Nil, Nil, _ => sdfx.core.plugins.api.Values.empty)
      val queued = JobBuilder.build(vm, plugins)
      val body = RequestPayload.freeze(Json.toJs(queued.params))
      vm.variationEnabled.set(true)
      assertEquals(queued.params.subseed, 99L)
      assertEquals(queued.params.subseedStrength, 0.0)
      assertEquals(RequestPayload.freeze(Json.toJs(queued.params)), body)
      assertEquals(vm.toParams.subseedStrength, 0.4)
    }
  }
