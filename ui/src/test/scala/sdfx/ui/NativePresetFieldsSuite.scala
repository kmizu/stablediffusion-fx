package sdfx.ui

import javafx.beans.property.SimpleBooleanProperty
import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Slider, Spinner, SpinnerValueFactory, TitledPane}
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.presets.*
import sdfx.ui.presets.NativePresetFields
import sdfx.ui.vm.*

/** Contract tests for the direct, lossless editor bindings. The attached hires widget is
  * separately exercised by RealNativePresetsPaneCheck after the pane is wired in Task 5.
  */
class NativePresetFieldsSuite extends munit.FunSuite:

  private def key(parts: String*): FieldKey = FieldKey(parts.toVector)
  private def atom(fields: NativePresetFields, parts: String*): PresetAtom =
    fields.capture().values(key(parts*))
  private def spec(fields: NativePresetFields, parts: String*): PresetFieldSpec =
    fields.schema.fields.find(_.key == key(parts*)).getOrElse(fail(s"missing field: ${parts.mkString(".")}"))
  private def fixture(mode: Mode, scripts: Seq[Plugin] = Nil,
                      alwaysOn: Seq[Plugin] = Nil): (GenerationVM, PluginsVM, CatalogVM, NativePresetFields) =
    val vm = new GenerationVM(mode)
    val plugins = new PluginsVM(mode, scripts, alwaysOn, _ => Values.empty)
    val catalog = new CatalogVM
    val fields = new NativePresetFields(vm, plugins, catalog, new SimpleBooleanProperty(false))
    (vm, plugins, catalog, fields)

  private final class ProbePlugin(val id: String, val kind: PluginKind,
                                  override val controls: Seq[Control]) extends Plugin:
    var processed = 0
    var ran = 0
    def name: String = id
    override def process(p: GenerationParams, v: Values): GenerationParams =
      processed += 1
      p
    override def run(p: GenerationParams, v: Values, ctx: PluginContext): JobOutput =
      ran += 1
      JobOutput(Nil)

  test("all_control_kinds_defaults_and_disabled_values") {
    val plugin = new ProbePlugin("probe", PluginKind.AlwaysOn, Seq(
      Control.Check("enabled", "Enabled"),
      Control.Slider("strength", "Strength", 0, 2, 0.05, 0.75),
      Control.IntField("count", "Count", 1, 10, 3),
      Control.Choice("style", "Style", Seq("a", "b"), "b"),
      Control.DynamicChoice("model", "Model", ChoiceSource.FaceRestorers, "None"),
      Control.Text("text", "Text", "draft"),
      Control.ImageFile("image", "Image"),
      Control.Note("note", "Not a value"),
      Control.Section("section", "Layout only")
    ))
    val (_, plugins, catalog, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(plugin))
    catalog.choices(ChoiceSource.FaceRestorers).setAll("GFPGAN")
    val p = plugins.form("probe").get
    val expected = Map(
      "enabled" -> PresetAtom.Flag(false),
      "strength" -> PresetAtom.Decimal(BigDecimal("0.75")),
      "count" -> PresetAtom.Integer(3L),
      "style" -> PresetAtom.Text("b"),
      "model" -> PresetAtom.Text("None"),
      "text" -> PresetAtom.Text("draft"),
      "image" -> PresetAtom.Text("")
    )
    expected.foreach { (control, value) =>
      assertEquals(spec(fields, "plugin", "probe", control).defaultValue, value)
      assertEquals(atom(fields, "plugin", "probe", control), value)
    }
    assert(!fields.schema.fields.exists(_.key == key("plugin", "probe", "note")))
    assert(!fields.schema.fields.exists(_.key == key("plugin", "probe", "section")))
    assertEquals(spec(fields, "plugin", "probe", "strength").min, Some(BigDecimal(0)))
    assertEquals(spec(fields, "plugin", "probe", "strength").max, Some(BigDecimal(2)))
    assertEquals(spec(fields, "plugin", "probe", "style").choices,
      Vector(PresetAtom.Text("a"), PresetAtom.Text("b")))
    assertEquals(spec(fields, "plugin", "probe", "model").choices,
      Vector(PresetAtom.Text("None"), PresetAtom.Text("GFPGAN")))

    val changes = Map(
      key("plugin", "probe", "strength") -> PresetAtom.Decimal(BigDecimal("1.25")),
      key("plugin", "probe", "count") -> PresetAtom.Integer(8L),
      key("plugin", "probe", "style") -> PresetAtom.Text("a"),
      key("plugin", "probe", "model") -> PresetAtom.Text("GFPGAN"),
      key("plugin", "probe", "text") -> PresetAtom.Text("still disabled"),
      key("plugin", "probe", "image") -> PresetAtom.Text("C:/images/source.png")
    )
    assert(fields.write(changes).isRight)
    changes.foreach((k, v) => assertEquals(fields.capture().values(k), v))
    assertEquals(p.prop("enabled").get.get, "false")
    assertEquals(plugin.processed, 0)
    assertEquals(plugin.ran, 0)
  }

  test("native_segments_roundtrip_external_plugin_keys") {
    val id = "external.a%plugin"
    val control = "x.y%value"
    val plugin = new ProbePlugin(id, PluginKind.AlwaysOn, Seq(Control.Text(control, "Value")))
    val (_, _, _, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(plugin))
    val k = key("plugin", id, control)
    assertEquals(FieldKey.parse(k.render), Right(k))
    assert(fields.schema.fields.exists(_.key == k))
    assert(fields.write(Map(k -> PresetAtom.Text("exact identity"))).isRight)
    assertEquals(fields.capture().values(k), PresetAtom.Text("exact identity"))
    assert(!fields.schema.fields.exists(_.key == key("plugin", "external", "a%plugin", control)))
  }

  test("external_keys_at_512_byte_boundary_roundtrip_and_513_are_unavailable") {
    val examples = Seq(
      ("a" * 499, "a" * 500),
      ("." * 165 + "abcd", "." * 166 + "ab")
    )
    examples.foreach { (acceptedId, rejectedId) =>
      val acceptedKey = key("plugin", acceptedId, "draft")
      val rejectedKey = key("plugin", rejectedId, "draft")
      assertEquals(acceptedKey.render.getBytes(java.nio.charset.StandardCharsets.UTF_8).length, 512)
      assertEquals(rejectedKey.render.getBytes(java.nio.charset.StandardCharsets.UTF_8).length, 513)
      assertEquals(FieldKey.parse(acceptedKey.render), Right(acceptedKey))
      assert(FieldKey.parse(rejectedKey.render).isLeft)
      val accepted = new ProbePlugin(acceptedId, PluginKind.AlwaysOn, Seq(Control.Text("draft", "Draft")))
      val rejected = new ProbePlugin(rejectedId, PluginKind.AlwaysOn, Seq(Control.Text("draft", "Draft")))
      val (_, plugins, _, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(accepted, rejected))
      assert(fields.schema.fields.exists(_.key == acceptedKey))
      assert(!fields.schema.fields.exists(_.key == rejectedKey))
      assert(fields.schema.issues.nonEmpty)
      assert(fields.write(Map(acceptedKey -> PresetAtom.Text("saved"))).isRight)
      assertEquals(fields.capture().values(acceptedKey), PresetAtom.Text("saved"))
      val rejectedProp = plugins.form(rejectedId).get.prop("draft").get
      assert(fields.write(Map(rejectedKey -> PresetAtom.Text("must not write"))).isLeft)
      assertEquals(rejectedProp.get, "")
      assert(!fields.capture().values.contains(rejectedKey))
    }
  }

  test("very_large_external_identity_has_bounded_diagnostics_and_no_binding") {
    val id = "x" * 100000
    val plugin = new ProbePlugin(id, PluginKind.AlwaysOn, Seq(Control.Text("draft", "Draft")))
    val (_, plugins, _, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(plugin))
    val invalidKey = key("plugin", id, "draft")
    assert(!fields.schema.fields.exists(_.key == invalidKey))
    assert(fields.schema.issues.nonEmpty)
    assert(fields.schema.issues.forall(i => i.path.length <= 512 && i.message.length <= 512))
    val result = fields.write(Map(invalidKey -> PresetAtom.Text("must not write")))
    assert(result.isLeft)
    assert(result.swap.toOption.get.issues.forall(i => i.path.length <= 512 && i.message.length <= 512))
    assertEquals(plugins.form(id).get.prop("draft").get.get, "")
  }

  test("registry_limit_counts_standard_fields_and_all_controlnet_slots") {
    Mode.values.foreach { mode =>
      val (_, _, _, base) = fixture(mode)
      val maximum = PresetLimits().maxRegisteredFieldsPerMode
      val nativeCount = base.schema.fields.size
      assert(nativeCount > 21)
      assertEquals(base.schema.fields.count(_.key.segments.head == "controlnet"), 21)
      val room = maximum - nativeCount
      def pluginWith(n: Int) = new ProbePlugin(s"many-${mode.key}", PluginKind.AlwaysOn,
        (0 until n).map(i => Control.Text(s"draft-$i", s"Draft $i")).toVector)

      val atLimit = pluginWith(room)
      val (_, _, _, accepted) = fixture(mode, alwaysOn = Seq(atLimit))
      val lastKey = key("plugin", atLimit.id, s"draft-${room - 1}")
      assertEquals(accepted.schema.fields.size, maximum)
      assertEquals(accepted.capture().values.size, maximum)
      assert(accepted.write(Map(lastKey -> PresetAtom.Text("boundary"))).isRight)
      assertEquals(accepted.capture().values(lastKey), PresetAtom.Text("boundary"))

      val overflow = pluginWith(room + 1)
      val (vm, forms, _, unavailable) = fixture(mode, alwaysOn = Seq(overflow))
      val issue = unavailable.schema.issues.find(_.code.contains("limit"))
      assert(issue.nonEmpty, s"mode $mode needs a visible over-limit issue")
      assertEquals(unavailable.schema.fields.size, 0)
      assertEquals(unavailable.capture().values.size, 0)
      val prompt = vm.prompt.get
      val attempted = unavailable.write(Map(key("generation", "prompt") -> PresetAtom.Text("must not write")))
      assert(attempted.isLeft)
      assertEquals(vm.prompt.get, prompt)
      assertEquals(forms.form(overflow.id).get.prop(s"draft-$room").get.get, "")
    }
  }

  test("duplicate_plugin_or_control_ids_are_unavailable") {
    val a = new ProbePlugin("duplicate", PluginKind.AlwaysOn, Seq(Control.Text("x", "X")))
    val b = new ProbePlugin("duplicate", PluginKind.AlwaysOn, Seq(Control.Text("x", "Other X")))
    val c = new ProbePlugin("duplicate-control", PluginKind.AlwaysOn,
      Seq(Control.Text("x", "X"), Control.Text("x", "Other X")))
    val (_, _, _, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(a, b, c))
    assert(!fields.schema.fields.exists(_.key == key("plugin", "duplicate", "x")))
    assert(!fields.schema.fields.exists(_.key == key("plugin", "duplicate-control", "x")))
    assert(fields.schema.issues.nonEmpty)
    assert(fields.write(Map(key("plugin", "duplicate", "x") -> PresetAtom.Text("ambiguous"))).isLeft)
  }

  test("three_controlnet_slots_keep_disabled_drafts") {
    Mode.values.foreach { mode =>
      val (vm, _, catalog, fields) = fixture(mode)
      catalog.choices(ChoiceSource.ControlNets).setAll("model-0", "model-1", "model-2")
      catalog.choices(ChoiceSource.Preprocessors).setAll("pre-0", "pre-1", "pre-2")
      vm.controlNet.zipWithIndex.foreach { (unit, n) =>
        unit.enabled.set(false)
        unit.model.set(s"model-$n")
        unit.image.set(s"C:/image-$n.png")
        unit.preprocessor.set(s"pre-$n")
        unit.weight.set(0.25 + n * 0.5)
        unit.guidanceStart.set(n * 0.1)
        unit.guidanceEnd.set(1 - n * 0.1)
      }
      val captured = fields.capture()
      (0 until 3).foreach { n =>
        assertEquals(captured.values(key("controlnet", n.toString, "enabled")), PresetAtom.Flag(false))
        assertEquals(captured.values(key("controlnet", n.toString, "model")), PresetAtom.Text(s"model-$n"))
        assertEquals(captured.values(key("controlnet", n.toString, "image")), PresetAtom.Text(s"C:/image-$n.png"))
        assertEquals(captured.values(key("controlnet", n.toString, "preprocessor")), PresetAtom.Text(s"pre-$n"))
        assertEquals(captured.values(key("controlnet", n.toString, "weight")), PresetAtom.Decimal(BigDecimal(0.25 + n * 0.5)))
      }
      assertEquals(vm.toParams.controlnet.size, 0)
      vm.controlNet.foreach(_.model.set("cleared"))
      val drafts = captured.values.filter((k, _) => k.segments.headOption.contains("controlnet"))
      assert(fields.write(drafts).isRight)
      (0 until 3).foreach(n => assertEquals(vm.controlNet(n).model.get, s"model-$n"))
    }
  }

  test("quicksettings_and_image_buffers_have_no_bindings") {
    Mode.values.foreach { mode =>
      val (vm, _, _, fields) = fixture(mode)
      vm.shared.checkpoint.set("checkpoint")
      vm.shared.vae.set("vae")
      vm.shared.clipSkip.set(2)
      vm.initImage.set("init.png")
      vm.maskImage.set("mask.png")
      vm.hasMask.set(true)
      val ids = fields.schema.fields.map(_.key.render).toSet
      Seq("generation.checkpoint", "generation.vae", "generation.clipSkip",
        "generation.initImage", "generation.maskImage", "generation.hasMask",
        "setting_sd_model_checkpoint", "setting_sd_vae", "setting_CLIP_stop_at_last_layers")
        .foreach(id => assert(!ids.contains(id), s"unexpected preset binding: $id"))
      assert(!fields.capture().values.keys.exists(k => k.render.contains("maskImage") || k.render.contains("initImage")))
    }
  }

  test("mode_layout_and_declared_defaults_do_not_follow_loaded_values") {
    Mode.values.foreach { mode =>
      val (vm, _, _, fields) = fixture(mode)
      vm.steps.set(48)
      vm.width.set(768)
      assertEquals(spec(fields, "generation", "steps").defaultValue, PresetAtom.Integer(20L))
      assertEquals(spec(fields, "generation", "width").defaultValue, PresetAtom.Integer(512L))
      assertEquals(atom(fields, "generation", "steps"), PresetAtom.Integer(48L))
      assertEquals(atom(fields, "generation", "width"), PresetAtom.Integer(768L))
      val ids = fields.schema.fields.map(_.key).toSet
      assertEquals(ids.contains(key("generation", "hiresScale")), mode == Mode.Txt2Img)
      assertEquals(ids.contains(key("generation", "resizeMode")), mode != Mode.Txt2Img)
      assertEquals(ids.contains(key("generation", "maskBlur")), mode == Mode.Inpaint)
      assertEquals(ids.contains(key("generation", "inpaintingFill")), mode == Mode.Inpaint)
      assert(ids.contains(key("ui", "variationExpanded")))
      assert(fields.write(Map(key("generation", "steps") -> PresetAtom.Integer(151L))).isLeft)
      assertEquals(vm.steps.get, 48)
    }
  }

  test("capture_is_an_immutable_raw_editor_snapshot") {
    val (vm, _, _, fields) = fixture(Mode.Txt2Img)
    vm.prompt.set("before")
    vm.seed.set(-1L)
    vm.hiresEnabled.set(false)
    vm.hiresScale.set(5.0)
    val captured = fields.capture()
    vm.prompt.set("after")
    vm.hiresScale.set(2.0)
    assertEquals(captured.values(key("generation", "prompt")), PresetAtom.Text("before"))
    assertEquals(captured.values(key("generation", "seed")), PresetAtom.Integer(-1L))
    assertEquals(captured.values(key("generation", "hiresScale")), PresetAtom.Decimal(BigDecimal(5)))
    assertEquals(fields.capture().values(key("generation", "prompt")), PresetAtom.Text("after"))
  }

  test("extra_default_false_differs_from_legacy_native_true") {
    Mode.values.foreach { mode =>
      val (vm, _, _, fields) = fixture(mode)
      assert(vm.variationEnabled.get)
      assertEquals(spec(fields, "generation", "variationEnabled").defaultValue, PresetAtom.Flag(false))
      vm.subseed.set(99L)
      vm.subseedStrength.set(0.4)
      assert(fields.write(Map(key("generation", "variationEnabled") -> PresetAtom.Flag(false))).isRight)
      assertEquals(atom(fields, "generation", "subseed"), PresetAtom.Integer(99L))
      assertEquals(atom(fields, "generation", "subseedStrength"), PresetAtom.Decimal(BigDecimal("0.4")))
      assertEquals(vm.toParams.subseedStrength, 0.0)
      assert(fields.write(Map(key("generation", "subseedStrength") -> PresetAtom.Decimal(BigDecimal("0.6")))).isRight)
      assert(!vm.variationEnabled.get)
      assertEquals(atom(fields, "generation", "subseedStrength"), PresetAtom.Decimal(BigDecimal("0.6")))
      assert(fields.write(Map(key("generation", "variationEnabled") -> PresetAtom.Flag(true))).isRight)
      assertEquals(vm.toParams.subseedStrength, 0.6)
    }
  }

  test("logical_denoise_binds_by_mode") {
    Mode.values.foreach { mode =>
      val (vm, _, _, fields) = fixture(mode)
      val logical = key("generation", "denoisingStrength")
      assert(fields.schema.fields.exists(_.key == logical))
      assert(!fields.schema.fields.exists(_.key == key("generation", "hiresDenoise")))
      assert(!fields.schema.fields.exists(_.key == key("generation", "denoise")))
      assert(fields.write(Map(logical -> PresetAtom.Decimal(BigDecimal("0.4")))).isRight)
      assertEquals(fields.capture().values(logical), PresetAtom.Decimal(BigDecimal("0.4")))
      if mode == Mode.Txt2Img then
        assertEquals(vm.hiresDenoise.get, 0.4)
        assertEquals(vm.denoise.get, 0.75)
      else
        assertEquals(vm.denoise.get, 0.4)
        assertEquals(vm.hiresDenoise.get, 0.55)
    }
  }

  test("script_selected_and_inactive_script_drafts_roundtrip") {
    val script = new ProbePlugin("script.with.dot", PluginKind.Script,
      Seq(Control.Text("draft", "Draft", "original")))
    val (_, plugins, _, fields) = fixture(Mode.Txt2Img, scripts = Seq(script))
    assertEquals(atom(fields, "script", "selected"), PresetAtom.Text(""))
    assert(fields.write(Map(key("plugin", script.id, "draft") -> PresetAtom.Text("retained"))).isRight)
    assertEquals(plugins.selectedScript.get, "")
    assertEquals(atom(fields, "plugin", script.id, "draft"), PresetAtom.Text("retained"))
    assert(fields.write(Map(key("script", "selected") -> PresetAtom.Text(script.id))).isRight)
    assertEquals(plugins.selectedScript.get, script.id)
    assertEquals(script.ran, 0)
  }

  test("long_control_free_script_ids_are_scalar_choices_not_plugin_field_keys") {
    val selected = key("script", "selected")
    val validIds = Seq(
      ("a" * 504, 504),
      ("a" * 512, 512),
      ("あ" * 170 + "ab", 512) // 512 UTF-8 bytes in a short UTF-16 string
    )
    validIds.foreach { (id, expectedBytes) =>
      assertEquals(id.getBytes(java.nio.charset.StandardCharsets.UTF_8).length, expectedBytes)
      val script = new ProbePlugin(id, PluginKind.Script, Seq.empty)
      val (_, plugins, _, fields) = fixture(Mode.Txt2Img, scripts = Seq(script))
      assert(fields.schema.fields.find(_.key == selected).get.choices.contains(PresetAtom.Text(id)))
      plugins.selectedScript.set(id)
      assertEquals(fields.capture().values(selected), PresetAtom.Text(id))
      plugins.selectedScript.set("")
      assert(fields.write(Map(selected -> PresetAtom.Text(id))).isRight)
      assertEquals(plugins.selectedScript.get, id)
      assertEquals(fields.capture().values(selected), PresetAtom.Text(id))
    }

    Seq("b" * 513, "c" * 100000).foreach { id =>
      val script = new ProbePlugin(id, PluginKind.Script, Seq.empty)
      val (_, plugins, _, fields) = fixture(Mode.Txt2Img, scripts = Seq(script))
      assert(!fields.schema.fields.find(_.key == selected).get.choices.contains(PresetAtom.Text(id)))
      assert(fields.schema.issues.nonEmpty)
      assert(fields.schema.issues.forall(i => i.path.length <= 512 && i.message.length <= 512))
      plugins.selectedScript.set(id)
      assertEquals(fields.capture().values(selected), PresetAtom.Text(id))
      plugins.selectedScript.set("")
      val rejected = fields.write(Map(selected -> PresetAtom.Text(id)))
      assert(rejected.isLeft)
      assert(rejected.swap.toOption.get.issues.forall(i => i.path.length <= 512 && i.message.length <= 512))
      assertEquals(plugins.selectedScript.get, "")
    }
  }

  test("dynamic_catalog_choices_refresh_without_service_calls") {
    val plugin = new ProbePlugin("dynamic", PluginKind.AlwaysOn,
      Seq(Control.DynamicChoice("model", "Model", ChoiceSource.FaceRestorers, "None")))
    val (_, _, catalog, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(plugin))
    catalog.choices(ChoiceSource.FaceRestorers).setAll("GFPGAN")
    val before = fields.schema
    assertEquals(spec(fields, "plugin", "dynamic", "model").choices,
      Vector(PresetAtom.Text("None"), PresetAtom.Text("GFPGAN")))
    catalog.choices(ChoiceSource.FaceRestorers).setAll("CodeFormer")
    assertEquals(spec(fields, "plugin", "dynamic", "model").choices,
      Vector(PresetAtom.Text("None"), PresetAtom.Text("CodeFormer")))
    assertNotEquals(fields.schema.revision, before.revision)
  }

  test("generation_restore_faces_binds_the_actual_always_on_form") {
    val restore = new sdfx.plugins.FaceRestorePlugin
    val (_, plugins, _, fields) = fixture(Mode.Img2Img, alwaysOn = Seq(restore))
    val enabled = key("plugin", "face_restore", "enabled")
    assert(fields.schema.fields.exists(_.key == enabled))
    assert(fields.write(Map(enabled -> PresetAtom.Flag(true))).isRight)
    assertEquals(plugins.form("face_restore").get.prop("enabled").get.get, "true")
    assertEquals(fields.capture().values(enabled), PresetAtom.Flag(true))
    assert(!plugins.form("face_restore").get.isActive) // no restoration model selected
  }

  test("hires_scale_schema_accepts_five_rejects_above_five") {
    val (vm, _, _, fields) = fixture(Mode.Txt2Img)
    val scale = key("generation", "hiresScale")
    assertEquals(spec(fields, "generation", "hiresScale").min, Some(BigDecimal(1)))
    assertEquals(spec(fields, "generation", "hiresScale").max, Some(BigDecimal(5)))
    for value <- Seq("1", "5") do
      assert(fields.write(Map(scale -> PresetAtom.Decimal(BigDecimal(value)))).isRight)
      assertEquals(fields.capture().values(scale), PresetAtom.Decimal(BigDecimal(value)))
    assert(fields.write(Map(scale -> PresetAtom.Decimal(BigDecimal("5.01")))).isLeft)
    assertEquals(vm.hiresScale.get, 5.0)
    assert(fields.write(Map(scale -> PresetAtom.Decimal(BigDecimal("0.99")))).isLeft)
    assertEquals(vm.hiresScale.get, 5.0)
    assertEquals(vm.hiresTarget, (2560, 2560))
  }

  test("raw_asymmetric_tiling_controls_are_independent_drafts") {
    val tiling = new sdfx.plugins.AsymmetricTilingPlugin
    val (_, plugins, _, fields) = fixture(Mode.Txt2Img, alwaysOn = Seq(tiling))
    val changes = Map(
      key("plugin", "asymmetric_tiling", "enabled") -> PresetAtom.Flag(true),
      key("plugin", "asymmetric_tiling", "tile_x") -> PresetAtom.Flag(false),
      key("plugin", "asymmetric_tiling", "tile_y") -> PresetAtom.Flag(true),
      key("plugin", "asymmetric_tiling", "start_step") -> PresetAtom.Integer(4L),
      key("plugin", "asymmetric_tiling", "stop_step") -> PresetAtom.Integer(9L)
    )
    assert(fields.write(changes).isRight)
    changes.foreach((k, v) => assertEquals(fields.capture().values(k), v))
    assertEquals(plugins.form("asymmetric_tiling").get.prop("enabled").get.get, "true")
    assertEquals(plugins.form("asymmetric_tiling").get.prop("tile_x").get.get, "false")
  }

/** Called by Task 5/6's real pane helper after the hires pane is attached to a shown Stage.
  * This checks the actual bound controls, which a property-only munit test cannot prove.
  */
object NativePresetFieldsWidgetAssertions:
  def assertAttachedHiresScaleFive(pane: TitledPane, vm: GenerationVM,
                                   fields: NativePresetFields): Unit =
    require(Platform.isFxApplicationThread, "attached widget check must run on FX thread")
    require(pane.getScene != null && pane.getScene.getWindow != null && pane.getScene.getWindow.isShowing,
      "hires pane must be attached to a shown Stage")
    require(pane.isExpanded && pane.getContent.getLayoutBounds.getWidth > 0,
      "allow an FX layout pulse with the hires section expanded before checking bound widgets")
    val scale = FieldKey(Vector("generation", "hiresScale"))
    assert(fields.write(Map(scale -> PresetAtom.Decimal(BigDecimal(5)))).isRight)
    val nodes = descendants(pane.getContent)
    val sliders = nodes.collect { case s: Slider if s.getMin == 1.0 && s.getMax == 5.0 => s }
    val spinners = nodes.collect {
      case s: Spinner[?] if s.getValueFactory.isInstanceOf[SpinnerValueFactory.DoubleSpinnerValueFactory] &&
        s.getValueFactory.asInstanceOf[SpinnerValueFactory.DoubleSpinnerValueFactory].getMin == 1.0 &&
        s.getValueFactory.asInstanceOf[SpinnerValueFactory.DoubleSpinnerValueFactory].getMax == 5.0 => s
    }
    assert(sliders.size == 1, "actual hires scale slider with 1..5 range")
    assert(spinners.size == 1, "actual hires scale spinner with 1..5 range")
    assert(vm.hiresScale.get == 5.0)
    assert(sliders.head.getValue == 5.0)
    assert(spinners.head.getValue.asInstanceOf[java.lang.Double].doubleValue == 5.0)
    assert(fields.capture().values(scale) == PresetAtom.Decimal(BigDecimal(5)))

  private def descendants(node: Node): Vector[Node] = node match
    case p: Parent => Vector(node) ++ p.getChildrenUnmodifiable.toArray.toVector.flatMap(x => descendants(x.asInstanceOf[Node]))
    case _ => Vector(node)
