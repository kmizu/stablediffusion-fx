package sdfx.ui

import java.nio.charset.StandardCharsets.UTF_8
import javafx.beans.property.SimpleBooleanProperty
import sdfx.core.plugins.api.Values
import sdfx.core.presets.*
import sdfx.ui.presets.{NativePresetFields, NativePresetsController}
import sdfx.ui.vm.*

/** Source enum routing against the actual native integer-backed Choice fields, without a stage or engine. */
class NativePresetSourceChoiceSuite extends munit.FunSuite:
  private val resize = FieldKey.parse("generation.resizeMode").toOption.get
  private val fill = FieldKey.parse("generation.inpaintingFill").toOption.get
  private val limits = PresetLimits()

  private def form(mode: Mode): (GenerationVM, NativePresetFields, NativePresetsController) =
    val vm = new GenerationVM(mode)
    val plugins = new PluginsVM(mode, Nil, Nil, _ => Values.empty)
    val fields = new NativePresetFields(vm, plugins, new CatalogVM,
      new SimpleBooleanProperty(false))
    (vm, fields, new NativePresetsController(fields, () => false, () => true))

  private def document(json: String, format: String, mode: Option[PresetMode]): SourceDocument =
    SourceDocument(format, mode, PresetJson.readSource(json.getBytes(UTF_8), limits).toOption.get,
      "source-choice-fixture")

  private def imported(json: String, format: String, mode: Option[PresetMode],
      schemas: Map[PresetMode, PresetSchema]): CompatibilityResult =
    PresetCompatibility.importRecords(document(json, format, mode), schemas, Vector.empty)
      .toOption.get.head

  private def schemas(img: NativePresetFields, inpaint: NativePresetFields):
      Map[PresetMode, PresetSchema] =
    Map(PresetMode.Img2Img -> img.schema, PresetMode.Inpaint -> inpaint.schema)

  private val validSources = Vector(
    ("config numeric", "config-presets", Some(PresetMode.Img2Img),
      """{"One":{"resize_mode":2,"img2img_inpainting_fill":1}}"""),
    ("config labels", "config-presets", Some(PresetMode.Img2Img),
      """{"One":{"resize_mode":"Resize and fill","img2img_inpainting_fill":"original"}}"""),
    ("utilities numeric", "preset-utilities", None,
      """{"One":{"Resize mode":2,"Masked content":1}}"""),
    ("utilities labels", "preset-utilities", None,
      """{"One":{"Resize mode":"Resize and fill","Masked content":"original"}}"""))

  test("both source formats and scalar forms plan and apply to real indexed native choices") {
    validSources.foreach { (name, format, sourceMode, json) =>
      val (imgVm, imgFields, imgController) = form(Mode.Img2Img)
      val (paintVm, paintFields, paintController) = form(Mode.Inpaint)
      val available = schemas(imgFields, paintFields)
      val imgChoice = imgFields.schema.fields.find(_.key == resize).get
      val resizeChoice = paintFields.schema.fields.find(_.key == resize).get
      val fillChoice = paintFields.schema.fields.find(_.key == fill).get
      assertEquals(imgChoice.kind, PresetFieldKind.Choice)
      assertEquals(imgChoice.defaultValue, PresetAtom.Integer(0))
      assertEquals(resizeChoice.choices,
        Vector(PresetAtom.Integer(0), PresetAtom.Integer(1), PresetAtom.Integer(2)))
      assertEquals(fillChoice.kind, PresetFieldKind.Choice)
      assertEquals(fillChoice.defaultValue, PresetAtom.Integer(1))
      assertEquals(fillChoice.choices,
        Vector(PresetAtom.Integer(0), PresetAtom.Integer(1),
          PresetAtom.Integer(2), PresetAtom.Integer(3)))

      val result = imported(json, format, sourceMode, available)
      assert(result.issues.isEmpty, s"$name: ${result.issues}")
      assertEquals(result.record.fields.get(resize), Some(PresetAtom.Integer(2)), name)
      assertEquals(result.record.fields.get(fill), Some(PresetAtom.Integer(1)), name)

      paintVm.resizeMode.set(0)
      paintVm.inpaintingFill.set(3)
      val plan = paintController.preview(result.record, Set(resize, fill), ApplyBase.Current)
        .toOption.get
      assertEquals(plan.changes,
        Map(resize -> PresetAtom.Integer(2), fill -> PresetAtom.Integer(1)), name)
      assert(paintController.apply(result.record, Set(resize, fill), ApplyBase.Current).isRight,
        name)
      assertEquals(paintVm.resizeMode.get, 2, name)
      assertEquals(paintVm.inpaintingFill.get, 1, name)

      imgVm.resizeMode.set(0)
      val resizeOnly = result.record.copy(fields = Map(resize -> PresetAtom.Integer(2)))
      assertEquals(imgController.preview(resizeOnly, Set(resize), ApplyBase.Current)
        .toOption.get.changes, Map(resize -> PresetAtom.Integer(2)), name)
      assert(imgController.apply(resizeOnly, Set(resize), ApplyBase.Current).isRight, name)
      assertEquals(imgVm.resizeMode.get, 2, name)
    }
  }

  test("source indices beyond native options and unknown labels stay opaque") {
    val (_, imgFields, _) = form(Mode.Img2Img)
    val (vm, paintFields, controller) = form(Mode.Inpaint)
    val available = schemas(imgFields, paintFields)
    val cases = Vector(
      ("config-presets", Some(PresetMode.Img2Img),
        """{"One":{"resize_mode":3,"img2img_inpainting_fill":4}}""",
        Set("resize_mode", "img2img_inpainting_fill")),
      ("preset-utilities", None,
        """{"One":{"Resize mode":3,"Masked content":4}}""",
        Set("Resize mode", "Masked content")),
      ("preset-utilities", None,
        """{"One":{"Resize mode":"not a resize option","Masked content":"not a fill option"}}""",
        Set("Resize mode", "Masked content")))
    cases.foreach { (format, sourceMode, json, sourceKeys) =>
      val result = imported(json, format, sourceMode, available)
      assert(!result.record.fields.contains(resize))
      assert(!result.record.fields.contains(fill))
      assertEquals(result.record.opaque.map(_._1).toSet, sourceKeys)
      assertEquals(result.issues.size, 2)
      val plan = controller.preview(result.record, Set(resize, fill), ApplyBase.Current)
        .toOption.get
      assertEquals(plan.changes, Map.empty[FieldKey, PresetAtom])
      assert(controller.apply(result.record, Set(resize, fill), ApplyBase.Current).isRight)
      assertEquals(vm.resizeMode.get, 0)
      assertEquals(vm.inpaintingFill.get, 1)
    }
    val invalidNative = imported(validSources.head._4, "config-presets",
      Some(PresetMode.Img2Img), available).record.copy(
      fields = Map(resize -> PresetAtom.Integer(3), fill -> PresetAtom.Integer(4)),
      dirty = Set(resize, fill))
    assert(PresetCompatibility.exportRecords(Vector(invalidNative), "config-presets",
      Some(PresetMode.Img2Img), Vector.empty, supportedOnly = false).isLeft)
  }

  test("dirty native indexed choices export through fixed source tables and reimport as integers") {
    val (_, imgFields, _) = form(Mode.Img2Img)
    val (_, paintFields, _) = form(Mode.Inpaint)
    val available = schemas(imgFields, paintFields)
    val cases = Vector(
      ("config-presets", Some(PresetMode.Img2Img), validSources.head._4,
        "\"resize_mode\":1", "\"img2img_inpainting_fill\":2"),
      ("preset-utilities", None, validSources.last._4,
        "\"Resize mode\":\"Crop and resize\"", "\"Masked content\":\"latent noise\""))
    cases.foreach { (format, sourceMode, json, resizeToken, fillToken) =>
      val original = imported(json, format, sourceMode, available).record
      assertEquals(original.fields.get(resize), Some(PresetAtom.Integer(2)))
      assertEquals(original.fields.get(fill), Some(PresetAtom.Integer(1)))
      val changed = original.copy(
        fields = original.fields.updated(resize, PresetAtom.Integer(1))
          .updated(fill, PresetAtom.Integer(2)),
        dirty = Set(resize, fill))
      val output = PresetCompatibility.exportRecords(Vector(changed), format,
        sourceMode, Vector.empty, supportedOnly = false).toOption.get
      val text = new String(output, UTF_8)
      assert(text.contains(resizeToken), text)
      assert(text.contains(fillToken), text)
      val again = imported(text, format, sourceMode, available)
      assertEquals(again.record.fields.get(resize), Some(PresetAtom.Integer(1)))
      assertEquals(again.record.fields.get(fill), Some(PresetAtom.Integer(2)))
    }
  }
