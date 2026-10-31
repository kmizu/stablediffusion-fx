package sdfx.ui.vm

import javafx.beans.property.{SimpleStringProperty, StringProperty}
import sdfx.core.image.GridNumbers
import sdfx.core.infotext.Infotext
import sdfx.core.plugins.api.*
import sdfx.core.payload.PayloadJsonBudget

/** How a declarative [[Control]] is rendered by the generic plugin form. */
enum Widget:
  case SliderSpinner, IntSpinner, CheckBox, Combo, DynamicCombo, TextField, TextArea, Label, Section, ImagePicker, ImageList

/** Editable values of one plugin for one tab. Values are kept as strings exactly like [[Values]]. */
final class PluginValuesVM(val plugin: Plugin, initial: Values = Values.empty):
  val props: Map[String, StringProperty] =
    plugin.controls.collect {
      case c if c.hasValue => c.key -> new SimpleStringProperty(initial.map.getOrElse(c.key, c.default))
    }.toMap

  def prop(key: String): Option[StringProperty] = props.get(key)
  def values: Values = Values(props.view.mapValues(_.get).toMap)
  def isActive: Boolean = plugin.isActive(values)
  def hasEnabledToggle: Boolean = props.contains("enabled")

  def load(v: Values): Unit = props.foreach((k, p) => v.map.get(k).foreach(p.set))
  def reset(): Unit = load(PluginValuesVM.defaults(plugin.controls))

object PluginValuesVM:
  def defaults(controls: Seq[Control]): Values = Values.defaults(controls)

  def widgetFor(c: Control): Widget = c match
    case _: Control.Slider                => Widget.SliderSpinner
    case _: Control.IntField              => Widget.IntSpinner
    case _: Control.Check                 => Widget.CheckBox
    case _: Control.Choice                => Widget.Combo
    case _: Control.DynamicChoice         => Widget.DynamicCombo
    case t: Control.Text if t.multiline   => Widget.TextArea
    case _: Control.Text                  => Widget.TextField
    case _: Control.Note                  => Widget.Label
    case _: Control.Section               => Widget.Section
    case i: Control.ImageFile if i.multiple => Widget.ImageList
    case _: Control.ImageFile             => Widget.ImagePicker

  /** Options of a choice control; dynamic ones are resolved through `choices` and get "None" when allowed. */
  def options(c: Control, choices: ChoiceSource => Seq[String]): Seq[String] = c match
    case ch: Control.Choice         => ch.options
    case dc: Control.DynamicChoice  => (if dc.allowNone then Seq("None") else Nil) ++ choices(dc.source)
    case _                          => Nil

/** Plugins of one tab: the selected script and every AlwaysOn plugin, with their values. */
final class PluginsVM(val mode: Mode, scripts: Seq[Plugin], alwaysOnPlugins: Seq[Plugin], initial: Plugin => Values):
  val scriptForms: Seq[PluginValuesVM] = scripts.map(p => new PluginValuesVM(p, initial(p)))
  val alwaysOnForms: Seq[PluginValuesVM] = alwaysOnPlugins.map(p => new PluginValuesVM(p, initial(p)))
  /** Selected script plugin id, "" = None. */
  val selectedScript = new SimpleStringProperty("")

  def form(id: String): Option[PluginValuesVM] = (scriptForms ++ alwaysOnForms).find(_.plugin.id == id)

  /** Replay only this script's own settings; ordinary infotext leaves other script choices alone. */
  def restoreLatentUpscale(fields: Map[String, String]): Unit =
    sdfx.plugins.LatentUpscalePlugin.replay(fields) match
      case Some(values) => form("latent_upscale").foreach { f =>
        f.load(values)
        selectedScript.set("latent_upscale")
      }
      case None => if selectedScript.get == "latent_upscale" then selectedScript.set("")

  def scriptJob: Option[(Plugin, Values)] =
    scriptForms.find(_.plugin.id == selectedScript.get).map(f => f.plugin -> f.values)

  def alwaysOnJob: Seq[(Plugin, Values)] = alwaysOnForms.filter(_.isActive).map(f => f.plugin -> f.values)

  /** All applicable AlwaysOn forms, including disabled values, frozen on actual queue submission. */
  def payloadConfigurationJson(limits: PayloadJsonBudget.Limits = PayloadJsonBudget.Limits()): String =
    def form(f: PluginValuesVM): ujson.Obj =
      val values = f.values
      ujson.Obj("id" -> f.plugin.id, "title" -> f.plugin.name,
        "values" -> ujson.Obj.from(values.map.toSeq.sortBy(_._1).map((k, v) => k -> ujson.Str(v))),
        "active" -> f.plugin.isActive(values))
    val configuration = ujson.Obj("ui_forms" -> "available",
      "selected_script" -> scriptForms.find(_.plugin.id == selectedScript.get).map(form).getOrElse(ujson.Null),
      "always_on" -> ujson.Arr.from(alwaysOnForms.map(form)))
    PayloadJsonBudget.write(configuration, limits).getOrElse("{\"status\":\"unavailable_limits\"}")

  /** Restore native display controls from PNG parameters. GenerationVM deliberately clears extra fields;
    * the form must own restored values so both its checkboxes and the next queued job agree.
    */
  def restoreInfotext(text: String): Unit =
    val fields = Infotext.parse(text).fields
    sdfx.plugins.ConditioningHighresFixPlugin.replay(fields).foreach(values => form("conditioning_highres_fix").foreach(_.load(values)))
    sdfx.plugins.Img2ImgHiresFixPlugin.replay(fields).foreach(values => form("img2img_hires_fix").foreach(_.load(values)))
    if fields.contains("Model keyword settings") || fields.get("Model keyword replay").exists(_.equalsIgnoreCase("true")) then
      form("model_keyword").foreach(_.load(Values(Map("enabled" -> "false"))))
    fields.get("Hires seed").flatMap(sdfx.core.infotext.HiresSeedMetadata.parse).foreach { h =>
      val values = Values(h.value.toMap.map { (key, value) => key -> (value match
        case ujson.Bool(b) => b.toString
        case ujson.Num(n) if n == math.floor(n) => n.toLong.toString
        case ujson.Num(n) => n.toString
        case other => other.str)
      })
      form("hires_fix_tweaks").foreach(_.load(values))
    }
    sdfx.plugins.TokenDownsamplingPlugin.replay(fields).foreach(values => form("token_downsampling").foreach(_.load(values)))
    def flag(key: String): Option[Boolean] = fields.get(key).flatMap { v =>
      v.trim.toLowerCase(java.util.Locale.ROOT) match
        case "true" => Some(true)
        case "false" => Some(false)
        case _ => None
    }
    val numbers = flag(GridNumbers.NumbersKey)
    val xyz = flag(GridNumbers.XyzKey)
    if numbers.isDefined || xyz.isDefined then
      form("grid_numbers").foreach(_.load(Values(Map(
        "enabled" -> "true", "image_numbers" -> numbers.getOrElse(true).toString,
        "xyz_info" -> xyz.getOrElse(false).toString))))

  /** Apply the validated prompt/seed restore result to the owned native Hires form. */
  def restoreHiresParams(params: sdfx.core.model.GenerationParams): Unit =
    form("hires_fix_tweaks").foreach { f =>
      f.reset()
      params.hooks.get("hires_fix_tweaks").collect { case o: ujson.Obj => o }.foreach { h =>
        val values = h.value.toSeq.collect {
          case (k, ujson.Str(s)) => k -> s
          case (k, v: ujson.Bool) => k -> v.bool.toString
          case (k, ujson.Num(n)) => k -> (if n == math.floor(n) then n.toLong.toString else n.toString)
          case ("styles", v: ujson.Arr) => "styles" -> v.render()
        }
        f.load(Values(values.toMap))
      }
    }

  /** plugin id -> values, for persistence. */
  def snapshot: Map[String, Map[String, String]] = (scriptForms ++ alwaysOnForms).map(f => f.plugin.id -> f.values.map).toMap
