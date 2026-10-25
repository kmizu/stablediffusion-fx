package sdfx.ui.vm

import javafx.beans.property.{SimpleStringProperty, StringProperty}
import sdfx.core.plugins.api.*

/** How a declarative [[Control]] is rendered by the generic plugin form. */
enum Widget:
  case SliderSpinner, IntSpinner, CheckBox, Combo, DynamicCombo, TextField, TextArea, Label, ImagePicker, ImageList

/** Editable values of one plugin for one tab. Values are kept as strings exactly like [[Values]]. */
final class PluginValuesVM(val plugin: Plugin, initial: Values = Values.empty):
  val props: Map[String, StringProperty] =
    plugin.controls.collect {
      case c if !c.isInstanceOf[Control.Note] => c.key -> new SimpleStringProperty(initial.map.getOrElse(c.key, c.default))
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

  def scriptJob: Option[(Plugin, Values)] =
    scriptForms.find(_.plugin.id == selectedScript.get).map(f => f.plugin -> f.values)

  def alwaysOnJob: Seq[(Plugin, Values)] = alwaysOnForms.filter(_.isActive).map(f => f.plugin -> f.values)

  /** plugin id -> values, for persistence. */
  def snapshot: Map[String, Map[String, String]] = (scriptForms ++ alwaysOnForms).map(f => f.plugin.id -> f.values.map).toMap
