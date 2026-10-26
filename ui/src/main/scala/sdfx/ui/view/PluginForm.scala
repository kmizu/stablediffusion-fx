package sdfx.ui.view

import javafx.beans.property.*
import javafx.collections.{FXCollections, ObservableList}
import javafx.scene.Node
import javafx.scene.control.{Control as _, *}
import javafx.scene.layout.VBox
import sdfx.core.plugins.api.*
import sdfx.ui.vm.{PluginValuesVM, PluginsVM, Widget}

/** Renders plugin [[Control]]s generically and binds them to a [[PluginValuesVM]]. */
object PluginForm:
  def render(ctx: UiContext, form: PluginValuesVM): Node =
    val v = new VBox(8)
    if form.plugin.description.nonEmpty then
      val d = W.label(form.plugin.description, "muted", "plugin-description")
      d.setWrapText(true)
      v.getChildren.add(d)
    // controls before the first Section go straight into the panel, each Section becomes a collapsible group
    val groups = PluginForm.groups(form.plugin.controls)
    groups.foreach { (section, controls) =>
      val grid = renderControls(ctx, form, controls)
      section match
        case None => v.getChildren.add(grid)
        case Some(s) =>
          val tp = new TitledPane(s.label, grid)
          tp.setExpanded(s.expanded)
          tp.setAnimated(false)
          tp.getStyleClass.add("plugin-subsection")
          v.getChildren.add(tp)
    }
    v

  /** Splits controls at Section markers: (None, leading controls), (Some(section), its controls)... */
  def groups(controls: Seq[Control]): Seq[(Option[Control.Section], Seq[Control])] =
    val out = scala.collection.mutable.ArrayBuffer[(Option[Control.Section], Seq[Control])]()
    var cur: Option[Control.Section] = None
    val buf = scala.collection.mutable.ArrayBuffer[Control]()
    def flush(): Unit =
      if buf.nonEmpty || cur.nonEmpty then out += cur -> buf.toSeq
      buf.clear()
    controls.foreach {
      case s: Control.Section => flush(); cur = Some(s)
      case c                  => buf += c
    }
    flush()
    out.toSeq

  private def renderControls(ctx: UiContext, form: PluginValuesVM, controls: Seq[Control]): Node =
    val f = W.form()
    controls.foreach { c =>
      val widget = PluginValuesVM.widgetFor(c)
      widget match
        case Widget.Section => ()
        case Widget.Label =>
          val l = W.label(c.label, "plugin-note")
          l.setWrapText(true)
          f.full(l)
        case Widget.CheckBox =>
          // the AlwaysOn "enabled" toggle lives in the section header
          if !(c.key == "enabled" && form.plugin.kind == PluginKind.AlwaysOn) then
            f.full(W.check(c.label, boolProp(form.props(c.key))))
        case _ =>
          val node = control(ctx, c, widget, form.props(c.key))
          bindEnabled(form, c.key, node)
          f.add(c.label, node)
    }
    f.grid

  /** Greys a control out while the checkbox named by `Plugin.enabledBy` is off. */
  private def bindEnabled(form: PluginValuesVM, key: String, node: Node): Unit =
    form.plugin.enabledBy.get(key).flatMap(form.props.get).foreach { gate =>
      def update(v: String): Unit = node.setDisable(!Values(Map("v" -> v)).bool("v"))
      update(gate.get)
      gate.addListener((_, _, v) => update(v))
    }

  private def control(ctx: UiContext, c: Control, widget: Widget, p: StringProperty): Node = (c, widget) match
    case (s: Control.Slider, _) =>
      W.doubleSlider(doubleProp(p, s.defaultValue), s.min, s.max, s.step)
    case (i: Control.IntField, _) =>
      if i.max - i.min <= 4096 then W.intSlider(intProp(p, i.defaultValue), i.min, i.max) else W.intSpinner(intProp(p, i.defaultValue), i.min, i.max)
    case (ch: Control.Choice, _) =>
      W.combo(FXCollections.observableArrayList(ch.options*), p)
    case (dc: Control.DynamicChoice, _) =>
      W.combo(dynamicItems(ctx, dc), p)
    case (t: Control.Text, Widget.TextArea) =>
      val a = new TextArea()
      a.textProperty.bindBidirectional(p)
      a.setPromptText(t.placeholder)
      a.setPrefRowCount(4)
      a.setWrapText(true)
      a
    case (t: Control.Text, _) =>
      W.textField(p, t.placeholder)
    case (_: Control.ImageFile, Widget.ImageList) =>
      val a = new TextArea()
      a.textProperty.bindBidirectional(p)
      a.setPromptText("One image path per line")
      a.setPrefRowCount(3)
      val add = W.button("Add images…", "Choose one or more image files") {
        val fs = ImageFiles.chooser("Add images").showOpenMultipleDialog(ctx.stage)
        if fs != null && !fs.isEmpty then
          val cur = p.get.split("[\\r\\n]+").map(_.trim).filter(_.nonEmpty).toSeq
          p.set((cur ++ (0 until fs.size).map(i => fs.get(i).toPath.toString)).distinct.mkString("\n"))
      }
      new VBox(4, a, add)
    case (_: Control.ImageFile, _) =>
      new ImageDropZone(ctx, p, "Image", 140).node
    case _ => W.textField(p)

  /** Dynamic choice options that follow the catalog. */
  private def dynamicItems(ctx: UiContext, dc: Control.DynamicChoice): ObservableList[String] =
    val out = FXCollections.observableArrayList[String]()
    def refresh(): Unit = out.setAll(PluginValuesVM.options(dc, s => ctx.catalog.choices(s).toArray(Array.empty[String]).toSeq)*)
    refresh()
    ctx.catalog.choices(dc.source).addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) => refresh())
    out

  // String-valued plugin props mirrored into typed properties for the numeric widgets.
  private def doubleProp(p: StringProperty, d: Double): DoubleProperty =
    val dp = new SimpleDoubleProperty(p.get.trim.toDoubleOption.getOrElse(d))
    dp.addListener((_, _, v) => if !p.get.trim.toDoubleOption.contains(v.doubleValue) then p.set(Values.fmt(v.doubleValue)))
    p.addListener((_, _, v) => v.trim.toDoubleOption.foreach(x => if x != dp.get then dp.set(x)))
    dp

  private def intProp(p: StringProperty, d: Int): IntegerProperty =
    val ip = new SimpleIntegerProperty(p.get.trim.toDoubleOption.map(_.toInt).getOrElse(d))
    ip.addListener((_, _, v) => if !p.get.trim.toIntOption.contains(v.intValue) then p.set(v.toString))
    p.addListener((_, _, v) => v.trim.toDoubleOption.map(_.toInt).foreach(x => if x != ip.get then ip.set(x)))
    ip

  def boolProp(p: StringProperty): BooleanProperty =
    val bp = new SimpleBooleanProperty(Values(Map("v" -> p.get)).bool("v"))
    bp.addListener((_, _, v) => p.set(v.toString))
    p.addListener((_, _, v) => bp.set(Values(Map("v" -> v)).bool("v")))
    bp

  /** "Script" dropdown with the selected script's controls rendered below. */
  def scriptSection(ctx: UiContext, plugins: PluginsVM): Node =
    val names = FXCollections.observableArrayList[String]("None")
    names.addAll(plugins.scriptForms.map(_.plugin.name)*)
    val combo = new ComboBox[String](names)
    combo.setMaxWidth(Double.MaxValue)
    def idOf(name: String) = plugins.scriptForms.find(_.plugin.name == name).map(_.plugin.id).getOrElse("")
    def nameOf(id: String) = plugins.scriptForms.find(_.plugin.id == id).map(_.plugin.name).getOrElse("None")
    val body = new VBox(8)
    // each script form is rendered once and reused: re-rendering would leak listeners on the long-lived props
    val rendered = scala.collection.mutable.Map.empty[String, Node]
    def show(id: String): Unit =
      body.getChildren.setAll(plugins.scriptForms.find(_.plugin.id == id).map(f => rendered.getOrElseUpdate(id, render(ctx, f))).toSeq*)
    combo.setValue(nameOf(plugins.selectedScript.get))
    show(plugins.selectedScript.get)
    combo.valueProperty.addListener((_, _, n) => plugins.selectedScript.set(idOf(Option(n).getOrElse("None"))))
    plugins.selectedScript.addListener((_, _, id) =>
      if combo.getValue != nameOf(id) then combo.setValue(nameOf(id))
      show(id)
    )
    W.tip(combo, "A script takes over the run (X/Y/Z plot, prompt matrix, loopback…). Only one can be active.")
    val v = new VBox(8, combo, body)
    v.getStyleClass.add("script-section")
    v

  /** Collapsible pane for an AlwaysOn plugin with its "enabled" checkbox in the header. */
  def alwaysOnSection(ctx: UiContext, form: PluginValuesVM, expanded: Boolean): TitledPane =
    val pane = W.section(form.plugin.name, render(ctx, form), expanded, form.plugin.description)
    form.prop("enabled").foreach { en =>
      val cb = new CheckBox()
      cb.selectedProperty.bindBidirectional(boolProp(en))
      W.tip(cb, s"Enable ${form.plugin.name}")
      pane.setGraphic(cb)
      pane.setContentDisplay(ContentDisplay.LEFT)
      cb.selectedProperty.addListener((_, _, on) =>
        if on then pane.getStyleClass.add("section-active") else pane.getStyleClass.remove("section-active")
      )
      if cb.isSelected then pane.getStyleClass.add("section-active")
    }
    pane
