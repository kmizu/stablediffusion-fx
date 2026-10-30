package sdfx.ui.view

import javafx.collections.{FXCollections, ListChangeListener}
import javafx.scene.Node
import javafx.scene.control.{ListView, SelectionMode}
import javafx.scene.layout.{HBox, VBox}
import sdfx.core.model.Styles
import sdfx.ui.vm.{GenerationVM, PluginValuesVM}
import scala.jdk.CollectionConverters.*
import scala.util.Try

private[view] object HiresTweaksControls:
  def showWhen(ctx: UiContext, node: Node)(visible: sdfx.core.config.HiresFixTweaksPreferences => Boolean): Unit =
    def update(): Unit =
      val show = visible(ctx.hiresPreferences.get)
      node.setVisible(show)
      node.setManaged(show)
    update()
    ctx.hiresPreferences.addListener((_, _, _) => update())

  private def modeControl(ctx: UiContext, form: PluginValuesVM, key: String, label: String): Node =
    val property = form.props(key)
    val choices = sdfx.core.model.HiresPromptModes.modes
    val combo = W.combo(FXCollections.observableArrayList(choices*), property)
    combo.setId(s"hires-tweaks-$key-dropdown")
    val group = new javafx.scene.control.ToggleGroup()
    val radios = new javafx.scene.layout.FlowPane(8, 6)
    radios.setId(s"hires-tweaks-$key-radio")
    val buttons = choices.map { value =>
      val button = new javafx.scene.control.RadioButton(value)
      button.setToggleGroup(group)
      button.setSelected(property.get == value)
      button.setOnAction(_ => property.set(value))
      radios.getChildren.add(button)
      value -> button
    }
    property.addListener((_, _, value) => buttons.foreach((name, button) => button.setSelected(name == value)))
    showWhen(ctx, radios)(_.promptModeRadio)
    showWhen(ctx, combo)(p => !p.promptModeRadio)
    new VBox(4, W.label(label, "hint"), radios, combo)

  def render(ctx: UiContext, vm: GenerationVM, form: PluginValuesVM): Node =
    val box = new VBox(8)
    PluginForm.groups(form.plugin.controls).foreach { (section, controls) =>
      val content = new VBox(8)
      if section.exists(_.key == "prompt_modes") then
        val modes = new VBox(8,
          modeControl(ctx, form, "prompt_mode", "Positive prompt"),
          modeControl(ctx, form, "negative_mode", "Negative prompt"),
          PluginForm.renderControls(ctx, form, controls.filter(c => Set("append_separator", "prepend_separator", "marker",
            "empty_prompt", "empty_negative_prompt")(c.key))))
        val remove = PluginForm.renderControls(ctx, form, controls.filter(_.key == "remove_networks"))
        showWhen(ctx, modes)(_.showPromptModes)
        showWhen(ctx, remove)(_.showRemoveNetworks)
        content.getChildren.addAll(modes, remove)
      else content.getChildren.add(PluginForm.renderControls(ctx, form, controls.filterNot(_.key == "styles")))
      if section.exists(_.key == "batch_seed") then
        val actions = new javafx.scene.layout.FlowPane(8, 8,
          W.button("Random seed", "Choose a new Hires seed when generating") { form.props("seed").set("-1") },
          W.button("Use first-pass seed", "Inherit the first-pass seed for Hires") { form.props("seed").set("0") },
          W.button("Random variation", "Choose a new Hires variation seed when generating") { form.props("subseed").set("-1") },
          W.button("Reuse last Hires seed", "Use the actual seed of the last saved Hires image") {
            ctx.hiresSeedHistory.seed match
              case Some(seed) => form.props("seed").set(seed.toString); form.props("seed_enabled").set("true")
              case None => ctx.info("No Hires image generated yet")
          },
          W.button("Reuse last variation seed", "Use the actual variation seed of the last saved Hires image") {
            ctx.hiresSeedHistory.subseed match
              case Some(seed) =>
                form.props("subseed").set(seed.toString)
                form.props("seed_enabled").set("true")
                form.props("extras").set("true")
              case None => ctx.info("No Hires image generated yet")
          })
        content.getChildren.add(actions)
      if section.exists(_.key == "hr_styles") then
        val list = new ListView[String]()
        list.setId("hires-tweaks-styles")
        list.setPrefHeight(110)
        list.getSelectionModel.setSelectionMode(SelectionMode.MULTIPLE)
        var updating = false
        def selectedNames = Try(ujson.read(form.props("styles").get).arr.map(_.str).toSeq).getOrElse(Seq.empty)
        val order = W.label("", "hint")
        order.setId("hires-tweaks-style-order")
        order.setWrapText(true)
        def showOrder(): Unit = order.setText("Apply order: " + (if selectedNames.isEmpty then "none" else selectedNames.mkString(" → ")))
        showOrder()
        def refresh(): Unit =
          updating = true
          try
            val selected = selectedNames
            list.setItems(FXCollections.observableArrayList(ctx.styles.asScala.map(_.name).toSeq*))
            selected.foreach(name => list.getSelectionModel.select(name))
          finally updating = false
        refresh()
        ctx.styles.addListener((_: ListChangeListener.Change[? <: sdfx.core.model.PromptStyle]) => refresh())
        form.props("styles").addListener((_, _, _) => { showOrder(); if !updating then refresh() })
        list.getSelectionModel.getSelectedItems.addListener((_: ListChangeListener.Change[? <: String]) =>
          if !updating then
            updating = true
            try
              val current = list.getSelectionModel.getSelectedItems.asScala.toSeq
              val previous = selectedNames
              val ordered = previous.filter(current.contains) ++ current.filterNot(previous.contains)
              form.props("styles").set(ujson.Arr.from(ordered).render())
            finally updating = false)
        content.getChildren.addAll(list, order, W.button("Apply styles to Hires prompts", "Insert selected styles into the visible Hires prompts and clear the selection") {
          val (positive, negative) = Styles.apply(ctx.styles.asScala.toSeq, selectedNames, vm.hiresPrompt.get, vm.hiresNegative.get)
          vm.hiresPrompt.set(positive)
          vm.hiresNegative.set(negative)
          form.props("styles").set("[]")
        }, W.label("With separate styles enabled, an empty selection removes inherited first-pass styles.", "hint"))
      section match
        case Some(s) =>
          val pane = W.section(s.label, content, false)
          pane.setId(s"hires-tweaks-${s.key}")
          showWhen(ctx, pane)(p => s.key match
            case "batch_seed" => p.showBatchSeed
            case "prompt_modes" => p.showPromptModes || p.showRemoveNetworks
            case "hr_styles" => p.showStyles
            case _ => true)
          box.getChildren.add(pane)
        case None => box.getChildren.add(content)
    }
    box
