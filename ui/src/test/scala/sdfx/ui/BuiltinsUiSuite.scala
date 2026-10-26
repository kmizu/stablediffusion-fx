package sdfx.ui

import sdfx.core.plugins.api.*
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{PluginValuesVM, PromptEdit, Widget}

/** UI parts of the reForge built-in rewrites: prompt-bracket-checker, image controls, plugin forms. */
class BuiltinsUiSuite extends munit.FunSuite:

  test("bracket checker reports unbalanced brackets and ignores escaped ones"):
    assertEquals(PromptEdit.bracketIssues("a (b:1.2), [c], {d|e}"), "")
    assertEquals(PromptEdit.bracketIssues("a ((b), c"), "Unbalanced brackets: 1 extra '('")
    assertEquals(PromptEdit.bracketIssues("a b]"), "Unbalanced brackets: 1 extra ']'")
    assertEquals(PromptEdit.bracketIssues("""a \(literal\) and (ok)"""), "")
    assertEquals(PromptEdit.bracketIssues("((x) [y"), "Unbalanced brackets: 1 extra '(', 1 extra '['")

  test("image controls get image widgets, single and multiple"):
    assertEquals(PluginValuesVM.widgetFor(Control.ImageFile("image", "Image")), Widget.ImagePicker)
    assertEquals(PluginValuesVM.widgetFor(Control.ImageFile("images", "Images", multiple = true)), Widget.ImageList)

  test("every standard plugin control has a widget and a default value"):
    StandardPlugins.withOwnAxes.foreach { p =>
      p.controls.foreach(c => PluginValuesVM.widgetFor(c))
      val vm = new PluginValuesVM(p)
      p.controls.filter(_.hasValue).foreach(c => assert(vm.prop(c.key).isDefined, s"${p.id}.${c.key}"))
    }

  test("image-based AlwaysOn plugins are not active from defaults alone"):
    val ids = Set("reference", "revision", "ip_adapter", "photomaker")
    StandardPlugins.alwaysOn.filter(p => ids.contains(p.id)).foreach { p =>
      val vm = new PluginValuesVM(p)
      vm.prop("enabled").foreach(_.set("true"))
      assert(!vm.isActive, p.id)
    }
