package sdfx.ui

import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.*
import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.{Node, Parent}
import javafx.scene.control.*
import javafx.stage.{Stage, Window}
import javax.imageio.ImageIO
import sdfx.core.config.{AppHome, Settings}
import sdfx.core.presets.*
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins
import sdfx.ui.presets.{NativePresetFields, NativePresetFileDialogs}
import sdfx.ui.vm.{Mode, UiState}
import javafx.beans.property.SimpleBooleanProperty

/** Root-only offline acceptance of the actual native controls. No Generate, Queue,
  * inference, model load, or clipboard operation occurs here. Arg: output directory.
  */
object RealNativePresetsPaneCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body)
    Platform.runLater(task)
    task.get(30, TimeUnit.SECONDS)

  private def eventually(label: String)(condition: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while !condition && System.nanoTime() < end do Thread.sleep(20)
    require(condition, s"Timed out: $label")

  private def sha(path: Path): String =
    val bytes = Files.readAllBytes(path)
    MessageDigest.getInstance("SHA-256").digest(bytes).map(b => f"${b & 0xff}%02x").mkString

  def main(args: Array[String]): Unit =
    require(args.length == 1, "output-directory")
    val output = Paths.get(args(0)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val artifacts = Files.createTempDirectory(output, "pane-run-")
    val home = AppHome(Files.createTempDirectory(output, "native-presets-home-"))
    val services = new AppServices(home, Settings(), dryRun = true,
      builtins = StandardPlugins.withOwnAxes, persistSettings = false, installShutdownHook = false)
    val jobEvents = new AtomicInteger(0)
    val removeQueueListener = services.queue.addListener(_ => { jobEvents.incrementAndGet(); () })
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS), "JavaFX startup timed out")
    var window: MainWindow = null
    var failure: Throwable = null
    var checks = Vector.empty[String]
    var screenshots = Vector.empty[Path]
    val openChoices = new java.util.concurrent.ConcurrentLinkedQueue[Path]()
    val saveChoices = new java.util.concurrent.ConcurrentLinkedQueue[Path]()
    val openedExternal = ArrayBuffer.empty[Path]
    val fileDialogs = new NativePresetFileDialogs:
      def chooseOpen(owner: Window): Option[Path] = Option(openChoices.poll())
      def chooseSave(owner: Window): Option[Path] = Option(saveChoices.poll())
      def openExternal(path: Path): Unit = { openedExternal += path; () }
    val library = home.configDir.resolve("native-presets").resolve("library-v1.json")
    require(!Files.exists(library.getParent), "fresh default preset directory unexpectedly pre-exists")
    val sourceConfig = artifacts.resolve("source-config-txt2img.json")
    Files.writeString(sourceConfig,
      """{"my 4K copy":{"txt2img_steps":38,"txt2img_hr_scale":5,"unknown_source_number":9.001e+25}}""")
    val sourceLongPrompt = artifacts.resolve("source-long-prompt.json")
    Files.writeString(sourceLongPrompt,
      "{\"Late source ID\":{\"txt2img_prompt\":\"" + ("p" * 8192) +
        "\",\"txt2img_steps\":39,\"unknown_after_prompt\":9.001e+25}}")
    val sourceTracked = artifacts.resolve("source-custom-tracked.txt")
    Files.writeString(sourceTracked, "# source custom IDs\ntxt2img_steps\nunknown_custom_component\n")
    val sourceAlias = artifacts.resolve("source-custom-alias.json")
    Files.writeString(sourceAlias, """{"Alias from source":{"my-height":512}}""")
    val sourceAdditional = artifacts.resolve("source-additional-components.json")
    Files.writeString(sourceAdditional,
      """{"additionalComponents":{"Steps":"txt2img_steps","Future control":null}}""")
    val sourceCustomAdditional = artifacts.resolve("source-custom-additional-components.json")
    Files.writeString(sourceCustomAdditional,
      """{"additionalComponents":{"My height":"txt2img_height","Future control":null}}""")
    val sourcePolicyAdditional = artifacts.resolve("source-policy-additional-components.json")
    Files.writeString(sourcePolicyAdditional,
      """{"additionalComponents":{"My height":null,"Manual height":"txt2img_height","Sampling steps":"txt2img_width","Unknown slot":"unknown_component_id"}}""")
    val sourceCustomUtilities = artifacts.resolve("source-custom-utilities.json")
    Files.writeString(sourceCustomUtilities, """{"Custom":{"My height":320}}""")
    val sourceOpaqueSweep = artifacts.resolve("source-opaque-sweep.json")
    val opaqueSweepMembers = Vector.tabulate(210)(i =>
      "\"unknown_source_%03d\":%d".format(i, i + 1)).mkString(",")
    Files.writeString(sourceOpaqueSweep,
      s"""{"Opaque sweep":{"img2img_steps":28,$opaqueSweepMembers}}""")
    val sourceConfigChoices = artifacts.resolve("source-config-index-choices.json")
    Files.writeString(sourceConfigChoices,
      """{"I2 Config numeric":{"resize_mode":2,"img2img_inpainting_fill":1},"I2 Config labels":{"resize_mode":"Resize and fill","img2img_inpainting_fill":"original"}}""")
    val sourceUtilityChoices = artifacts.resolve("source-utilities-index-choices.json")
    Files.writeString(sourceUtilityChoices,
      """{"I2 Utilities numeric":{"Resize mode":2,"Masked content":1},"I2 Utilities labels":{"Resize mode":"Resize and fill","Masked content":"original"}}""")
    def checked(label: String): Unit =
      checks :+= label
      println("PRESETS PANE PASS: " + label)
    def stage: Stage = window.presetsInspectorStage.getOrElse(sys.error("Preset stage unavailable"))
    def node[A <: Node](id: String): A =
      val found = Window.getWindows.asScala.collectFirst {
        case s: Stage if s.isShowing && s.getScene != null && s.getScene.lookup("#" + id) != null =>
          s.getScene.lookup("#" + id)
      }.orNull
      require(found != null, s"Missing actual native control #$id")
      found.asInstanceOf[A]
    def visibleText(item: Node): String = item match
      case input: TextInputControl => input.getText
      case labeled: Labeled => labeled.getText
      case parent: Parent => parent.getChildrenUnmodifiable.asScala.map(visibleText).mkString(" ")
      case _ => ""
    def descendants(item: Node): Vector[Node] = item match
      case parent: Parent => Vector(item) ++ parent.getChildrenUnmodifiable.asScala.toVector.flatMap(descendants)
      case _ => Vector(item)
    def fieldCheck(key: String): CheckBox =
      val id = "presets-field-check-" + key
      val table = fx { node[TableView[?]]("presets-fields") }
      var found: Option[CheckBox] = None
      for row <- 0 until table.getItems.size if found.isEmpty do
        fx { table.scrollTo(row) }
        found = fx {
          stage.getScene.getRoot.layout()
          descendants(table).collectFirst { case box: CheckBox if box.getId == id => box }
        }
      found.getOrElse(sys.error(s"No actual visible checklist checkbox $id"))
    def fieldRowText(key: String): (String, Double) =
      val check = fieldCheck(key)
      fx {
        val graphic = check.getParent
        val description = descendants(graphic).collectFirst {
          case value: Label if value.getText.contains(key) => value
        }.getOrElse(sys.error(s"No materialized field description for $key"))
        (description.getText, check.getParent.getBoundsInParent.getHeight)
      }
    def noWork(): Unit =
      require(services.supervisor.pid.isEmpty && services.supervisor.loadedModel.isEmpty,
        "preset operation started the engine or loaded a model")
      require(services.queue.isIdle && services.payloads.snapshots.isEmpty,
        "preset operation changed queue or submitted request captures")
      require(jobEvents.get == 0, "preset operation emitted a queue job event")
    def saved(name: String, collectionId: String): PresetRecord =
      val parsed = PresetJson.readNative(Files.readAllBytes(library), PresetLimits())
        .fold(e => sys.error(s"Cannot parse authoritative native library: $e"), identity)
      parsed.collections.find(_.id == collectionId).getOrElse(sys.error(s"Missing collection $collectionId"))
        .presets.find(_.name == name).getOrElse(sys.error(s"Missing saved preset $collectionId/$name"))
    def choose(combo: ComboBox[Any], fragment: String): Unit =
      val item = combo.getItems.toArray.toVector.find(x => x.toString.toLowerCase.contains(fragment.toLowerCase))
        .getOrElse(sys.error(s"Missing choice '$fragment' in ${combo.getId}: ${combo.getItems}"))
      combo.getSelectionModel.select(item)
    def seekImportPreview(fragment: String): Int =
      eventually(s"import review ready for $fragment") { fx {
        !node[Button]("presets-import-confirm").isDisabled
      }}
      var turns = 0
      var found = fx { visibleText(node[Node]("presets-import-preview")).contains(fragment) }
      while !found && turns < 1000 && fx { !node[Button]("presets-import-next").isDisabled } do
        fx { node[Button]("presets-import-next").fire() }
        turns += 1
        found = fx { visibleText(node[Node]("presets-import-preview")).contains(fragment) }
      require(found, s"Import review did not expose '$fragment' through the actual Next page control")
      turns
    def seekDiagnostic(previewId: String, nextId: String, fragment: String): Int =
      var turns = 0
      var found = fx { visibleText(node[Node](previewId)).contains(fragment) }
      while !found && turns < 1000 && fx { !node[Button](nextId).isDisabled } do
        fx { node[Button](nextId).fire() }
        turns += 1
        found = fx {
          val shown = visibleText(node[Node](previewId))
          require(shown.length <= 4096, s"#$previewId exceeded bounded diagnostic page")
          shown.contains(fragment)
        }
      require(found, s"$fragment was unreachable through actual #$nextId pages")
      turns
    def screenshot(name: String, width: Int, height: Int): Unit =
      fx {
        val scene = stage.getScene
        stage.setWidth(width + math.max(0.0, stage.getWidth - scene.getWidth))
        stage.setHeight(height + math.max(0.0, stage.getHeight - scene.getHeight))
      }
      eventually(s"settled scene $width x $height") { fx {
        math.round(stage.getScene.getWidth).toInt == width && math.round(stage.getScene.getHeight).toInt == height
      }}
      // Layout and snapshot run in a later FX pulse than resizing. The saved PNG is
      // checked against actual client dimensions, not guessed window decorations.
      fx {
        val scene = stage.getScene
        scene.getRoot.applyCss()
        scene.getRoot.layout()
        val png = SwingFXUtils.fromFXImage(scene.snapshot(null), null)
        require(png.getWidth == width && png.getHeight == height,
          s"Screenshot ${png.getWidth}x${png.getHeight} differs from settled client $width x $height")
        val path = artifacts.resolve(name)
        require(ImageIO.write(png, "png", path.toFile), s"PNG write failed: $path")
        val saved = ImageIO.read(path.toFile)
        require(saved != null && saved.getWidth == width && saved.getHeight == height,
          s"Saved PNG dimensions differ: $path")
        screenshots :+= path
      }
    try
      window = fx {
        val result = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => (), fileDialogs)
        result.show()
        result
      }
      fx {
        Mode.values.foreach { mode =>
          window.select(mode)
          val entry = window.scene.lookup(s"#presets-entry-${mode.key}").asInstanceOf[TitledPane]
          require(entry != null && !entry.isExpanded, s"${mode.key} preset entry should start collapsed")
          entry.setExpanded(true)
          val open = window.scene.lookup(s"#presets-open-${mode.key}").asInstanceOf[Button]
          require(open != null, s"missing ${mode.key} native open button")
          open.fire()
          require(stage.isShowing, s"${mode.key} entry did not open the preset stage")
          require(node[Label]("presets-destination").getText.toLowerCase.contains(mode.key),
            s"${mode.key} destination label disagrees with invoking mode")
        }
      }
      noWork()
      checked("all three collapsed mode entries open one owned native stage with destination labels")

      fx {
        window.openPresets(Mode.Txt2Img)
        choose(node[ComboBox[Any]]("presets-profile"), "Config")
        require(node[Label]("presets-collection").getText.contains("config-txt2img"))
        val tabs = node[TabPane]("presets-tabs")
        require(tabs.getTabs.size >= 4, "Quick, Detailed, Inspect and About must be available")
        require(tabs.getTabs.toArray.toVector.take(4).map(_.asInstanceOf[Tab].getText.toLowerCase) ==
          Vector("quick", "detailed", "inspect", "about"), "native tab order or labels differ")
      }
      eventually("fresh default directory and complete Config seed") {
        Files.isRegularFile(library) && fx {
          node[ComboBox[Any]]("presets-select").getItems.size >= 14
        }
      }
      for index <- 0 until 4 do
        fx { node[TabPane]("presets-tabs").getSelectionModel.select(index) }
        fx {
          index match
            case 0 =>
              node[ComboBox[Any]]("presets-select")
              node[Button]("presets-apply")
            case 1 =>
              node[Node]("presets-fields")
              node[TextField]("presets-name")
              node[Button]("presets-save")
              node[Button]("presets-delete")
              node[Button]("presets-import")
              node[Button]("presets-export")
            case 2 => node[Button]("presets-inspect")
            case _ => ()
        }
        screenshot(s"config-tab-$index-1040x740.png", 1040, 740)
      fx {
        val tabs = node[TabPane]("presets-tabs")
        tabs.getSelectionModel.select(0)
        val selector = node[ComboBox[Any]]("presets-select")
        require(selector.getItems.size >= 14, "complete source Config txt2img defaults were not seeded")
        val vm = window.vms(Mode.Txt2Img)
        vm.width.set(512)
        choose(selector, "with Refiner")
        require(vm.width.get == 1024 &&
          node[Label]("presets-status").getText.contains("compatibility warnings"),
          "built-in Refiner applied dimensions without disclosing its unresolved source value")
        require(visibleText(node[Node]("presets-apply-preview"))
          .contains("txt2img_enable-checkbox"),
          "Refiner source key was absent from actual Apply diagnostics")
        vm.width.set(512)
        node[Button]("presets-reapply").fire()
        require(vm.width.get == 1024 &&
          visibleText(node[Node]("presets-apply-preview")).contains("txt2img_enable-checkbox"),
          "Refiner Reapply lost typed changes or opaque warning")
        choose(selector, "Flux.1 Dev")
        require(vm.steps.get == 20 &&
          node[Label]("presets-status").getText.contains("compatibility warnings"),
          "built-in Flux preset did not apply steps and disclose partial compatibility")
      }
      seekDiagnostic("presets-apply-preview", "presets-apply-next",
        "txt2img_distilled_cfg_scale")
      fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.steps.set(37)
        node[Button]("presets-reapply").fire()
        require(vm.steps.get == 20,
          "Flux Reapply lost its known typed Steps value")
      }
      seekDiagnostic("presets-apply-preview", "presets-apply-next",
        "txt2img_distilled_cfg_scale")
      noWork()
      checked("built-in Refiner and Flux selector/Reapply disclose retained opaque source keys")

      fx {
        val selector = node[ComboBox[Any]]("presets-select")
        choose(selector, "4k") // user selection must apply through the real Quick selector
        val vm = window.vms(Mode.Txt2Img)
        require(vm.width.get == 768 && vm.height.get == 432 && vm.hiresScale.get == 5.0,
          "original 4K preset did not apply exact native base size and scale")
        require(vm.hiresEnabled.get && vm.hiresResizeX.get == 0 && vm.hiresResizeY.get == 0,
          "original 4K preset lost scale-based target")
        val hires = window.panels(Mode.Txt2Img).sections("txt2img.hires")
        hires.setExpanded(true)
        val fields = new NativePresetFields(vm, window.plugins(Mode.Txt2Img), window.catalogVM,
          new SimpleBooleanProperty(false))
        NativePresetFieldsWidgetAssertions.assertAttachedHiresScaleFive(hires, vm, fields)
        require(vm.width.get * vm.hiresScale.get == 3840.0 && vm.height.get * vm.hiresScale.get == 2160.0,
          "original 4K target math changed")
        node[Button]("presets-reapply").fire()
        require(vm.hiresScale.get == 5.0, "reapply clamped attached hires scale")
        vm.width.set(1024)
        node[Button]("presets-refresh").fire()
      }
      eventually("successful Refresh completed without apply") { fx {
        window.vms(Mode.Txt2Img).width.get == 1024 &&
          node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      fx {
        node[Button]("presets-reapply").fire()
        require(window.vms(Mode.Txt2Img).width.get == 768,
          "explicit Reapply after Refresh did not restore width")
        window.vms(Mode.Txt2Img).width.set(1024)
        choose(node[ComboBox[Any]]("presets-profile"), "Utilities")
        require(window.vms(Mode.Txt2Img).width.get == 1024, "profile change applied a preset")
        choose(node[ComboBox[Any]]("presets-profile"), "Config")
        require(window.vms(Mode.Txt2Img).width.get == 1024, "profile return reapplied a preset")
        choose(node[ComboBox[Any]]("presets-select"), "4k")
        if window.vms(Mode.Txt2Img).width.get != 768 then
          node[Button]("presets-reapply").fire()
        require(window.vms(Mode.Txt2Img).width.get == 768,
          "explicit 4K reselection or Reapply did not restore width")
      }
      noWork()
      checked("original 4K Quick apply, explicit reapply and passive refresh/profile changes")

      fx {
        val tabs = node[TabPane]("presets-tabs")
        tabs.getSelectionModel.select(1)
        val name = node[TextField]("presets-name")
        name.setText("my 4K copy")
        node[Button]("presets-save").fire()
      }
      eventually("4K saved through native Detailed form") {
        scala.util.Try(saved("my 4K copy", "config-txt2img")).isSuccess
      }
      val saved4k = saved("my 4K copy", "config-txt2img")
      require(saved4k.fields.get(FieldKey(Vector("generation", "hiresScale"))) ==
        Some(PresetAtom.Decimal(BigDecimal(5))), "Detailed save omitted scale 5")
      require(saved4k.fields.get(FieldKey(Vector("generation", "width"))) ==
        Some(PresetAtom.Integer(768)), "Detailed save omitted width 768")
      require(saved4k.fields.get(FieldKey(Vector("generation", "height"))) ==
        Some(PresetAtom.Integer(432)), "Detailed save omitted height 432")
      fx { node[TabPane]("presets-tabs").getSelectionModel.select(0) }
      fx { node[Button]("presets-refresh").fire() }
      eventually("save survives refresh") { fx {
        node[ComboBox[Any]]("presets-select").getItems.toArray.exists(_.toString.contains("my 4K copy"))
      }}
      fx {
        choose(node[ComboBox[Any]]("presets-select"), "my 4K copy")
        require(window.vms(Mode.Txt2Img).hiresScale.get == 5.0, "saved 4K copy failed to reapply")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      checked("native Detailed save, refresh and Quick reapply preserve 4K geometry")

      fx {
        val spinner = window.scene.lookup("#generation-steps-spinner-txt2img")
          .asInstanceOf[Spinner[Integer]]
        require(spinner != null, "actual Steps spinner is required for busy apply admission")
        val vm = window.vms(Mode.Txt2Img)
        val before = vm.steps.get
        val originalCommit = window.ctx.commitPresetEditors
        var commits = 0
        spinner.getEditor.setText("47")
        window.ctx.commitPresetEditors = m => { commits += 1; originalCommit(m) }
        try
          window.withPresetMaskExportBusyForCheck {
            node[TabPane]("presets-tabs").getSelectionModel.select(0)
            node[Button]("presets-apply").fire()
            require(node[Label]("presets-status").getText.contains("mask export"),
              "busy Apply did not show its admission error")
            node[Button]("presets-reapply").fire()
            require(node[Label]("presets-status").getText.contains("mask export"),
              "busy Reapply did not show its admission error")
            choose(node[ComboBox[Any]]("presets-select"), "None")
            require(node[Label]("presets-status").getText.contains("mask export"),
              "busy selector apply did not show its admission error")
            require(commits == 0 && vm.steps.get == before && spinner.getEditor.getText == "47",
              "rejected Apply/Reapply/selection committed a numeric draft before busy admission")
          }
        finally
          window.ctx.commitPresetEditors = originalCommit
          spinner.getEditor.setText(before.toString)
        choose(node[ComboBox[Any]]("presets-select"), "my 4K copy")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      noWork()
      checked("busy Apply, Reapply and selection reject before numeric editor commit")

      val validHash = sha(library)
      fx {
        val name = node[TextField]("presets-name")
        name.setText("   ")
        node[Button]("presets-save").fire()
        require(name.getText == "   ", "invalid name draft was erased during failed Save")
      }
      require(sha(library) == validHash, "invalid editor draft changed the native library")
      fx { node[Button]("presets-reload-tools").fire() }
      eventually("invalid draft retained across tools reload") { fx {
        node[TextField]("presets-name").getText == "   " &&
          node[Label]("presets-status").getText.toLowerCase.contains("name")
      }}
      fx { node[TextField]("presets-name").setText("my 4K copy") }
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        window.vms(Mode.Txt2Img).hiresScale.set(2.0)
        choose(node[ComboBox[Any]]("presets-select"), "my 4K copy")
        node[Button]("presets-reapply").fire()
        require(window.vms(Mode.Txt2Img).hiresScale.get == 5.0 &&
          window.vms(Mode.Txt2Img).width.get == 768 && window.vms(Mode.Txt2Img).height.get == 432,
          "saved 4K preset lost exact geometry after Reload preset tools")
        val vm = window.vms(Mode.Txt2Img)
        val fields = new NativePresetFields(vm, window.plugins(Mode.Txt2Img), window.catalogVM,
          new SimpleBooleanProperty(false))
        NativePresetFieldsWidgetAssertions.assertAttachedHiresScaleFive(
          window.panels(Mode.Txt2Img).sections("txt2img.hires"), vm, fields)
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      noWork()
      checked("invalid draft survives reload; saved 4K geometry survives reapply in attached widgets")

      fx {
        window.select(Mode.Txt2Img)
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      val stepsCheck = fieldCheck("generation.steps")
      fx { if !stepsCheck.isSelected then stepsCheck.fire() }
      val stepsSpinner = fx {
        val found = window.scene.lookup("#generation-steps-spinner-txt2img")
        require(found != null, "actual Sampling Steps spinner lacks its stable ID")
        found.asInstanceOf[Spinner[Integer]]
      }
      val stepsBeforeDraft = fx { window.vms(Mode.Txt2Img).steps.get }
      val hashBeforeNumericDraft = sha(library)
      fx {
        node[TextField]("presets-name").setText("Numeric draft")
        stepsSpinner.getEditor.setText("-") // a real intermediate integer edit, not a hidden VM write
        node[Button]("presets-save").fire()
        require(stepsSpinner.getEditor.getText == "-" &&
          window.vms(Mode.Txt2Img).steps.get == stepsBeforeDraft,
          "failed Save erased an invalid numeric draft or captured the stale Steps value")
        val status = node[Label]("presets-status").getText.toLowerCase
        require(status.contains("step") || status.contains("number") || status.contains("invalid"),
          "failed Save did not explain the invalid spinner editor")
      }
      require(sha(library) == hashBeforeNumericDraft &&
        scala.util.Try(saved("Numeric draft", "config-txt2img")).isFailure,
        "failed numeric Save wrote a stale valid preset")
      fx {
        stepsSpinner.getEditor.setText(stepsBeforeDraft.toString)
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        choose(node[ComboBox[Any]]("presets-select"), "Low quality")
        val committed = window.vms(Mode.Txt2Img).steps.get
        require(committed == 10, "Steps-bearing source preset did not establish the apply fixture")
        stepsSpinner.getEditor.setText("-")
        node[Button]("presets-reapply").fire()
        require(stepsSpinner.getEditor.getText == "-" &&
          window.vms(Mode.Txt2Img).steps.get == committed,
          "failed Apply reset invalid text or mutated the previously committed Steps")
        node[Button]("presets-reload-tools").fire()
      }
      eventually("invalid numeric editor remains after Reload preset tools") { fx {
        stepsSpinner.getEditor.getText == "-" && window.vms(Mode.Txt2Img).steps.get == 10 &&
          node[Label]("presets-status").getText.toLowerCase.contains("invalid")
      }}
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        node[TextField]("presets-name").setText("Spinner commit")
        stepsSpinner.getEditor.setText("44")
        node[Button]("presets-save").fire()
      }
      eventually("valid spinner editor committed before capture") {
        scala.util.Try(saved("Spinner commit", "config-txt2img")).toOption.exists { record =>
          record.fields.get(FieldKey(Vector("generation", "steps"))) == Some(PresetAtom.Integer(44))
        }
      }
      require(fx { window.vms(Mode.Txt2Img).steps.get == 44 &&
        stepsSpinner.getEditor.getText == "44" },
        "valid text did not commit into the actual Steps VM and spinner")
      noWork()
      checked("actual Steps spinner preserves invalid intermediate text across failed Save/Apply/Reload and commits valid 44")

      val hiresScaleSpinner = fx {
        descendants(window.panels(Mode.Txt2Img).sections("txt2img.hires").getContent)
          .collectFirst { case s: Spinner[?]
              if s.getValueFactory.isInstanceOf[SpinnerValueFactory.DoubleSpinnerValueFactory] &&
                s.getValueFactory.asInstanceOf[SpinnerValueFactory.DoubleSpinnerValueFactory].getMin == 1.0 &&
                s.getValueFactory.asInstanceOf[SpinnerValueFactory.DoubleSpinnerValueFactory].getMax == 5.0 =>
            s.asInstanceOf[Spinner[java.lang.Double]]
          }.getOrElse(sys.error("actual Hires scale spinner missing"))
      }
      val beforeBadHires = sha(library)
      val hiresScaleBefore = fx { window.vms(Mode.Txt2Img).hiresScale.get }
      fx {
        node[TextField]("presets-name").setText("Invalid Hires draft")
        hiresScaleSpinner.getEditor.setText("NaN")
        node[Button]("presets-save").fire()
        require(hiresScaleSpinner.getEditor.getText == "NaN" &&
          window.vms(Mode.Txt2Img).hiresScale.get == hiresScaleBefore,
          "nonfinite Hires spinner draft was erased or changed the VM")
      }
      require(sha(library) == beforeBadHires,
        "nonfinite Hires draft wrote a stale preset")
      fx { hiresScaleSpinner.getEditor.setText(hiresScaleBefore.toString) }
      noWork()
      checked("actual nonfinite Hires spinner blocks Save without changing the file or editor draft")

      val savedId = saved("my 4K copy", "config-txt2img").id
      def previewConfigImport(collision: String): Unit =
        openChoices.add(sourceConfig)
        fx {
          choose(node[ComboBox[Any]]("presets-import-format"), "Config")
          node[Button]("presets-import").fire()
        }
        eventually("Config import review is visible") { fx {
          val text = visibleText(node[Node]("presets-import-preview"))
          text.contains("my 4K copy") && text.contains("txt2img_steps") &&
            text.contains("unknown_source_number") &&
            text.toLowerCase.contains("conflict")
        }}
        fx {
          if collision == "Skip" then
            require(Option(node[ComboBox[Any]]("presets-import-collision").getValue)
              .exists(_.toString.contains("Skip")), "collision default is not Skip")
          choose(node[ComboBox[Any]]("presets-import-collision"), collision)
          node[Button]("presets-import-confirm").fire()
        }
      val beforeSkip = sha(library)
      previewConfigImport("Skip")
      eventually("Skip import finished") { fx {
        !node[Label]("presets-status").getText.toLowerCase.contains("loading")
      }}
      require(saved("my 4K copy", "config-txt2img").fields == saved4k.fields,
        "default Skip collision changed the saved preset")
      require(sha(library) == beforeSkip, "Skip collision rewrote the authoritative file")
      previewConfigImport("Overwrite")
      eventually("Overwrite import installed changed source steps") {
        saved("my 4K copy", "config-txt2img").fields.get(FieldKey(Vector("generation", "steps"))) ==
          Some(PresetAtom.Integer(38))
      }
      val imported = saved("my 4K copy", "config-txt2img")
      require(imported.id == savedId, "same-name source overwrite changed stable native identity")
      require(imported.opaque.exists(_._1 == "unknown_source_number"),
        "unknown source entry was lost on import")
      noWork()
      checked("explicit Config import review defaults to Skip and Overwrite retains identity and opaque values")

      // A bounded source preview must enumerate IDs after a large first value;
      // showing only a prefix of the original JSON would hide the known Steps ID.
      openChoices.add(sourceLongPrompt)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Config")
        node[Button]("presets-import").fire()
      }
      eventually("source ID after long Prompt remains visible in import review") { fx {
        val preview = visibleText(node[Node]("presets-import-preview"))
        preview.contains("Late source ID") && preview.contains("txt2img_prompt") &&
          preview.contains("txt2img_steps") && preview.contains("generation.steps") &&
          preview.contains("unknown_after_prompt") && preview.length <= 4096
      }}
      noWork()
      checked("bounded Config import review enumerates source IDs after a long first Prompt")

      val beforeTracked = sha(library)
      openChoices.add(sourceTracked)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Tracked")
        node[Button]("presets-import").fire()
      }
      eventually("tracked text review shows known and unavailable IDs") { fx {
        val text = visibleText(node[Node]("presets-import-preview"))
        text.contains("txt2img_steps") && text.contains("unknown_custom_component")
      }}
      fx {
        val addTracked = node[CheckBox]("presets-import-add-tracked")
        if !addTracked.isSelected then addTracked.fire()
        node[Button]("presets-import-confirm").fire()
      }
      eventually("tracked text import committed its retained diagnostics") {
        Files.isRegularFile(library) && sha(library) != beforeTracked
      }
      noWork()
      checked("explicit custom tracked text import displays known and unavailable source IDs")

      // An external declarative edit adds a native-only tracked value and a
      // custom reverse alias. Refresh and Inspect must show both without apply.
      val configStore = new PresetStore(library.getParent, PresetLimits())
      val currentConfig = configStore.load().toOption.get
      val configIndex = currentConfig.library.collections.indexWhere(_.id == "config-txt2img")
      val configCollection = currentConfig.library.collections(configIndex)
      val importedIndex = configCollection.presets.indexWhere(_.name == "my 4K copy")
      val nativeOnly = FieldKey(Vector("generation", "hiresScheduler"))
      val customSteps = FieldKey(Vector("generation", "steps"))
      val customHeight = FieldKey(Vector("generation", "height"))
      val customRecord = configCollection.presets(importedIndex).copy(
        fields = configCollection.presets(importedIndex).fields.updated(nativeOnly, PresetAtom.Text("Automatic")),
        dirty = configCollection.presets(importedIndex).dirty + nativeOnly)
      val customCollection = configCollection.copy(
        tracked = configCollection.tracked + nativeOnly,
        bindings = configCollection.bindings ++ Vector(
          SourceBinding("config-presets", "my_custom_steps", customSteps),
          SourceBinding("config-presets", "my-height", customHeight)),
        presets = configCollection.presets.updated(importedIndex, customRecord))
      val externalConfig = currentConfig.library.copy(collections =
        currentConfig.library.collections.updated(configIndex, customCollection))
      require(configStore.save(currentConfig.revision, externalConfig).isRight,
        "external declarative mapping fixture failed")
      fx { node[Button]("presets-refresh").fire() }
      eventually("external custom mapping loaded") { fx {
        !node[Label]("presets-status").getText.toLowerCase.contains("loading")
      }}
      fx {
        node[Button]("presets-open-backing").fire()
        node[Button]("presets-open-fields").fire()
        require(openedExternal.takeRight(2).toVector == Vector(library, library),
          "open backing and field-list actions did not resolve the fixed native file")
        node[TabPane]("presets-tabs").getSelectionModel.select(2)
        node[Button]("presets-inspect").fire()
        val search = node[TextField]("presets-inspect-search")
        Vector(
          "generation.steps" -> Vector("generation.steps"),
          "my_custom_steps" -> Vector("generation.steps", "my_custom_steps"),
          "generation.hiresScheduler" -> Vector("generation.hiresScheduler")
        ).foreach { (query, expected) =>
          search.setText(query)
          node[Button]("presets-inspect").fire()
          val facts = visibleText(node[Node]("presets-inspect-preview"))
          require(expected.forall(facts.contains),
            s"Inspect search $query omitted ${expected.mkString(", ")}: $facts")
        }
        search.setText("")
      }
      noWork()
      checked("external custom mapping refresh, fixed-path Open actions and actual native Inspect")

      fx {
        choose(node[ComboBox[Any]]("presets-profile"), "Utilities")
        val stale = visibleText(node[Node]("presets-inspect-preview"))
        require(!stale.contains("Collection config-txt2img") && !stale.contains("my_custom_steps"),
          "Inspect carried Config facts into Utilities without a new Gather")
        node[Button]("presets-inspect").fire()
        require(visibleText(node[Node]("presets-inspect-preview")).contains("Collection utilities"),
          "Utilities Gather did not show its own collection")
        window.openPresets(Mode.Img2Img)
        val afterDestination = visibleText(node[Node]("presets-inspect-preview"))
        require(!afterDestination.contains("mode txt2img"),
          "Inspect kept the previous destination's facts after destination change")
        node[Button]("presets-inspect").fire()
        require(visibleText(node[Node]("presets-inspect-preview")).contains("mode img2img"),
          "explicit Gather did not use the new destination")
        window.openPresets(Mode.Inpaint)
        require(!visibleText(node[Node]("presets-inspect-preview")).contains("mode img2img"),
          "Inspect kept Img2Img facts after changing destination to Inpaint")
        node[Button]("presets-inspect").fire()
        require(visibleText(node[Node]("presets-inspect-preview")).contains("mode inpaint"),
          "explicit Gather did not use Inpaint destination")
        node[Button]("presets-reload-tools").fire()
      }
      eventually("successful Reload invalidates previously gathered Inspect facts") { fx {
        val preview = visibleText(node[Node]("presets-inspect-preview"))
        val status = node[Label]("presets-status").getText.toLowerCase
        preview.startsWith("Gather / inspect to show the current profile") &&
          !preview.contains("mode inpaint") &&
          status.contains("invalid numeric editor") && status.contains("nan")
      }}
      fx {
        window.openPresets(Mode.Txt2Img)
        choose(node[ComboBox[Any]]("presets-profile"), "Config")
      }
      noWork()
      checked("Inspect invalidates old profile, destination and successful Reload facts")

      val unknownBeforeReapply = saved("my 4K copy", "config-txt2img").opaque
        .find(_._1 == "unknown_source_number").get._2.text
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        val selector = node[ComboBox[Any]]("presets-select")
        choose(selector, "None")
        window.vms(Mode.Txt2Img).steps.set(11)
        choose(selector, "my 4K copy")
        require(window.vms(Mode.Txt2Img).steps.get == 38 &&
          node[Label]("presets-status").getText.contains("compatibility warnings") &&
          visibleText(node[Node]("presets-apply-preview")).contains("unknown_source_number"),
          "saved/reloaded import applied known Steps without warning about its unknown source key")
        window.vms(Mode.Txt2Img).steps.set(12)
        node[Button]("presets-reapply").fire()
        require(window.vms(Mode.Txt2Img).steps.get == 38 &&
          visibleText(node[Node]("presets-apply-preview")).contains("unknown_source_number"),
          "Reapply lost imported unknown-source warning or known Steps value")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      require(saved("my 4K copy", "config-txt2img").opaque
        .find(_._1 == "unknown_source_number").get._2.text == unknownBeforeReapply,
        "native Reload/Apply changed an opaque source numeric lexeme")
      noWork()
      checked("imported unknown source survives native Reload and warns on selector/Reapply")

      fx { node[TabPane]("presets-tabs").getSelectionModel.select(1) }
      val sourceExport = artifacts.resolve("export-config-txt2img.json")
      saveChoices.add(sourceExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Config")
        node[Button]("presets-export").fire()
      }
      eventually("source export omission review") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("generation.hiresScheduler")
      }}
      require(!Files.exists(sourceExport), "source export silently omitted native-only field")
      fx {
        val supported = node[CheckBox]("presets-export-supported-only")
        require(!supported.isSelected, "lossy supported-only export was preselected")
        supported.fire()
        node[Button]("presets-export-confirm").fire()
      }
      eventually("supported-only source file written") { Files.isRegularFile(sourceExport) }
      val sourceBytes = Files.readString(sourceExport)
      require(sourceBytes.contains("unknown_source_number") && sourceBytes.contains("9.001e+25") &&
        sourceBytes.contains("txt2img_steps") && !sourceBytes.contains("hiresScheduler"),
        "supported-only source export lost opaque data or invented a native-only source key")
      val nativeExport = artifacts.resolve("export-native-library.json")
      saveChoices.add(nativeExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Native")
        node[Button]("presets-export").fire()
      }
      eventually("native export destination review") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Ready to export")
      }}
      fx {
        node[Button]("presets-export-confirm").fire()
      }
      eventually("lossless native export file written") { Files.isRegularFile(nativeExport) }
      require(PresetJson.readNative(Files.readAllBytes(nativeExport), PresetLimits()).toOption.get
        .collections.find(_.id == "config-txt2img").get.presets.exists(_.name == "my 4K copy"),
        "native full export did not retain the Config record")
      val trackedExport = artifacts.resolve("export-config-tracked.txt")
      saveChoices.add(trackedExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Tracked")
        node[Button]("presets-export").fire()
      }
      eventually("tracked text export lists native-only omission") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("generation.hiresScheduler")
      }}
      fx {
        val supported = node[CheckBox]("presets-export-supported-only")
        require(!supported.isSelected, "tracked text export silently selected lossy mode")
        supported.fire()
        node[Button]("presets-export-confirm").fire()
      }
      eventually("tracked source text written") { Files.isRegularFile(trackedExport) }
      require(PresetSourceFiles.readTrackedIds(Files.readAllBytes(trackedExport), PresetLimits())
        .toOption.get.contains("txt2img_steps"), "tracked text roundtrip lost source Steps ID")
      noWork()
      checked("explicit source/native/tracked export, omission review and supported-only confirmation")

      openChoices.add(sourceAlias)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Config")
        node[Button]("presets-import").fire()
      }
      eventually("custom alias import review shows mapped field") { fx {
        val text = visibleText(node[Node]("presets-import-preview"))
        text.contains("Alias from source") && text.contains("my-height") &&
          text.contains("generation.height")
      }}
      fx { node[Button]("presets-import-confirm").fire() }
      eventually("custom alias imported with original source provenance") {
        scala.util.Try(saved("Alias from source", "config-txt2img")).toOption.exists { record =>
          record.fields.get(FieldKey(Vector("generation", "height"))) ==
            Some(PresetAtom.Integer(512)) && record.dirty.isEmpty && record.source.nonEmpty
        }
      }
      val aliasId = saved("Alias from source", "config-txt2img").id
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        choose(node[ComboBox[Any]]("presets-select"), "Alias from source")
        require(window.vms(Mode.Txt2Img).height.get == 512,
          "custom source alias did not apply to actual Height VM")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        require(node[TextField]("presets-name").getText == "Alias from source",
          "Detailed editor did not populate imported alias name")
      }
      val heightCheck = fieldCheck("generation.height")
      require(fx { heightCheck.isSelected }, "imported custom alias was not checked for Detailed save")
      fx {
        val found = window.scene.lookup("#generation-height-spinner-txt2img")
        require(found != null, "actual Size Height spinner lacks its stable ID")
        val spinner = found.asInstanceOf[Spinner[Integer]]
        spinner.getEditor.setText("768")
        node[Button]("presets-save").fire()
      }
      eventually("UI Height spinner edit marks imported alias field dirty") {
        scala.util.Try(saved("Alias from source", "config-txt2img")).toOption.exists { record =>
          record.id == aliasId &&
            record.fields.get(FieldKey(Vector("generation", "height"))) == Some(PresetAtom.Integer(768)) &&
            record.dirty.contains(FieldKey(Vector("generation", "height")))
        }
      }
      require(fx { window.vms(Mode.Txt2Img).height.get == 768 },
        "actual Height spinner edit was not committed before native Save")
      val aliasExport = artifacts.resolve("export-custom-alias.json")
      saveChoices.add(aliasExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Config")
        node[Button]("presets-export").fire()
      }
      eventually("custom alias export omission review") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("generation.hiresScheduler")
      }}
      fx {
        val supported = node[CheckBox]("presets-export-supported-only")
        if !supported.isSelected then supported.fire()
        node[Button]("presets-export-confirm").fire()
      }
      eventually("edited custom alias source export") { Files.isRegularFile(aliasExport) }
      require(Files.readString(aliasExport).contains("\"my-height\":768"),
        "edited custom alias did not reverse-map through explicit binding")
      openChoices.add(aliasExport)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Config")
        node[Button]("presets-import").fire()
      }
      try
        val advanced = seekImportPreview("Alias from source")
        if advanced > 0 then fx {
          node[Button]("presets-import-previous").fire()
          node[Button]("presets-import-next").fire()
          require(visibleText(node[Node]("presets-import-preview")).contains("Alias from source"),
            "Previous/Next review navigation lost the source alias row")
        }
      catch case e: IllegalArgumentException =>
        val diagnostic = fx {
          val preview = visibleText(node[Node]("presets-import-preview"))
          val status = node[Label]("presets-status").getText
          val confirmDisabled = node[Button]("presets-import-confirm").isDisabled
          val scene = stage.getScene
          scene.getRoot.applyCss()
          scene.getRoot.layout()
          val png = SwingFXUtils.fromFXImage(scene.snapshot(null), null)
          val path = artifacts.resolve("alias-reimport-review-failure.png")
          require(ImageIO.write(png, "png", path.toFile), s"Failure PNG write failed: $path")
          (status, preview, confirmDisabled, path)
        }
        println(s"PRESETS PANE DIAGNOSTIC: alias reimport status=${diagnostic._1}; " +
          s"confirmDisabled=${diagnostic._3}; openChoices=${openChoices.size()}; " +
          s"previewChars=${diagnostic._2.length}; footer=${diagnostic._2.linesIterator.toVector.lastOption.getOrElse("")}; " +
          s"failurePng=${diagnostic._4}")
        println("PRESETS PANE DIAGNOSTIC preview:\n" + diagnostic._2)
        throw e
      fx {
        choose(node[ComboBox[Any]]("presets-import-collision"), "Overwrite")
        node[Button]("presets-import-confirm").fire()
      }
      eventually("reimport resets dirty provenance while retaining custom mapping and value") {
        scala.util.Try(saved("Alias from source", "config-txt2img")).toOption.exists { record =>
          record.id == aliasId && record.dirty.isEmpty &&
            record.fields.get(FieldKey(Vector("generation", "height"))) ==
              Some(PresetAtom.Integer(768)) &&
            record.source.exists(_.originalFields.text.contains("\"my-height\":768"))
        }
      }
      noWork()
      checked("custom source alias imports, actual Height spinner edit saves dirty, exports alias and reimports clean")

      eventually("custom alias reimport callback reached the visible editor") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ")
      }}

      val removalBefore = saved("my 4K copy", "config-txt2img")
      require(removalBefore.fields.contains(
        FieldKey(Vector("generation", "steps"))),
        "same-name removal fixture needs a previously saved known Steps field")

      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        choose(node[ComboBox[Any]]("presets-select"), "my 4K copy")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        node[Button]("presets-edit").fire()
      }
      val removalCheck = fieldCheck("generation.steps")
      require(fx { removalCheck.isSelected }, "source Steps field was not checked before overwrite")
      val beforeCancel = sha(library)
      val beforeCancelSteps = fx { window.vms(Mode.Txt2Img).steps.get }
      fx {
        removalCheck.fire()
        val preview = visibleText(node[Node]("presets-save-preview"))
        require(preview.contains("my 4K copy") && preview.toLowerCase.contains("overwrite") &&
          preview.contains("generation.steps") && preview.toLowerCase.contains("remove") &&
          preview.toLowerCase.contains("opaque") && preview.toLowerCase.contains("selected"),
          "Detailed pre-click preview hid overwrite scope, known removal or opaque retention")
        node[Button]("presets-cancel").fire()
        require(!node[Node]("presets-editor").isVisible,
          "Cancel did not collapse the Config editor")
        require(window.vms(Mode.Txt2Img).steps.get == beforeCancelSteps,
          "Cancel applied or captured a generation value")
      }
      require(sha(library) == beforeCancel, "Cancel wrote the native library")
      fx {
        node[Button]("presets-edit").fire()
        require(node[Node]("presets-editor").isVisible,
          "Edit did not reopen the Config editor")
      }
      val stepsAgain = fieldCheck("generation.steps")
      fx {
        if stepsAgain.isSelected then stepsAgain.fire()
        require(visibleText(node[Node]("presets-save-preview")).contains("generation.steps"),
          "reopened Detailed editor did not disclose the known field removal")
        node[Button]("presets-save").fire()
      }
      eventually("same-name Detailed overwrite removes disclosed Steps only") {
        scala.util.Try(saved("my 4K copy", "config-txt2img")).toOption.exists { record =>
          record.id == savedId &&
            record.fields.keySet == removalBefore.fields.keySet - FieldKey(Vector("generation", "steps")) &&
            record.opaque == removalBefore.opaque &&
            record.opaque.exists(_._1 == "unknown_source_number")
        }
      }
      eventually("same-name Detailed save callback finished before another mutation") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      noWork()
      checked("Config Edit and Cancel preserve form/library; pre-click overwrite discloses removal and retains identity/opaque")

      fx {
        node[TextField]("presets-name").setText("Overlap guard")
        node[Button]("presets-save").fire()
        node[Button]("presets-delete").fire()
        require(node[Label]("presets-status").getText.toLowerCase.contains("busy"),
          "overlapping Delete did not report Busy while Save owns the mutation gate")
        openChoices.add(sourceConfig)
        node[Button]("presets-import").fire()
        require(openChoices.size() == 1 &&
          node[Label]("presets-status").getText.toLowerCase.contains("busy"),
          "overlapping Import opened the file chooser or escaped the Busy gate")
        openChoices.clear()
      }
      eventually("accepted save survives rejected overlapping delete") {
        scala.util.Try(saved("Overlap guard", "config-txt2img")).isSuccess
      }
      require(saved("my 4K copy", "config-txt2img").id == savedId,
        "overlapping save/delete removed an older record")
      noWork()
      checked("same FX-turn Save, Delete and Import reject Busy without losing records")

      fx {
        choose(node[ComboBox[Any]]("presets-profile"), "Utilities")
        require(node[Label]("presets-collection").getText.contains("utilities"))
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        val quick = node[ComboBox[Any]]("presets-utilities-quick-select")
        val quickStack = node[CheckBox]("presets-utilities-quick-stackable")
        require(quickStack.isSelected, "Quick Stackable default must be true")
        quickStack.fire()
        require(!quickStack.isSelected, "Quick Stackable did not toggle")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        require(node[CheckBox]("presets-utilities-detailed-stackable").isSelected,
          "Quick Stackable changed Detailed default")
        val detailedCount = node[ComboBox[Any]]("presets-utilities-detailed-select").getItems.size
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        quickStack.fire()
        require(quick.getItems.size == detailedCount, "Utilities selectors diverged")
        window.vms(Mode.Txt2Img).steps.set(37)
        window.vms(Mode.Txt2Img).cfgScale.set(9.0)
        node[TextField]("presets-quick-name").setText("Quick all")
        node[Button]("presets-quick-save").fire()
      }
      eventually("Utilities Quick captured all available tracked fields") {
        Files.isRegularFile(library) &&
          scala.util.Try(saved("Quick all", "utilities")).toOption.exists { record =>
            record.fields.get(FieldKey(Vector("generation", "steps"))) == Some(PresetAtom.Integer(37)) &&
              record.fields.get(FieldKey(Vector("generation", "cfgScale"))) ==
                Some(PresetAtom.Decimal(BigDecimal(9)))
          }
      }
      val quickRecord = saved("Quick all", "utilities")
      require(quickRecord.fields.size > 2, "Quick save captured a selected subset instead of all available tracked fields")
      eventually("Quick all save callback updated the visible collection") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ") &&
          node[ComboBox[Any]]("presets-utilities-quick-select").getItems.toArray
            .exists(_.toString == "Quick all")
      }}
      fx {
        node[TextField]("presets-quick-name").setText("Quick all")
        val preview = visibleText(node[Node]("presets-quick-save-preview"))
        require(preview.contains("Quick all") && preview.toLowerCase.contains("overwrite") &&
          preview.toLowerCase.contains("all") && preview.toLowerCase.contains("tracked") &&
          preview.toLowerCase.contains("opaque"),
          "Quick pre-click save did not disclose exact-name overwrite, all-tracked scope or opaque retention")
        val quick = node[ComboBox[Any]]("presets-utilities-quick-select")
        require(quick.getItems.toArray.exists(_.toString.contains("Quick all")),
          "Quick selector was not updated")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        val detailed = node[ComboBox[Any]]("presets-utilities-detailed-select")
        require(detailed.getItems.toArray.exists(_.toString.contains("Quick all")),
          "Quick save did not update both Utilities selectors")
      }
      checked("Utilities Quick save captures all available tracked fields and synchronizes both selectors")

      // Simulate another editor adding a one-field Utilities record. Refresh must
      // reveal it without applying, then Detailed selection must populate real checks.
      val store = new PresetStore(library.getParent, PresetLimits())
      val externalBase = store.load().toOption.get
      val utilIndex = externalBase.library.collections.indexWhere(_.id == "utilities")
      require(utilIndex >= 0, "Utilities collection missing after Quick save")
      val utilCollection = externalBase.library.collections(utilIndex)
      val fixtureName = "Steps only fixture"
      val fixture = PresetRecord(PresetRecord.initialId("utilities", fixtureName), fixtureName,
        Map(FieldKey(Vector("generation", "steps")) -> PresetAtom.Integer(41)),
        None, Set.empty, Vector.empty, Vector.empty)
      val editedCollection = utilCollection.copy(presets = utilCollection.presets :+ fixture)
      val editedLibrary = externalBase.library.copy(collections =
        externalBase.library.collections.updated(utilIndex, editedCollection))
      require(store.save(externalBase.revision, editedLibrary).isRight,
        "external editor fixture failed to write a valid revision")
      fx { node[Button]("presets-refresh").fire() }
      eventually("external Utilities addition appears in Detailed selector") { fx {
        node[ComboBox[Any]]("presets-utilities-detailed-select").getItems.toArray
          .exists(_.toString.contains(fixtureName))
      }}
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        choose(node[ComboBox[Any]]("presets-utilities-detailed-select"), fixtureName)
        require(node[TextField]("presets-name").getText == fixtureName,
          "Detailed selection did not populate the saved name")
      }
      val utilitiesStepsCheck = fieldCheck("generation.steps")
      require(fx { utilitiesStepsCheck.isSelected },
        "Detailed selection did not check its recognized saved Steps field")
      val cfgCheck = fieldCheck("generation.cfgScale")
      require(!fx { cfgCheck.isSelected },
        "Detailed selection checked absent CFG")
      fx {
        require(node[CheckBox]("presets-utilities-detailed-stackable").isSelected,
          "Detailed Stackable should remain independently true")
        window.vms(Mode.Txt2Img).steps.set(43)
        window.vms(Mode.Txt2Img).cfgScale.set(15.0)
        node[TextField]("presets-name").setText("Detailed one")
        node[Button]("presets-save").fire()
      }
      eventually("Detailed captured only checked Steps") {
        scala.util.Try(saved("Detailed one", "utilities")).toOption.exists { record =>
          record.fields.keySet == Set(FieldKey(Vector("generation", "steps"))) &&
            record.fields(FieldKey(Vector("generation", "steps"))) == PresetAtom.Integer(43)
        }
      }
      eventually("Detailed save result synchronized the actual selector") { fx {
        val detailed = node[ComboBox[Any]]("presets-utilities-detailed-select")
        Option(detailed.getValue).exists(_.toString == "Detailed one") &&
          node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      fx {
        window.vms(Mode.Txt2Img).steps.set(20)
        window.vms(Mode.Txt2Img).cfgScale.set(16.0)
        val detailed = node[ComboBox[Any]]("presets-utilities-detailed-select")
        choose(detailed, fixtureName)
        require(window.vms(Mode.Txt2Img).steps.get == 41 && window.vms(Mode.Txt2Img).cfgScale.get == 16.0,
          "Detailed Stackable fixture selection changed an absent CFG field")
        choose(detailed, "Detailed one")
        require(window.vms(Mode.Txt2Img).steps.get == 43 && window.vms(Mode.Txt2Img).cfgScale.get == 16.0,
          "Detailed Stackable overlay changed an absent tracked field")
        val detailedStack = node[CheckBox]("presets-utilities-detailed-stackable")
        detailedStack.fire()
        require(!detailedStack.isSelected, "Detailed Stackable did not toggle")
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        require(node[CheckBox]("presets-utilities-quick-stackable").isSelected,
          "Detailed toggle changed Quick Stackable")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        window.vms(Mode.Txt2Img).cfgScale.set(16.0)
        choose(node[ComboBox[Any]]("presets-utilities-detailed-select"), fixtureName)
        choose(node[ComboBox[Any]]("presets-utilities-detailed-select"), "Detailed one")
        require(window.vms(Mode.Txt2Img).steps.get == 43 && window.vms(Mode.Txt2Img).cfgScale.get == 7.0,
          "Detailed defaults base did not reset absent tracked CFG")
      }
      noWork()
      checked("Detailed selection populates name/checks and its own Stackable toggle controls apply base")

      val beforeAdditional = sha(library)
      openChoices.add(sourceAdditional)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Additional")
        node[Button]("presets-import").fire()
      }
      eventually("additional-components import review") { fx {
        val text = visibleText(node[Node]("presets-import-preview"))
        text.contains("Steps") && text.contains("Future control")
      }}
      fx { node[Button]("presets-import-confirm").fire() }
      eventually("additional-components import commits") {
        Files.isRegularFile(library) && sha(library) != beforeAdditional
      }
      val additionalExport = artifacts.resolve("export-additional-components.json")
      saveChoices.add(additionalExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Additional")
        node[Button]("presets-export").fire()
      }
      eventually("additional components export destination review") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Ready to export")
      }}
      fx {
        node[Button]("presets-export-confirm").fire()
      }
      eventually("additional-components JSON export") { Files.isRegularFile(additionalExport) }
      val additionalEntries = PresetSourceFiles.readAdditionalComponents(
        Files.readAllBytes(additionalExport), PresetLimits()).toOption.get
      require(additionalEntries.contains("Steps" -> Some("txt2img_steps")) &&
        additionalEntries.contains("Future control" -> None),
        "additionalComponents label/ID/null roundtrip changed")
      saveChoices.add(additionalExport)
      fx { node[Button]("presets-export").fire() }
      eventually("existing export target revision appears in visible review") { fx {
        val preview = visibleText(node[Node]("presets-export-preview"))
        preview.contains("Destination revision") && preview.contains("Ready to export")
      }}
      Files.writeString(additionalExport, "external target edit after review")
      fx { node[Button]("presets-export-confirm").fire() }
      eventually("external export target edit is rejected") { fx {
        val text = node[Label]("presets-status").getText.toLowerCase
        text.contains("conflict") || text.contains("revision")
      }}
      require(Files.readString(additionalExport) == "external target edit after review",
        "stale export review overwrote a changed target file")
      noWork()
      checked("additionalComponents roundtrip and stale external export-target guard")

      // The Additional Components file supplies the source label -> component ID
      // mapping used by a separate Preset Utilities JSON import. The mapping must
      // remain usable after Reload, including when its label is not a built-in one.
      val beforeCustomAdditional = sha(library)
      openChoices.add(sourceCustomAdditional)
      fx {
        val add = node[CheckBox]("presets-import-add-tracked")
        if add.isSelected then add.fire()
        choose(node[ComboBox[Any]]("presets-import-format"), "Additional")
        node[Button]("presets-import").fire()
      }
      eventually("custom additional-components review") { fx {
        val preview = visibleText(node[Node]("presets-import-preview"))
        preview.contains("My height") && preview.contains("txt2img_height") &&
          preview.contains("Future control")
      }}
      fx {
        require(!node[CheckBox]("presets-import-add-tracked").isSelected,
          "custom mapping import unexpectedly changed the Add tracked choice")
        node[Button]("presets-import-confirm").fire()
      }
      eventually("custom additional-components import commits") {
        Files.isRegularFile(library) && sha(library) != beforeCustomAdditional
      }
      eventually("custom Additional Components binding reached the pane") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ")
      }}

      openChoices.add(sourceCustomUtilities)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Preset Utilities")
        node[Button]("presets-import").fire()
      }
      eventually("custom Utilities source label resolves through imported component ID") { fx {
        val preview = visibleText(node[Node]("presets-import-preview"))
        preview.contains("Custom") && preview.contains("My height") &&
          preview.contains("generation.height")
      }}
      fx {
        val add = node[CheckBox]("presets-import-add-tracked")
        if !add.isSelected then add.fire()
        node[Button]("presets-import-confirm").fire()
      }
      eventually("custom Utilities preset stores mapped Height and provenance") {
        scala.util.Try(saved("Custom", "utilities")).toOption.exists { record =>
          record.fields.get(FieldKey(Vector("generation", "height"))) ==
            Some(PresetAtom.Integer(320)) && record.source.nonEmpty && record.dirty.isEmpty &&
            !record.opaque.exists(_._1 == "My height")
        }
      }
      val customId = saved("Custom", "utilities").id
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        window.vms(Mode.Txt2Img).height.set(512)
        choose(node[ComboBox[Any]]("presets-utilities-quick-select"), "Custom")
        require(window.vms(Mode.Txt2Img).height.get == 320,
          "imported custom Utilities label did not apply to actual Height VM")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        choose(node[ComboBox[Any]]("presets-utilities-detailed-select"), "Custom")
        require(node[TextField]("presets-name").getText == "Custom",
          "Detailed editor did not populate imported custom Utilities name")
      }
      val customHeightCheck = fieldCheck("generation.height")
      require(fx { customHeightCheck.isSelected },
        "imported custom Utilities Height was not checked for Detailed save")
      fx {
        val spinner = window.scene.lookup("#generation-height-spinner-txt2img")
          .asInstanceOf[Spinner[Integer]]
        require(spinner != null, "actual Height spinner lacks its stable ID")
        spinner.getEditor.setText("768")
        node[Button]("presets-save").fire()
      }
      eventually("custom Utilities Height spinner edit saves with dirty provenance") {
        scala.util.Try(saved("Custom", "utilities")).toOption.exists { record =>
          record.id == customId &&
            record.fields.get(FieldKey(Vector("generation", "height"))) ==
              Some(PresetAtom.Integer(768)) &&
            record.dirty.contains(FieldKey(Vector("generation", "height")))
        }
      }

      val customUtilitiesExport = artifacts.resolve("export-custom-utilities.json")
      saveChoices.add(customUtilitiesExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Preset Utilities")
        node[Button]("presets-export").fire()
      }
      eventually("custom Utilities source export review lists omissions") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Omission")
      }}
      fx {
        val supported = node[CheckBox]("presets-export-supported-only")
        if !supported.isSelected then supported.fire()
        node[Button]("presets-export-confirm").fire()
      }
      eventually("custom Utilities source export writes edited label") {
        Files.isRegularFile(customUtilitiesExport)
      }
      require(Files.readString(customUtilitiesExport).contains("\"My height\":768"),
        "edited custom Utilities Height did not export through its imported label")

      fx { node[Button]("presets-reload-tools").fire() }
      eventually("custom Utilities mapping survives Reload preset tools") { fx {
        val status = node[Label]("presets-status").getText
        status.contains("Loaded") &&
          node[ComboBox[Any]]("presets-utilities-detailed-select").getItems.toArray
            .exists(_.toString.contains("Custom"))
      }}
      openChoices.add(customUtilitiesExport)
      fx {
        choose(node[ComboBox[Any]]("presets-import-format"), "Preset Utilities")
        node[Button]("presets-import").fire()
      }
      seekImportPreview("Custom")
      seekImportPreview("My height")
      require(fx { visibleText(node[Node]("presets-import-preview")).contains("generation.height") },
        "reimport did not resolve custom Utilities label after Reload")
      fx {
        choose(node[ComboBox[Any]]("presets-import-collision"), "Overwrite")
        node[Button]("presets-import-confirm").fire()
      }
      eventually("custom Utilities reimport clears dirty and preserves source provenance") {
        scala.util.Try(saved("Custom", "utilities")).toOption.exists { record =>
          record.id == customId && record.dirty.isEmpty &&
            record.fields.get(FieldKey(Vector("generation", "height"))) ==
              Some(PresetAtom.Integer(768)) &&
            record.source.exists(_.originalFields.text.contains("\"My height\":768"))
        }
      }
      val customAdditionalExport = artifacts.resolve("export-custom-additional-components.json")
      saveChoices.add(customAdditionalExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Additional")
        node[Button]("presets-export").fire()
      }
      eventually("custom Additional Components destination review") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Ready to export")
      }}
      fx { node[Button]("presets-export-confirm").fire() }
      eventually("custom Additional Components export") { Files.isRegularFile(customAdditionalExport) }
      val customAdditionalEntries = PresetSourceFiles.readAdditionalComponents(
        Files.readAllBytes(customAdditionalExport), PresetLimits()).toOption.get
      require(customAdditionalEntries.contains("My height" -> Some("txt2img_height")) &&
        customAdditionalEntries.contains("Future control" -> None),
        "custom label/component ID or null was lost after Reload and source reimport")
      noWork()
      checked("Additional Components custom label resolves, applies, edits, exports and survives Reload/reimport")

      // A later null removes only the binding created by Additional Components.
      // Existing manual bindings, conflicting labels and unknown IDs retain
      // their own provenance and never acquire an arbitrary new target.
      eventually("custom Additional export callback delivered") { fx {
        node[Label]("presets-status").getText.startsWith("Exported additional")
      }}
      val beforePolicy = store.load().toOption.get
      val beforePolicyIndex = beforePolicy.library.collections.indexWhere(_.id == "utilities")
      require(beforePolicyIndex >= 0, "Utilities collection missing before Additional ownership check")
      val beforePolicyCollection = beforePolicy.library.collections(beforePolicyIndex)
      val heightKey = FieldKey(Vector("generation", "height"))
      val widthKey = FieldKey(Vector("generation", "width"))
      require(beforePolicyCollection.bindings.contains(
        SourceBinding("preset-utilities", "My height", heightKey)),
        "Additional Components did not create an owned custom Utilities binding")
      val manualBinding = SourceBinding("preset-utilities", "Manual height", widthKey)
      require(!beforePolicyCollection.bindings.exists(_.sourceKey == manualBinding.sourceKey),
        "manual conflict fixture label already exists")
      val withManual = beforePolicyCollection.copy(bindings = beforePolicyCollection.bindings :+ manualBinding)
      val withManualLibrary = beforePolicy.library.copy(collections =
        beforePolicy.library.collections.updated(beforePolicyIndex, withManual))
      require(store.save(beforePolicy.revision, withManualLibrary).isRight,
        "external manual Utilities binding fixture failed")
      fx { node[Button]("presets-refresh").fire() }
      eventually("external manual binding reached the pane") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      val beforePolicyImport = sha(library)
      openChoices.add(sourcePolicyAdditional)
      fx {
        val add = node[CheckBox]("presets-import-add-tracked")
        if add.isSelected then add.fire()
        choose(node[ComboBox[Any]]("presets-import-format"), "Additional")
        node[Button]("presets-import").fire()
      }
      eventually("Additional ownership and collision review") { fx {
        val preview = visibleText(node[Node]("presets-import-preview"))
        preview.contains("My height -> null") && preview.contains("Manual height") &&
          preview.contains("existing binding") && preview.contains("Sampling steps") &&
          preview.contains("standard route") && preview.contains("Unknown slot") &&
          preview.contains("retained without binding")
      }}
      fx { node[Button]("presets-import-confirm").fire() }
      eventually("Additional null and conflict metadata committed") {
        sha(library) != beforePolicyImport && fx {
          node[Label]("presets-status").getText.startsWith("Loaded ")
        }
      }
      val policyLibrary = PresetJson.readNative(Files.readAllBytes(library), PresetLimits()).toOption.get
      val policyCollection = policyLibrary.collections.find(_.id == "utilities").get
      require(!policyCollection.bindings.exists(b => b.format == "preset-utilities" &&
        b.sourceKey == "My height") && policyCollection.bindings.contains(manualBinding) &&
        !policyCollection.bindings.exists(b => b.format == "preset-utilities" &&
          (b.sourceKey == "Unknown slot" || b.sourceKey == "Sampling steps")),
        "null did not remove only the owned binding or a conflicting/unknown binding was installed")
      require(policyCollection.tracked == beforePolicyCollection.tracked,
        "Add tracked OFF changed the tracked set while importing mappings")
      val ownedRaw = policyCollection.extra.find(_._1 ==
        "sdfx.native-presets.additional-owned-bindings.v1").get._2
      require(PresetSourceFiles.readAdditionalComponents(
        ownedRaw.text.getBytes(java.nio.charset.StandardCharsets.UTF_8), PresetLimits()).toOption.get.isEmpty,
        "removed Additional-owned binding remained in ownership metadata")
      val policyExport = artifacts.resolve("export-policy-additional-components.json")
      saveChoices.add(policyExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Additional")
        node[Button]("presets-export").fire()
      }
      eventually("policy Additional export review") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Ready to export")
      }}
      fx { node[Button]("presets-export-confirm").fire() }
      eventually("policy Additional export written") { Files.isRegularFile(policyExport) }
      val policyEntries = PresetSourceFiles.readAdditionalComponents(
        Files.readAllBytes(policyExport), PresetLimits()).toOption.get
      require(policyEntries.contains("My height" -> None) &&
        policyEntries.contains("Manual height" -> Some("txt2img_height")) &&
        policyEntries.contains("Sampling steps" -> Some("txt2img_width")) &&
        policyEntries.contains("Unknown slot" -> Some("unknown_component_id")),
        "null, conflicting or unknown Additional Components metadata was lost")
      noWork()
      checked("Additional ownership null removes only its binding; manual/standard conflicts and unknown IDs stay inert")

      val omissionStore = new PresetStore(library.getParent, PresetLimits())
      val omissionBase = omissionStore.load().toOption.get
      val omissionIndex = omissionBase.library.collections.indexWhere(_.id == "utilities")
      val omissionCollection = omissionBase.library.collections(omissionIndex)
      val omitted = (0 until 210).map { n =>
        FieldKey(Vector("future", f"omission_$n%03d")) -> PresetAtom.Integer(n.toLong)
      }.toMap
      val bulkName = "Bulk omitted fields"
      val bulkRecord = PresetRecord(PresetRecord.initialId("utilities", bulkName), bulkName,
        omitted.updated(FieldKey(Vector("generation", "steps")), PresetAtom.Integer(37)),
        None, Set.empty, Vector.empty, Vector.empty)
      val omissionLibrary = omissionBase.library.copy(collections = omissionBase.library.collections.updated(
        omissionIndex, omissionCollection.copy(tracked = omissionCollection.tracked ++ omitted.keySet,
          presets = Vector(bulkRecord))))
      require(omissionStore.save(omissionBase.revision, omissionLibrary).isRight,
        "210 native-only omission fixture failed")
      fx { node[Button]("presets-refresh").fire() }
      eventually("bulk omission library loaded") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ") &&
          node[ComboBox[Any]]("presets-utilities-detailed-select").getItems.toArray
            .exists(_.toString == bulkName)
      }}
      val bulkExport = artifacts.resolve("export-bulk-supported-utilities.json")
      saveChoices.add(bulkExport)
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        choose(node[ComboBox[Any]]("presets-export-format"), "Preset Utilities")
        node[Button]("presets-export").fire()
      }
      eventually("bounded bulk omission review begins") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Omission") &&
          !node[Button]("presets-export-confirm").isDisabled
      }}
      val omissionTurns = seekDiagnostic("presets-export-preview", "presets-export-next",
        "future.omission_209")
      require(omissionTurns > 0, "final omission after row/character bounds was not reached")
      fx {
        val total = node[Label]("presets-export-page").getText
        require(total.contains("210") &&
          visibleText(node[Node]("presets-export-preview")).length <= 4096,
          "export review did not show exact 210 omissions in bounded pages")
        val supported = node[CheckBox]("presets-export-supported-only")
        require(!supported.isSelected, "bulk supported-only export was silently enabled")
        supported.fire()
        node[Button]("presets-export-confirm").fire()
      }
      eventually("bulk supported-only export written") { Files.isRegularFile(bulkExport) }
      eventually("bulk supported-only export callback completed") { fx {
        node[Label]("presets-status").getText.contains("Exported preset-utilities")
      }}
      val bulkText = Files.readString(bulkExport)
      require(bulkText.contains(bulkName) && !bulkText.contains("future.omission_") &&
        !bulkText.contains("omission_209") &&
        saved(bulkName, "utilities").fields.size == 211,
        "supported-only output lost a supported record or changed the native full record")
      val afterBulk = omissionStore.load().toOption.get
      require(omissionStore.save(afterBulk.revision,
        afterBulk.library.copy(collections = afterBulk.library.collections.updated(
          omissionIndex, omissionCollection))).isRight,
        "pre-bulk Utilities records could not be restored for the remaining workflows")
      fx { node[Button]("presets-refresh").fire() }
      eventually("Utilities records restored after isolated omission fixture") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ") &&
          node[ComboBox[Any]]("presets-utilities-detailed-select").getItems.toArray
            .exists(_.toString == "Custom")
      }}
      noWork()
      checked("210 export omissions remain paged and exact; supported-only source omits reviewed fields only")

      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(2)
        node[TextField]("presets-inspect-search").setText("")
        node[Button]("presets-inspect").fire()
        require(visibleText(node[Node]("presets-inspect-preview")).contains("Collection utilities"),
          "Utilities screenshot would show facts from a different collection")
      }
      for index <- 0 until 4 do
        fx { node[TabPane]("presets-tabs").getSelectionModel.select(index) }
        screenshot(s"utilities-tab-$index-1040x740.png", 1040, 740)
      fx { node[TabPane]("presets-tabs").getSelectionModel.select(1) }
      screenshot("utilities-detailed-small-760x560.png", 760, 560)
      fx { node[TabPane]("presets-tabs").getSelectionModel.select(2) }
      screenshot("utilities-inspect-small-760x560.png", 760, 560)
      screenshot("utilities-small-760x560.png", 760, 560)
      noWork()
      checked("Utilities has synchronized selectors, independent Stackable toggles and readable tab screenshots")

      // Imported-but-untracked, checked-but-unsaved, and native Extra's initial
      // value are different facts. Inspect must report them separately.
      val factStore = new PresetStore(library.getParent, PresetLimits())
      val factBase = factStore.load().toOption.get
      val factIndex = factBase.library.collections.indexWhere(_.id == "config-txt2img")
      val factCollection = factBase.library.collections(factIndex)
      val factRecordIndex = factCollection.presets.indexWhere(_.name == "my 4K copy")
      val seedKey = FieldKey(Vector("generation", "seed"))
      val factRecord = factCollection.presets(factRecordIndex).copy(
        fields = factCollection.presets(factRecordIndex).fields.updated(seedKey, PresetAtom.Integer(123)))
      val factLibrary = factBase.library.copy(collections = factBase.library.collections.updated(
        factIndex, factCollection.copy(presets = factCollection.presets.updated(factRecordIndex, factRecord))))
      require(factStore.save(factBase.revision, factLibrary).isRight,
        "Inspect independent-state fixture failed to save")
      fx {
        window.openPresets(Mode.Txt2Img)
        choose(node[ComboBox[Any]]("presets-profile"), "Config")
        node[Button]("presets-refresh").fire()
      }
      eventually("Inspect independent-state fixture loaded") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        choose(node[ComboBox[Any]]("presets-select"), "my 4K copy")
        window.vms(Mode.Txt2Img).variationEnabled.set(false)
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      val savedSeedCheck = fieldCheck("generation.seed")
      fx { if savedSeedCheck.isSelected then savedSeedCheck.fire() }
      val newCfgCheck = fieldCheck("generation.cfgScale")
      fx { if !newCfgCheck.isSelected then newCfgCheck.fire() }
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(2)
        val search = node[TextField]("presets-inspect-search")
        search.setText("generation.seed")
        node[Button]("presets-inspect").fire()
        val seedFacts = visibleText(node[Node]("presets-inspect-preview"))
        require(seedFacts.contains("generation.seed") && seedFacts.contains("tracked=false") &&
          seedFacts.contains("checked=false") && seedFacts.contains("saved=true") &&
          seedFacts.contains("available=true"),
          "Inspect merged imported-but-untracked, checked-save, saved-present or available state")
        search.setText("generation.cfgScale")
        node[Button]("presets-inspect").fire()
        val cfgFacts = visibleText(node[Node]("presets-inspect-preview"))
        require(cfgFacts.contains("generation.cfgScale") && cfgFacts.contains("tracked=true") &&
          cfgFacts.contains("checked=true") && cfgFacts.contains("saved=false") &&
          cfgFacts.contains("available=true"),
          "Inspect omitted a checked new field's four independent states")
        search.setText("generation.variationEnabled")
        node[Button]("presets-inspect").fire()
        val extraFacts = visibleText(node[Node]("presets-inspect-preview"))
        require(extraFacts.contains("Source preset default=false") &&
          extraFacts.contains("Native initial value=true") && extraFacts.contains("current=false"),
          "Extra Inspect conflated source reset default, native initial value and edited current value")
        search.setText("")
        node[Button]("presets-inspect").fire()
      }
      screenshot("config-inspect-populated-1040x740.png", 1040, 740)
      screenshot("config-inspect-populated-760x560.png", 760, 560)
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      screenshot("config-detailed-populated-1040x740.png", 1040, 740)
      screenshot("config-detailed-populated-760x560.png", 760, 560)
      noWork()
      checked("Inspect distinguishes tracked, checked, saved and available and shows Extra's two baselines")

      val longPrompt = ("P" * 1000 + "\n") * 260
      val longBase = factStore.load().toOption.get
      val longIndex = longBase.library.collections.indexWhere(_.id == "config-txt2img")
      val longCollection = longBase.library.collections(longIndex)
      val longRecordIndex = longCollection.presets.indexWhere(_.name == "my 4K copy")
      val promptKey = FieldKey(Vector("generation", "prompt"))
      val manyAliases = (0 until 300).map { n =>
        SourceBinding("config-presets", f"long_prompt_alias_$n%03d", promptKey)
      }.toVector
      val longRecord = longCollection.presets(longRecordIndex).copy(fields =
        longCollection.presets(longRecordIndex).fields.updated(promptKey, PresetAtom.Text(longPrompt)))
      val longLibrary = longBase.library.copy(collections = longBase.library.collections.updated(
        longIndex, longCollection.copy(tracked = longCollection.tracked + promptKey,
          bindings = longCollection.bindings ++ manyAliases,
          presets = longCollection.presets.updated(longRecordIndex, longRecord))))
      require(factStore.save(longBase.revision, longLibrary).isRight,
        "valid long scalar and many aliases fixture failed to save")
      fx {
        node[Button]("presets-refresh").fire()
      }
      eventually("long scalar and aliases loaded for actual Detailed row") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        window.vms(Mode.Txt2Img).prompt.set("before long preset")
        choose(node[ComboBox[Any]]("presets-select"), "None")
        choose(node[ComboBox[Any]]("presets-select"), "my 4K copy")
        require(window.vms(Mode.Txt2Img).prompt.get == longPrompt,
          "long saved Prompt was not applied through the actual selector")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      val (longRow, longRowHeight) = fieldRowText("generation.prompt")
      require(longRow.length <= 4096 && longRow.toLowerCase.contains("truncated") &&
        longRowHeight <= 160.0,
        s"Detailed Prompt row was unbounded or inaccessible: ${longRow.length} chars, $longRowHeight px")
      screenshot("config-detailed-long-prompt-1040x740.png", 1040, 740)
      screenshot("config-detailed-long-prompt-760x560.png", 760, 560)
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(2)
        node[TextField]("presets-inspect-search").setText("generation.prompt")
        node[Button]("presets-inspect").fire()
        val preview = visibleText(node[Node]("presets-inspect-preview"))
        require(preview.length <= 4096 && preview.toLowerCase.contains("truncated"),
          "Inspect built an unbounded Prompt/alias row or silently cut it")
        node[TextField]("presets-inspect-search").setText("long_prompt_alias_299")
        node[Button]("presets-inspect").fire()
        val lateAlias = visibleText(node[Node]("presets-inspect-preview"))
        require(lateAlias.contains("generation.prompt") &&
          lateAlias.contains("long_prompt_alias_299") && lateAlias.length <= 4096,
          "Inspect search matched a later alias without showing its actual source ID")
        node[TextField]("presets-inspect-search").setText("generation.prompt")
        node[Button]("presets-inspect").fire()
      }
      screenshot("config-inspect-long-prompt-760x560.png", 760, 560)
      fx {
        node[TextField]("presets-inspect-search").setText("")
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
      }
      val longNativeExport = artifacts.resolve("export-long-prompt-native.json")
      saveChoices.add(longNativeExport)
      fx {
        choose(node[ComboBox[Any]]("presets-export-format"), "Native")
        node[Button]("presets-export").fire()
      }
      eventually("full-value native export review after bounded field preview") { fx {
        visibleText(node[Node]("presets-export-preview")).contains("Ready to export")
      }}
      fx { node[Button]("presets-export-confirm").fire() }
      eventually("full-value native export written") { Files.isRegularFile(longNativeExport) }
      eventually("full-value native export callback completed") { fx {
        node[Label]("presets-status").getText.contains("Exported native")
      }}
      val longExported = PresetJson.readNative(Files.readAllBytes(longNativeExport), PresetLimits())
        .toOption.get.collections.find(_.id == "config-txt2img").get.presets
        .find(_.name == "my 4K copy").get
      require(longExported.fields.get(promptKey).contains(PresetAtom.Text(longPrompt)),
        "bounded Detailed preview truncated the saved or exported Prompt")
      noWork()
      checked("long valid Prompt and 300 aliases stay bounded in Detailed while full native export survives")

      fx {
        window.openPresets(Mode.Img2Img)
        choose(node[ComboBox[Any]]("presets-profile"), "Config")
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        require(node[Label]("presets-collection").getText.contains("config-img2img"))
        require(node[ComboBox[Any]]("presets-select").getItems.size >= 4)
        window.vms(Mode.Img2Img).denoise.set(0.1)
        window.vms(Mode.Inpaint).denoise.set(0.2)
        choose(node[ComboBox[Any]]("presets-select"), "Medium denoising")
        require(window.vms(Mode.Img2Img).denoise.get == 0.4 &&
          window.vms(Mode.Inpaint).denoise.get == 0.2,
          "Img2Img preset selection mutated the Inpaint VM")
      }
      screenshot("config-img2img-1040x740.png", 1040, 740)
      fx {
        window.openPresets(Mode.Inpaint)
        require(node[Label]("presets-collection").getText.contains("config-img2img"))
        require(window.vms(Mode.Inpaint).denoise.get == 0.2,
          "opening Inpaint programmatically reapplied the shared collection")
        choose(node[ComboBox[Any]]("presets-select"), "None")
        choose(node[ComboBox[Any]]("presets-select"), "Medium denoising")
        require(window.vms(Mode.Inpaint).denoise.get == 0.4 &&
          window.vms(Mode.Img2Img).denoise.get == 0.4,
          "Inpaint selection failed to apply to only its own VM")
      }
      screenshot("config-inpaint-1040x740.png", 1040, 740)
      noWork()
      checked("Img2Img and Inpaint share Config collection while keeping distinct destinations")

      val warningStore = new PresetStore(library.getParent, PresetLimits())
      val warningBase = warningStore.load().toOption.get
      val warningIndex = warningBase.library.collections.indexWhere(_.id == "config-img2img")
      val warningCollection = warningBase.library.collections(warningIndex)
      val warningValues = Map(
        FieldKey(Vector("generation", "hiresEnabled")) -> PresetAtom.Flag(true),
        FieldKey(Vector("generation", "hiresPrompt")) -> PresetAtom.Text("kept"),
        FieldKey(Vector("generation", "hiresResizeX")) -> PresetAtom.Integer(640),
        FieldKey(Vector("generation", "hiresResizeY")) -> PresetAtom.Integer(640),
        FieldKey(Vector("generation", "hiresSampler")) -> PresetAtom.Text("Euler"),
        FieldKey(Vector("generation", "hiresScale")) -> PresetAtom.Decimal(BigDecimal(2)),
        FieldKey(Vector("generation", "hiresScheduler")) -> PresetAtom.Text("Normal"),
        FieldKey(Vector("generation", "hiresSteps")) -> PresetAtom.Integer(12),
        FieldKey(Vector("generation", "hiresUpscaler")) -> PresetAtom.Text("Latent"))
      val warningName = "Unavailable hires facts"
      val warningRecord = PresetRecord(PresetRecord.initialId("config-img2img", warningName),
        warningName, warningValues, None, Set.empty, Vector.empty, Vector.empty)
      val warningLibrary = warningBase.library.copy(collections = warningBase.library.collections.updated(
        warningIndex, warningCollection.copy(tracked = warningCollection.tracked ++ warningValues.keySet,
          presets = warningCollection.presets :+ warningRecord)))
      require(warningStore.save(warningBase.revision, warningLibrary).isRight,
        "nine unavailable canonical fields fixture failed")
      fx {
        window.openPresets(Mode.Img2Img)
        node[Button]("presets-refresh").fire()
      }
      eventually("nine warning fixture loaded in Config Img2Img") { fx {
        node[ComboBox[Any]]("presets-select").getItems.toArray.exists(_.toString == warningName) &&
          node[Label]("presets-status").getText.startsWith("Loaded ")
      }}
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        choose(node[ComboBox[Any]]("presets-select"), warningName)
        require(node[Label]("presets-status").getText.contains("9 compatibility warnings"),
          "Apply did not retain the exact count of nine unavailable saved fields")
      }
      seekDiagnostic("presets-apply-preview", "presets-apply-next", "generation.hiresUpscaler")
      fx {
        require(node[Label]("presets-apply-page").getText.contains("9"),
          "Apply diagnostics did not show exact total")
        node[TabPane]("presets-tabs").getSelectionModel.select(2)
        node[TextField]("presets-inspect-search").setText("generation.hiresScale")
        node[Button]("presets-inspect").fire()
        val facts = visibleText(node[Node]("presets-inspect-preview"))
        require(facts.contains("generation.hiresScale") && facts.contains("tracked=true") &&
          facts.contains("checked=false") && facts.contains("saved=true") &&
          facts.contains("available=false"),
          "Inspect dropped a saved canonical field unavailable in Img2Img")
      }
      noWork()
      checked("all nine Apply warnings are reachable; saved unavailable canonical field retains four Inspect states")

      openChoices.add(sourceOpaqueSweep)
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        choose(node[ComboBox[Any]]("presets-import-format"), "Config")
        node[Button]("presets-import").fire()
      }
      seekImportPreview("unknown_source_209")
      fx {
        require(visibleText(node[Node]("presets-import-preview")).length <= 4096,
          "opaque source review exceeded its visible page bound")
        node[Button]("presets-import-confirm").fire()
      }
      eventually("210 opaque source keys saved in the native collection") {
        scala.util.Try(saved("Opaque sweep", "config-img2img")).toOption.exists { record =>
          record.opaque.size == 210 && record.opaque.last._1 == "unknown_source_209" &&
            record.opaque.last._2.text == "210"
        }
      }
      eventually("opaque sweep import reached the visible selector") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ") &&
          node[ComboBox[Any]]("presets-select").getItems.toArray
            .exists(_.toString == "Opaque sweep")
      }}
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        val selector = node[ComboBox[Any]]("presets-select")
        choose(selector, "None")
        window.vms(Mode.Img2Img).steps.set(18)
        choose(selector, "Opaque sweep")
        require(window.vms(Mode.Img2Img).steps.get == 28 &&
          node[Label]("presets-status").getText.contains("210 compatibility warnings"),
          "known Steps were not applied with an exact per-opaque-key warning count")
      }
      seekDiagnostic("presets-apply-preview", "presets-apply-next", "unknown_source_209")
      fx {
        val vm = window.vms(Mode.Img2Img)
        vm.steps.set(19)
        node[Button]("presets-reapply").fire()
        require(vm.steps.get == 28 &&
          node[Label]("presets-status").getText.contains("210 compatibility warnings"),
          "Reapply lost a known value or an opaque warning after paging")
        node[Button]("presets-reload-tools").fire()
      }
      eventually("opaque sweep survives native Reload") { fx {
        node[Label]("presets-status").getText.startsWith("Loaded ") &&
          node[ComboBox[Any]]("presets-select").getItems.toArray
            .exists(_.toString == "Opaque sweep")
      }}
      fx {
        val selector = node[ComboBox[Any]]("presets-select")
        choose(selector, "None")
        window.vms(Mode.Img2Img).steps.set(20)
        choose(selector, "Opaque sweep")
        require(window.vms(Mode.Img2Img).steps.get == 28 &&
          node[Label]("presets-status").getText.contains("210 compatibility warnings"),
          "native Reload erased opaque source warnings or known Steps")
      }
      seekDiagnostic("presets-apply-preview", "presets-apply-next", "unknown_source_209")
      require(saved("Opaque sweep", "config-img2img").opaque.last._2.text == "210",
        "native Reload/Apply changed the last opaque source value")
      noWork()
      checked("210 imported opaque keys survive native Reload and last Apply/Reapply warning remains reachable")

      val resizeKey = FieldKey(Vector("generation", "resizeMode"))
      val fillKey = FieldKey(Vector("generation", "inpaintingFill"))
      def importSourceChoices(path: Path, profileName: String, format: String,
          collectionId: String, names: Vector[String]): Unit =
        openChoices.add(path)
        fx {
          window.openPresets(Mode.Img2Img)
          choose(node[ComboBox[Any]]("presets-profile"), profileName)
          node[TabPane]("presets-tabs").getSelectionModel.select(1)
          choose(node[ComboBox[Any]]("presets-import-format"), format)
          val add = node[CheckBox]("presets-import-add-tracked")
          if !add.isSelected then add.fire()
          node[Button]("presets-import").fire()
        }
        eventually(s"$format indexed-choice import review") { fx {
          val preview = visibleText(node[Node]("presets-import-preview"))
          preview.contains(names.head) && preview.contains("resizeMode") &&
            preview.contains("inpaintingFill") &&
            !node[Button]("presets-import-confirm").isDisabled
        }}
        fx { node[Button]("presets-import-confirm").fire() }
        eventually(s"$format indexed-choice import reached native selectors") { fx {
          val selectorId = if profileName == "Utilities" then
            "presets-utilities-quick-select" else "presets-select"
          val choices = node[ComboBox[Any]](selectorId).getItems.toArray.map(_.toString).toSet
          node[Label]("presets-status").getText.startsWith("Loaded ") &&
            names.forall(choices.contains)
        }}
        names.foreach { name =>
          val record = saved(name, collectionId)
          require(record.fields.get(resizeKey) == Some(PresetAtom.Integer(2)) &&
            record.fields.get(fillKey) == Some(PresetAtom.Integer(1)) &&
            !record.opaque.exists((id, _) =>
              id == "resize_mode" || id == "img2img_inpainting_fill" ||
                id == "Resize mode" || id == "Masked content"),
            s"$format $name did not store both source choices as native integer atoms")
        }
      importSourceChoices(sourceConfigChoices, "Config", "Config", "config-img2img",
        Vector("I2 Config numeric", "I2 Config labels"))
      importSourceChoices(sourceUtilityChoices, "Utilities", "Preset Utilities", "utilities",
        Vector("I2 Utilities numeric", "I2 Utilities labels"))
      for (profileName, names) <- Vector(
          "Config" -> Vector("I2 Config numeric", "I2 Config labels"),
          "Utilities" -> Vector("I2 Utilities numeric", "I2 Utilities labels")) do
        for destinationMode <- Vector(Mode.Img2Img, Mode.Inpaint) do
          fx {
            window.openPresets(destinationMode)
            choose(node[ComboBox[Any]]("presets-profile"), profileName)
            node[TabPane]("presets-tabs").getSelectionModel.select(0)
            val selector = node[ComboBox[Any]](
              if profileName == "Utilities" then "presets-utilities-quick-select" else "presets-select")
            val neutral = if profileName == "Utilities" then "Reset common controls" else "None"
            names.foreach { name =>
              choose(selector, neutral)
              val vm = window.vms(destinationMode)
              vm.resizeMode.set(0)
              vm.inpaintingFill.set(3)
              choose(selector, name)
              require(vm.resizeMode.get == 2,
                s"$profileName $name did not apply Resize mode index 2 in $destinationMode")
              require(vm.inpaintingFill.get == (if destinationMode == Mode.Inpaint then 1 else 3),
                s"$profileName $name changed Masked content outside Inpaint or missed index 1")
            }
          }
      fx {
        window.openPresets(Mode.Img2Img)
        choose(node[ComboBox[Any]]("presets-profile"), "Config")
      }
      noWork()
      checked("Config and Utilities numeric/label indexed source choices apply to real Img2Img and Inpaint VMs")

      val hiddenImgSteps = fx {
        window.panels(Mode.Img2Img).content.lookup("#generation-steps-spinner-img2img")
          .asInstanceOf[Spinner[Integer]]
      }
      require(hiddenImgSteps != null, "Img2Img destination's actual Steps spinner is missing")
      val imgStepsBefore = fx { window.vms(Mode.Img2Img).steps.get }
      val hiddenDraftHash = sha(library)
      fx {
        window.openPresets(Mode.Img2Img)
        window.select(Mode.Txt2Img) // destination's editor is now in a hidden main tab
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        node[TextField]("presets-name").setText("Hidden mode draft")
        hiddenImgSteps.getEditor.setText("-")
        node[Button]("presets-save").fire()
        require(hiddenImgSteps.getEditor.getText == "-" &&
          window.vms(Mode.Img2Img).steps.get == imgStepsBefore,
          "hidden destination editor was reset or its stale value was captured")
      }
      require(sha(library) == hiddenDraftHash,
        "hidden destination invalid draft changed the library")
      fx { hiddenImgSteps.getEditor.setText(imgStepsBefore.toString) }
      noWork()
      checked("hidden destination mode's actual spinner draft blocks stale capture without changing other tabs")

      eventually("authoritative native library") { Files.isRegularFile(library) }
      // External edit down to one Img2Img entry, then delete through the actual
      // native button. The other two collections must remain byte-valid.
      val lastStore = new PresetStore(library.getParent, PresetLimits())
      val beforeLast = lastStore.load().toOption.get
      val imgIndex = beforeLast.library.collections.indexWhere(_.id == "config-img2img")
      require(imgIndex >= 0 && beforeLast.library.collections(imgIndex).presets.nonEmpty)
      val lastOnly = beforeLast.library.collections(imgIndex).copy(
        presets = beforeLast.library.collections(imgIndex).presets.take(1))
      val oneLast = beforeLast.library.copy(collections =
        beforeLast.library.collections.updated(imgIndex, lastOnly))
      require(lastStore.save(beforeLast.revision, oneLast).isRight, "one-last external edit failed")
      fx { node[Button]("presets-refresh").fire() }
      eventually("one Img2Img record shown") { fx {
        node[ComboBox[Any]]("presets-select").getItems.size == 1
      }}
      fx {
        val selector = node[ComboBox[Any]]("presets-select")
        selector.getSelectionModel.select(0)
        node[TabPane]("presets-tabs").getSelectionModel.select(1)
        node[Button]("presets-delete").fire()
      }
      eventually("last Config Img2Img preset deleted") {
        val parsed = PresetJson.readNative(Files.readAllBytes(library), PresetLimits()).toOption.get
        parsed.collections.find(_.id == "config-img2img").exists(_.presets.isEmpty)
      }
      fx {
        node[TabPane]("presets-tabs").getSelectionModel.select(0)
        require(node[Button]("presets-apply").isDisabled,
          "Apply remained available after deleting the final selected preset")
      }
      require(saved("my 4K copy", "config-txt2img").id == savedId,
        "deleting the last Img2Img preset damaged Txt2Img collection")
      noWork()
      checked("external edit refresh and actual Delete of final entry leave a healthy empty collection")

      val beforeMissing = sha(library)
      val missingBackup = library.resolveSibling("library-v1.json.bak")
      require(Files.isRegularFile(missingBackup), "backup missing before disappearance test")
      val missingBackupHash = sha(missingBackup)
      Files.delete(library)
      fx { node[Button]("presets-refresh").fire() }
      eventually("missing previously loaded file reports conflict") { fx {
        val status = node[Label]("presets-status").getText.toLowerCase
        status.contains("missing") || status.contains("conflict")
      }}
      require(!Files.exists(library), "missing established library was silently reseeded")
      fx { node[Button]("presets-restore-backup").fire() }
      eventually("missing file restored from exact backup") {
        Files.isRegularFile(library) && sha(library) == missingBackupHash
      }
      noWork()
      checked("missing previously loaded library blocks reseed and explicit Restore recovers backup")

      val hashBefore = sha(library)
      fx { node[Button]("presets-reload-tools").fire() }
      eventually("native reload completed") { fx { !node[Label]("presets-status").getText.toLowerCase.contains("loading") } }
      require(sha(library) == hashBefore, "view-only reload unexpectedly rewrote the library")
      noWork()
      checked("Reload preset tools does not restart engine, submit requests or rewrite source")

      val backup = library.resolveSibling("library-v1.json.bak")
      require(Files.isRegularFile(backup), "native save did not retain a recoverable backup")
      val backupHash = sha(backup)
      Files.writeString(library, "{malformed-external-edit")
      fx { node[Button]("presets-refresh").fire() }
      eventually("malformed external edit is visible") { fx {
        val status = node[Label]("presets-status").getText.toLowerCase
        status.contains("error") || status.contains("invalid") || status.contains("malformed")
      }}
      require(Files.readString(library) == "{malformed-external-edit",
        "refresh silently replaced corrupt user bytes")
      fx { node[Button]("presets-restore-backup").fire() }
      eventually("explicit restore recovers exact backup") { Files.isRegularFile(library) && sha(library) == backupHash }
      val siblings = Files.list(library.getParent)
      try
        require(siblings.toArray.exists(_.toString.contains(".rejected-")),
          "restore did not preserve rejected malformed bytes")
      finally siblings.close()
      noWork()
      checked("corrupt external edit blocks refresh writes and explicit Restore preserves rejected bytes")

      Files.writeString(output.resolve("native-presets-pane-report.json"), ujson.write(ujson.Obj(
        "status" -> "passed", "home" -> home.root.toString, "artifacts" -> artifacts.toString,
        "library" -> library.toString,
        "librarySha256BeforeMissing" -> beforeMissing,
        "missingBackupSha256" -> missingBackupHash,
        "librarySha256BeforeCorruption" -> hashBefore, "restoredBackupSha256" -> backupHash,
        "checks" -> ujson.Arr.from(checks),
        "screenshots" -> ujson.Arr.from(screenshots.map(_.toString))), indent = 2))
    catch
      case error: Throwable =>
        failure = error
        if window != null then
          try fx {
            window.presetsInspectorStage.filter(_.isShowing).foreach { failedStage =>
              val scene = failedStage.getScene
              scene.getRoot.applyCss()
              scene.getRoot.layout()
              ImageIO.write(SwingFXUtils.fromFXImage(scene.snapshot(null), null), "png",
                artifacts.resolve("failure-current-pane.png").toFile)
            }
          }
          catch case _: Throwable => ()
    finally
      NativeAcceptanceSafety.cleanup(failure)(
        () => if window != null then fx { window.dispose(); window.stage.close() },
        () => removeQueueListener(),
        () => services.shutdown(),
        () => Platform.exit())
    if failure != null then throw failure
