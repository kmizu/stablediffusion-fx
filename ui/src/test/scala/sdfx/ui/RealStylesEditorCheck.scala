package sdfx.ui

import javafx.application.Platform
import javafx.embed.swing.SwingFXUtils
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.input.{Clipboard, DataFormat, KeyCode, KeyEvent}
import javafx.stage.{Stage, Window}
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.infotext.Infotext
import sdfx.core.model.*
import sdfx.core.png.PngMetadata
import sdfx.engine.{AppServices, EngineLaunch, EngineTimeouts, Job}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, TimeUnit}
import java.util.concurrent.atomic.AtomicReference
import javax.imageio.ImageIO
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** MANUAL isolated-AppHome UI + actual style-conditioning gate. Never run by sbt test.
  * ui/Test/runMain sdfx.ui.RealStylesEditorCheck <python> <checkpoint-dir> <checkpoint-name> <out-dir>
  * Parent runs once per SD1.5/SDXL. All CSV/style edits occur inside a newly created home under out-dir.
  */
object RealStylesEditorCheck:
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A]:
      def call(): A = body
    )
    Platform.runLater(task)
    task.get(40, TimeUnit.SECONDS)
  private def eventually(label: String, timeout: FiniteDuration = 90.seconds)(test: => Boolean): Unit =
    val until = System.nanoTime() + timeout.toNanos
    while !test && System.nanoTime() < until do Thread.sleep(40)
    require(test, s"Timeout: $label")
  private def hash(path: Path) = MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(path)).map(b => f"${b & 255}%02x").mkString
  def main(args: Array[String]): Unit =
    require(args.length == 4 || (args.length == 5 && args(4) == "--dry-run"), "python checkpoint-dir checkpoint-name output-dir [--dry-run]")
    val dryRun = args.length == 5
    val output = Paths.get(args(3)).toAbsolutePath.normalize()
    Files.createDirectories(output)
    val engineDir = AppHome.detect().pythonDir
    val home = AppHome(Files.createTempDirectory(output, "styles-app-home-"))
    Files.createDirectories(home.configDir)
    val untouched = "Untouched,\"keep, \"\"quotes\"\"\",n,opaque\r\n"
    Files.writeString(home.stylesFile, "\uFEFFname,prompt,negative_prompt,private\r\nExisting,\"ink, {prompt}\",lowres,KEEP\r\n" + untouched)
    val services = new AppServices(home, Settings(outputDir = output.resolve("images").toString,
      modelDirs = ModelDirs(checkpoints = Seq(Paths.get(args(1)).toAbsolutePath.toString)), previewEvery = 0),
      dryRun = dryRun, builtins = StandardPlugins.withOwnAxes,
      launchOverride = Some(() => EngineLaunch(Seq(Paths.get(args(0)).toAbsolutePath.toString, "-m", "sdfx") ++ (if dryRun then Seq("--dry-run") else Nil), engineDir,
        Map("PYTHONPATH" -> engineDir.toString, "USE_TF" -> "0"))), persistSettings = false,
      timeouts = EngineTimeouts(load = 10.minutes), installShutdownHook = false)
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    var clipboardBackup: Option[java.util.HashMap[DataFormat, Object]] = None
    var primaryFailure: Throwable = null
    var skips = Vector.empty[String]
    var checks = Vector.empty[String]
    def checked(label: String): Unit = { checks :+= label; println(s"STYLES CHECK: $label") }
    def find[A <: Node](id: String): A =
      val node = window.scene.lookup(s"#$id")
      require(node != null, s"Missing native control #$id")
      node.asInstanceOf[A]
    def idle(): Unit = eventually("editor idle") { fx { !window.stylesEditor.busy.get && window.stylesEditor.document.nonEmpty } }
    def named(name: String): Boolean = fx { window.stylesEditor.document.exists(_.promptStyles.exists(_.name == name)) }
    def select(name: String): Unit = fx {
      val editor = window.stylesEditor
      val doc = editor.document.get
      val row = doc.entries.find(e => doc.value(e, "name") == name).getOrElse(sys.error(s"Missing row $name"))
      editor.table.getSelectionModel.clearSelection()
      editor.table.getSelectionModel.select(row)
    }
    def dialog(button: String, name: Option[String] = None, accept: Boolean = true): Unit =
      val error = new AtomicReference[Throwable]()
      fx {
        Platform.runLater(() =>
          var pane: DialogPane = null
          try
            pane = Window.getWindows.asScala.filter(w => w.isShowing && (w ne window.stage)).flatMap { w =>
              w.getScene.getRoot match
                case p: DialogPane => Some(p)
                case n => Option(n.lookup(".dialog-pane")).collect { case p: DialogPane => p }
            }.headOption.getOrElse(sys.error(s"No dialog for $button"))
            name.foreach(value => pane.lookup(".text-field").asInstanceOf[TextField].setText(value))
            pane.lookupButton(if accept then ButtonType.OK else ButtonType.CANCEL).asInstanceOf[Button].fire()
          catch case e: Throwable =>
            error.set(e)
            if pane != null then pane.getScene.getWindow.hide()
        )
        find[Button](button).fire()
      }
      Option(error.get).foreach(throw _)
      idle()
    def screenshot(name: String, wholeWindow: Boolean = false): Unit = fx {
      window.root.applyCss(); window.root.layout()
      val node = if wholeWindow then window.root else window.stylesEditor.node
      ImageIO.write(SwingFXUtils.fromFXImage(node.snapshot(null, null), null), "png", output.resolve(name).toFile)
    }
    try
      window = fx {
        val w = new MainWindow(new Stage(), services, UiState(selectedTab = "styles"), CliOptions())
        w.show(); w
      }
      val hello = services.start()
      require(hello.dryRun == dryRun)
      val loaded = services.loadCheckpoint(args(2), remember = false)
      require(Set("sd15", "sdxl")(loaded.kind))
      fx { window.model.checkpoint.set(loaded.name); window.model.vae.set("Automatic") }
      idle()
      require(fx(window.state.selectedTab == "styles"))
      select("Existing")
      fx { window.stylesEditor.positive.setText("cinematic, {prompt}"); window.stylesEditor.notes.setText("日本語 notes") }
      eventually("actual detail autosave") { Styles.load(home.stylesFile).find(_.name == "Existing").exists(_.prompt == "cinematic, {prompt}") }
      idle()
      require(Files.readString(home.stylesFile).contains(untouched))
      eventually("live style dropdown data") { fx(window.ctx.styles.asScala.exists(s => s.name == "Existing" && s.prompt == "cinematic, {prompt}")) }
      checked("actual native edit/autosave, Unicode notes, untouched source record and live dropdown refresh")
      val beforeConflict = Files.readAllBytes(home.stylesFile)
      val external = Files.readString(home.stylesFile) + "External editor,external prompt,,external value\r\n"
      Files.writeString(home.stylesFile, external)
      fx { window.stylesEditor.positive.setText("unsaved conflict draft, {prompt}") }
      eventually("failed autosave preserved before navigation") { fx {
        !window.stylesEditor.busy.get && window.stylesEditor.lastRecoveryPath.nonEmpty &&
          find[Label]("styles-status").getText.startsWith("Not saved:")
      }}
      val recovery = fx(window.stylesEditor.lastRecoveryPath.get)
      require(Styles.load(recovery).find(_.name == "Existing").get.prompt == "unsaved conflict draft, {prompt}")
      require(Files.readString(home.stylesFile) == external, "Conflict overwrote external edits")
      select("Untouched")
      require(fx(window.stylesEditor.name.getText == "Untouched"))
      require(Styles.load(recovery).find(_.name == "Existing").get.prompt == "unsaved conflict draft, {prompt}")
      require(Files.readString(home.stylesFile) == external)
      Files.write(home.stylesFile, beforeConflict)
      fx { find[Button]("styles-reload").fire() }; idle()
      select("Existing")
      checked("failed native autosave retains exact on-disk recovery draft before row navigation and preserves external edits")
      val recoveryDirectory = home.configDir.resolve("style-drafts")
      val parkedRecovery = home.configDir.resolve("style-drafts-before-double-failure")
      val beforeDoubleFailure = Files.readAllBytes(home.stylesFile)
      val expectedPendingBytes = fx {
        val doc = window.stylesEditor.document.get
        doc.edit(window.stylesEditor.table.getSelectionModel.getSelectedItem.id,
          Map("notes" -> "double failure table note")).bytes
      }
      Files.move(recoveryDirectory, parkedRecovery)
      Files.writeString(recoveryDirectory, "Deliberately block recovery directory inside isolated acceptance home")
      try
        Files.writeString(home.stylesFile, Files.readString(home.stylesFile) + "External during double failure,external,,keep\r\n")
        fx {
          val table = window.stylesEditor.table
          val column = table.getColumns.get(4)
          val position = new TablePosition[StyleEntry, String](table, table.getSelectionModel.getSelectedIndex,
            column.asInstanceOf[TableColumn[StyleEntry, String]])
          val event = new TableColumn.CellEditEvent[StyleEntry, String](table, position,
            TableColumn.editCommitEvent[StyleEntry, String](), "double failure table note")
          javafx.event.Event.fireEvent(column, event)
        }
        eventually("save and draft recovery both fail") { fx {
          !window.stylesEditor.busy.get && find[Label]("styles-status").getText.startsWith("Not saved:")
        }}
        fx {
          require(window.stylesEditor.notes.isDisabled, "Detail editing must be locked when both persistence and recovery fail")
          require(!find[Button]("styles-save-changes").isDisabled, "Retry Save must remain usable")
          val previous = window.stylesEditor.notes.getText
          javafx.event.Event.fireEvent(window.stylesEditor.notes,
            new KeyEvent(KeyEvent.KEY_TYPED, "must not replace pending document", "", KeyCode.UNDEFINED, false, false, false, false))
          require(window.stylesEditor.notes.getText == previous, "Disabled details accepted editing after double failure")
        }
        Files.delete(recoveryDirectory)
        Files.move(parkedRecovery, recoveryDirectory)
        Files.write(home.stylesFile, beforeDoubleFailure)
        fx { find[Button]("styles-save-changes").fire() }; idle()
        require(Files.readAllBytes(home.stylesFile).sameElements(expectedPendingBytes),
          "Retry Save changed or lost the pending document after double failure")
      finally
        if Files.isRegularFile(recoveryDirectory) then Files.delete(recoveryDirectory)
        if Files.exists(parkedRecovery) then Files.move(parkedRecovery, recoveryDirectory)
        Files.write(home.stylesFile, beforeDoubleFailure)
      fx { find[Button]("styles-reload").fire() }; idle()
      select("Existing")
      checked("double persistence/recovery failure locks detail editing but Retry Save preserves the exact pending cell edit")
      def shortcut(code: KeyCode): Unit = fx {
        val table = window.stylesEditor.table
        table.getFocusModel.focus(table.getSelectionModel.getSelectedIndex, table.getColumns.get(4))
        javafx.event.Event.fireEvent(table, new KeyEvent(KeyEvent.KEY_PRESSED, "", "", code, false, true, false, false))
      }
      fx(NativeAcceptanceSafety.captureClipboard()) match
        case Right(saved) =>
          clipboardBackup = Some(saved)
          shortcut(KeyCode.C)
          require(fx(Clipboard.getSystemClipboard.getString == "日本語 notes"))
          shortcut(KeyCode.X); idle()
          require(fx { val d = window.stylesEditor.document.get; d.value(d.entries.find(e => d.value(e, "name") == "Existing").get, "notes").isEmpty })
          shortcut(KeyCode.V); idle()
          require(fx { val d = window.stylesEditor.document.get; d.value(d.entries.find(e => d.value(e, "name") == "Existing").get, "notes") == "日本語 notes" })
          checked("actual native table-cell Ctrl-C/X/V; clipboard restoration attempted on exit")
        case Left(reason) =>
          val skip = "native table-cell Ctrl-C/X/V: " + reason + "; original clipboard left unchanged"
          skips :+= skip
          println("STYLES SKIP: " + skip)
      fx {
        window.vms(Mode.Txt2Img).prompt.set("a cat"); window.vms(Mode.Txt2Img).negativePrompt.set("bad")
        find[Button]("styles-apply").fire()
        require(window.vms(Mode.Txt2Img).prompt.get == "cinematic, a cat")
        require(window.vms(Mode.Txt2Img).negativePrompt.get == "bad, lowres")
        window.ctx.openStylesEditor()
      }; idle()
      checked("native prompt-style Apply preserves placeholder/negative append semantics")
      fx { find[Button]("styles-duplicate").fire() }
      eventually("native duplicate") { named("Existingx") }; idle()
      fx { window.stylesEditor.search.setText("cinematic") }
      require(fx(window.stylesEditor.table.getItems.size == 2))
      fx { window.stylesEditor.filterMode.setValue("regex"); window.stylesEditor.search.setText("[") }
      require(fx(find[Label]("styles-status").getText.startsWith("Invalid regex:")))
      fx { window.stylesEditor.search.clear(); window.stylesEditor.filterMode.setValue("Case insensitive") }
      select("Existing")
      fx { find[Button]("styles-down").fire() }; idle()
      fx { find[TextField]("styles-find").setText("cinematic"); find[TextField]("styles-replacement").setText("watercolor") }
      dialog("styles-replace")
      require(Styles.load(home.stylesFile).find(_.name == "Existing").get.prompt == "watercolor, {prompt}")
      select("Existingx")
      val beforeCancel = hash(home.stylesFile)
      dialog("styles-delete", accept = false)
      require(hash(home.stylesFile) == beforeCancel)
      dialog("styles-delete")
      require(!named("Existingx"))
      checked("native duplicate/reorder, regex recovery, literal replace and confirmed/canceled deletion")
      fx { window.vms(Mode.Txt2Img).prompt.set("saved cat"); window.vms(Mode.Txt2Img).negativePrompt.set("bad") }
      dialog("styles-from-prompts", Some("From current prompts"))
      require(Styles.load(home.stylesFile).find(_.name == "From current prompts").get.prompt == "saved cat")
      select("Existing")
      dialog("styles-move-category", Some("Artists"))
      require(Styles.load(home.stylesFile).exists(_.name == "Artists::Existing"))
      fx { window.stylesEditor.category.setValue("Artists") }; idle()
      require(named("Existing"))
      fx { find[TitledPane]("styles-backup-panel").setExpanded(true); window.root.applyCss(); window.root.layout() }
      fx { find[CheckBox]("styles-backup-encrypt").setSelected(true); find[PasswordField]("styles-backup-password").setText("test-only-password")
        find[Button]("styles-backup-now").fire() }
      idle()
      val backup = window.ctx.styleLibrary.backups().find(_.toString.endsWith(".aes")).get
      fx { window.stylesEditor.positive.setText("newer category value") }
      eventually("new category save") { window.ctx.styleLibrary.loadCategory("Artists").promptStyles.head.prompt == "newer category value" }; idle()
      fx { find[ComboBox[Path]]("styles-backups").setValue(backup) }
      dialog("styles-restore-backup")
      require(window.ctx.styleLibrary.loadCategory("Artists").promptStyles.head.prompt == "watercolor, {prompt}")
      require(Files.readString(home.stylesFile).contains(untouched))
      checked("creation from prompts, category move/live merge, real AES backup and complete category restore")
      fx {
        window.vms(Mode.Txt2Img).prompt.set("preset cat"); window.vms(Mode.Txt2Img).negativePrompt.set("preset bad")
        window.vms(Mode.Txt2Img).width.set(384); window.vms(Mode.Txt2Img).height.set(512)
        window.stylesEditor.scope.setValue("Extended presets")
      }; idle()
      dialog("styles-from-prompts", Some("Same name"))
      dialog("styles-from-prompts", Some("Same name"))
      val presets = window.ctx.styleLibrary.extended()
      require(presets.size == 2 && presets.map(_.id).distinct.size == 2)
      fx { find[CheckBox]("styles-target").setSelected(true) }
      require(fx(window.stylesEditor.table.getItems.size == 2))
      fx { window.model.checkpoint.set("not-current"); find[CheckBox]("styles-target").setSelected(false); find[CheckBox]("styles-target").setSelected(true) }
      require(fx(window.stylesEditor.table.getItems.isEmpty))
      fx { find[Button]("styles-reload").fire() }; idle()
      require(fx(window.stylesEditor.table.getItems.size == 2))
      fx {
        window.stylesEditor.table.getSelectionModel.selectFirst()
        val vm = window.vms(Mode.Txt2Img)
        vm.prompt.set("changed"); vm.width.set(640); vm.height.set(640); vm.steps.set(11); vm.cfgScale.set(9); vm.seed.set(777)
        vm.controlNet.head.model.set("disabled draft"); vm.controlNet.head.enabled.set(false)
        window.model.vae.set("None")
        find[Button]("styles-apply").fire()
      }
      eventually("actual extended apply/model/VAE callback") { fx {
        val vm = window.vms(Mode.Txt2Img)
        vm.prompt.get == "preset cat" && vm.width.get == 384 && vm.height.get == 512 && window.model.checkpoint.get == loaded.name && window.model.vae.get == "Automatic"
      }}
      require(fx { val vm = window.vms(Mode.Txt2Img); vm.steps.get == 11 && vm.cfgScale.get == 9 && vm.seed.get == 777 && vm.controlNet.head.model.get == "disabled draft" })
      require(services.supervisor.loadedModel.exists(_.name == loaded.name))
      checked("UUID-distinct same-name extended presets, model/VAE filter + refresh, exact six-field apply preserves unrelated drafts")
      fx { window.ctx.openStylesEditor() }; idle()
      val blankPresetId = fx {
        window.stylesEditor.table.getSelectionModel.selectFirst()
        val d = window.stylesEditor.document.get
        val id = d.value(window.stylesEditor.table.getSelectionModel.getSelectedItem, "id")
        window.stylesEditor.vae.clear()
        id
      }
      eventually("native blank VAE edit saved") {
        window.ctx.styleLibrary.extended().find(_.id == blankPresetId).exists(_.vae.isEmpty) &&
          fx(!window.stylesEditor.busy.get)
      }
      val blankPreset = window.ctx.styleLibrary.extended().find(_.id == blankPresetId).get
      fx {
        window.model.vae.set("None")
        window.ctx.applyExtendedStyle(blankPreset)
        require(window.ctx.extendedStyleApplying.get, "Apply must remain guarded through model/VAE worker")
        require(find[Button]("styles-apply").isDisabled, "Native Apply remained enabled during model/VAE worker")
        find[Button]("styles-apply").fire() // disabled native handler cannot enqueue another preset
        window.ctx.applyExtendedStyle(blankPreset.copy(prompt = "REENTRANT APPLY MUST NOT WIN"))
        require(window.ctx.extendedStyleApplying.get)
      }
      eventually("blank VAE is Automatic in both UI and next generation request") { fx {
        !window.ctx.extendedStyleApplying.get && window.model.vae.get == "Automatic" &&
          window.vms(Mode.Txt2Img).prompt.get == blankPreset.prompt
      }}
      require(fx(window.vms(Mode.Txt2Img).toParams.vae.contains("Automatic")))
      require(services.supervisor.loadedModel.exists(_.vaeName.isEmpty), "Blank preset did not reset the backend VAE to checkpoint default")
      checked("blank preset VAE restores Automatic for backend/UI/next request; disabled native Apply and direct callback reject reentrant application")
      val p = GenerationParams(prompt = "a cat", negativePrompt = "bad", seed = 321, steps = 4, width = 384, height = 384,
        checkpoint = Some(loaded.name), vae = Some("Automatic"))
      val (positive, negative) = Styles.apply(Styles.load(home.stylesFile), Seq("Artists::Existing"), p.prompt, p.negativePrompt)
      def run(params: GenerationParams) = Await.result(services.queue.submit(Job(params)), 10.minutes).images.head
      val namedStyle = run(p.copy(styles = Seq("Artists::Existing")))
      val explicit = run(p.copy(prompt = positive, negativePrompt = negative))
      def pixels(path: Path): Seq[Int] =
        val i = ImageIO.read(path.toFile); i.getRGB(0, 0, i.getWidth, i.getHeight, null, 0, i.getWidth).toSeq
      require(pixels(namedStyle.path) == pixels(explicit.path), "named style differs from explicit conditioning")
      require(Infotext.parse(PngMetadata.readParameters(namedStyle.path).get).prompt == positive)
      checked(if dryRun then "CPU dry-run named-style request expansion and PNG metadata" else "actual real-model named-style conditioning equals explicit expanded prompt and PNG metadata")
      fx { window.ctx.openStylesEditor() }; idle()
      fx { window.stylesEditor.scope.setValue("Prompt styles") }; idle()
      screenshot("styles-editor.png")
      fx { window.stylesEditor.scope.setValue("Extended presets") }; idle()
      screenshot("extended-presets.png")
      fx { window.stage.setWidth(1280); window.stage.setHeight(800) }
      screenshot("styles-minimum-viewport.png", wholeWindow = true)
      require(!UiState.toJson(fx(window.state)).contains("test-only-password"))
      val report = ujson.Obj("status" -> "passed", "mode" -> (if dryRun then "cpu_dry_run" else "real_model"), "checkpoint" -> loaded.name, "kind" -> loaded.kind,
        "isolated_home" -> home.root.toString, "checks" -> ujson.Arr.from(checks), "skips" -> ujson.Arr.from(skips), "same_name_ids" -> ujson.Arr.from(presets.map(p => ujson.Str(p.id))),
        "named_image" -> namedStyle.path.toString, "explicit_image" -> explicit.path.toString,
        "styles_sha256" -> hash(home.stylesFile), "password_persisted" -> false)
      Files.writeString(output.resolve("styles-editor-report.json"), report.render(indent = 2))
      println(s"STYLES EDITOR VERIFIED: ${output.resolve("styles-editor-report.json")}")
    catch
      case error: Throwable =>
        primaryFailure = error
        throw error
    finally
      NativeAcceptanceSafety.cleanup(primaryFailure)(
        () => fx { clipboardBackup.foreach(NativeAcceptanceSafety.restore) },
        () => { if window != null then fx { window.dispose() } },
        () => { if window != null then fx { window.stage.close() } },
        () => services.shutdown(),
        () => Platform.exit())
