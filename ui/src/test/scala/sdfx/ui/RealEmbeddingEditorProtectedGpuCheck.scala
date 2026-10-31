package sdfx.ui

import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, CheckBox, ComboBox, Label, Slider, TabPane, TextArea, TextField}
import javafx.stage.Stage
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.png.ScrambledPng
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, EditorPreviewPayload, EngineTimeouts, Job, JobEvent}
import sdfx.core.model.Img2ImgParams
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}
import scala.concurrent.duration.*

/** Real protected native editor preview; writes artifacts only below the supplied output directory. */
object RealEmbeddingEditorProtectedGpuCheck:
  def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(45, TimeUnit.SECONDS)
  def waitFor(label: String, seconds: Int = 120)(condition: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds)
    while !condition && System.nanoTime() < end do Thread.sleep(100)
    require(condition, s"Timed out: $label")

  def main(args: Array[String]): Unit =
    require(args.length >= 3 && args.length <= 5, "output-dir python checkpoint [vram-mode [source-python-dir]]")
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val python = Path.of(args(1)).toAbsolutePath; val checkpoint = Path.of(args(2)).toAbsolutePath
    val mode = if args.length >= 4 then args(3) else "gpu"
    require(Set("gpu", "model_offload", "sequential_offload")(mode))
    val run = Files.createTempDirectory(root, "editor-protected-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    Files.createDirectories(home.pythonDir)
    val source = (if args.length == 5 then Path.of(args(4)).resolve("sdfx") else Path.of("python", "sdfx")).toAbsolutePath
    require(Files.isRegularFile(source.resolve("embedding_editor_generation.py")), "editor source missing")
    val sources = Files.walk(source)
    try sources.iterator().asScala.filterNot(_.toString.contains("__pycache__")).foreach { path =>
      val out = home.pythonDir.resolve("sdfx").resolve(source.relativize(path).toString)
      if Files.isDirectory(path) then Files.createDirectories(out) else Files.copy(path, out)
    } finally sources.close()
    val embeddings = Files.createDirectories(run.resolve("embeddings"))
    val service = new AppServices(home, Settings(pythonExe = Some(python.toString), hfOffline = true,
      allowCustomCode = true, vramMode = mode,
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString), embeddings = Seq(embeddings.toString))),
      dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false,
      installShutdownHook = false, timeouts = EngineTimeouts(load = 10.minutes))
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    val logs = new java.util.concurrent.CopyOnWriteArrayList[String]()
    val previewJobs = new java.util.concurrent.CopyOnWriteArrayList[String]()
    val previewPayloads = new java.util.concurrent.CopyOnWriteArrayList[EditorPreviewPayload]()
    val draftCanary = "0.123456789123"
    service.queue.addListener {
      case e: JobEvent.JobStarted if e.job.editorPreview.nonEmpty =>
        previewJobs.add(e.job.id); previewPayloads.add(e.job.editorPreview.get); ()
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case e: JobEvent.Log => logs.add(e.message); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def nodes(n: Node): Seq[Node] = Seq(n) ++ (n match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.root).find(_.getId == id).getOrElse(sys.error(s"Missing $id"))
    def label(id: String): String = fx(find(id).asInstanceOf[Label].getText)
    def waitReply(id: String): Unit = waitFor(id)(!label(id).startsWith("Queued"))
    def image(): OutputImage =
      val value = finished.poll(300, TimeUnit.SECONDS)
      require(value != null, "generation timed out")
      val images = value.fold(message => throw new IllegalStateException(message), identity)
      require(images.size == 1); waitFor("idle")(service.queue.isIdle && fx(!window.progress.get.running))
      images.head
    try
      require(!service.start().dryRun)
      val loaded = service.loadCheckpoint(checkpoint.toString, remember = false)
      require(Set("sd15", "sdxl")(loaded.kind))
      fx {
        window = new MainWindow(new Stage(), service, UiState(), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(loaded.name)
        val vm = window.vms(Mode.Txt2Img)
        vm.seed.set(173); vm.steps.set(if mode == "sequential_offload" then 2 else 5)
        vm.width.set(512); vm.height.set(512); vm.prompt.set("cat")
        window.select(Mode.Txt2Img); window.statusBar.generateBtn.fire()
      }
      image()
      val embedding = embeddings.resolve("nuieditorfixture.safetensors")
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
        find("embedding-inspector-rows").asInstanceOf[TextArea].setText("cat|1")
        find("embedding-inspector-output").asInstanceOf[TextField].setText(embedding.toString)
        find("embedding-inspector-save").asInstanceOf[Button].fire()
      }
      waitReply("embedding-inspector-status")
      require(Files.isRegularFile(embedding), label("embedding-inspector-status"))
      service.catalog.refresh()
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-editor-tab").get)
        find("embedding-editor-name").asInstanceOf[TextField].setText("nuieditorfixture")
        find("embedding-editor-open").asInstanceOf[Button].fire()
      }
      waitReply("embedding-editor-status")
      require(label("embedding-editor-status").startsWith("Opened"), label("embedding-editor-status"))
      val original = Files.readAllBytes(embedding)
      fx {
        service.imageScrambleSession.unlock("editor-test-password")
        window.vms(Mode.Txt2Img).prompt.set("nuieditorfixture")
        window.select(Mode.Txt2Img); window.statusBar.generateBtn.fire()
      }
      val before = image()
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-editor-tab").get)
        val exact = find("embedding-editor-coordinate-value-0").asInstanceOf[TextField]
        val first = find("embedding-editor-coordinate-0").asInstanceOf[Slider]
        exact.setText("0.125"); exact.fireEvent(new javafx.event.ActionEvent())
        require(first.getValue == 0.125, "exact coordinate entry was not applied")
        exact.requestFocus(); exact.setText("0.25")
        find("embedding-editor-name").requestFocus()
        require(first.getValue == 0.25, "focus loss discarded typed coordinate")
        exact.requestFocus(); exact.setText("NaN")
        find("embedding-editor-preview").asInstanceOf[Button].fire()
        require(first.getValue == 0.25, "invalid coordinate entry changed the draft")
        require(service.queue.isIdle, "invalid coordinate entry queued a preview")
        (0 until 24).foreach { i =>
          val slider = find(s"embedding-editor-coordinate-$i").asInstanceOf[Slider]
          slider.setValue(if slider.getValue - slider.getMin > slider.getMax - slider.getValue then slider.getMin else slider.getMax)
        }
        exact.setText(draftCanary); exact.fireEvent(new javafx.event.ActionEvent())
        find("embedding-editor-guidance").asInstanceOf[TextField].setText("cat:#FF0000, dog:#00FF00")
        find("embedding-editor-update-guidance").asInstanceOf[Button].fire()
      }
      waitFor("loaded guidance reply")(label("embedding-editor-status").startsWith("Guidance updated"))
      fx {
        require(find("embedding-editor-coordinate-0").asInstanceOf[Slider].getValue == draftCanary.toDouble,
          "guidance refresh discarded unsaved coordinates")
        require(find("embedding-editor-guidance-bar-0").asInstanceOf[Parent].getChildrenUnmodifiable.size == 2,
          "real tokenizer guidance markers missing")
        find("embedding-editor-preview").asInstanceOf[Button].fire()
      }
      val generated = image()
      val decoded = ScrambledPng.read(generated.path, Some("editor-test-password"))
      require(decoded.scrambled && decoded.passwordVerified.contains(true))
      fx { window.select(Mode.Txt2Img); window.statusBar.generateBtn.fire() }
      val after = image()
      def pixels(path: Path): Array[Int] =
        val picture = ScrambledPng.read(path, Some("editor-test-password")).image
        picture.getRGB(0, 0, picture.getWidth, picture.getHeight, null, 0, picture.getWidth)
      require(!java.util.Arrays.equals(pixels(before.path), pixels(generated.path)), "edited preview had no visible effect")
      require(java.util.Arrays.equals(pixels(before.path), pixels(after.path)), "model was not restored after preview")
      require(java.util.Arrays.equals(original, Files.readAllBytes(embedding)), "preview changed source file")
      val snapshots = service.generator.payloads.snapshots
      require(snapshots.forall(s => !s.bodyJson.exists(_.contains("embedding_editor_preview"))))
      require(previewJobs.size() == 1)
      val previewCapture = snapshots.find(_.key.jobId == previewJobs.get(0)).getOrElse(sys.error("missing preview diagnostic"))
      require(previewCapture.bodyJson.isEmpty, "preview request body was retained")
      require(!previewCapture.toString.contains(draftCanary), "preview diagnostic leaked a coordinate")
      require(!logs.asScala.exists(_.contains(draftCanary)), "job log leaked a coordinate")
      require(!new String(Files.readAllBytes(generated.path), java.nio.charset.StandardCharsets.ISO_8859_1).contains(draftCanary),
        "image bytes or metadata leaked a coordinate")
      require(logs.asScala.exists(_.contains("exact request capture is disabled")))
      // Exercise the worker-only Alternative inversion route using the actual UI draft.
      // The native editor deliberately rejects Script combinations, so submit the
      // typed job directly while retaining the same protected session and queue.
      val inversionParams = fx(window.vms(Mode.Txt2Img).toParams).copy(
        steps = 3, img2img = Some(Img2ImgParams(initImage = before.path.toString, denoisingStrength = 0.6)),
        hooks = Map("img2imgalt" -> ujson.Obj("override_prompt" -> "nuieditorfixture",
          "override_negative" -> "blurry", "inversion_steps" -> 3, "decode_cfg_scale" -> 1.0, "randomness" -> 0.0)))
      def inversion(edited: Boolean): OutputImage =
        service.queue.submit(Job(Job.newId(), inversionParams, origin = "editor-inversion-check",
          imageSecret = service.imageScrambleSession.snapshot,
          editorPreview = Option.when(edited)(previewPayloads.get(0))))
        image()
      val inversionBefore = inversion(false)
      val inversionPreview = inversion(true)
      val inversionAfter = inversion(false)
      require(java.util.Arrays.equals(pixels(inversionBefore.path), pixels(inversionAfter.path)),
        "inversion override preview did not restore the loaded vectors")
      require(!java.util.Arrays.equals(pixels(inversionBefore.path), pixels(inversionPreview.path)),
        "inversion override editor preview had no effect")
      require(java.util.Arrays.equals(original, Files.readAllBytes(embedding)), "inversion preview changed source file")
      // The preview checks above finish before explicit publication changes the fixture.
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-editor-tab").get)
        window.root.applyCss(); window.root.layout()
        javax.imageio.ImageIO.write(javafx.embed.swing.SwingFXUtils.fromFXImage(window.scene.snapshot(null), null),
          "png", run.resolve("editor-before-save.png").toFile)
        find("embedding-editor-save").asInstanceOf[Button].fire()
        require(find("embedding-editor-status").asInstanceOf[Label].getText.contains("overwrite consent"))
        require(java.util.Arrays.equals(original, Files.readAllBytes(embedding)), "unapproved save modified source")
        find("embedding-editor-overwrite").asInstanceOf[CheckBox].setSelected(true)
        find("embedding-editor-save").asInstanceOf[Button].fire()
      }
      waitReply("embedding-editor-status")
      require(label("embedding-editor-status").startsWith("Saved and reloaded."), label("embedding-editor-status"))
      require(!java.util.Arrays.equals(original, Files.readAllBytes(embedding)), "save did not publish edited vectors")
      val backupFiles = Files.list(embeddings)
      val backups = try backupFiles.iterator().asScala.filter(_.getFileName.toString.startsWith(
        embedding.getFileName.toString + ".before-edit-")).toVector finally backupFiles.close()
      require(backups.size == 1 && java.util.Arrays.equals(original, Files.readAllBytes(backups.head)),
        "save did not preserve the exact original in its backup")
      fx { window.select(Mode.Txt2Img); window.statusBar.generateBtn.fire() }
      val savedImage = image()
      require(java.util.Arrays.equals(pixels(generated.path), pixels(savedImage.path)),
        "saved embedding reload differs from the identical unsaved preview")
      val inversionSaved = inversion(false)
      require(java.util.Arrays.equals(pixels(inversionPreview.path), pixels(inversionSaved.path)),
        "saved vectors differ from preview during inversion override encoding")
      // Independent owners named base and base_1 must not be mistaken for a multi-vector embedding.
      val ownerBase = embeddings.resolve("ownerprobe.safetensors")
      val ownerSuffix = embeddings.resolve("ownerprobe_1.safetensors")
      Files.copy(embedding, ownerBase); Files.copy(embedding, ownerSuffix)
      val ownerBytes = Files.readAllBytes(ownerBase)
      service.catalog.refresh()
      def openOwner(name: String): Unit =
        fx {
          val tabs = find("main-tabs").asInstanceOf[TabPane]
          tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-editor-tab").get)
          find("embedding-editor-catalog").asInstanceOf[ComboBox[String]].setValue(
            embeddings.resolve(name + ".safetensors").toString)
          find("embedding-editor-name").asInstanceOf[TextField].setText(name)
          find("embedding-editor-open").asInstanceOf[Button].fire()
        }
        waitReply("embedding-editor-status")
        require(label("embedding-editor-status").startsWith("Opened"), label("embedding-editor-status"))
      openOwner("ownerprobe"); openOwner("ownerprobe_1"); openOwner("ownerprobe")
      fx {
        val exact = find("embedding-editor-coordinate-value-0").asInstanceOf[TextField]
        exact.setText("0.5"); exact.fireEvent(new javafx.event.ActionEvent())
        find("embedding-editor-overwrite").asInstanceOf[CheckBox].setSelected(true)
        find("embedding-editor-save").asInstanceOf[Button].fire()
      }
      waitReply("embedding-editor-status")
      require(label("embedding-editor-status").toLowerCase.contains("collision"), label("embedding-editor-status"))
      require(java.util.Arrays.equals(ownerBytes, Files.readAllBytes(ownerBase)) &&
        java.util.Arrays.equals(ownerBytes, Files.readAllBytes(ownerSuffix)), "collision refusal changed a source")
      require(fx(find("embedding-editor-coordinate-value-0").asInstanceOf[TextField].getText.toDouble) == 0.5,
        "collision refusal discarded the unsaved draft")
      val collisionFiles = Files.list(embeddings)
      try require(!collisionFiles.iterator().asScala.exists(_.getFileName.toString.startsWith("ownerprobe.safetensors.before-edit-")),
        "collision refusal reached backup/publication") finally collisionFiles.close()
      Files.writeString(run.resolve("receipt.json"), ujson.Obj("family" -> loaded.kind, "vram_mode" -> mode,
        "before" -> before.path.toString, "protected_image" -> generated.path.toString,
        "after" -> after.path.toString, "preview_source_unchanged" -> true,
        "inversion_before" -> inversionBefore.path.toString, "inversion_preview" -> inversionPreview.path.toString,
        "inversion_saved" -> inversionSaved.path.toString, "inversion_saved_matches_preview" -> true,
        "saved_image" -> savedImage.path.toString, "backup" -> backups.head.toString,
        "save_consent_enforced" -> true, "saved_reload_matches_preview" -> true, "independent_owner_collision_refused" -> true).render(2))
      println(s"PASS protected editor preview $run")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      service.shutdown(); Platform.exit()
