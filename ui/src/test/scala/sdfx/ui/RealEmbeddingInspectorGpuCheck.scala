package sdfx.ui

import java.nio.file.{Files, Path}
import java.util.concurrent.{Callable, CountDownLatch, FutureTask, LinkedBlockingQueue, TimeUnit}
import javafx.application.Platform
import javafx.scene.{Node, Parent}
import javafx.scene.control.{Button, CheckBox, Label, TabPane, TextArea, TextField}
import javafx.stage.Stage
import scala.concurrent.Await
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*
import sdfx.core.config.{AppHome, ModelDirs, Settings}
import sdfx.core.plugins.api.OutputImage
import sdfx.engine.{AppServices, EngineTimeouts, JobEvent}
import sdfx.plugins.StandardPlugins
import sdfx.ui.vm.{Mode, UiState}

/** Actual mounted inspector plus generated image using its saved embedding. Args: output-dir python checkpoint. */
object RealEmbeddingInspectorGpuCheck:
  // Read Accelerate's offloaded backing tensor without detaching hooks or moving the model.
  private val readWeightCode = """
inspector_test_states = []
def inspector_test_weight(encoder):
    module = encoder.get_input_embeddings()
    weight = module.weight
    hook = getattr(module, '_hf_hook', None)
    hooks = tuple(getattr(hook, 'hooks', [hook]))
    devices = tuple(str(getattr(item, 'execution_device', None)) for item in hooks)
    if weight.is_meta:
        backing = None
        for item in hooks:
            mapping = getattr(item, 'weights_map', None)
            if mapping is not None:
                try:
                    backing = mapping['weight'].detach()
                    break
                except (KeyError, TypeError):
                    pass
        if backing is None:
            raise RuntimeError('cannot locate offloaded text embedding backing weights')
    else:
        backing = weight.detach()
    assert module.weight is weight and getattr(module, '_hf_hook', None) is hook
    assert tuple(getattr(hook, 'hooks', [hook])) == hooks
    assert tuple(str(getattr(item, 'execution_device', None)) for item in hooks) == devices
    inspector_test_states.append({'parameter_device': str(weight.device),
        'backing_device': str(backing.device), 'execution_devices': devices,
        'hook_types': [type(item).__name__ for item in hooks]})
    return backing
"""
  private def fx[A](body: => A): A =
    val task = new FutureTask[A](new Callable[A] { def call(): A = body })
    Platform.runLater(task); task.get(45, TimeUnit.SECONDS)
  private def waitFor(label: String, seconds: Int = 120)(condition: => Boolean): Unit =
    val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(seconds)
    while !condition && System.nanoTime() < end do Thread.sleep(100)
    require(condition, s"Timed out: $label")
  def main(args: Array[String]): Unit =
    require(args.length == 3 || args.length == 4, "output-dir python checkpoint-file [vram-mode]")
    val vramMode = if args.length == 4 then args(3) else "gpu"
    require(Set("gpu", "model_offload", "sequential_offload")(vramMode))
    val root = Path.of(args(0)).toAbsolutePath; Files.createDirectories(root)
    val python = Path.of(args(1)).toAbsolutePath
    val checkpoint = Path.of(args(2)).toAbsolutePath
    require(Files.isRegularFile(python) && Files.isRegularFile(checkpoint))
    val run = Files.createTempDirectory(root, "embedding-inspector-gpu-")
    val home = AppHome(Files.createDirectories(run.resolve("home")))
    Files.createDirectories(home.pythonDir)
    val source = Path.of("python", "sdfx").toAbsolutePath
    val target = home.pythonDir.resolve("sdfx")
    val sources = Files.walk(source)
    try sources.iterator().asScala.filterNot(_.toString.contains("__pycache__")).foreach { path =>
      val out = target.resolve(source.relativize(path).toString)
      if Files.isDirectory(path) then Files.createDirectories(out) else Files.copy(path, out)
    }
    finally sources.close()
    val embeddings = Files.createDirectories(run.resolve("embeddings"))
    val services = new AppServices(home, Settings(pythonExe = Some(python.toString), hfOffline = true, allowCustomCode = true, vramMode = vramMode,
      modelDirs = ModelDirs(checkpoints = Seq(checkpoint.getParent.toString), embeddings = Seq(embeddings.toString))),
      dryRun = false, builtins = StandardPlugins.withOwnAxes, persistSettings = false,
      installShutdownHook = false, timeouts = EngineTimeouts(load = 10.minutes))
    val finished = new LinkedBlockingQueue[Either[String, Seq[OutputImage]]]()
    services.queue.addListener {
      case e: JobEvent.JobFinished => finished.offer(Right(e.output.images)); ()
      case e: JobEvent.JobFailed => finished.offer(Left(e.message)); ()
      case _ => ()
    }
    val ready = new CountDownLatch(1)
    Platform.startup(() => { Platform.setImplicitExit(false); ready.countDown() })
    require(ready.await(30, TimeUnit.SECONDS))
    var window: MainWindow = null
    def nodes(node: Node): Seq[Node] = Seq(node) ++ (node match
      case p: Parent => p.getChildrenUnmodifiable.asScala.toSeq.flatMap(nodes)
      case _ => Nil)
    def find(id: String): Node = nodes(window.root).find(_.getId == id).getOrElse(sys.error(s"Missing $id"))
    def status: String = fx(find("embedding-inspector-status").asInstanceOf[Label].getText)
    def awaitDone(): Unit =
      waitFor("inspector response") {
        val text = status
        !text.startsWith("Queued") && !text.startsWith("Saved and reloaded; refreshing")
      }
      require(status.startsWith("Completed"), s"Inspector failed: $status")
    def generate(prompt: String): OutputImage =
      fx { window.select(Mode.Txt2Img); window.vms(Mode.Txt2Img).prompt.set(prompt); window.statusBar.generateBtn.fire() }
      val value = finished.poll(300, TimeUnit.SECONDS)
      require(value != null, "Generation timed out")
      val images = value.fold(message => throw new IllegalStateException(message), identity)
      require(images.size == 1)
      waitFor("generation idle")(services.queue.isIdle && fx(!window.progress.get.running))
      images.head
    try
      val hello = services.start(); require(!hello.dryRun)
      val loaded = services.loadCheckpoint(checkpoint.toString, remember = false)
      require(Set("sd15", "sdxl")(loaded.kind))
      fx {
        window = new MainWindow(new Stage(), services, UiState(), CliOptions(), _ => ())
        window.show(); window.model.checkpoint.set(loaded.name)
        val vm = window.vms(Mode.Txt2Img)
        vm.seed.set(173); vm.steps.set(if vramMode == "sequential_offload" then 2 else 5); vm.width.set(512); vm.height.set(512)
      }
      val started = System.nanoTime()
      def phase(label: String): Unit = println(f"PHASE $label at ${(System.nanoTime() - started) / 1e9}%.2fs")
      phase("baseline-start")
      val baseline = generate("cat")
      phase("baseline-complete")
      if vramMode != "gpu" then
        phase("no-refresh-control-start")
        val control = generate("cat")
        val before = javax.imageio.ImageIO.read(baseline.path.toFile)
        val after = javax.imageio.ImageIO.read(control.path.toFile)
        require(java.util.Arrays.equals(before.getRGB(0, 0, before.getWidth, before.getHeight, null, 0, before.getWidth),
          after.getRGB(0, 0, after.getWidth, after.getHeight, null, 0, after.getWidth)), "No-refresh control differs")
        phase("no-refresh-control-complete")
      val destination = embeddings.resolve("nuiinspectorcheck.safetensors")
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
        find("embedding-inspector-input").asInstanceOf[TextField].setText("cat")
        find("embedding-inspector-inspect").asInstanceOf[Button].fire()
      }
      awaitDone()
      val report = fx(find("embedding-inspector-report").asInstanceOf[TextArea].getText)
      require(report.contains("clip_l") && (loaded.kind != "sdxl" || report.contains("clip_g")))
      phase("refresh-start")
      fx {
        find("embedding-inspector-rows").asInstanceOf[TextArea].setText("cat|1")
        find("embedding-inspector-output").asInstanceOf[TextField].setText(destination.toString)
        find("embedding-inspector-eval").asInstanceOf[TextField].setText("v*1")
        find("embedding-inspector-save").asInstanceOf[Button].fire()
      }
      awaitDone(); require(Files.isRegularFile(destination))
      phase("refresh-complete")
      services.catalog.refresh()
      require(services.catalog.embeddings.exists(_.path == destination.toString))
      phase("after-refresh-generation-start")
      val generated = generate("nuiinspectorcheck")
      phase("after-refresh-generation-complete")
      val diagnostic = Await.result(services.client.raw("run_code", ujson.Obj("code" -> (readWeightCode + """
import json
from sdfx import prompt
with torch.no_grad():
    a, ap = prompt.encode_sdxl(pipe, 'cat') if engine.sdxl else (None, None)
    b, bp = prompt.encode_sdxl(pipe, 'nuiinspectorcheck') if engine.sdxl else (None, None)
    rows = []
    for tok, enc in [(pipe.tokenizer, pipe.text_encoder)] + ([(pipe.tokenizer_2, pipe.text_encoder_2)] if engine.sdxl else []):
        original = tok.encode('cat', add_special_tokens=False)
        added = tok.encode('nuiinspectorcheck', add_special_tokens=False)
        weights = inspector_test_weight(enc)
        rows.append({'original': original, 'added': added, 'tokenizer_eos': tok.eos_token_id,
                     'encoder_eos': enc.text_model.eos_token_id,
                     'row_equal': bool(torch.equal(weights[original[0]], weights[added[0]]))})
    result = json.dumps({'states': inspector_test_states, 'rows': rows,
                         'hidden_equal': None if a is None else bool(torch.equal(a,b)),
                         'hidden_max_diff': None if a is None else float((a-b).abs().max()),
                         'pooled_equal': None if ap is None else bool(torch.equal(ap,bp)),
                         'pooled_max_diff': None if ap is None else float((ap-bp).abs().max())})
"""))), 120.seconds)
      Files.writeString(run.resolve("conditioning-diagnostic.json"), diagnostic("result").str)
      println(s"CONDITIONING_DIAGNOSTIC ${diagnostic("result").str}")
      val a = javax.imageio.ImageIO.read(baseline.path.toFile)
      val b = javax.imageio.ImageIO.read(generated.path.toFile)
      require(a != null && b != null)
      val same = java.util.Arrays.equals(a.getRGB(0, 0, a.getWidth, a.getHeight, null, 0, a.getWidth),
        b.getRGB(0, 0, b.getWidth, b.getHeight, null, 0, b.getWidth))
      Files.writeString(run.resolve("result.json"), ujson.Obj("family" -> loaded.kind, "vram_mode" -> vramMode,
        "baseline" -> baseline.path.toString, "generated" -> generated.path.toString,
        "embedding" -> destination.toString, "pixel_equal" -> same).render(2))
      require(same, "Saved token did not match original token generation")
      // Load two separately named files through the normal generation path. The
      // suffix is an independent embedding, not a second row of the base one.
      val collisionBase = embeddings.resolve("nuicollision.safetensors")
      val collisionSuffix = embeddings.resolve("nuicollision_1.safetensors")
      Await.result(services.client.raw("run_code", ujson.Obj("base" -> collisionBase.toString,
        "suffix" -> collisionSuffix.toString, "code" -> (readWeightCode + """
from safetensors.torch import save_file
left = inspector_test_weight(pipe.text_encoder)[2368].detach().cpu().float().unsqueeze(0).contiguous()
if engine.sdxl:
    right = inspector_test_weight(pipe.text_encoder_2)[2368].detach().cpu().float().unsqueeze(0).contiguous()
    data = {'clip_l': left, 'clip_g': right}
else:
    data = {'emb_params': left}
save_file(data, params['base'], metadata={'step': '42', 'sd_checkpoint_name': 'fixture-checkpoint'})
save_file({key: value.clone() for key, value in data.items()}, params['suffix'])
result = 'independent files prepared'
"""))), 120.seconds)
      services.catalog.refresh()
      if vramMode != "gpu" then
        Await.result(services.client.raw("run_code", ujson.Obj("trace" -> run.resolve("offload-reapply-trace.jsonl").toString,
          "mode" -> vramMode, "code" -> """
import json
method = 'enable_model_cpu_offload' if params['mode'] == 'model_offload' else 'enable_sequential_cpu_offload'
original_offload_method = getattr(pipe, method)
trace_path = params['trace']
def traced_offload(*args, **kwargs):
    state = {'method': method, 'args': [str(a) for a in args], 'kwargs': {k: str(v) for k,v in kwargs.items()},
             'encoder_devices': [str(pipe.text_encoder.device)] + ([str(pipe.text_encoder_2.device)] if engine.sdxl else [])}
    with open(trace_path, 'a', encoding='utf-8') as output: output.write(json.dumps(state) + '\n')
    return original_offload_method(*args, **kwargs)
setattr(pipe, method, traced_offload)
result = 'offload reapply tracing enabled'
""")), 120.seconds)
      phase("collision-base-load-start")
      generate("nuicollision")
      phase("collision-suffix-load-start")
      generate("nuicollision_1")
      phase("collision-loaded")
      val collisionBefore = Await.result(services.client.raw("run_code", ujson.Obj("code" -> (readWeightCode + """
import json
names = ('nuicollision', 'nuicollision_1')
streams = [('clip_l', pipe.tokenizer, pipe.text_encoder)]
if engine.sdxl: streams.append(('clip_g', pipe.tokenizer_2, pipe.text_encoder_2))
engine._collision_test_rows = {}
for key, tok, enc in streams:
    for name in names:
        assert engine.embedding_token_owners[name][key] == [name]
        assert name in tok.get_added_vocab()
        engine._collision_test_rows[(key, name)] = inspector_test_weight(enc)[tok.convert_tokens_to_ids(name)].detach().clone()
result = json.dumps(engine.embedding_token_owners)
"""))), 120.seconds)
      Files.writeString(run.resolve("collision-ownership-before.json"), collisionBefore("result").str)
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
        find("embedding-inspector-list").asInstanceOf[Button].fire()
      }
      awaitDone()
      val listReport = fx(find("embedding-inspector-report").asInstanceOf[TextArea].getText)
      require(listReport.contains("Loaded embeddings") && listReport.contains("nuicollision:") &&
        listReport.contains("nuicollision_1:") && listReport.contains("file step: 42") &&
        listReport.contains("file sd_checkpoint_name: fixture-checkpoint"), s"Incomplete loaded list: $listReport")
      Files.writeString(run.resolve("loaded-list-report.txt"), listReport)
      val oldFile = Files.readAllBytes(collisionBase)
      val inspectParams = ujson.Obj("operation" -> "inspect", "text" -> "nuicollision", "limit" -> 2,
        "embedding_paths" -> ujson.Arr.from(services.catalog.embeddings.map(e => ujson.Str(e.path))))
      val loadedBefore = Await.result(services.client.raw("embedding_inspector", inspectParams), 120.seconds)
      Await.result(services.client.raw("run_code", ujson.Obj("path" -> collisionBase.toString, "code" -> """
import gc
from safetensors.torch import load_file, save_file
changed = load_file(params['path'])
zeros = {key: torch.zeros_like(value) for key,value in changed.items()}
del changed
gc.collect()
save_file(zeros, params['path'])
result = 'catalog file changed after loading'
""")), 120.seconds)
      val loadedAfter = Await.result(services.client.raw("embedding_inspector", inspectParams), 120.seconds)
      require(loadedAfter("metadata")("source").str == "loaded model" &&
        loadedAfter("metadata")("file_metadata_status").str.contains("changed"), "Changed catalog file was not flagged")
      for (key, data) <- loadedBefore("encoders").obj do
        require(data("vectors") == loadedAfter("encoders")(key)("vectors"), s"Changed catalog file replaced live $key vectors")
      Files.write(collisionBase, oldFile)
      println(s"PASS ${loaded.kind} inspect used loaded rows and flagged changed catalog file")
      val oldSuffixFile = Files.readAllBytes(collisionSuffix)
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
        find("embedding-inspector-rows").asInstanceOf[TextArea].setText("cat|1")
        find("embedding-inspector-eval").asInstanceOf[TextField].clear()
        find("embedding-inspector-overwrite").asInstanceOf[CheckBox].setSelected(true)
        find("embedding-inspector-output").asInstanceOf[TextField].setText(collisionBase.toString)
        find("embedding-inspector-save").asInstanceOf[Button].fire()
      }
      waitFor("collision rejection") { !status.startsWith("Queued") }
      require(status.contains("embedding token collision: nuicollision_1"), s"Missing collision explanation: $status")
      require(java.util.Arrays.equals(oldFile, Files.readAllBytes(collisionBase)), "Collision changed the existing file")
      require(java.util.Arrays.equals(oldSuffixFile, Files.readAllBytes(collisionSuffix)), "Collision changed the separate suffix file")
      val collisionAfter = Await.result(services.client.raw("run_code", ujson.Obj("code" -> (readWeightCode + """
checks = []
streams = [('clip_l', pipe.tokenizer, pipe.text_encoder)]
if engine.sdxl: streams.append(('clip_g', pipe.tokenizer_2, pipe.text_encoder_2))
for key, tok, enc in streams:
    for name in ('nuicollision', 'nuicollision_1'):
        checks.append(name in tok.get_added_vocab() and engine.embedding_token_owners[name][key] == [name]
                      and torch.equal(inspector_test_weight(enc)[tok.convert_tokens_to_ids(name)], engine._collision_test_rows[(key, name)]))
result = str(all(checks))
"""))), 120.seconds)
      require(collisionAfter("result").str == "True", "Collision changed a loaded embedding")
      phase("collision-rejected-preserved")
      println(s"PASS ${loaded.kind} independent suffix collision rejected before overwrite; both files and live encoder rows preserved")
      if loaded.kind == "sdxl" then
        val partialPath = embeddings.resolve("nuipartial.safetensors")
        Files.copy(collisionBase, partialPath)
        services.catalog.refresh()
        Await.result(services.client.raw("run_code", ujson.Obj("code" -> """
engine._ordinary_test_original = pipe.load_textual_inversion
def fail_second_ordinary(*args, **kwargs):
    if kwargs.get('tokenizer') is pipe.tokenizer:
        raise RuntimeError('injected ordinary second encoder failure')
    return engine._ordinary_test_original(*args, **kwargs)
pipe.load_textual_inversion = fail_second_ordinary
result = 'ordinary partial load failure armed'
""")), 120.seconds)
        phase("ordinary-partial-failure-start")
        fx { window.select(Mode.Txt2Img); window.vms(Mode.Txt2Img).prompt.set("nuipartial"); window.statusBar.generateBtn.fire() }
        val partial = finished.poll(300, TimeUnit.SECONDS)
        require(partial != null && partial.isLeft && partial.swap.toOption.get.contains("reload the checkpoint"),
          s"Expected partial ordinary load to abort and require checkpoint reload: $partial")
        waitFor("partial load failure idle")(services.queue.isIdle && fx(!window.progress.get.running))
        require(services.supervisor.loadedModel.isEmpty, "Supervisor still reports the invalidated model loaded")
        services.loadCheckpoint(checkpoint.toString, remember = false)
        generate("cat")
        generate("nuiinspectorcheck")
        phase("ordinary-partial-failure-recovered")
        println("PASS SDXL ordinary partial load invalidated model and checkpoint reload recovered")
      if loaded.kind == "sdxl" then
        val sourcePair = embeddings.resolve("nuizerosource.safetensors")
        val mixedPair = embeddings.resolve("nuizeromixed.safetensors")
        Await.result(services.client.raw("run_code", ujson.Obj("source_path" -> sourcePair.toString, "code" -> (readWeightCode + """
from safetensors.torch import save_file
left = inspector_test_weight(pipe.text_encoder)[2368].cpu().float()
right = inspector_test_weight(pipe.text_encoder_2)[2368].cpu().float()
save_file({'clip_l': torch.stack([left, torch.zeros_like(left)]),
           'clip_g': torch.stack([torch.zeros_like(right), right])}, params['source_path'])
result = 'saved source pair'
"""))), 120.seconds)
        services.catalog.refresh()
        fx {
          val tabs = find("main-tabs").asInstanceOf[TabPane]
          tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
          find("embedding-inspector-rows").asInstanceOf[TextArea].setText("nuizerosource|1")
          find("embedding-inspector-eval").asInstanceOf[TextField].clear()
          find("embedding-inspector-output").asInstanceOf[TextField].setText(mixedPair.toString)
          find("embedding-inspector-save").asInstanceOf[Button].fire()
        }
        awaitDone(); require(Files.isRegularFile(mixedPair))
        val paired = Await.result(services.client.raw("run_code", ujson.Obj("source_path" -> sourcePair.toString,
          "mixed_path" -> mixedPair.toString, "code" -> """
from safetensors.torch import load_file
original, mixed = load_file(params['source_path']), load_file(params['mixed_path'])
result = str(all(torch.equal(original[k], mixed[k]) for k in ('clip_l', 'clip_g')))
""")), 120.seconds)
        require(paired("result").str == "True", "Mixed zero-row encoder positions did not survive")
        services.catalog.refresh()
        phase("paired-generation-start")
        val pairedImage = generate("nuizeromixed")
        phase("paired-generation-complete")
        Files.writeString(run.resolve("paired-zero-image.txt"), pairedImage.path.toString)
        println(s"PASS SDXL paired zero rows: saved exactly and generated; ${pairedImage.path}")
      fx {
        val tabs = find("main-tabs").asInstanceOf[TabPane]
        tabs.getSelectionModel.select(tabs.getTabs.asScala.find(_.getId == "embedding-inspector-tab").get)
        find("embedding-inspector-rows").asInstanceOf[TextArea].setText("cat|1\ndog|1")
        find("embedding-inspector-eval").asInstanceOf[TextField].clear()
        find("embedding-inspector-concat").asInstanceOf[CheckBox].setSelected(true)
        find("embedding-inspector-overwrite").asInstanceOf[CheckBox].setSelected(true)
        find("embedding-inspector-output").asInstanceOf[TextField].setText(destination.toString)
        find("embedding-inspector-save").asInstanceOf[Button].fire()
      }
      awaitDone()
      val overwriteCheck = Await.result(services.client.raw("run_code", ujson.Obj("saved" -> destination.toString,
        "other" -> embeddings.resolve("nuizeromixed.safetensors").toString, "code" -> (readWeightCode + """
import json
from safetensors.torch import load_file
saved=load_file(params['saved'])
streams=[('clip_l', pipe.tokenizer, pipe.text_encoder)]
if engine.sdxl: streams.append(('clip_g', pipe.tokenizer_2, pipe.text_encoder_2))
checks={}
for key,tok,enc in streams:
    ids=[tok.convert_tokens_to_ids('nuiinspectorcheck'), tok.convert_tokens_to_ids('nuiinspectorcheck_1')]
    checks[key]=bool(torch.equal(inspector_test_weight(enc)[ids].cpu().float(), saved[key].float()))
    if engine.sdxl:
        other=load_file(params['other'])
        other_ids=[tok.convert_tokens_to_ids('nuizeromixed'), tok.convert_tokens_to_ids('nuizeromixed_1')]
        checks[key+'_other']=bool(torch.equal(inspector_test_weight(enc)[other_ids].cpu().float(), other[key].float()))
result=json.dumps(checks)
"""))), 120.seconds)
      val overwriteResult = ujson.read(overwriteCheck("result").str)
      require(overwriteResult.obj.values.forall(_.bool), s"Overwritten live token mismatch: $overwriteResult")
      phase("overwrite-generation-start")
      val overwritten = generate("nuiinspectorcheck")
      phase("overwrite-generation-complete")
      Files.writeString(run.resolve("overwritten-image.txt"), overwritten.path.toString)
      println(s"PASS ${loaded.kind} live overwrite: two-vector rows reloaded; generated ${overwritten.path}")
      Await.result(services.client.raw("run_code", ujson.Obj("code" -> (readWeightCode + """
engine._refresh_test_original = pipe.load_textual_inversion
engine._refresh_test_rows = []
streams = [(pipe.tokenizer, pipe.text_encoder)] + ([(pipe.tokenizer_2, pipe.text_encoder_2)] if engine.sdxl else [])
for tok, enc in streams:
    ids = [tok.convert_tokens_to_ids('nuiinspectorcheck'), tok.convert_tokens_to_ids('nuiinspectorcheck_1')]
    engine._refresh_test_rows.append(inspector_test_weight(enc)[ids].detach().clone())
def fail_after_second(*args, **kwargs):
    value = engine._refresh_test_original(*args, **kwargs)
    if not engine.sdxl or kwargs.get('tokenizer') is pipe.tokenizer_2: raise RuntimeError('injected encoder failure')
    return value
pipe.load_textual_inversion = fail_after_second
result = 'failure injection armed'
"""))), 120.seconds)
      val bad = ujson.Obj("operation" -> "save", "rows" -> ujson.Arr(ujson.Obj("name" -> "cat", "weight" -> 1)),
        "output" -> destination.toString, "overwrite" -> true,
        "embedding_paths" -> ujson.Arr.from(services.catalog.embeddings.map(e => ujson.Str(e.path))))
      val failed = scala.util.Try(Await.result(services.client.raw("embedding_inspector", bad), 120.seconds)).failed.toOption
      require(failed.exists(_.getMessage.contains("saved but live refresh failed")), s"Expected explicit failure, got $failed")
      val restored = Await.result(services.client.raw("run_code", ujson.Obj("code" -> (readWeightCode + """
checks=[]
streams = [(pipe.tokenizer, pipe.text_encoder)] + ([(pipe.tokenizer_2, pipe.text_encoder_2)] if engine.sdxl else [])
for index,(tok,enc) in enumerate(streams):
    ids=[tok.convert_tokens_to_ids('nuiinspectorcheck'),tok.convert_tokens_to_ids('nuiinspectorcheck_1')]
    checks.append(bool(torch.equal(inspector_test_weight(enc)[ids],engine._refresh_test_rows[index])))
result=str(all(checks))
"""))), 120.seconds)
      require(restored("result").str == "True", "Failed refresh did not restore encoder rows")
      fx { window.select(Mode.Txt2Img); window.vms(Mode.Txt2Img).prompt.set("cat"); window.statusBar.generateBtn.fire() }
      val blocked = finished.poll(120, TimeUnit.SECONDS)
      require(blocked != null && blocked.isLeft && blocked.swap.toOption.get.contains("reload the checkpoint"),
        s"Expected generation to fail closed after refresh failure: $blocked")
      waitFor("failed job idle")(services.queue.isIdle && fx(!window.progress.get.running))
      services.loadCheckpoint(checkpoint.toString, remember = false)
      phase("recovery-generation-start")
      val recovered = generate("cat")
      phase("recovery-generation-complete")
      Files.writeString(run.resolve("refresh-recovery-image.txt"), recovered.path.toString)
      println(s"PASS ${loaded.kind} refresh failure rollback, blocked generation and checkpoint recovery; ${recovered.path}")
      println(s"PASS mounted inspector ${loaded.kind}: inspect, authorized eval, save, catalog reload and original-token RGB parity; $run")
    finally
      if window != null then fx { window.dispose(); window.stage.close() }
      services.shutdown(); Platform.exit()
