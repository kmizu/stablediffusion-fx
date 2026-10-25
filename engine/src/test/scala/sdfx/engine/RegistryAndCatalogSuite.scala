package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.plugins.api.*
import sdfx.core.protocol.*
import sdfx.engine.testplugins.{MarkerAlwaysOn, RepeatScript}

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.util.jar.{JarEntry, JarOutputStream}
import scala.util.Using

class RegistryAndCatalogSuite extends munit.FunSuite:
  import FakeEngine.*

  /** Hides the test plugin classes so they can only come from the jars (child-first for that package). */
  private final class HidingLoader(delegate: ClassLoader) extends ClassLoader(null: ClassLoader):
    override def loadClass(name: String, resolve: Boolean): Class[?] =
      if name.startsWith("sdfx.engine.testplugins.") then throw new ClassNotFoundException(name)
      else delegate.loadClass(name)
    override def findResource(name: String): java.net.URL = delegate.getResource(name)
    override def findResources(name: String): java.util.Enumeration[java.net.URL] = delegate.getResources(name)

  private def writeJar(path: java.nio.file.Path, services: Seq[String], classes: Seq[Class[?]]): Unit =
    Using.resource(new JarOutputStream(Files.newOutputStream(path))) { jar =>
      jar.putNextEntry(new JarEntry(s"META-INF/services/${classOf[Plugin].getName}"))
      jar.write(services.mkString("# test\n", "\n", "\n").getBytes(StandardCharsets.UTF_8))
      jar.closeEntry()
      classes.foreach { c =>
        val res = c.getName.replace('.', '/') + ".class"
        jar.putNextEntry(new JarEntry(res))
        jar.write(Using.resource(getClass.getClassLoader.getResourceAsStream(res))(_.readAllBytes()))
        jar.closeEntry()
      }
    }

  test("PluginRegistry loads plugin classes from external jars via ServiceLoader") {
    val home = tempHome()
    try
      Files.createDirectories(home.pluginsDir)
      writeJar(home.pluginsDir.resolve("a-ext.jar"),
        Seq("sdfx.engine.testplugins.ExtPlugin", "sdfx.engine.testplugins.DoesNotExist", "sdfx.engine.testplugins.DuplicateIdPlugin"),
        Seq(classOf[testplugins.ExtPlugin], classOf[testplugins.DuplicateIdPlugin]))
      var settings = Settings()
      val hiding = new HidingLoader(getClass.getClassLoader)
      val reg = new PluginRegistry(Seq(new RepeatScript), home, () => settings, hiding)
      assertEquals(reg.all.map(_.id), Seq("repeat", "ext-test"))
      val ext = reg.get("ext-test").get
      assert(ext.getClass.getClassLoader.isInstanceOf[java.net.URLClassLoader], ext.getClass.getClassLoader.toString)
      assert(ext.getClass ne classOf[testplugins.ExtPlugin], "class must come from the jar, not the test classpath")
      assertEquals(reg.get("repeat").map(_.name), Some("Repeat"))
      assert(reg.loadErrors.exists(_.contains("DoesNotExist")), reg.loadErrors.toString)
      assertEquals(reg.xyzAxes.map(_.name), Seq("Repeat axis", "Ext axis"))
      assertEquals(reg.scripts(Tab.Txt2Img).map(_.id), Seq("repeat"))
      assertEquals(reg.alwaysOn(Tab.Txt2Img).map(_.id), Seq("ext-test"))

      // reload picks up a jar added later
      writeJar(home.pluginsDir.resolve("b-ext.jar"), Seq("sdfx.engine.testplugins.MarkerAlwaysOn"), Seq(classOf[MarkerAlwaysOn]))
      reg.reload()
      assertEquals(reg.externalPlugins.map(_.id).toSet, Set("ext-test", "marker"))
      assertEquals(reg.alwaysOn(Tab.Txt2Img).map(_.id).toSet, Set("ext-test", "marker"))

      settings = settings.copy(disabledPlugins = Seq("ext-test"), pluginValues = Map("repeat" -> Map("n" -> "7")))
      assert(!reg.isEnabled("ext-test"))
      assertEquals(reg.enabled.map(_.id).toSet, Set("repeat", "marker"))
      assertEquals(reg.xyzAxes.map(_.name), Seq("Repeat axis"))
      assertEquals(reg.values(reg.get("repeat").get).int("n"), 7)
      assertEquals(reg.values(reg.get("repeat").get).bool("grid"), false)
      assertEquals(reg.unsupported(Nil), Nil)
      reg.close()
    finally deleteTree(home.root)
  }

  test("a broken jar never breaks registry construction") {
    val home = tempHome()
    try
      Files.createDirectories(home.pluginsDir)
      Files.write(home.pluginsDir.resolve("garbage.jar"), Array[Byte](1, 2, 3, 4))
      writeJar(home.pluginsDir.resolve("only-bad.jar"), Seq("not a class name!", "java.lang.String"), Nil)
      val reg = new PluginRegistry(Seq(new RepeatScript), home, () => Settings(), new HidingLoader(getClass.getClassLoader))
      assertEquals(reg.all.map(_.id), Seq("repeat"))
      assert(reg.loadErrors.nonEmpty)
      reg.close()
    finally deleteTree(home.root)
  }

  test("PluginRegistry without plugins-ext dir") {
    val home = tempHome()
    try
      val reg = new PluginRegistry(Seq(new RepeatScript), home, () => Settings())
      assertEquals(reg.all.map(_.id), Seq("repeat"))
      assertEquals(reg.loadErrors, Nil)
    finally deleteTree(home.root)
  }

  private def catalog() =
    val home = tempHome()
    val c = new ModelCatalog(null, home, () => Settings())
    c.set(CatalogSnapshot(
      checkpoints = Seq(ModelEntry("foo", "/m/foo.safetensors", "sd15"), ModelEntry("sdxl/Bar_XL", "/m/sdxl/Bar_XL.safetensors", "sdxl")),
      vaes = Seq(FileEntry("vae-ft-mse", "/v/vae-ft-mse.safetensors")),
      loras = Seq(FileEntry("My Lora", "/l/My Lora.safetensors"), FileEntry("sub/Detail_v2", "/l/sub/Detail_v2.safetensors")),
      controlnets = Seq(FileEntry("control_canny", "/c/control_canny.pth")),
      upscalers = Seq(FileEntry("4x-Ultra", "/u/4x-Ultra.pth")),
      samplers = SamplerList(Seq(SamplerInfo("Euler a"), SamplerInfo("Bad", supported = false)), Seq("Automatic"), Seq("Latent", "Lanczos"), Seq("canny"))
    ))
    (home, c)

  test("ModelCatalog name resolution") {
    val (home, c) = catalog()
    try
      assertEquals(c.findLora("My Lora").map(_.path), Some("/l/My Lora.safetensors"))
      assertEquals(c.findLora("my lora").map(_.path), Some("/l/My Lora.safetensors"))
      assertEquals(c.findLora("Detail_v2").map(_.path), Some("/l/sub/Detail_v2.safetensors"))
      assertEquals(c.findLora("detail_V2").map(_.path), Some("/l/sub/Detail_v2.safetensors"))
      assertEquals(c.findLora("sub\\Detail_v2").map(_.path), Some("/l/sub/Detail_v2.safetensors"))
      assertEquals(c.findLora("/l/My Lora.safetensors").map(_.name), Some("My Lora"))
      assertEquals(c.findLora("nothing"), None)
      assertEquals(c.loraPaths(Seq("My Lora", "nothing")), Map("My Lora" -> "/l/My Lora.safetensors"))

      assertEquals(c.findCheckpoint("Bar_XL").map(_.name), Some("sdxl/Bar_XL"))
      assertEquals(c.findCheckpoint("Bar_XL.safetensors").map(_.name), Some("sdxl/Bar_XL"))
      assertEquals(c.findCheckpoint("foo [abcdef1234]").map(_.name), Some("foo"))
      assertEquals(c.findCheckpoint("abcdef12"), None)
      c.rememberHash(LoadedModel("foo", "/m/foo.safetensors", "sd15", "abcdef1234567890"))
      assertEquals(c.findCheckpoint("abcdef12").map(_.name), Some("foo"))
      assertEquals(c.findCheckpoint("renamed [abcdef1234]").map(_.name), Some("foo"))
      assertEquals(c.findVae("vae-ft-mse.safetensors").map(_.name), Some("vae-ft-mse"))
      assertEquals(c.findControlNet("control_canny [a1b2c3d4]").map(_.name), Some("control_canny"))
      assertEquals(c.findUpscaler("4x-ultra").map(_.path), Some("/u/4x-Ultra.pth"))
    finally deleteTree(home.root)
  }

  test("ModelCatalog choices") {
    val (home, c) = catalog()
    try
      assertEquals(c.choices(ChoiceSource.Samplers), Seq("Euler a"))
      assertEquals(c.choices(ChoiceSource.Checkpoints), Seq("foo", "sdxl/Bar_XL"))
      assertEquals(c.choices(ChoiceSource.Upscalers), Seq("Lanczos", "4x-Ultra"))
      assertEquals(c.choices(ChoiceSource.HiresUpscalers).take(1), Seq("Latent"))
      assert(c.choices(ChoiceSource.HiresUpscalers).contains("4x-Ultra"))
      assertEquals(c.choices(ChoiceSource.Preprocessors), Seq("canny"))
      assertEquals(c.choices(ChoiceSource.ControlNets), Seq("control_canny"))
    finally deleteTree(home.root)
  }

  test("ModelCatalog refresh keeps previous lists when one listing fails") {
    val home = tempHome()
    val sup = new EngineSupervisor(() => launch("--fail-list", "lora"))
    try
      sup.start()
      val c = new ModelCatalog(new EngineClient(sup), home, () => Settings())
      val oldLora = FileEntry("Old", "/l/Old.safetensors")
      c.set(CatalogSnapshot(loras = Seq(oldLora)))
      val snap = c.refresh()
      assertEquals(snap.loras, Seq(oldLora))
      assertEquals(c.findLora("Old").map(_.path), Some(oldLora.path))
      assertEquals(snap.checkpoints.map(_.name).toSet, Set("foo", "sdxl/Bar_XL"))
      assert(c.lastErrors.exists(_.startsWith("lora")), c.lastErrors.toString)
      // checkpoint listing failure: nothing published, future fails
      sup.stop()
      val before = c.snapshot
      intercept[Exception](c.refresh())
      assertEquals(c.snapshot, before)
      assert(c.lastErrors.exists(_.startsWith("checkpoints")), c.lastErrors.toString)
    finally
      sup.close()
      deleteTree(home.root)
  }

  test("PythonEnv bootstrap commands, probe caching and configured interpreter") {
    val home = tempHome()
    try
      Files.createDirectories(home.pythonDir)
      Files.writeString(home.pythonDir.resolve(PythonEnv.TorchRequirements), "torch\n")
      Files.writeString(home.pythonDir.resolve(PythonEnv.Requirements), "numpy\n")
      val cmds = PythonEnv.bootstrapCommands(home)
      assertEquals(cmds.size, 3)
      assert(cmds(1).containsSlice(Seq("-r", PythonEnv.TorchRequirements, "--index-url", "https://download.pytorch.org/whl/cu128")), cmds(1).toString)
      assert(cmds(2).containsSlice(Seq("-r", PythonEnv.Requirements)), cmds(2).toString)
      assert(cmds.forall(_.contains("--progress-bar")))
      assert(!PythonEnv.works(Seq("definitely-not-a-python-sdfx")))
      assert(!PythonEnv.works(Seq("definitely-not-a-python-sdfx")))
      val bad = Settings(pythonExe = Some("definitely-not-a-python-sdfx"))
      assertEquals(PythonEnv.locate(home, bad), None)
      assert(PythonEnv.configuredProblem(bad).isDefined)
      val e = intercept[IllegalStateException](PythonEnv.engineCommand(home, bad, dryRun = false))
      assert(e.getMessage.contains("definitely-not-a-python-sdfx"))
      assertEquals(PythonEnv.configuredProblem(Settings()), None)
    finally deleteTree(home.root)
  }

  test("PythonEnv command and bootstrap stamp") {
    val home = tempHome()
    try
      val cmd = PythonEnv.engineCommand(home, Settings(), dryRun = true)
      assertEquals(cmd.takeRight(3), Seq("-m", "sdfx", "--dry-run"))
      assertEquals(PythonEnv.engineLaunch(home, Settings(), dryRun = false).workDir, home.pythonDir)
      assert(PythonEnv.needsBootstrap(home))
      val s1 = PythonEnv.requirementsStamp(home)
      Files.createDirectories(home.pythonDir)
      Files.writeString(home.pythonDir.resolve("requirements.txt"), "numpy\n")
      assertNotEquals(PythonEnv.requirementsStamp(home), s1)
      val custom = PythonEnv.locate(home, Settings(pythonExe = Some(FakeEngine.python.head)))
      assert(custom.isDefined)
    finally deleteTree(home.root)
  }
