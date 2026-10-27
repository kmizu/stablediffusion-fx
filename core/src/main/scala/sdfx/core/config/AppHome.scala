package sdfx.core.config

import java.nio.file.{Files, Path, Paths}

/** The application root (`stablediffusion-nui/`). Everything the app owns lives below it. */
final case class AppHome(root: Path):
  def resolve(p: String): Path =
    val path = Paths.get(p)
    if path.isAbsolute then path else root.resolve(path).normalize()
  def pythonDir: Path = root.resolve("python")
  def configDir: Path = root.resolve("config")
  def settingsFile: Path = configDir.resolve("settings.json")
  def stylesFile: Path = configDir.resolve("styles.csv")
  def pluginsDir: Path = root.resolve("plugins-ext")
  def outputs(settings: Settings): Path = resolve(settings.outputDir)
  def tmpDir(settings: Settings): Path =
    val d = outputs(settings).resolve(".tmp")
    Files.createDirectories(d)
    d

object AppHome:
  /** Keep legacy settings keys and folder support when discovering the renamed application. */
  def detect(): AppHome =
    def valid(p: Path) = Files.isDirectory(p.resolve("python").resolve("sdfx"))
    val candidates = Seq(
      Option(System.getProperty("sdfx.home")).map(Paths.get(_)),
      Option(System.getenv("SDFX_HOME")).map(Paths.get(_)),
      Some(Paths.get("").toAbsolutePath),
      Some(Paths.get("").toAbsolutePath.resolve("stablediffusion-nui")),
      Some(Paths.get("").toAbsolutePath.resolve("stablediffusion-fx")),
      jarDir.map(_.getParent),
      jarDir
    ).flatten.map(_.toAbsolutePath.normalize())
    AppHome(candidates.find(valid).getOrElse(Paths.get("").toAbsolutePath))

  private def jarDir: Option[Path] =
    try
      val src = classOf[AppHome].getProtectionDomain.getCodeSource
      Option(src).map(s => Paths.get(s.getLocation.toURI)).map(p => if Files.isDirectory(p) then p else p.getParent)
    catch case _: Exception => None
