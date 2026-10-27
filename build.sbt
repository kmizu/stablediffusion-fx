ThisBuild / scalaVersion := "3.3.6"
ThisBuild / version := "0.03"
ThisBuild / organization := "sdfx"
ThisBuild / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-encoding", "utf8")

val upickleV = "4.1.0"
val munitV = "1.1.1"
val fxV = "21.0.6"

lazy val fxOs: String = {
  val n = System.getProperty("os.name").toLowerCase
  val arch = System.getProperty("os.arch")
  if (n.contains("win")) "win"
  else if (n.contains("mac")) { if (arch == "aarch64") "mac-aarch64" else "mac" }
  else "linux"
}
lazy val fxDeps = Seq("base", "graphics", "controls", "swing").map(m => "org.openjfx" % s"javafx-$m" % fxV classifier fxOs)

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "com.lihaoyi" %% "upickle" % upickleV,
    "org.scalameta" %% "munit" % munitV % Test
  ),
  Test / fork := true,
  Test / baseDirectory := (ThisBuild / baseDirectory).value,
  Test / parallelExecution := false,
  Compile / run / fork := true,
  Compile / run / baseDirectory := (ThisBuild / baseDirectory).value,
  run / connectInput := true
)

lazy val core = project.settings(commonSettings, name := "sdfx-core", Test / javaOptions ++= Seq("-Djava.awt.headless=true", "-Dfile.encoding=UTF-8"))

lazy val engine = project.dependsOn(core).settings(commonSettings, name := "sdfx-engine")

lazy val plugins = project.dependsOn(core, engine).settings(commonSettings, name := "sdfx-plugins")

lazy val mcp = project.dependsOn(core, engine, plugins).settings(commonSettings, name := "sdfx-mcp")

lazy val ui = project
  .dependsOn(core, engine, plugins, mcp)
  .settings(
    commonSettings,
    name := "sdfx-ui",
    libraryDependencies ++= fxDeps,
    Compile / mainClass := Some("sdfx.ui.Launcher"),
    run / javaOptions ++= Seq("-Dprism.allowhidpi=true", "-Xmx2g", "-Dfile.encoding=UTF-8"),
    Test / javaOptions ++= Seq("-Djava.awt.headless=true"),
    assembly / mainClass := Some("sdfx.ui.Launcher"),
    assembly / assemblyJarName := "stablediffusion-nui.jar",
    assembly / assemblyOutputPath := (ThisBuild / baseDirectory).value / "target" / "stablediffusion-nui.jar",
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", "services", _*) => MergeStrategy.concat
      case PathList("META-INF", _*)              => MergeStrategy.discard
      case x if x.endsWith("module-info.class")  => MergeStrategy.discard
      case _                                     => MergeStrategy.first
    }
  )

lazy val root = (project in file("."))
  .aggregate(core, engine, plugins, mcp, ui)
  .settings(name := "stablediffusion-nui", publish / skip := true)
