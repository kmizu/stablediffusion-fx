package sdfx.ui

import javafx.application.Application

/** Command line options of the desktop app. Hidden dev flags: --screenshot, --exit-after, --autogen, --tab. */
final case class CliOptions(
    dryRun: Boolean = false,
    home: Option[String] = None,
    mcp: Boolean = false,
    screenshot: Option[String] = None,
    exitAfter: Option[Double] = None,
    autogen: Boolean = false,
    demoMask: Boolean = false,
    tab: Option[String] = None,
    configPresetsDir: Option[String] = None
)

object CliOptions:
  def parse(args: Seq[String]): CliOptions =
    @annotation.tailrec
    def go(rest: List[String], o: CliOptions): CliOptions = rest match
      case "--dry-run" :: t            => go(t, o.copy(dryRun = true))
      case "--home" :: v :: t          => go(t, o.copy(home = Some(v)))
      case "--mcp" :: t                => go(t, o.copy(mcp = true))
      case "--screenshot" :: v :: t    => go(t, o.copy(screenshot = Some(v)))
      case "--exit-after" :: v :: t    => go(t, o.copy(exitAfter = v.toDoubleOption))
      case "--autogen" :: t            => go(t, o.copy(autogen = true))
      case "--demo-mask" :: t          => go(t, o.copy(demoMask = true))
      case "--tab" :: v :: t           => go(t, o.copy(tab = Some(v)))
      case "--configpresets-dir" :: v :: t => go(t, o.copy(configPresetsDir = Some(v)))
      case _ :: t                      => go(t, o)
      case Nil                         => o
    go(args.toList, CliOptions())

/** Entry point. Not an `Application` subclass so the fat jar starts without the JavaFX module path. */
object Launcher:
  def main(args: Array[String]): Unit =
    val opts = CliOptions.parse(args.toSeq)
    opts.home.foreach(h => System.setProperty("sdfx.home", h))
    if opts.mcp then runMcp(args)
    else Application.launch(classOf[SdfxApp], args*)

  private def runMcp(args: Array[String]): Unit =
    try
      val cls = Class.forName("sdfx.mcp.McpMain")
      cls.getMethod("main", classOf[Array[String]]).invoke(null, args)
    catch
      case _: ClassNotFoundException =>
        System.err.println("[sdfx] MCP server is not available in this build (sdfx.mcp.McpMain missing)")
        sys.exit(2)
      case e: java.lang.reflect.InvocationTargetException =>
        val c = Option(e.getCause).getOrElse(e)
        System.err.println(s"[sdfx] MCP server failed: ${c.getMessage}")
        throw c
