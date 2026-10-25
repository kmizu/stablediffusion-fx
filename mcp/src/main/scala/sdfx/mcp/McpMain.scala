package sdfx.mcp

import sdfx.core.config.{AppHome, Settings}
import sdfx.engine.AppServices
import sdfx.plugins.StandardPlugins

import java.io.{BufferedOutputStream, FileDescriptor, FileOutputStream, PrintStream}
import java.nio.charset.StandardCharsets
import java.nio.file.{Path, Paths}
import scala.util.control.NonFatal

/** Headless MCP server: boots its own engine and speaks MCP over stdio (default) or HTTP (`--http [port]`).
  *
  * Options: `--home DIR`, `--dry-run`, `--http [PORT]`, `--token TOKEN` (or env `SDFX_MCP_TOKEN`),
  * `--allow-dir DIR` (repeatable; also env `SDFX_MCP_ALLOWED_DIRS`), `--drain-seconds N` (how long in-flight calls
  * may finish after stdin closes, default 2). Unknown options (e.g. `--mcp` from the launcher) are ignored.
  * stdout carries only MCP messages; every log line goes to stderr.
  */
object McpMain:
  final case class Options(
      home: Option[String] = None,
      dryRun: Boolean = false,
      http: Option[Option[Int]] = None,
      token: Option[String] = None,
      allowDirs: Seq[String] = Nil,
      drainSeconds: Int = 2
  )

  def parseArgs(args: Seq[String]): Options =
    @annotation.tailrec
    def go(rest: List[String], o: Options): Options = rest match
      case "--home" :: dir :: tail                              => go(tail, o.copy(home = Some(dir)))
      case "--dry-run" :: tail                                  => go(tail, o.copy(dryRun = true))
      case "--token" :: t :: tail                               => go(tail, o.copy(token = Some(t)))
      case "--allow-dir" :: d :: tail                           => go(tail, o.copy(allowDirs = o.allowDirs :+ d))
      case "--drain-seconds" :: n :: tail if n.toIntOption.isDefined => go(tail, o.copy(drainSeconds = n.toInt.max(0)))
      case "--http" :: p :: tail if p.toIntOption.isDefined    => go(tail, o.copy(http = Some(p.toIntOption)))
      case "--http" :: tail                                     => go(tail, o.copy(http = Some(None)))
      case _ :: tail                                            => go(tail, o)
      case Nil                                                  => o
    go(args.toList, Options())

  def main(args: Array[String]): Unit =
    // Capture the real stdout for the protocol, then route everything else (println, libraries) to stderr.
    val protocolOut = new BufferedOutputStream(new FileOutputStream(FileDescriptor.out), 1 << 16)
    System.setOut(new PrintStream(new FileOutputStream(FileDescriptor.err), true, StandardCharsets.UTF_8))
    System.setProperty("java.awt.headless", "true")
    val log: String => Unit = s => System.err.println(s)
    val opts = parseArgs(args.toSeq)
    val home = opts.home.map(h => AppHome(Paths.get(h).toAbsolutePath.normalize())).getOrElse(AppHome.detect())
    val settings = Settings.load(home.settingsFile)
    val allowed: Seq[Path] = PathPolicy.fromEnv() ++ opts.allowDirs.filterNot(PathPolicy.isNetworkPath).map(d => Paths.get(d).toAbsolutePath.normalize())
    log(s"[mcp] home=${home.root} dryRun=${opts.dryRun}${if allowed.nonEmpty then s" allowedDirs=${allowed.mkString(";")}" else ""}")

    lazy val services: AppServices = new AppServices(home, settings, opts.dryRun, StandardPlugins.all(() => services.registry.xyzAxes))
    services.addLogListener(log)
    val started = services.startAsync()
    started.failed.foreach(e => log(s"[mcp] engine start failed: ${e.getMessage}"))(using scala.concurrent.ExecutionContext.global)

    val tools = new McpTools(services, McpTools.awaitStart(started), allowedDirs = allowed).all
    opts.http match
      case Some(portOpt) =>
        val port = portOpt.getOrElse(settings.mcpHttpPort)
        val token = opts.token.orElse(Option(System.getenv("SDFX_MCP_TOKEN"))).orElse(settings.mcpToken).filter(_.trim.nonEmpty)
        val t =
          try new HttpTransport(() => new McpServer(tools), port, token, log)
          catch
            case NonFatal(e) =>
              log(s"[mcp] cannot listen on 127.0.0.1:$port: $e")
              services.shutdown()
              System.exit(1)
              throw e
        log(s"[mcp] HTTP server listening on ${t.url}${if token.isEmpty then " (no token: any local process can use it)" else ""}")
        // runs until the JVM is terminated; the hook cleans up
        Runtime.getRuntime.addShutdownHook(new Thread(() => {
          t.close()
          try services.shutdown()
          catch case NonFatal(_) => ()
        }))
        while true do Thread.sleep(Long.MaxValue)
      case None =>
        val code =
          try
            new StdioTransport(System.in, protocolOut, new McpServer(tools), log).run(opts.drainSeconds * 1000L)
            log("[mcp] stdin closed, shutting down")
            0
          catch
            case NonFatal(e) =>
              log(s"[mcp] fatal: $e")
              1
          finally
            try services.shutdown()
            catch case NonFatal(_) => ()
        System.exit(code)
