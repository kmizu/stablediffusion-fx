package sdfx.engine

import sdfx.core.config.{AppHome, Settings}
import sdfx.core.plugins.api.*

import java.net.{URL, URLClassLoader}
import java.nio.file.{Files, Path}
import java.util.ServiceLoader
import scala.jdk.CollectionConverters.*
import scala.util.Using
import scala.util.control.NonFatal

/** Built-in plugins plus external ones from `home/plugins-ext/ *.jar` (java.util.ServiceLoader of [[Plugin]]).
  * Built-ins win on duplicate ids. Enabled state comes from `Settings.disabledPlugins`.
  */
final class PluginRegistry(
    builtins: Seq[Plugin],
    home: AppHome,
    settings: () => Settings,
    parent: ClassLoader = classOf[PluginRegistry].getClassLoader
):
  @volatile private var external: Seq[Plugin] = Nil
  @volatile private var errors: Seq[String] = Nil
  @volatile private var loader: Option[URLClassLoader] = None
  @volatile private var retired: List[URLClassLoader] = Nil

  reload()

  def jarFiles: Seq[Path] =
    val dir = home.pluginsDir
    if !Files.isDirectory(dir) then Nil
    else Using.resource(Files.list(dir))(_.iterator().asScala.filter(p => p.getFileName.toString.toLowerCase.endsWith(".jar")).toSeq.sortBy(_.toString))

  /** Rescans the external plugin jars. Never throws: bad jars / provider entries end up in [[loadErrors]].
    * Previous class loaders are kept open until [[close]] because plugin instances from them may still be running.
    */
  def reload(): Unit = synchronized {
    val errs = Seq.newBuilder[String]
    val jars =
      try jarFiles
      catch
        case NonFatal(e) =>
          errs += s"cannot list ${home.pluginsDir}: ${e.getMessage}"
          Nil
    val found =
      if jars.isEmpty then Nil
      else
        try
          val l = new URLClassLoader(jars.map(_.toUri.toURL).toArray[URL], parent)
          loader.foreach(old => retired = old :: retired)
          loader = Some(l)
          val it = ServiceLoader.load(classOf[Plugin], l).iterator()
          val b = Seq.newBuilder[Plugin]
          var more = true
          var guard = 0
          while more && guard < 10000 do
            guard += 1
            try
              if it.hasNext then b += it.next()
              else more = false
            catch
              case e: java.util.ServiceConfigurationError => errs += s"plugin provider: ${e.getMessage}"
              case e: Throwable if !e.isInstanceOf[VirtualMachineError] => errs += s"plugin load failed: $e"
          b.result()
        catch
          case e: Throwable if !e.isInstanceOf[VirtualMachineError] =>
            errs += s"plugin scan failed: $e"
            Nil
    def safeId(p: Plugin): Option[String] =
      try Option(p.id).filter(_.nonEmpty)
      catch
        case e: Throwable if !e.isInstanceOf[VirtualMachineError] =>
          errs += s"${p.getClass.getName}: id failed: $e"
          None
    val builtinIds = builtins.flatMap(safeId).toSet
    val withIds = found.flatMap(p => safeId(p).map(_ -> p))
    external = withIds.filterNot((id, _) => builtinIds.contains(id)).groupBy(_._1).values.map(_.head._2).toSeq
      .sortBy(p =>
        try p.name
        catch case NonFatal(_) => ""
      )
    errors = errs.result()
  }

  def loadErrors: Seq[String] = errors
  def all: Seq[Plugin] = builtins ++ external
  def externalPlugins: Seq[Plugin] = external
  def isEnabled(id: String): Boolean = !settings().disabledPlugins.contains(id)
  def enabled: Seq[Plugin] = all.filter(p => isEnabled(p.id))
  def get(id: String): Option[Plugin] = all.find(_.id == id)
  def scripts(tab: Tab): Seq[Plugin] = enabled.filter(p => p.kind == PluginKind.Script && p.tabs.contains(tab))
  def alwaysOn(tab: Tab): Seq[Plugin] = enabled.filter(p => p.kind == PluginKind.AlwaysOn && p.tabs.contains(tab))
  def xyzAxes: Seq[XyzAxis] = enabled.flatMap(p =>
    try p.xyzAxes
    catch case NonFatal(_) => Nil
  )

  /** Values for a plugin: saved settings over control defaults. */
  def values(p: Plugin): Values =
    Values(Values.defaults(p.controls).map ++ settings().pluginValues.getOrElse(p.id, Map.empty))

  /** Plugins whose required engine features are missing from the hello. */
  def unsupported(features: Seq[String]): Seq[(Plugin, Seq[String])] =
    all.map(p => p -> p.requiresEngineFeatures.filterNot(features.contains)).filter(_._2.nonEmpty)

  def close(): Unit = (loader.toList ++ retired).foreach(l =>
    try l.close()
    catch case NonFatal(_) => ()
  )
