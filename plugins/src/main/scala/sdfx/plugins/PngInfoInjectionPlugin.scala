package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

class PngInfoInjectionPlugin extends HookPlugin:
  def id = "pnginfo_injection"
  def hook = id
  def name = "PNG resource hashes"
  override def description = "Record local resource hashes in PNG generation metadata for resource identification."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Note("scope", "Records hashes of confirmed loaded resources. Missing files and unsupported resource directories are reported instead of guessed. First use may read model files to calculate SHA-256.")
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj()
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("PNG resource hashes" -> "True")
