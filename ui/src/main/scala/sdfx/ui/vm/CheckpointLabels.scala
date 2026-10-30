package sdfx.ui.vm

import sdfx.core.protocol.ModelEntry

object CheckpointLabels:
  val modes = Seq("Default", "With paths (Hires fix)", "Without paths (Hires fix)", "With paths (Global)", "Without paths (Global)")
  private def basename(path: String): String = path.replace('\\', '/').split('/').lastOption.getOrElse(path)
  def label(entry: ModelEntry, entries: Seq[ModelEntry], mode: String, hires: Boolean = false): String =
    val applies = mode.endsWith("(Global)") || (hires && mode.endsWith("(Hires fix)"))
    if !applies then entry.name
    else if mode.startsWith("With paths") then entry.path
    else if mode.startsWith("Without paths") then
      val name = basename(entry.path)
      val collisions = entries.count(e => basename(e.path).equalsIgnoreCase(name))
      if collisions > 1 then s"$name [${entry.name}]" else name
    else entry.name
