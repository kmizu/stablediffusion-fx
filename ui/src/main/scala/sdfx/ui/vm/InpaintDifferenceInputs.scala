package sdfx.ui.vm

import javafx.beans.property.{SimpleStringProperty, StringProperty}
import sdfx.core.plugins.api.Values

/** Keep hand-painted and difference inputs independent while the active VM path follows the mode. */
final class InpaintDifferenceInputs(actual: StringProperty, enabled: StringProperty,
    savedPainted: StringProperty = new SimpleStringProperty(""),
    savedAltered: StringProperty = new SimpleStringProperty(""),
    validSaved: String => Boolean = _ => true):
  private def active = Values(Map("enabled" -> Option(enabled.get).getOrElse("false"))).bool("enabled")
  val painted = savedPainted
  val altered = savedAltered
  if Option(painted.get).exists(s => s.nonEmpty && !validSaved(s)) then painted.set("")
  if Option(altered.get).exists(s => s.nonEmpty && !validSaved(s)) then altered.set("")
  if active && Option(altered.get).forall(_.isEmpty) then altered.set(actual.get)
  if !active && Option(painted.get).forall(_.isEmpty) then painted.set(actual.get)
  actual.set(if active then altered.get else painted.get)
  private var syncing = false
  private def sync(f: => Unit): Unit = if !syncing then
    syncing = true
    try f finally syncing = false
  actual.addListener((_, _, value) => sync {
    if active then altered.set(value) else painted.set(value)
  })
  painted.addListener((_, _, value) => sync { if !active then actual.set(value) })
  altered.addListener((_, _, value) => sync { if active then actual.set(value) })
  enabled.addListener((_, _, _) => sync { actual.set(if active then altered.get else painted.get) })
