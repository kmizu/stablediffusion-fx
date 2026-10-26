package sdfx.plugins

import sdfx.core.image.GridNumbers
import sdfx.core.model.GenerationParams
import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Native grid display settings; no Python sampling hook is needed. */
class GridNumbersPlugin extends Plugin:
  def id = "grid_numbers"
  def name = "Grid image numbers"
  def kind = PluginKind.AlwaysOn
  override def description = "Identify grid tiles by their saved image numbers and X/Y/Z values."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Check("image_numbers", "Show saved image numbers", true),
    Note("numbers_help", "The number is the leading part of each saved filename, including zeros. It appears at the bottom left of its grid tile. Original images stay unchanged."),
    Check("xyz_info", "Show X/Y/Z values"),
    Note("axes_help", "X appears at the top left, Y at the top right and Z at the bottom right. Every native axis is supported, including plugin axes. Missing axes are omitted."),
    Note("layout_help", "White text uses translucent black panels. Long text shrinks from 24 to 9 px, then tries the value alone; text that still cannot fit is omitted."),
    Note("batch_help", "Enabled batches with more than one image also save a grid. Existing script grids keep their normal layout, legends and margins. No extra grid is made for a single image.")
  )
  override def enabledBy = Map("image_numbers" -> "enabled", "xyz_info" -> "enabled")
  override def process(p: GenerationParams, v: Values): GenerationParams =
    if !isActive(v) then p
    else p.copy(extra = p.extra ++ Map(
      GridNumbers.NumbersKey -> (if v.bool("image_numbers", true) then "True" else "False"),
      GridNumbers.XyzKey -> (if v.bool("xyz_info") then "True" else "False")))
