package sdfx.ui.vm

import javafx.beans.property.SimpleObjectProperty
import javafx.collections.{FXCollections, ObservableList}
import sdfx.core.plugins.api.ChoiceSource
import sdfx.engine.CatalogSnapshot

/** Observable lists derived from the engine model catalog. `update` must run on the FX thread once views exist. */
final class CatalogVM:
  val snapshot = new SimpleObjectProperty[CatalogSnapshot](CatalogSnapshot())
  private val lists = ChoiceSource.values.map(s => s -> FXCollections.observableArrayList[String]()).toMap
  val samplers: ObservableList[String] = FXCollections.observableArrayList(CatalogVM.FallbackSamplers*)
  val schedulers: ObservableList[String] = FXCollections.observableArrayList(CatalogVM.FallbackSchedulers*)
  val hiresUpscalers: ObservableList[String] = FXCollections.observableArrayList("Latent")
  val upscalers: ObservableList[String] = FXCollections.observableArrayList[String]()
  val vaes: ObservableList[String] = FXCollections.observableArrayList("Automatic", "None")

  def choices(src: ChoiceSource): ObservableList[String] = lists(src)

  def update(snap: CatalogSnapshot, resolve: ChoiceSource => Seq[String]): Unit =
    snapshot.set(snap)
    ChoiceSource.values.foreach(s => replace(lists(s), resolve(s)))
    replace(samplers, nonEmpty(resolve(ChoiceSource.Samplers), CatalogVM.FallbackSamplers))
    replace(schedulers, nonEmpty(withAutomatic(resolve(ChoiceSource.Schedulers)), CatalogVM.FallbackSchedulers))
    replace(hiresUpscalers, nonEmpty(resolve(ChoiceSource.HiresUpscalers), Seq("Latent")))
    replace(upscalers, resolve(ChoiceSource.Upscalers))
    replace(vaes, Seq("Automatic", "None") ++ snap.vaes.map(_.name))

  private def withAutomatic(xs: Seq[String]) = if xs.isEmpty || xs.contains("Automatic") then xs else "Automatic" +: xs
  private def nonEmpty(xs: Seq[String], fallback: Seq[String]) = if xs.isEmpty then fallback else xs
  private def replace(l: ObservableList[String], xs: Seq[String]): Unit =
    if !l.toArray.toSeq.equals(xs) then l.setAll(xs*)

object CatalogVM:
  val FallbackSamplers: Seq[String] = Seq("DPM++ 2M", "DPM++ SDE", "DPM++ 2M SDE", "Euler a", "Euler", "DDIM", "UniPC")
  val FallbackSchedulers: Seq[String] = Seq("Automatic", "Karras", "Exponential", "Uniform", "SGM Uniform")
