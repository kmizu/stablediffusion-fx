package sdfx.ui.vm

import javafx.beans.property.{SimpleIntegerProperty, SimpleStringProperty}
import javafx.collections.transformation.FilteredList
import javafx.collections.{FXCollections, ObservableList}
import sdfx.core.gallery.GalleryItem

import java.nio.file.Path

/** Gallery items (newest first) with a prompt / file name filter. */
final class GalleryVM:
  val items: ObservableList[GalleryItem] = FXCollections.observableArrayList()
  val filtered: FilteredList[GalleryItem] = new FilteredList(items, _ => true)
  val query = new SimpleStringProperty("")
  /** Empty selection means historical images; ZIP is only enabled for a concrete job. */
  val generations: ObservableList[String] = FXCollections.observableArrayList("")
  val selectedGeneration = new SimpleStringProperty("")
  val selectedGenerationCount = new SimpleIntegerProperty(0)
  private val resultPaths = scala.collection.mutable.Map.empty[String, Vector[Path]]

  private def refreshFilter(): Unit =
    val group = Option(selectedGeneration.get).getOrElse("")
    val paths = resultPaths.getOrElse(group, Vector.empty).toSet
    filtered.setPredicate(i => (group.isEmpty || paths(i.path)) && GalleryVM.matches(i, query.get))
    selectedGenerationCount.set(paths.size)
  query.addListener((_, _, _) => refreshFilter())
  selectedGeneration.addListener((_, _, _) => refreshFilter())

  def beginGeneration(jobId: String): Unit =
    require(jobId.nonEmpty, "Generation id is empty")
    if !resultPaths.contains(jobId) then
      resultPaths(jobId) = Vector.empty
      generations.add(1, jobId)
    query.set("")
    selectedGeneration.set(jobId)

  /** Only individual final results belong to a generation's export; grids stay in history. */
  def addResult(jobId: String, item: GalleryItem, grid: Boolean = false): Unit =
    if !resultPaths.contains(jobId) then beginGeneration(jobId)
    if !grid then
      val existing = resultPaths(jobId)
      if !existing.contains(item.path) then resultPaths(jobId) = existing :+ item.path
    refreshFilter()
    addNewest(item)

  /** Immutable snapshot in generation order, independent of the prompt search. */
  def selectedGenerationPaths: Seq[Path] = resultPaths.getOrElse(selectedGeneration.get, Vector.empty)

  def setAll(xs: Seq[GalleryItem]): Unit = items.setAll(xs*)

  /** Replaces the list with a fresh scan but keeps items added since `since` (images saved while scanning). */
  def merge(scanned: Seq[GalleryItem], since: Long): Unit =
    val paths = scanned.map(_.path).toSet
    val retained = resultPaths.valuesIterator.flatten.toSet
    val extra = items.toArray(Array.empty[GalleryItem]).toSeq.filter(i => !paths.contains(i.path) && (i.modified >= since || retained(i.path)))
    items.setAll((extra ++ scanned).sortBy(i => -i.modified)*)

  /** Inserts a freshly saved image at the front (replacing an entry with the same path). */
  def addNewest(item: GalleryItem): Unit =
    val idx = indexOf(item.path)
    if idx >= 0 then items.remove(idx)
    items.add(0, item)

  def remove(path: Path): Unit =
    resultPaths.keys.toVector.foreach(id => resultPaths(id) = resultPaths(id).filterNot(_ == path))
    val idx = indexOf(path)
    if idx >= 0 then items.remove(idx)
    refreshFilter()

  def indexOf(path: Path): Int =
    var i = 0
    while i < items.size && items.get(i).path != path do i += 1
    if i < items.size then i else -1

  /** Neighbour in the filtered list (`delta` -1 = newer, +1 = older). */
  def neighbor(path: Path, delta: Int): Option[GalleryItem] =
    var i = 0
    while i < filtered.size && filtered.get(i).path != path do i += 1
    if i >= filtered.size then None
    else
      val j = i + delta
      Option.when(j >= 0 && j < filtered.size)(filtered.get(j))

object GalleryVM:
  /** Every whitespace separated term must occur in the file name or the parameters (case-insensitive). */
  def matches(i: GalleryItem, query: String): Boolean =
    val terms = Option(query).getOrElse("").toLowerCase.split("\\s+").filter(_.nonEmpty)
    terms.isEmpty || {
      val hay = (i.fileName + "\n" + i.parameters.getOrElse("")).toLowerCase
      terms.forall(hay.contains)
    }
