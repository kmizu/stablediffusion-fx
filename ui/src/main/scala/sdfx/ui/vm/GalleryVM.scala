package sdfx.ui.vm

import javafx.beans.property.SimpleStringProperty
import javafx.collections.transformation.FilteredList
import javafx.collections.{FXCollections, ObservableList}
import sdfx.core.gallery.GalleryItem

import java.nio.file.Path

/** Gallery items (newest first) with a prompt / file name filter. */
final class GalleryVM:
  val items: ObservableList[GalleryItem] = FXCollections.observableArrayList()
  val filtered: FilteredList[GalleryItem] = new FilteredList(items, _ => true)
  val query = new SimpleStringProperty("")

  query.addListener((_, _, q) => filtered.setPredicate(i => GalleryVM.matches(i, q)))

  def setAll(xs: Seq[GalleryItem]): Unit = items.setAll(xs*)

  /** Replaces the list with a fresh scan but keeps items added since `since` (images saved while scanning). */
  def merge(scanned: Seq[GalleryItem], since: Long): Unit =
    val paths = scanned.map(_.path).toSet
    val extra = items.toArray(Array.empty[GalleryItem]).toSeq.filter(i => !paths.contains(i.path) && i.modified >= since)
    items.setAll((extra ++ scanned).sortBy(i => -i.modified)*)

  /** Inserts a freshly saved image at the front (replacing an entry with the same path). */
  def addNewest(item: GalleryItem): Unit =
    val idx = indexOf(item.path)
    if idx >= 0 then items.remove(idx)
    items.add(0, item)

  def remove(path: Path): Unit =
    val idx = indexOf(path)
    if idx >= 0 then items.remove(idx)

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
