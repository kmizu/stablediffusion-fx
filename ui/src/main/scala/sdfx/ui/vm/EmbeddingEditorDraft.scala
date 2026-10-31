package sdfx.ui.vm

/** Unsaved edits retain every encoder and vector until save, discard or preview. */
final class EmbeddingEditorDraft private (initial: Map[String, Vector[Vector[Double]]]):
  private val original = initial
  private var current = initial

  def dirty: Boolean = current != original
  def values: Map[String, Vector[Vector[Double]]] = current

  def vector(encoder: String, index: Int): Vector[Double] =
    val rows = current.getOrElse(encoder, throw new IllegalArgumentException("unknown encoder"))
    require(index >= 0 && index < rows.size, "invalid vector index")
    rows(index)

  def setCoordinate(encoder: String, index: Int, coordinate: Int, value: Double): Unit =
    val row = vector(encoder, index)
    require(coordinate >= 0 && coordinate < row.size, "invalid coordinate index")
    require(value.isFinite, "coordinate must be finite")
    val rows = current(encoder)
    current = current.updated(encoder, rows.updated(index, row.updated(coordinate, value)))

  def resetVector(encoder: String, index: Int): Unit =
    vector(encoder, index)
    current = current.updated(encoder, current(encoder).updated(index, original(encoder)(index)))

  def resetAll(): Unit = current = original

object EmbeddingEditorDraft:
  def apply(values: Map[String, Vector[Vector[Double]]]): EmbeddingEditorDraft =
    require(values.keySet == Set("clip_l") || values.keySet == Set("clip_l", "clip_g"), "invalid encoder set")
    values.values.foreach { rows =>
      require(rows.nonEmpty && rows.head.nonEmpty, "empty vector")
      require(rows.forall(row => row.size == rows.head.size && row.forall(_.isFinite)), "invalid vector dimensions or values")
    }
    require(values.size == 1 || values("clip_l").size == values("clip_g").size, "encoder vector counts differ")
    new EmbeddingEditorDraft(values)
