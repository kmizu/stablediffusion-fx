package sdfx.ui.presets

import javafx.application.Platform
import sdfx.core.presets.*

import java.nio.charset.StandardCharsets.UTF_8
import java.util.UUID
import scala.util.control.NonFatal

/** Coordinates one synchronous form transaction on the JavaFX application thread. */
final class NativePresetsController private[ui] (
  fields: NativePresetFields,
  busy: () => Boolean,
  onFxThread: () => Boolean
):
  def this(fields: NativePresetFields, busy: () => Boolean) =
    this(fields, busy, () => Platform.isFxApplicationThread)

  private var applying = false
  private val limits = PresetLimits()

  private def error(code: String, path: String, message: String): PresetError =
    PresetError(Vector(PresetIssue(code, path, message)))

  private def exception(code: String, path: String, e: Throwable): PresetError =
    error(code, path, Option(e.getMessage).getOrElse(e.getClass.getSimpleName))

  private def guard(operation: String): Option[PresetError] =
    if !onFxThread() then Some(error("wrong-thread", operation,
      "Preset form operations must run on the JavaFX application thread"))
    else if applying then Some(error("reentrant", operation,
      "A preset application is already changing this form"))
    else None

  /** An immutable catalog/current pair for the pane. No form writes or service work. */
  def inspect(): Either[PresetError, (PresetSchema, FieldSnapshot)] =
    try
      guard("inspect") match
        case Some(problem) => Left(problem)
        case None => Right((fields.schema, fields.capture()))
    catch case NonFatal(e) => Left(exception("inspect-failed", "inspect", e))

  /** Read-only admission before the pane commits any destination editor draft.
    * apply() checks the same gate again immediately before form writes.
    */
  def admitApply(): Either[PresetError, Unit] =
    try
      guard("apply") match
        case Some(problem) => Left(problem)
        case None if busy() => Left(error("mask-export-busy", "apply",
          "Wait for the mask export to finish before applying a preset"))
        case None => Right(())
    catch case NonFatal(e) => Left(exception("apply-failed", "apply", e))

  private def plan(record: PresetRecord, tracked: Set[FieldKey],
                   base: ApplyBase): Either[PresetError, PresetApplyPlan] =
    PresetApply.plan(record, tracked, fields.schema, fields.capture(), base)

  def preview(record: PresetRecord, tracked: Set[FieldKey],
              base: ApplyBase): Either[PresetError, PresetApplyPlan] =
    try
      guard("preview") match
        case Some(problem) => Left(problem)
        case None => plan(record, tracked, base)
    catch case NonFatal(e) => Left(exception("preview-failed", "preview", e))

  private def currentPlan(record: PresetRecord, tracked: Set[FieldKey],
                          base: ApplyBase): Either[PresetError, PresetApplyPlan] =
    val firstSchema = fields.schema
    val firstSnapshot = fields.capture()
    val initial = PresetApply.plan(record, tracked, firstSchema, firstSnapshot, base)
    val secondSchema = fields.schema
    val secondSnapshot = fields.capture()
    val chosen =
      if secondSchema.revision == firstSchema.revision &&
          secondSnapshot.revision == firstSnapshot.revision then initial
      else PresetApply.plan(record, tracked, secondSchema, secondSnapshot, base)
    chosen.flatMap { candidate =>
      val latestSchema = fields.schema
      val latestSnapshot = fields.capture()
      if latestSchema.revision != candidate.schemaRevision ||
          latestSnapshot.revision != candidate.currentRevision then
        Left(error("stale-form", "apply", "Form or catalog changed again while planning the preset"))
      else Right(candidate)
    }

  private def unrestored(expected: Map[FieldKey, PresetAtom]): Vector[PresetIssue] =
    try
      val actual = fields.capture().values
      expected.keysIterator.filter(key => actual.get(key) != expected.get(key)).toVector
        .sortBy(_.render).map { key =>
          PresetIssue("rollback-failed", key.render, "Original form value was not restored")
        }
    catch case NonFatal(e) => Vector(PresetIssue("rollback-failed", "apply",
      s"Could not verify restored form values: ${Option(e.getMessage).getOrElse(e.getClass.getSimpleName)}"))

  def apply(record: PresetRecord, tracked: Set[FieldKey],
            base: ApplyBase): Either[PresetError, PresetApplyPlan] =
    try
      guard("apply") match
        case Some(problem) => Left(problem)
        case None =>
          if busy() then Left(error("mask-export-busy", "apply",
            "Wait for the mask export to finish before applying a preset"))
          else
            applying = true
            try
              currentPlan(record, tracked, base).flatMap { candidate =>
                if busy() then Left(error("mask-export-busy", "apply",
                  "Wait for the mask export to finish before applying a preset"))
                else
                  val before = fields.capture()
                  if before.revision != candidate.currentRevision then
                    Left(error("stale-form", "apply", "Form changed before the preset could be written"))
                  else
                    val touched = before.values.filter((key, _) => candidate.changes.contains(key))
                    // NativePresetFields.write prevalidates the complete map and rolls back
                    // exact touched setters on exception, including separate rollback errors.
                    fields.write(candidate.changes) match
                      case Left(failure) => Left(PresetError(failure.issues ++ unrestored(touched)))
                      case Right(_) =>
                        val after = fields.capture()
                        val didNotStick = candidate.changes.keysIterator.filter { key =>
                          after.values.get(key) != candidate.changes.get(key)
                        }.toVector.sortBy(_.render)
                        if didNotStick.isEmpty then Right(candidate)
                        else
                          val mismatch = PresetIssue("write-mismatch", "apply",
                            s"Preset field changed during apply: ${didNotStick.map(_.render).mkString(", ")}")
                          val rollback = fields.write(touched).left.toOption.toVector.flatMap(_.issues)
                            .map(i => i.copy(code = "rollback-failed"))
                          Left(PresetError(Vector(mismatch) ++ rollback ++ unrestored(touched)))
              }
            finally applying = false
    catch case NonFatal(e) => Left(exception("apply-failed", "apply", e))

  private def validName(name: String): Boolean =
    if name == null then false
    else
      val cps = name.codePoints().toArray
      cps.length <= limits.maxNameCodePoints &&
        cps.exists(cp => !Character.isWhitespace(cp)) &&
        !cps.exists(cp => Character.isISOControl(cp) || (cp >= 0xd800 && cp <= 0xdfff))

  private def freeId(collection: PresetCollection, name: String): Either[PresetError, UUID] =
    val occupied = collection.presets.iterator.map(_.id).toSet
    val original = PresetRecord.initialId(collection.id, name)
    if !occupied(original) then Right(original)
    else
      val attempts = math.min(occupied.size + 1, limits.maxRecords + 1)
      (1 to attempts).iterator.map { counter =>
        val material = s"${collection.id}\u0000$name\u0000sdfx-native-presets-collision-v1\u0000$counter"
        UUID.nameUUIDFromBytes(material.getBytes(UTF_8))
      }.find(id => !occupied(id)).toRight(error("duplicate-id", "name",
        "Could not allocate a free deterministic preset ID within the bounded candidate search"))

  def captureSelected(name: String, collection: PresetCollection,
                      selected: Set[FieldKey]): Either[PresetError, PresetRecord] =
    try
      guard("captureSelected") match
        case Some(problem) => Left(problem)
        case None if !validName(name) =>
          Left(error("invalid-name", "name", "Preset name must contain printable non-whitespace text"))
        case None =>
          val untracked = selected.diff(collection.tracked)
          if untracked.nonEmpty then
            Left(error("untracked-selection", untracked.toVector.map(_.render).sorted.mkString(", "),
              "Selected fields must first be added to this collection's tracked set"))
          else
            val schema = fields.schema
            val snapshot = fields.capture()
            val available = schema.fields.map(_.key).toSet
            val missing = selected.diff(available).union(selected.diff(snapshot.values.keySet))
            if missing.nonEmpty then
              Left(error("unavailable-selection", missing.toVector.map(_.render).sorted.mkString(", "),
                "Selected field is unavailable in this mode"))
            else
              val matches = collection.presets.filter(_.name == name)
              if matches.size > 1 then
                Left(error("duplicate-name", "name", "Collection contains duplicate preset names"))
              else
                val existing = matches.headOption
                val retained = existing.toVector.flatMap(_.fields).filter { (key, _) =>
                  !collection.tracked(key) || !available(key)
                }.toMap
                val captured = snapshot.values.filter((key, _) => selected(key))
                val nextFields = retained ++ captured
                val oldFields = existing.map(_.fields).getOrElse(Map.empty[FieldKey, PresetAtom])
                val modified = (oldFields.keySet ++ nextFields.keySet).filter { key =>
                  oldFields.get(key) != nextFields.get(key)
                }
                val dirty = existing.map(_.dirty).getOrElse(Set.empty[FieldKey]) ++ modified
                if nextFields.size > limits.maxFieldsPerRecord || dirty.size > limits.maxFieldsPerRecord then
                  Left(error("limit", "fields", "Preset has too many captured fields"))
                else existing match
                  case Some(saved) => Right(saved.copy(fields = nextFields, dirty = dirty))
                  case None => freeId(collection, name).map { id =>
                    PresetRecord(id, name, nextFields, None, dirty, Vector.empty, Vector.empty)
                  }
    catch case NonFatal(e) => Left(exception("capture-failed", "captureSelected", e))
