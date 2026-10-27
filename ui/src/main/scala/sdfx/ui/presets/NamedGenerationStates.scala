package sdfx.ui.presets

import java.nio.file.{Files, Path}
import java.util.UUID
import scala.util.control.NonFatal
import sdfx.core.presets.*
import sdfx.ui.vm.Mode

/** Full typed snapshots; unavailable saved fields abort instead of becoming partial overlays. */
object NamedGenerationStates:
  val scope = "Saves this mode's generation settings, shared checkpoint/VAE/clip skip, all script and extension controls, and ControlNet input paths. Init image and painted mask pixels are NOT saved or restored; current inputs stay in place. Other input paths are references, not copied files. Window layout, history, outputs and global Settings are excluded."

  def collectionId(mode: Mode): String = s"state-${mode.key}"
  def collection(library: PresetLibrary, mode: Mode): PresetCollection =
    library.collections.find(_.id == collectionId(mode)).getOrElse(
      PresetCollection(collectionId(mode), Set.empty, Vector.empty, Vector.empty, Vector.empty))

  def capture(name: String, fields: NativePresetFields): Either[PresetError, PresetRecord] =
    val clean = name.trim
    if clean.isEmpty || clean.codePointCount(0, clean.length) > 256 || clean.exists(c => Character.isISOControl(c)) then
      Left(PresetError(Vector(PresetIssue("name", "name", "Enter a printable name of 1–256 characters"))))
    else if fields.schema.issues.nonEmpty then Left(PresetError(fields.schema.issues))
    else Right(PresetRecord(UUID.randomUUID(), clean, fields.capture().values, None,
      Set.empty, Vector.empty, Vector.empty))

  def put(library: PresetLibrary, mode: Mode, record: PresetRecord): PresetLibrary =
    val old = collection(library, mode)
    val existing = old.presets.find(_.name == record.name)
    val next = record.copy(id = existing.map(_.id).getOrElse(record.id))
    val records = old.presets.filterNot(_.name == next.name) :+ next
    val updated = old.copy(tracked = records.flatMap(_.fields.keys).toSet, presets = records)
    library.copy(collections = library.collections.filterNot(_.id == old.id) :+ updated)

  def remove(library: PresetLibrary, mode: Mode, name: String): PresetLibrary =
    val old = collection(library, mode)
    val remaining = old.presets.filterNot(_.name == name)
    library.copy(collections = library.collections.filterNot(_.id == old.id) :+
      old.copy(tracked = remaining.flatMap(_.fields.keys).toSet, presets = remaining))

  def preflight(record: PresetRecord, fields: NativePresetFields): Either[PresetError, PresetApplyPlan] =
    val schema = fields.schema
    val before = fields.capture()
    PresetApply.plan(record, record.fields.keySet, schema, before, ApplyBase.Current).flatMap { plan =>
      // Blank, already blank optional plugin choices do not require installed resources.
      val issues = plan.issues.filterNot(i => i.code == "unavailable-choice" &&
        FieldKey.parse(i.path).toOption.exists(k => record.fields.get(k).contains(PresetAtom.Text("")) &&
          before.values.get(k).contains(PresetAtom.Text(""))))
      val paths = schema.fields.filter(_.kind == PresetFieldKind.PathText).flatMap { spec =>
        record.fields.get(spec.key).collect { case PresetAtom.Text(path) if path.nonEmpty => (spec.key, path) }
      }
      val missing = paths.flatMap { (key, path) =>
        val exists = try path.split("[\\r\\n]+").filter(_.nonEmpty).forall(p => Files.isRegularFile(Path.of(p.trim))) catch case NonFatal(_) => false
        Option.unless(exists)(PresetIssue("missing-input", key.render, s"Input file is missing: $path"))
      }
      val batchInput = record.fields.get(FieldKey(Vector("batch", "input")))
      val batchMissing = if record.fields.get(FieldKey(Vector("batch", "enabled"))).contains(PresetAtom.Flag(true)) then
        val exists = batchInput.exists {
          case PresetAtom.Text(path) => try path.nonEmpty && Files.isDirectory(Path.of(path)) catch case NonFatal(_) => false
          case _ => false
        }
        Option.unless(exists)(PresetIssue("missing-input", "batch.input", "WD batch input folder is missing")).toVector
      else Vector.empty
      val all = issues ++ missing ++ batchMissing ++ fields.namedStateResourceIssues(record.fields)
      if all.nonEmpty then Left(PresetError(all)) else Right(plan)
    }

  def describe(error: PresetError): String = error.issues.map(i => s"${i.path}: ${i.message}").mkString("\n")
