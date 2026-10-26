package sdfx.core.presets

enum PresetFieldKind:
  case Text, Integer, Decimal, Flag, Choice, PathText

/** A field registered by the destination form, independent of a saved record's tracked set. */
final case class PresetFieldSpec(
  key: FieldKey,
  label: String,
  kind: PresetFieldKind,
  defaultValue: PresetAtom,
  choices: Vector[PresetAtom],
  min: Option[BigDecimal],
  max: Option[BigDecimal]
)

final case class PresetSchema(
  mode: PresetMode,
  revision: String,
  fields: Vector[PresetFieldSpec],
  issues: Vector[PresetIssue]
)

final case class FieldSnapshot(values: Map[FieldKey, PresetAtom], revision: String)
