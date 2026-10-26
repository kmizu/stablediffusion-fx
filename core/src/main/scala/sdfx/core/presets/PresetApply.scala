package sdfx.core.presets

/** A complete, immutable decision about one destination form. The UI must check both revisions again before writing. */
final case class PresetApplyPlan(
  changes: Map[FieldKey, PresetAtom],
  issues: Vector[PresetIssue],
  schemaRevision: String,
  currentRevision: String
)

object PresetApply:
  private def issue(code: String, key: FieldKey, message: String): PresetIssue =
    PresetIssue(code, key.render, message)

  private def sameAtomKind(a: PresetAtom, b: PresetAtom): Boolean = (a, b) match
    case (_: PresetAtom.Text, _: PresetAtom.Text) => true
    case (_: PresetAtom.Integer, _: PresetAtom.Integer) => true
    case (_: PresetAtom.Decimal, _: PresetAtom.Decimal) => true
    case (_: PresetAtom.Flag, _: PresetAtom.Flag) => true
    case _ => false

  private def kindMatches(spec: PresetFieldSpec, value: PresetAtom): Boolean =
    spec.kind match
      case PresetFieldKind.Text | PresetFieldKind.PathText => value.isInstanceOf[PresetAtom.Text]
      case PresetFieldKind.Integer => value.isInstanceOf[PresetAtom.Integer]
      case PresetFieldKind.Decimal => value.isInstanceOf[PresetAtom.Decimal]
      case PresetFieldKind.Flag => value.isInstanceOf[PresetAtom.Flag]
      case PresetFieldKind.Choice => sameAtomKind(spec.defaultValue, value)

  private def numericValue(value: PresetAtom): Option[BigDecimal] = value match
    case PresetAtom.Integer(n) => Some(BigDecimal(n))
    case PresetAtom.Decimal(n) => Some(n)
    case _ => None

  private def inRange(spec: PresetFieldSpec, value: PresetAtom): Boolean =
    numericValue(value) match
      case Some(number) =>
        spec.min.forall(number >= _) && spec.max.forall(number <= _)
      case None => true

  /** Never returns a partial mutation when a known, tracked value has an invalid type or numeric range. */
  def plan(
    record: PresetRecord,
    tracked: Set[FieldKey],
    schema: PresetSchema,
    current: FieldSnapshot,
    base: ApplyBase
  ): Either[PresetError, PresetApplyPlan] =
    val specs = schema.fields.groupBy(_.key)
    val skipped = Vector.newBuilder[PresetIssue]
    val invalid = Vector.newBuilder[PresetIssue]
    val changes = Map.newBuilder[FieldKey, PresetAtom]

    record.fields.keysIterator.filterNot(tracked).toVector.sortBy(_.render).foreach { key =>
      skipped += issue("untracked", key, "saved field is not in this collection's tracked set")
    }

    tracked.toVector.sortBy(_.render).foreach { key =>
      specs.get(key) match
        case None =>
          if record.fields.contains(key) || base == ApplyBase.Defaults then
            skipped += issue("unavailable", key, "field is not available in this destination mode")
        case Some(candidates) if candidates.size != 1 =>
          skipped += issue("ambiguous", key, "multiple destination bindings have this field ID")
        case Some(candidates) =>
          val spec = candidates.head
          val candidate = record.fields.get(key).orElse {
            Option.when(base == ApplyBase.Defaults)(spec.defaultValue)
          }
          candidate.foreach { value =>
            current.values.get(key) match
              case None =>
                skipped += issue("unavailable", key, "current destination value is unavailable")
              case Some(before) =>
                if !kindMatches(spec, value) then
                  invalid += issue("invalid-type", key, "saved value has the wrong field type")
                else if !inRange(spec, value) then
                  invalid += issue("invalid-range", key, "saved number is outside the destination range")
                else if spec.kind == PresetFieldKind.Choice && !spec.choices.contains(value) then
                  skipped += issue("unavailable-choice", key, "saved choice is not available in the current catalog")
                  if base == ApplyBase.Defaults then
                    val fallback = spec.defaultValue
                    val fallbackAvailable =
                      kindMatches(spec, fallback) && inRange(spec, fallback) && spec.choices.contains(fallback)
                    if !fallbackAvailable then
                      skipped += issue("unavailable-default", key,
                        "declared default choice is also unavailable; current value is retained")
                    else if fallback != before then changes += key -> fallback
                else if value != before then changes += key -> value
          }
    }

    record.opaque.foreach { (sourceKey, _) =>
      skipped += PresetIssue("opaque-unapplied", sourceKey,
        "retained unresolved source value; not applied")
    }

    val errors = invalid.result()
    if errors.nonEmpty then Left(PresetError(errors))
    else Right(PresetApplyPlan(changes.result(), schema.issues ++ skipped.result(),
      schema.revision, current.revision))
