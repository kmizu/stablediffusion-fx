package sdfx.core.modelcards

import java.nio.file.Path
import scala.util.control.NonFatal

final case class CardId private (kind: String, modelPath: String):
  require(kind != null && kind.nonEmpty, "Model kind is required")
  require(modelPath != null && Path.of(modelPath).isAbsolute, "Model path must be absolute")
  require(Path.of(modelPath).normalize.toString == modelPath, "Model path must be normalized")
  def storageKey: String = ujson.write(ujson.Arr(kind, modelPath))

object CardId:
  def apply(kind: String, modelPath: String): CardId =
    require(modelPath != null, "Model path is required")
    new CardId(kind, Path.of(modelPath).normalize.toString)

  def fromStorageKey(key: String): Either[String, CardId] =
    try
      ujson.read(key) match
        case a: ujson.Arr if a.value.length == 2 => (a.value(0), a.value(1)) match
          case (ujson.Str(kind), ujson.Str(path)) => Right(CardId(kind, path))
          case _ => Left("Model identity must contain two strings")
        case _ => Left("Model identity must be a two-element JSON array")
    catch case NonFatal(_) => Left("Invalid model identity")

final case class AncestorIdentity(path: Path, fileKey: String, reparseTarget: Option[Path])
final case class LocalPreviewIdentity(realPath: Path, fileKey: String, size: Long,
    modifiedMillis: Long, ancestorKeys: Vector[AncestorIdentity])

/** Only the validated resolver can construct provenance; consumers cannot replace its source URI. */
final class PreviewReference private[modelcards] (
    val localPath: Path, val source: String, val identity: LocalPreviewIdentity,
    val allowedRealParent: Option[Path])

final case class PreviewSourceSnapshot(bytes: Array[Byte], length: Int, sha256: String,
    identity: LocalPreviewIdentity)

final case class ModelCardRecord(id: CardId, name: String, family: Option[String],
    preview: Option[PreviewReference], problem: Option[String])


enum CardScanStatus:
  case Complete, Partial, Cancelled, Failed

final case class CardScanRequest(revision: Long, home: Path, settings: sdfx.core.config.Settings)
final case class CardScanResult(revision: Long, records: Vector[ModelCardRecord], status: CardScanStatus,
    problems: Vector[String], problemCount: Long, visited: Long)
