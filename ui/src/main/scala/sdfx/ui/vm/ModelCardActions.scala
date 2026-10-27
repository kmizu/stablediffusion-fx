package sdfx.ui.vm

import javafx.beans.property.{ReadOnlyLongProperty, ReadOnlyLongWrapper}
import sdfx.core.model.LoraSyntax
import sdfx.core.modelcards.{CardId, ModelCardRecord}
import sdfx.core.protocol.{FileEntry, ModelEntry}
import sdfx.engine.{CatalogSnapshot, EngineState}
import sdfx.ui.view.UiContext

import java.nio.file.Path
import scala.concurrent.{ExecutionContext, Future, Promise}
import scala.util.{Failure, Success}
import scala.util.control.NonFatal

final case class PromptInsertionTarget(modeKey: String, negative: Boolean, capturedText: String, caret: Int,
    current: () => (String, Int), insert: String => Unit)
final case class ModelCardActionState(canSelectCheckpoint: Boolean, canInsert: Boolean, reason: Option[String])
final case class CheckpointCardAdmission(allowed: Boolean, reason: Option[String], revision: Long)
final case class CheckpointAdmission(token: Long, modelPath: String)

/** All gate methods and callbacks except validate's work are called on the owner thread. */
private[ui] final case class CardGateSnapshot(engine: EngineState, queueIdle: Boolean, loadsEmpty: Boolean,
    extendedStyleApplying: Boolean, catalog: CatalogSnapshot)

private[ui] final class CheckpointCardGate(snapshot: () => CardGateSnapshot,
    validate: CardId => Future[Either[String, Unit]], onFx: (() => Unit) => Unit, start: String => Long,
    windowsPaths: Boolean):
  private val version = new ReadOnlyLongWrapper(0L)
  val revision: ReadOnlyLongProperty = version.getReadOnlyProperty
  private var serial = 0L
  private var pending: Option[(Long, Promise[Either[String, CheckpointAdmission]])] = None
  private var closed = false
  def stateChanged(): Unit = version.set(version.get + 1)

  private def entry(id: CardId, s: CardGateSnapshot): Either[String, ModelEntry] =
    if id.kind != "checkpoint" then Left("Only checkpoint cards can select a checkpoint")
    else
      val matches = s.catalog.checkpoints.filter(e => CardActionResolution.samePath(e.path, id.modelPath, windowsPaths))
      matches match
        case Seq(one) if s.catalog.checkpoints.count(_.name == one.name) == 1 &&
            CardActionResolution.find(s.catalog.checkpoints, one.name, _.name, _.path).contains(one) => Right(one)
        case _ => Left("Checkpoint is absent or ambiguous in the current catalog")

  private def eligibility(id: CardId, ignorePending: Boolean): Either[String, ModelEntry] =
    val s = snapshot()
    if closed then Left("The model browser is closed")
    else if pending.nonEmpty && !ignorePending then Left("Checkpoint validation is already pending")
    else if !s.engine.isInstanceOf[EngineState.Ready] then Left("Engine must be ready")
    else if !s.queueIdle then Left("Generation queue is busy")
    else if !s.loadsEmpty then Left("A model or VAE load is pending")
    else if s.extendedStyleApplying then Left("An extended style is being applied")
    else entry(id, s)

  def state(id: CardId): CheckpointCardAdmission =
    val result = eligibility(id, false)
    CheckpointCardAdmission(result.isRight, result.left.toOption, revision.get)

  def admit(id: CardId, expectedName: Option[String] = None): Future[Either[String, CheckpointAdmission]] =
    eligibility(id, false) match
      case Left(reason) => Future.successful(Left(reason))
      case Right(original) if expectedName.exists(_ != original.name) =>
        Future.successful(Left("Displayed checkpoint is stale; refresh the browser"))
      case Right(original) =>
        serial += 1
        val token = serial
        val result = Promise[Either[String, CheckpointAdmission]]()
        pending = Some(token -> result)
        stateChanged()
        def finish(validated: Either[String, Unit]): Unit =
          if pending.exists(_._1 == token) then
            val answer = validated.flatMap(_ => eligibility(id, true)).flatMap { e =>
              if e.name != original.name || !CardActionResolution.samePath(e.path, original.path, windowsPaths) then
                Left("Checkpoint catalog entry changed during validation")
              else
                try Right(CheckpointAdmission(start(e.name), id.modelPath))
                catch case NonFatal(ex) => Left(Option(ex.getMessage).getOrElse(ex.toString))
            }
            pending = None
            stateChanged()
            result.trySuccess(answer)
        try
          validate(id).onComplete {
            case Success(value) => onFx(() => finish(value))
            case Failure(ex) => onFx(() => finish(Left(Option(ex.getMessage).getOrElse(ex.toString))))
          }(using ExecutionContext.parasitic)
        catch case NonFatal(ex) => finish(Left(Option(ex.getMessage).getOrElse(ex.toString)))
        result.future

  def dispose(): Unit =
    closed = true
    pending.foreach(_._2.trySuccess(Left("The model browser is closed")))
    pending = None
    stateChanged()

/** Mirrors ModelCatalog's ordered name/path lookup so ambiguity can be rejected before insertion. */
private object CardActionResolution:
  def samePath(a: String, b: String, windows: Boolean): Boolean =
    try
      val left = Path.of(a).normalize.toString
      val right = Path.of(b).normalize.toString
      if windows then left.equalsIgnoreCase(right) else left == right
    catch case NonFatal(_) => false
  def stem(s: String): String =
    val base = s.replace('\\', '/').split('/').lastOption.getOrElse("")
    val i = base.lastIndexOf('.')
    if i > 0 then base.substring(0, i) else base
  private def norm(s: String): String = s.replace('\\', '/').trim
  def find[T](items: Seq[T], q: String, name: T => String, path: T => String): Option[T] =
    val query = norm(q)
    if query.isEmpty then None
    else
      val predicates: Seq[T => Boolean] = Seq(
        x => norm(name(x)) == query, x => norm(path(x)) == query,
        x => stem(name(x)) == query, x => stem(path(x)) == query, x => stem(name(x)) == stem(query),
        x => norm(name(x)).equalsIgnoreCase(query), x => norm(path(x)).equalsIgnoreCase(query),
        x => stem(name(x)).equalsIgnoreCase(query), x => stem(path(x)).equalsIgnoreCase(query),
        x => stem(path(x)).equalsIgnoreCase(stem(query)))
      predicates.iterator.map(p => items.find(p)).collectFirst { case Some(x) => x }

/** One coalesced owner-thread observation after a terminal event, until the real queue is idle. */
private[ui] final class CardIdleObserver(isIdle: () => Boolean,
    schedule: (() => Unit) => (() => Unit), changed: () => Unit):
  private var closed = false
  private var serial = 0L
  private var pending: Option[() => Unit] = None
  def terminal(): Unit =
    if !closed && pending.isEmpty then
      serial += 1
      val token = serial
      pending = Some(schedule(() =>
        if !closed && token == serial && pending.nonEmpty then
          pending = None
          if isIdle() then changed() else terminal()
      ))
  def dispose(): Unit =
    closed = true
    serial += 1
    pending.foreach(_())
    pending = None

final class ModelCardActions(ctx: UiContext, selectionState: CardId => CheckpointCardAdmission,
    admit: CardId => Future[Either[String, CheckpointAdmission]], val selectionRevision: ReadOnlyLongProperty,
    validate: CardId => Future[Either[String, Unit]], onFx: (() => Unit) => Unit, windowsPaths: Boolean,
    admitRecord: Option[(CardId, String) => Future[Either[String, CheckpointAdmission]]] = None):
  private var target: Option[PromptInsertionTarget] = None
  private var closed = false
  private var serial = 0L
  private var pending: Option[(Long, Promise[Either[String, Unit]])] = None
  private def invalidate(): Unit =
    serial += 1
    pending.foreach(_._2.trySuccess(Left("Insertion target changed or closed")))
    pending = None
  def setTarget(value: Option[PromptInsertionTarget]): Unit =
    invalidate()
    target = value
  def dispose(): Unit =
    closed = true
    invalidate()
    target = None

  private def catalog: CatalogSnapshot = if ctx.services == null then ctx.catalog.snapshot.get else ctx.services.catalog.snapshot
  private def checkpoint(record: ModelCardRecord): Either[String, ModelEntry] =
    val checkpoints = catalog.checkpoints
    val entries = checkpoints.filter(e => CardActionResolution.samePath(e.path, record.id.modelPath, windowsPaths))
    entries match
      case Seq(one) if one.name == record.name && checkpoints.count(_.name == one.name) == 1 &&
          CardActionResolution.find(checkpoints, one.name, _.name, _.path).contains(one) => Right(one)
      case _ => Left("Displayed checkpoint is stale or ambiguous; refresh the browser")

  private def insertion(record: ModelCardRecord): Either[String, (PromptInsertionTarget, String, FileEntry)] =
    for
      _ <- Either.cond(!closed, (), "The model browser is closed")
      t <- target.toRight("Open Networks from a prompt editor")
      _ <- Either.cond(t.current() == ((t.capturedText, t.caret)), (), "Prompt or caret changed; reopen Networks")
      items <- record.id.kind match
        case "lora" => Right(catalog.loras)
        case "embedding" => Right(catalog.embeddings)
        case _ => Left("This model kind is inspect-only")
      matches = items.filter(e => CardActionResolution.samePath(e.path, record.id.modelPath, windowsPaths))
      item <- matches match
        case Seq(one) => Right(one)
        case _ => Left("Model is absent or ambiguous in the current catalog")
      _ <- Either.cond(item.name == record.name, (), "Displayed model is stale; refresh the browser")
      token = CardActionResolution.stem(if record.id.kind == "embedding" then item.path else item.name)
      sameToken = items.filter(e => CardActionResolution.stem(e.name).equalsIgnoreCase(token) ||
        CardActionResolution.stem(e.path).equalsIgnoreCase(token))
      _ <- Either.cond(token.nonEmpty && sameToken.size == 1 &&
        CardActionResolution.find(items, token, _.name, _.path).contains(item), (), "Model token is ambiguous")
      resolved = if ctx.services == null then Some(item)
        else if record.id.kind == "lora" then ctx.services.catalog.findLora(token)
        else ctx.services.catalog.findEmbedding(token)
      _ <- Either.cond(resolved.contains(item), (), "Model token resolves to a different catalog entry")
    yield (t, if record.id.kind == "lora" then LoraSyntax.tag(token, 1.0) else token, item)

  def state(record: ModelCardRecord): ModelCardActionState =
    if record.id.kind == "checkpoint" then
      val s = selectionState(record.id)
      val current = checkpoint(record)
      ModelCardActionState(!closed && s.allowed && current.isRight, false,
        if closed then Some("The model browser is closed") else current.left.toOption.orElse(s.reason))
    else if record.id.kind == "lora" || record.id.kind == "embedding" then
      val result = insertion(record)
      ModelCardActionState(false, result.isRight && pending.isEmpty,
        result.left.toOption.orElse(Option.when(pending.nonEmpty)("Model validation is already pending")))
    else ModelCardActionState(false, false, Some("This model kind is inspect-only"))

  def select(record: ModelCardRecord): Future[Either[String, CheckpointAdmission]] =
    if record.id.kind != "checkpoint" then Future.successful(Left("This model kind is inspect-only"))
    else if closed then Future.successful(Left("The model browser is closed"))
    else checkpoint(record) match
      case Left(reason) => Future.successful(Left(reason))
      case Right(_) => admitRecord.fold(admit(record.id))(f => f(record.id, record.name))

  def insert(record: ModelCardRecord): Future[Either[String, Unit]] =
    if pending.nonEmpty then Future.successful(Left("Model validation is already pending"))
    else insertion(record) match
      case Left(reason) => Future.successful(Left(reason))
      case Right(original) =>
        serial += 1
        val token = serial
        val result = Promise[Either[String, Unit]]()
        pending = Some(token -> result)
        def finish(checked: Either[String, Unit]): Unit =
          if pending.exists(_._1 == token) then
            val outcome = checked.flatMap(_ => insertion(record)).flatMap { current =>
              if !(current._1 eq original._1) || current._2 != original._2 || current._3.name != original._3.name ||
                  !CardActionResolution.samePath(current._3.path, original._3.path, windowsPaths) then
                Left("Model or insertion target changed during validation")
              else
                try
                  current._1.insert(current._2)
                  Right(())
                catch case NonFatal(ex) => Left(Option(ex.getMessage).getOrElse(ex.toString))
            }
            pending = None
            result.trySuccess(outcome)
        try
          validate(record.id).onComplete {
            case Success(value) => onFx(() => finish(value))
            case Failure(ex) => onFx(() => finish(Left(Option(ex.getMessage).getOrElse(ex.toString))))
          }(using ExecutionContext.parasitic)
        catch case NonFatal(ex) => finish(Left(Option(ex.getMessage).getOrElse(ex.toString)))
        result.future
