package sdfx.ui.view

import javafx.application.Platform
import javafx.collections.FXCollections
import javafx.geometry.{Insets, Pos}
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.layout.{BorderPane, HBox, Priority, Region, VBox}
import javafx.stage.{FileChooser, Stage}
import sdfx.core.config.{Settings, SettingsSaveIntent}
import sdfx.core.modelcards.*
import sdfx.engine.{AppServices, CardScanLease, ModelCardCatalog}
import sdfx.ui.Fx
import sdfx.ui.vm.{ModelCardActions, UiCleanup}
import scala.collection.mutable
import scala.concurrent.ExecutionContext
import scala.util.{Failure, Success}

private[ui] final class ModelBrowserSettings(services: AppServices, onSaved: Settings => Unit = _ => ()):
  def currentMode: CardPreviewMode = services.cardPreviews.current
  def menuModes: Seq[CardPreviewMode] = CardPreviewMode.values.toSeq.filterNot(_ == currentMode)
  def setTemporary(mode: CardPreviewMode): Unit = services.cardPreviews.setTemporary(mode)
  def useSavedDefaultNow(): Unit = setTemporary(CardPreviewPolicy.startup(services.settings.nsfwCardBlurDefault))
  private def save(f: Settings => Settings, intent: SettingsSaveIntent = SettingsSaveIntent.Explicit): Either[String, Settings] =
    val result = services.updateSettingsChecked(f, intent)
    result.foreach(onSaved)
    result
  def saveDefault(value: String): Either[String, Settings] = save(_.copy(nsfwCardBlurDefault = value))
  def saveAssociation(id: CardId, localPath: String): Either[String, Settings] =
    val windows = LocalPreviewFiles.systemPlatform.windows
    val pathCheck =
      try
        val scheme = localPath != null && localPath.matches("(?i)^[a-z][a-z0-9+.-]*:.*")
        val drive = windows && localPath != null && localPath.matches("(?i)^[a-z]:[\\\\/].*")
        if localPath == null || localPath.isEmpty || localPath.length > 8192 || (scheme && !drive) then
          Left("Choose an ordinary local preview path")
        else if windows then LocalPreviewFiles.validateWindowsPath(localPath, allowRelative = false)
        else if !java.nio.file.Path.of(localPath).isAbsolute then Left("Choose an absolute local preview path")
        else Right(())
      catch case scala.util.control.NonFatal(_) => Left("Invalid local preview path")
    pathCheck.flatMap(_ => save(s => s.copy(modelPreviewOverrides = s.modelPreviewOverrides.updated(id.storageKey, localPath))))
  def removeAssociation(id: CardId): Either[String, Settings] =
    save(s => s.copy(modelPreviewOverrides = s.modelPreviewOverrides - id.storageKey))
  def saveRecoveredSettings(): Either[String, Settings] = save(identity, SettingsSaveIntent.Recover)
  def saveFolders(folders: Map[String, Seq[String]]): Either[String, Settings] = save(_.copy(modelPreviewFolders = folders))

private[ui] trait BrowserOwnedView:
  def revision: Long
  def close(): Unit
  def valid: Boolean = true

/** Owns pending and installed leases, with at most one deferred callback. */
private[ui] final class BrowserViewState(dispatch: (() => Unit) => Unit,
    install: BrowserOwnedView => Unit, clear: () => Unit):
  private var pending: Option[BrowserOwnedView] = None
  private var current: Option[BrowserOwnedView] = None
  private var scheduled = false
  private var disposed = false
  private var highest = Long.MinValue
  def offer(view: BrowserOwnedView): Unit = synchronized {
    if disposed || !view.valid || view.revision <= highest then view.close()
    else
      highest = view.revision
      pending.foreach(_.close())
      pending = Some(view)
      if !scheduled then
        scheduled = true
        dispatch(() => drain())
  }
  private def drain(): Unit = synchronized {
    scheduled = false
    if !disposed then
      pending.foreach { next =>
        pending = None
        if !next.valid then next.close()
        else
          clear()
          current.foreach(_.close())
          current = Some(next)
          install(next)
      }
  }
  def dispose(): Unit = synchronized {
    if !disposed then
      disposed = true
      pending.foreach(_.close()); pending = None
      try clear()
      finally
        current.foreach(_.close()); current = None
  }

private[ui] object ModelBrowserState:
  final class DerivedRows(initial: Vector[ModelCardRecord], release: () => Unit):
    private var value = initial
    private var closed = false
    def rows: Vector[ModelCardRecord] = value
    def close(): Unit = if !closed then { closed = true; value = Vector.empty; release() }
  def filter(rows: Vector[ModelCardRecord], kind: String, query: String): Vector[ModelCardRecord] =
    val terms = Option(query).getOrElse("").toLowerCase(java.util.Locale.ROOT).split("\\s+").filter(_.nonEmpty)
    rows.filter { row =>
      val text = (row.name + " " + row.id.modelPath).toLowerCase(java.util.Locale.ROOT)
      (kind == "all" || row.id.kind == kind) && terms.forall(text.contains)
    }
  def derive(rows: Vector[ModelCardRecord], kind: String, query: String,
      reserve: (Long, Long) => Either[String, () => Unit], materialize: () => Unit = () => ()): Either[String, DerivedRows] =
    // Covers row references, collection nodes and temporary search strings before constructing any.
    val chars = 2048L + rows.iterator.map(r => (r.name.length.toLong + r.id.modelPath.length + 1L) * 4L).sum +
      Option(query).fold(0L)(_.length.toLong * 4L)
    val bytes = 1024L + rows.size.toLong * 256L + chars * 2L
    reserve(chars, bytes).map { release =>
      try
        materialize()
        new DerivedRows(filter(rows, kind, query), release)
      catch
        case e: Throwable =>
          release()
          throw e
    }
  def statusText(status: CardScanStatus, problems: Vector[String], count: Long): String =
    val detail = problems.headOption.fold("")(p => s": $p")
    status match
      case CardScanStatus.Complete => if count > 0 then s"Scan complete ($count problems)$detail" else "Scan complete"
      case CardScanStatus.Partial => s"Partial results$detail. Choose a narrower folder."
      case CardScanStatus.Failed => s"Scan failed$detail"
      case CardScanStatus.Cancelled => "Scan cancelled"
  def retryVisibleCapacity[A](cards: Seq[A], available: () => Int, retry: A => Unit): Unit =
    val it = cards.iterator
    while it.hasNext && available() > 0 do retry(it.next())

/** A source and derived owner are released only after the list/cells drop their references. */
private[ui] final class CatalogBrowserView(val lease: CardScanLease, isCurrent: () => Boolean = () => true) extends BrowserOwnedView:
  override def valid: Boolean = isCurrent()
  def revision: Long = lease.revision
  private var derived: Option[ModelBrowserState.DerivedRows] = None
  def replaceDerived(next: ModelBrowserState.DerivedRows): Unit =
    derived.foreach(_.close()); derived = Some(next)
  def close(): Unit =
    derived.foreach(_.close()); derived = None
    lease.close()

final class ModelBrowser(ctx: UiContext, actions: ModelCardActions, catalog: ModelCardCatalog):
  private val settings = new ModelBrowserSettings(ctx.services)
  private val loaderLease = PreviewImageLoader.acquire()
  private val loader = loaderLease.loader
  private val stage = new Stage()
  stage.initOwner(ctx.stage)
  stage.setTitle("Models and networks")
  stage.setMinWidth(640); stage.setMinHeight(420)
  stage.setWidth(820); stage.setHeight(620)
  private val list = new ListView[ModelCardRecord]()
  private val items = FXCollections.observableArrayList[ModelCardRecord]()
  list.setItems(items)
  list.setFixedCellSize(210)
  private val cells = mutable.Set.empty[CardCell]
  private val typeChoice = new ComboBox[String](FXCollections.observableArrayList("all", "checkpoint", "lora", "embedding", "vae", "controlnet", "hypernetwork"))
  typeChoice.setValue("all")
  private val query = new TextField()
  query.setPromptText("Search name or path")
  private val status = new Label("No scan yet")
  status.setWrapText(true)
  status.getStyleClass.add("model-browser-status")
  private val mode = new MenuButton()
  private var current: Option[CatalogBrowserView] = None
  private var closed = false
  private val requestSerial = new java.util.concurrent.atomic.AtomicLong(0L)
  private val views = new BrowserViewState(run => Platform.runLater(() => run()),
    v => { current = Some(v.asInstanceOf[CatalogBrowserView]); updateRows() }, clearRows)
  private def clearRows(): Unit =
    current = None
    cells.toVector.foreach(_.dropRecord())
    items.clear()
    status.setText("")
  private def updateRows(): Unit = current.foreach { owner =>
    val prepared = owner.lease.withView { view =>
      val rows = ModelBrowserState.derive(view.records, typeChoice.getValue, query.getText,
        (chars, bytes) => owner.lease.reserveDerived(chars, bytes).map(d => () => d.close()))
      (view.status, view.problems, view.problemCount, rows)
    }
    prepared match
      case Left(reason) => status.setText(reason)
      case Right((_, _, _, Left(reason))) => status.setText(s"Partial results: $reason. Choose a narrower folder.")
      case Right((scanStatus, problems, count, Right(derived))) =>
        val message = ModelBrowserState.statusText(scanStatus, problems, count)
        cells.toVector.foreach(_.dropRecord()); items.clear()
        owner.replaceDerived(derived)
        items.addAll(derived.rows*)
        status.setText(if derived.rows.isEmpty then s"$message · No matching models" else message)
  }
  private def offerCatalog(lease: CardScanLease): Unit =
    val revision = lease.revision
    views.offer(new CatalogBrowserView(lease, () => catalog.snapshotInfo.exists(_.revision == revision)))
  private val catalogSubscription = catalog.addListener(offerCatalog)
  private def refresh(): Unit =
    status.setText("Scanning previews…")
    val request = requestSerial.incrementAndGet()
    catalog.refresh().onComplete {
      case Success(lease) => offerCatalog(lease)
      case Failure(_: java.util.concurrent.CancellationException) => ()
      case Failure(e) => Fx.run(if !closed && requestSerial.get() == request then status.setText(s"Scan failed: ${Fx.message(e)}"))
    }(using ExecutionContext.parasitic)
  private def updateMenu(): Unit =
    mode.setText(s"Named NSFW previews: ${settings.currentMode}")
    mode.getItems.clear()
    settings.menuModes.foreach { value =>
      val item = new MenuItem(value.toString)
      item.setOnAction(_ => settings.setTemporary(value))
      mode.getItems.add(item)
    }
    val saved = new MenuItem("Use saved default now")
    saved.setOnAction(_ => settings.useSavedDefaultNow())
    mode.getItems.add(saved)
  private val removeModeListener = ctx.services.cardPreviews.addListener(_ => Fx.run(if !closed then updateMenu()))
  updateMenu()
  private val removeCapacity = loader.addCapacityListener(() =>
    ModelBrowserState.retryVisibleCapacity(cells.toSeq, () => 32 - loader.metrics.visibleConsumers, _.retry()))
  private val revisionListener: javafx.beans.value.ChangeListener[java.lang.Number] = (_, _, _) => cells.foreach(_.refreshActions())
  actions.selectionRevision.addListener(revisionListener)
  typeChoice.valueProperty.addListener((_, _, _) => updateRows())
  query.textProperty.addListener((_, _, _) => updateRows())
  list.setCellFactory(_ => new CardCell())
  private val refreshButton = new Button("Refresh previews")
  refreshButton.setOnAction(_ => refresh())
  private val recovery = new Button("Save recovered settings")
  private val recoveryNotice = new Label()
  recoveryNotice.setWrapText(true)
  private def updateRecoveryNotice(): Unit =
    val issue = ctx.services.settingsRecovery
    recovery.setVisible(issue.nonEmpty); recovery.setManaged(issue.nonEmpty)
    recoveryNotice.setVisible(issue.nonEmpty); recoveryNotice.setManaged(issue.nonEmpty)
    recoveryNotice.setText(issue.fold("")(r => s"Settings recovery required: ${r.reason}"))
  updateRecoveryNotice()
  stage.focusedProperty.addListener((_, _, focused) => if focused then updateRecoveryNotice())
  recovery.setOnAction(_ => settings.saveRecoveredSettings() match
    case Left(reason) => status.setText(reason); updateRecoveryNotice()
    case Right(_) => updateRecoveryNotice(); status.setText("Recovered settings saved"))
  private val foldersButton = new Button("Inspect-only folders…")
  foldersButton.setOnAction(_ => {
    val dialog = new Dialog[ButtonType]()
    dialog.initOwner(stage)
    dialog.setTitle("Additional inspect-only model folders")
    dialog.setHeaderText("These folders are for the browser only; they do not change inference model folders.")
    val text = new TextArea(ctx.services.settings.modelPreviewFolders.toSeq.sortBy(_._1)
      .flatMap((kind, paths) => paths.map(path => s"$kind=$path")).mkString("\n"))
    text.setPromptText("One kind=local folder per line, for example hypernetwork=D:/models/hypernetworks")
    dialog.getDialogPane.setContent(text)
    dialog.getDialogPane.getButtonTypes.addAll(ButtonType.CANCEL, ButtonType.OK)
    dialog.getDialogPane.getStylesheets.setAll(scene.getStylesheets)
    val answer = dialog.showAndWait()
    if answer.isPresent && answer.get == ButtonType.OK then
      val entries = text.getText.linesIterator.map(_.trim).filter(_.nonEmpty).toVector
      val parsed = entries.map { line =>
        val i = line.indexOf('=')
        if i <= 0 || i == line.length - 1 then None
        else Some(line.substring(0, i).trim -> line.substring(i + 1).trim)
      }
      if parsed.exists(_.isEmpty) then status.setText("Enter one kind=folder per line")
      else
        val folders = parsed.flatten.groupMap(_._1)(_._2).view.mapValues(_.toSeq).toMap
        settings.saveFolders(folders) match
          case Left(e) => status.setText(e)
          case Right(_) => refresh()
  })
  private val header = new VBox(6, new HBox(8, typeChoice, query, refreshButton),
    new HBox(8, mode, foldersButton), new Label("Matches ‘nsfw’ in the preview file reference. Temporary for this app session."), recoveryNotice, recovery, status)
  header.setPadding(new Insets(10))
  HBox.setHgrow(query, Priority.ALWAYS)
  val node: Region = new BorderPane(list, header, null, null, null)
  node.getStyleClass.add("model-browser")
  private val scene = new Scene(node, 820, 620)
  scene.getStylesheets.setAll(ctx.stage.getScene.getStylesheets)
  stage.setScene(scene)
  stage.setOnHidden(_ => dispose())
  def show(initialKind: String): Unit =
    typeChoice.setValue(initialKind)
    stage.show()
    catalogSubscription.left.foreach(reason => status.setText(s"Model catalog updates unavailable: $reason"))
    if catalogSubscription.isRight then refresh()
  private val cleanup = new UiCleanup
  def dispose(): Unit =
    closed = true
    requestSerial.incrementAndGet()
    cleanup.run(Seq(
      "browser target" -> (() => actions.setTarget(None)),
      "browser catalog listener" -> (() => catalogSubscription.foreach(remove => remove())),
      "browser mode listener" -> (() => removeModeListener()),
      "browser capacity listener" -> (() => removeCapacity()),
      "browser action listener" -> (() => actions.selectionRevision.removeListener(revisionListener)),
      "browser views" -> (() => views.dispose()),
      "browser cells" -> (() => {
        cells.toVector.foreach(cell => try cell.disposeCard() catch case scala.util.control.NonFatal(_) => ())
        cells.clear()
      }),
      "browser loader lease" -> (() => loaderLease.close()),
      "browser stage" -> (() => if stage.isShowing then stage.close())
    ))

  private final class CardCell extends ListCell[ModelCardRecord]:
    private var card = new ModelPreviewCard(ctx.services.cardPreviews, loader)
    private val select = new Button("Select")
    private val insert = new Button("Insert")
    private val choose = new Button("Choose preview…")
    private val remove = new Button("Remove override")
    private val detail = new TextField()
    detail.setEditable(false)
    private val reason = new Label()
    reason.setWrapText(true)
    private val box = new VBox(4, card.node, detail, new HBox(6, select, insert, choose, remove), reason)
    cells += this
    sceneProperty.addListener((_, oldScene, newScene) =>
      if oldScene != null && newScene == null then
        disposeCard()
        cells -= this
    )
    def clearCard(): Unit =
      if card != null then card.clear()
      setGraphic(null)
      detail.clear(); reason.setText("")
    def dropRecord(): Unit =
      clearCard()
      select.setOnAction(null); insert.setOnAction(null); choose.setOnAction(null); remove.setOnAction(null)
      super.updateItem(null, true)
    def disposeCard(): Unit =
      dropRecord()
      if card != null then { card.dispose(); card = null }
    def retry(): Unit = if card != null then card.retryVisibleCapacity()
    def refreshActions(): Unit = Option(getItem).foreach { row =>
      val s = actions.state(row)
      select.setVisible(row.id.kind == "checkpoint"); select.setManaged(select.isVisible)
      insert.setVisible(row.id.kind == "lora" || row.id.kind == "embedding"); insert.setManaged(insert.isVisible)
      select.setDisable(!s.canSelectCheckpoint); insert.setDisable(!s.canInsert)
      reason.setText(s.reason.getOrElse(""))
    }
    override def updateItem(row: ModelCardRecord, empty: Boolean): Unit =
      super.updateItem(row, empty)
      clearCard()
      if !empty && row != null then
        if card == null then
          card = new ModelPreviewCard(ctx.services.cardPreviews, loader)
          box.getChildren.set(0, card.node)
          cells += this
        card.bind(row, current.fold(0L)(_.revision))
        detail.setText(row.id.modelPath)
        refreshActions()
        select.setOnAction(_ => current.foreach { owner =>
          owner.lease.retain() match
            case Left(e) => reason.setText(e)
            case Right(held) => Fx.onFx(actions.select(row)) { result =>
              try
                if !closed && getItem == row then result match
                  case Left(e) => reason.setText(e)
                  case Right(_) => reason.setText("Checkpoint load admitted")
              finally held.close()
            }(e => { held.close(); if !closed then reason.setText(Fx.message(e)) })
        })
        insert.setOnAction(_ => current.foreach { owner =>
          owner.lease.retain() match
            case Left(e) => reason.setText(e)
            case Right(held) => Fx.onFx(actions.insert(row)) { result =>
              try
                if !closed && getItem == row then result match
                  case Left(e) => reason.setText(e)
                  case Right(_) => reason.setText("Inserted")
              finally held.close()
            }(e => { held.close(); if !closed then reason.setText(Fx.message(e)) })
        })
        choose.setOnAction(_ => {
          val chooser = new FileChooser(); chooser.setTitle("Choose local preview")
          Option(chooser.showOpenDialog(stage)).foreach { file =>
            current.foreach { owner => owner.lease.retain() match
              case Left(e) => reason.setText(e)
              case Right(held) =>
                Fx.async(ModelPreviewResolver.validateExplicit(file.toPath)) { result =>
                  try
                    if !closed && current.contains(owner) && getItem == row then result match
                      case Left(e) => reason.setText(e)
                      case Right(ref) => settings.saveAssociation(row.id, ref.localPath.toString) match
                        case Left(e) => reason.setText(e)
                        case Right(_) => refresh()
                  finally held.close()
                }(e => { held.close(); if !closed then reason.setText(Fx.message(e)) })
            }
          }
        })
        remove.setOnAction(_ => settings.removeAssociation(row.id) match
          case Left(e) => reason.setText(e)
          case Right(_) => refresh())
        setGraphic(box)
