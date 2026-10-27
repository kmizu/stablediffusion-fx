package sdfx.ui.view

import javafx.geometry.{Insets, Orientation, Pos}
import javafx.scene.control.*
import javafx.scene.image.{Image, ImageView}
import javafx.scene.input.{ClipboardContent, KeyCode, TransferMode}
import javafx.scene.layout.{Priority, StackPane, VBox}
import javafx.stage.FileChooser
import javafx.util.StringConverter
import sdfx.core.gallery.GalleryItem
import sdfx.ui.{Fx, GalleryZipExporter}
import sdfx.ui.vm.{GalleryVM, Mode}

import java.nio.file.Path

/** Thumbnail strip of the outputs (newest first) with search, context menu and keyboard navigation. */
final class GalleryView(ctx: UiContext, vm: GalleryVM, actions: ImageActions, onSelect: GalleryItem => Unit):
  val ThumbSize = 112.0

  // small LRU cache of background-loaded thumbnails
  private val cache = new java.util.LinkedHashMap[Path, Image](256, 0.75f, true):
    override def removeEldestEntry(e: java.util.Map.Entry[Path, Image]): Boolean = size() > 400
  private def thumb(p: Path): Image =
    val c = cache.get(p)
    if c != null then c
    else
      val img = new Image(p.toUri.toString, ThumbSize * 2, ThumbSize * 2, true, true, true)
      cache.put(p, img)
      img

  def evict(p: Path): Unit = cache.remove(p)

  val list = new ListView[GalleryItem](vm.filtered)
  list.setOrientation(Orientation.HORIZONTAL)
  list.getStyleClass.add("gallery")
  list.setPrefHeight(ThumbSize + 26)
  list.setMinHeight(ThumbSize + 26)
  list.setPlaceholder(W.label("No images yet", "muted"))
  list.setFixedCellSize(ThumbSize + 8)
  list.setCellFactory(_ => new Cell)

  private final class Cell extends ListCell[GalleryItem]:
    private val iv = new ImageView()
    iv.setFitWidth(ThumbSize)
    iv.setFitHeight(ThumbSize)
    iv.setPreserveRatio(true)
    iv.setSmooth(true)
    private val box = new StackPane(iv)
    box.setPrefSize(ThumbSize, ThumbSize)
    box.getStyleClass.add("thumb")
    getStyleClass.add("gallery-cell")
    setOnDragDetected(e =>
      Option(getItem).foreach { it =>
        val db = startDragAndDrop(TransferMode.COPY)
        val c = new ClipboardContent()
        c.putFiles(java.util.List.of(it.path.toFile))
        db.setContent(c)
        Option(iv.getImage).filter(i => i.getProgress >= 1 && !i.isError).foreach(i => db.setDragView(i, ThumbSize / 2, ThumbSize / 2))
        e.consume()
      }
    )
    // one tooltip per cell (text updated); the context menu is built only when requested
    private val tip = new Tooltip()
    tip.setWrapText(true)
    tip.setMaxWidth(420)
    setOnContextMenuRequested(e =>
      Option(getItem).foreach(it => menu(it).show(this, e.getScreenX, e.getScreenY))
      e.consume()
    )
    override def updateItem(item: GalleryItem, empty: Boolean): Unit =
      super.updateItem(item, empty)
      setText(null)
      if empty || item == null then
        setGraphic(null)
        setTooltip(null)
      else
        iv.setImage(thumb(item.path))
        iv.setUserData(item.path)
        setGraphic(box)
        tip.setText(item.fileName + item.parameters.map(p => "\n\n" + p.take(400)).getOrElse(""))
        setTooltip(tip)

  private def menu(it: GalleryItem): ContextMenu =
    def mi(text: String)(f: => Unit) =
      val m = new MenuItem(text)
      m.setOnAction(_ => f)
      m
    new ContextMenu(
      mi("Send to img2img")(actions.sendTo(Mode.Img2Img, it.path, it.parameters)),
      mi("Send to Inpaint")(actions.sendTo(Mode.Inpaint, it.path, it.parameters)),
      mi("Compare → A")(actions.compare(false, it.path)),
      mi("Compare → B")(actions.compare(true, it.path)),
      mi("Restore parameters")(it.parameters.fold(ctx.info("This image has no generation parameters"))(actions.restore)),
      new SeparatorMenuItem(),
      mi("Copy image")(ImageActions.copyImage(it.path)),
      mi("Copy parameters")(it.parameters.foreach { p =>
        val c = new ClipboardContent()
        c.putString(p)
        javafx.scene.input.Clipboard.getSystemClipboard.setContent(c)
      }),
      mi("Show in folder")(ImageActions.openFolder(it.path)),
      new SeparatorMenuItem(),
      mi("Delete…")(actions.delete(it.path))
    )

  list.getSelectionModel.selectedItemProperty.addListener((_, _, it) => if it != null then onSelect(it))
  list.setOnKeyPressed(e =>
    if e.getCode == KeyCode.DELETE then Option(list.getSelectionModel.getSelectedItem).foreach(it => actions.delete(it.path))
  )

  private val search = new TextField()
  search.textProperty.bindBidirectional(vm.query)
  search.setPromptText("⌕  Search prompts / file names")
  search.getStyleClass.add("search-field")
  W.tip(search, "Filter the gallery by prompt text or file name (all words must match)")
  private val count = W.label("", "hint")
  val generationChoice = new ComboBox[String](vm.generations)
  generationChoice.setConverter(new StringConverter[String]:
    override def toString(id: String): String =
      if id == null || id.isEmpty then "All images" else s"Generation ${id.take(8)}"
    override def fromString(s: String): String = s
  )
  generationChoice.valueProperty.bindBidirectional(vm.selectedGeneration)
  generationChoice.setPrefWidth(190)
  W.tip(generationChoice, "Choose a generation from this session to view and save all of its individual images")
  val saveZipButton = new Button("Save ZIP…")
  private var exporting = false
  private def refreshExport(): Unit =
    saveZipButton.setDisable(exporting || vm.selectedGenerationPaths.isEmpty)
    saveZipButton.setText(if exporting then "Saving ZIP…" else s"Save ZIP… (${vm.selectedGenerationPaths.size})")
  W.tip(saveZipButton, "Save every individual image from the selected generation, including images hidden by search; excludes grids")
  saveZipButton.setOnAction(_ =>
    val paths = vm.selectedGenerationPaths.toVector
    if paths.nonEmpty && !exporting then
      val chooser = new FileChooser()
      chooser.setTitle("Save generation images as ZIP")
      chooser.getExtensionFilters.add(new FileChooser.ExtensionFilter("ZIP archive", "*.zip"))
      chooser.setInitialFileName(s"generation-${vm.selectedGeneration.get.take(8)}.zip")
      Option(chooser.showSaveDialog(ctx.stage)).foreach { file =>
        exporting = true
        refreshExport()
        Fx.async(GalleryZipExporter.write(paths, file.toPath)) { n =>
          exporting = false
          refreshExport()
          ctx.success(s"Saved $n images to ${file.getName}")
        } { error =>
          exporting = false
          refreshExport()
          ctx.error(s"Cannot save ZIP: ${Fx.message(error)}")
        }
      }
  )
  private def updateCount(): Unit =
    val n = vm.filtered.size
    val total = if Option(vm.selectedGeneration.get).forall(_.isEmpty) then vm.items.size else vm.selectedGenerationPaths.size
    count.setText(if vm.query.get.trim.isEmpty then s"$n image${if n == 1 then "" else "s"}" else s"$n / $total")
    refreshExport()
  vm.filtered.addListener((_: javafx.collections.ListChangeListener.Change[? <: GalleryItem]) => updateCount())
  vm.selectedGeneration.addListener((_, _, _) => updateCount())
  vm.selectedGenerationCount.addListener((_, _, _) => updateCount())
  updateCount()

  val node: VBox =
    val header = W.hbox(8, W.label("Gallery", "pane-title"), count, W.spacer(), W.hgrow(search))
    val tools = W.hbox(8, generationChoice, saveZipButton)
    search.setMaxWidth(320)
    val v = new VBox(6, header, tools, list)
    v.getStyleClass.add("gallery-box")
    v.setPadding(new Insets(6, 10, 8, 10))
    VBox.setVgrow(list, Priority.ALWAYS)
    v

  /** Selects an item without triggering a scroll jump when it is already visible. */
  def select(path: Path): Boolean =
    val i = (0 until vm.filtered.size).find(j => vm.filtered.get(j).path == path)
    i.foreach { j =>
      list.getSelectionModel.select(j)
      list.scrollTo(math.max(0, j - 2))
    }
    i.isDefined

  def selectNeighbor(delta: Int): Unit =
    val i = list.getSelectionModel.getSelectedIndex
    val j = (if i < 0 then 0 else i + delta).max(0).min(vm.filtered.size - 1)
    if j >= 0 && j != i then
      list.getSelectionModel.select(j)
      list.scrollTo(math.max(0, j - 2))
