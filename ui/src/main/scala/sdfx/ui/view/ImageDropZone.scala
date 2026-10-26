package sdfx.ui.view

import javafx.beans.property.StringProperty
import javafx.embed.swing.SwingFXUtils
import javafx.geometry.{Insets, Pos}
import javafx.scene.Node
import javafx.scene.control.{ContextMenu, Label, MenuItem}
import javafx.scene.image.{Image, ImageView}
import javafx.scene.input.{Clipboard, DragEvent, Dragboard, MouseButton, TransferMode}
import javafx.scene.layout.{HBox, StackPane}
import javafx.stage.FileChooser
import sdfx.ui.Fx

import java.nio.file.{Path, Paths}
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters.*
import scala.util.control.NonFatal

object ImageFiles:
  val Extensions: Seq[String] = Seq("png", "jpg", "jpeg", "webp", "bmp", "gif")
  def isImage(p: Path): Boolean = Extensions.exists(e => p.getFileName.toString.toLowerCase.endsWith("." + e))

  def chooser(title: String): FileChooser =
    val fc = new FileChooser()
    fc.setTitle(title)
    fc.getExtensionFilters.add(new FileChooser.ExtensionFilter("Images", Extensions.map("*." + _).asJava))
    fc

  /** An image file of a drag / clipboard is delivered at once; raw image data is PNG-encoded into the temp dir
    * on a background thread first. Returns false when there is no image at all.
    */
  def acquire(ctx: UiContext, files: java.util.List[java.io.File], image: Image, prefix: String)(onPath: Path => Unit): Boolean =
    Option(files).map(_.asScala.map(_.toPath).find(isImage)).flatten match
      case Some(p) =>
        onPath(p)
        true
      case None if image != null =>
        Fx.async(saveFxImage(ctx, image, prefix))(onPath)(e => ctx.error(s"Cannot store image: ${Fx.message(e)}"))
        true
      case None => false

  def fromDragboard(ctx: UiContext, db: Dragboard)(onPath: Path => Unit): Boolean =
    acquire(ctx, if db.hasFiles then db.getFiles else null, if db.hasImage then db.getImage else null, "drop-")(onPath)

  def fromClipboard(ctx: UiContext)(onPath: Path => Unit): Boolean =
    val cb = Clipboard.getSystemClipboard
    acquire(ctx, if cb.hasFiles then cb.getFiles else null, if cb.hasImage then cb.getImage else null, "paste-")(onPath)

  /** Blocking: call off the FX thread. */
  def saveFxImage(ctx: UiContext, img: Image, prefix: String): Path =
    val f = ctx.tmpFile(prefix, ".png")
    ImageIO.write(SwingFXUtils.fromFXImage(img, null), "png", f.toFile)
    f

  /** Background-loading image for a path (errors surface through `Image.isError`). */
  def load(path: String, background: Boolean = true): Option[Image] =
    Option(path).map(_.trim).filter(_.nonEmpty).flatMap(p =>
      try Some(new Image(Paths.get(p).toUri.toString, background))
      catch case NonFatal(_) => None
    )

  /** Runs `f` once the image finished loading (successfully or not). */
  def whenLoaded(img: Image)(f: => Unit): Unit =
    if img.getProgress >= 1 || img.isError then f
    else
      var done = false
      def fire(): Unit = if !done then { done = true; Fx.run(f) }
      img.progressProperty.addListener((_, _, v) => if v.doubleValue >= 1 then fire())
      img.errorProperty.addListener((_, _, e) => if e then fire())

/** Headless image input logic (drag & drop, Ctrl+V paste, open dialog) bound to a path property. */
final class ImageInput(ctx: UiContext, path: StringProperty):
  def install(target: Node): Unit =
    target.setOnDragOver((e: DragEvent) =>
      if e.getDragboard.hasFiles || e.getDragboard.hasImage then
        e.acceptTransferModes(TransferMode.COPY, TransferMode.LINK)
        if !target.getStyleClass.contains("drag-over") then target.getStyleClass.add("drag-over")
      e.consume()
    )
    target.setOnDragExited(_ => target.getStyleClass.remove("drag-over"))
    target.setOnDragDropped((e: DragEvent) =>
      val ok =
        try ImageFiles.fromDragboard(ctx, e.getDragboard)(p => path.set(p.toString))
        catch
          case NonFatal(ex) =>
            ctx.error(s"Cannot read dropped image: ${ex.getMessage}")
            false
      e.setDropCompleted(ok)
      e.consume()
    )

  def open(): Unit =
    val f = ImageFiles.chooser("Open image").showOpenDialog(ctx.stage)
    if f != null then path.set(f.toPath.toString)

  /** Pastes an image or image file from the clipboard; returns false when there is none. */
  def paste(): Boolean =
    try
      val ok = ImageFiles.fromClipboard(ctx)(p => path.set(p.toString))
      if !ok then ctx.info("The clipboard holds no image")
      ok
    catch
      case NonFatal(e) =>
        ctx.error(s"Cannot paste image: ${e.getMessage}")
        false

/** Image input: drag & drop (files, gallery thumbnails, browser images), Ctrl+V paste, click to open. */
final class ImageDropZone(ctx: UiContext, path: StringProperty, hint: String, height: Double = 260):
  private val input = new ImageInput(ctx, path)
  private val view = new ImageView()
  view.setPreserveRatio(true)
  view.setSmooth(true)
  private val placeholder = new Label(s"⤓\n$hint\nDrop · Ctrl+V · click to open")
  placeholder.getStyleClass.add("drop-hint")
  private val info = W.label("", "image-info")
  info.visibleProperty.bind(info.textProperty.isNotEmpty)
  private val clearBtn = W.iconButton("✕", "Remove the image")(path.set(""))
  private val openBtn = W.iconButton("…", "Open an image file")(open())
  private val tools = new HBox(4, info, openBtn, clearBtn)
  tools.setAlignment(Pos.CENTER_RIGHT)
  tools.setPickOnBounds(false)
  tools.setMaxHeight(javafx.scene.layout.Region.USE_PREF_SIZE)
  tools.setMaxWidth(javafx.scene.layout.Region.USE_PREF_SIZE)

  val node = new StackPane(placeholder, view, tools)
  node.getStyleClass.add("drop-zone")
  node.setMinHeight(height)
  node.setMinWidth(0)
  node.setPrefHeight(height)
  StackPane.setAlignment(tools, Pos.TOP_RIGHT)
  StackPane.setMargin(tools, new Insets(6))
  view.fitWidthProperty.bind(node.widthProperty.subtract(16))
  view.fitHeightProperty.bind(node.heightProperty.subtract(16))
  W.tip(node, s"$hint. Drag a file or a gallery image here, paste with Ctrl+V, or click to browse.")
  input.install(node)

  node.setOnMouseClicked(e =>
    if e.getButton == MouseButton.PRIMARY && e.getClickCount == 1 && view.getImage == null then open()
  )
  private val menu = new ContextMenu(
    item("Open…")(open()),
    item("Paste (Ctrl+V)")(paste()),
    item("Clear")(path.set(""))
  )
  node.setOnContextMenuRequested(e => menu.show(node, e.getScreenX, e.getScreenY))

  private def item(text: String)(f: => Unit): MenuItem =
    val m = new MenuItem(text)
    m.setOnAction(_ => f)
    m

  def open(): Unit = input.open()
  def paste(): Boolean = input.paste()

  private def showPlaceholder(text: String): Unit =
    view.setImage(null)
    placeholder.setText(text)
    placeholder.setVisible(true)
    clearBtn.setVisible(Option(path.get).exists(_.trim.nonEmpty))
    info.setText("")

  private def refresh(p: String): Unit =
    view.setUserData(Option(p).filter(_.nonEmpty).flatMap(s => scala.util.Try(Paths.get(s)).toOption).orNull)
    ImageFiles.load(p) match
      case None => showPlaceholder(s"⤓\n$hint\nDrop · Ctrl+V · click to open")
      case Some(img) =>
        view.setImage(img)
        placeholder.setVisible(false)
        clearBtn.setVisible(true)
        info.setText("")
        ImageFiles.whenLoaded(img) {
          if path.get == p then
            if img.isError then showPlaceholder(s"⚠\nImage not found or unreadable\n$p")
            else info.setText(f"${img.getWidth}%.0f×${img.getHeight}%.0f")
        }

  path.addListener((_, _, p) => refresh(p))
  refresh(path.get)
