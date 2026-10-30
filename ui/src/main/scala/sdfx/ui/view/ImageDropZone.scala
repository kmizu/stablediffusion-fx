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
        ctx.temps.rememberInput(p)
        onPath(p)
        true
      case None if image != null =>
        Fx.async(saveFxImage(ctx, image, prefix))(p => if ctx.temps.acceptInput then onPath(p))
          (e => if ctx.temps.acceptInput then ctx.error(s"Cannot store image: ${Fx.message(e)}"))
        true
      case None => false

  def fromDragboard(ctx: UiContext, db: Dragboard)(onPath: Path => Unit): Boolean =
    acquire(ctx, if db.hasFiles then db.getFiles else null, if db.hasImage then db.getImage else null, "drop-")(onPath)

  def fromClipboard(ctx: UiContext)(onPath: Path => Unit): Boolean =
    val cb = Clipboard.getSystemClipboard
    acquire(ctx, if cb.hasFiles then cb.getFiles else null, if cb.hasImage then cb.getImage else null, "paste-")(onPath)

  /** Blocking: call off the FX thread. */
  def saveFxImage(ctx: UiContext, img: Image, prefix: String): Path =
    val f = ctx.temps.createInputTemp(prefix, ".png")
    try
      if !ImageIO.write(SwingFXUtils.fromFXImage(img, null), "png", f.toFile) then
        throw new java.io.IOException("PNG encoder is unavailable")
      if !ctx.temps.completeInputWrite(f) then throw new java.util.concurrent.CancellationException("Image input closed")
      f
    catch
      case NonFatal(e) =>
        ctx.temps.forgetInput(f)
        try java.nio.file.Files.deleteIfExists(f) catch case NonFatal(_) => ()
        throw e

  /** Background-loading image for a path (errors surface through `Image.isError`). */
  def load(path: String, background: Boolean = true): Option[Image] =
    Option(path).map(_.trim).filter(_.nonEmpty).flatMap(p =>
      try Some(new Image(Paths.get(p).toUri.toString, background))
      catch case NonFatal(_) => None
    )

  final case class DisplayImage(image: Image, passwordVerified: Option[Boolean], scrambled: Boolean)

  /** Blocking: resolve compatible scrambled PNGs in memory; call off the FX thread. */
  def readForDisplay(path: Path, secret: Option[sdfx.core.png.ImageScrambleSecret]): DisplayImage =
    val texts = sdfx.core.png.PngMetadata.readTexts(path)
    if texts.contains("Encrypt") then
      val decoded = sdfx.core.png.ScrambledPng.read(path, secret.map(_.use(identity)))
      DisplayImage(SwingFXUtils.toFXImage(decoded.image, null), decoded.passwordVerified, true)
    else
      val img = load(path.toString, background = false).filterNot(_.isError)
        .getOrElse(throw new IllegalArgumentException("Image not found or unreadable"))
      DisplayImage(img, None, false)

  /** Blocking thumbnail read, bounded in cache size; call off the FX thread. */
  def readThumbnail(path: Path, secret: Option[sdfx.core.png.ImageScrambleSecret], maxSide: Int): DisplayImage =
    val texts = sdfx.core.png.PngMetadata.readTexts(path)
    if texts.contains("Encrypt") then
      val decoded = sdfx.core.png.ScrambledPng.read(path, secret.map(_.use(identity)))
      val src = decoded.image
      val factor = math.min(1.0, math.min(maxSide.toDouble / src.getWidth, maxSide.toDouble / src.getHeight))
      val scaled = if factor >= 1 then src else
        val w = math.max(1, math.round(src.getWidth * factor).toInt)
        val h = math.max(1, math.round(src.getHeight * factor).toInt)
        val out = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_ARGB)
        val graphics = out.createGraphics()
        try
          graphics.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR)
          graphics.drawImage(src, 0, 0, w, h, null)
        finally graphics.dispose()
        out
      DisplayImage(SwingFXUtils.toFXImage(scaled, null), decoded.passwordVerified, true)
    else
      val image = new Image(path.toUri.toString, maxSide, maxSide, true, true, false)
      if image.isError then throw new IllegalArgumentException("Image not found or unreadable")
      DisplayImage(image, None, false)

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
    if f != null then
      ctx.temps.rememberInput(f.toPath)
      path.set(f.toPath.toString)

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
  private val reads = new sdfx.ui.vm.ImageReadGate[ImageFiles.DisplayImage](ctx.services.imageScrambleSession,
    run => Fx.run(run()), (file, secret) => ImageFiles.readForDisplay(file, secret))(using Fx.bg)
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
    reads.invalidate()
    val file = Option(p).filter(_.trim.nonEmpty).flatMap(s => scala.util.Try(Paths.get(s)).toOption)
    view.setUserData(file.orNull)
    file match
      case None => showPlaceholder(s"⤓\n$hint\nDrop · Ctrl+V · click to open")
      case Some(f) =>
        showPlaceholder("Loading image…")
        reads.request(f) {
          case scala.util.Success(display) if path.get == p && view.getUserData == f =>
            val img = display.image
            view.setImage(img)
            placeholder.setVisible(false)
            clearBtn.setVisible(true)
            val dimensions = f"${img.getWidth}%.0f×${img.getHeight}%.0f"
            info.setText(if display.scrambled && display.passwordVerified.isEmpty then dimensions + " · password unverified" else dimensions)
          case scala.util.Failure(error) if path.get == p && view.getUserData == f =>
            showPlaceholder("Cannot display image: " + Fx.message(error).stripPrefix("requirement failed: "))
          case _ => ()
        }

  def refreshImageAccess(): Unit = refresh(path.get)

  path.addListener((_, _, p) =>
    Option(p).filter(_.nonEmpty).flatMap(s => scala.util.Try(Paths.get(s)).toOption).foreach(ctx.temps.rememberInput)
    refresh(p))
  refresh(path.get)
  ctx.watchImageAccess(node)(() => refreshImageAccess())
