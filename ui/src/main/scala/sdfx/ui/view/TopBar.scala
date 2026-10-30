package sdfx.ui.view

import javafx.geometry.{Insets, Pos}
import javafx.scene.control.*
import javafx.scene.layout.HBox
import sdfx.core.protocol.{MemoryInfo, ModelEntry}
import sdfx.engine.EngineState
import sdfx.ui.vm.ModelVM

/** Checkpoint / VAE / clip skip selection, engine status and settings. */
final class TopBar(
    ctx: UiContext,
    model: ModelVM,
    onCheckpoint: String => Unit,
    onVae: String => Unit,
    onRefresh: () => Unit,
    onRestart: () => Unit,
    onSettings: () => Unit
):
  // ---- checkpoint picker -------------------------------------------------------------------------------
  private val ckptName = W.label("", "picker-name")
  private val ckptBadge = W.label("", "badge")
  private val ckptBtn = new Button()
  ckptBtn.setGraphic(W.hbox(8, ckptName, ckptBadge, W.label("▾", "picker-arrow")))
  ckptBtn.getStyleClass.add("picker")
  ckptBtn.setMinWidth(260)
  ckptBtn.setPrefWidth(340)
  ckptBtn.setMaxWidth(460)
  W.tip(ckptBtn, "Stable Diffusion checkpoint (click to search)")

  private val popup = new SearchPopup[ModelEntry](
    "Checkpoint",
    () => ctx.catalog.snapshot.get.checkpoints,
    m => ctx.checkpointLabel(m.name),
    m => Option(m.kind).filter(_.nonEmpty),
    m => onCheckpoint(m.name),
    "No checkpoints found.\nPut .safetensors in models/checkpoints or add folders in Settings."
  )
  ckptBtn.setOnAction(_ => popup.show(ckptBtn))

  private def kindOf(name: String): Option[String] =
    ctx.catalog.snapshot.get.checkpoints.find(_.name == name).map(_.kind).filter(_.nonEmpty)

  def refreshCheckpoint(): Unit =
    val n = model.checkpoint.get
    ckptName.setText(if n.isEmpty then "Select checkpoint…" else ctx.checkpointLabel(n))
    val k = kindOf(n)
    ckptBadge.setText(k.getOrElse(""))
    ckptBadge.setVisible(k.isDefined)
    ckptBadge.getStyleClass.removeIf(_.startsWith("badge-"))
    k.foreach(x => ckptBadge.getStyleClass.add(s"badge-${x.toLowerCase.replaceAll("[^a-z0-9]", "")}"))
  model.checkpoint.addListener((_, _, _) => refreshCheckpoint())
  ctx.catalog.snapshot.addListener((_, _, _) => refreshCheckpoint())
  ctx.hiresPreferences.addListener((_, _, _) => refreshCheckpoint())
  refreshCheckpoint()

  private val refreshBtn = W.iconButton("⟳", "Rescan model folders")(onRefresh())

  // ---- VAE / clip skip ---------------------------------------------------------------------------------
  private val vae = new ComboBox[String](ctx.catalog.vaes)
  vae.setValue(model.vae.get)
  vae.setPrefWidth(190)
  W.tip(vae, "VAE: Automatic = the checkpoint's own VAE")
  private var syncing = false
  vae.valueProperty.addListener((_, _, v) =>
    if !syncing && v != null && v != model.vae.get then
      model.vae.set(v)
      onVae(v)
  )
  model.vae.addListener((_, _, v) =>
    if v != vae.getValue then
      syncing = true
      try vae.setValue(v)
      finally syncing = false
  )
  ctx.catalog.vaes.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) =>
    syncing = true
    try vae.setValue(model.vae.get)
    finally syncing = false
  )

  private val clip = W.intSpinner(model.clipSkip, 1, 12, 1, 70)
  W.tip(clip, "Clip skip: 1 = normal, 2 is common for anime models")

  // ---- engine status -----------------------------------------------------------------------------------
  private val chip = W.label("● Starting", "chip", "chip-starting")
  private val restartBtn = W.button("⟲ Restart", "Restart the inference engine", Seq("danger-button"))(onRestart())
  restartBtn.setVisible(false)
  restartBtn.setManaged(false)
  private val vram = W.label("", "vram")
  W.tip(vram, "GPU memory used by the engine (allocated / total)")
  private val settingsBtn = W.iconButton("⚙", "Settings")(onSettings())
  private val dryBadge = W.label("DRY RUN", "badge", "badge-dry")
  dryBadge.setVisible(ctx.services.dryRun)
  dryBadge.setManaged(ctx.services.dryRun)
  W.tip(dryBadge, "Dry-run engine: placeholder images, no GPU needed")

  def setState(s: EngineState, loading: Option[String]): Unit =
    chip.getStyleClass.removeIf(_.startsWith("chip-"))
    val (text, cls, tipText) = (s, loading) match
      case (EngineState.Crashed(msg), _) => ("● Engine crashed", "chip-crashed", msg)
      case (EngineState.Stopped, _)      => ("● Stopped", "chip-crashed", "The engine is not running")
      case (EngineState.Starting, _)     => ("● Starting…", "chip-starting", "Starting the Python engine")
      case (_, Some(what))               => (s"● Loading $what", "chip-busy", s"Loading $what")
      case (EngineState.Busy, _)         => ("● Busy", "chip-busy", "The engine is working")
      case (EngineState.Ready(h), _) =>
        val dev = if h.dryRun then "dry run" else h.device
        (s"● Ready · $dev", "chip-ready", s"Engine ${h.engineVersion} · Python ${h.python} · torch ${h.torch.getOrElse("-")} · diffusers ${h.diffusers.getOrElse("-")}")
    chip.setText(text)
    chip.getStyleClass.add(cls)
    W.tip(chip, tipText)
    val crashed = s match
      case EngineState.Crashed(_) | EngineState.Stopped => true
      case _                                            => false
    restartBtn.setVisible(crashed)
    restartBtn.setManaged(crashed)

  def setMemory(m: Option[MemoryInfo]): Unit =
    vram.setText(m.filter(_.totalMb > 0).map(x => f"VRAM ${x.allocatedMb / 1024}%.1f / ${x.totalMb / 1024}%.1f GB").getOrElse(""))

  private val mcpChip = W.label("", "chip", "chip-mcp")
  mcpChip.setVisible(false)
  mcpChip.setManaged(false)

  /** MCP HTTP server status: Some(url) while it listens. */
  def setMcp(url: Option[String]): Unit =
    mcpChip.setVisible(url.isDefined)
    mcpChip.setManaged(url.isDefined)
    url.foreach { u =>
      val port = u.split(':').lastOption.map(_.takeWhile(_.isDigit)).getOrElse("")
      mcpChip.setText(s"⇄ MCP :$port")
      W.tip(mcpChip, s"MCP HTTP server listening on $u (Streamable HTTP). Configure it in Settings › MCP.")
    }

  private val title = W.label("SDNUI", "app-title")
  title.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE)

  val node: HBox =
    val h = new HBox(
      8,
      title,
      W.label("Checkpoint", "bar-label"),
      ckptBtn,
      W.button("Models…", "Browse model previews")(ctx.openModelBrowser("checkpoint", None)),
      refreshBtn,
      W.label("VAE", "bar-label"),
      vae,
      W.label("Clip skip", "bar-label"),
      clip,
      W.spacer(),
      dryBadge,
      vram,
      mcpChip,
      chip,
      restartBtn,
      settingsBtn
    )
    h.setAlignment(Pos.CENTER_LEFT)
    h.getStyleClass.add("top-bar")
    h.setPadding(new Insets(8, 12, 8, 12))
    h
