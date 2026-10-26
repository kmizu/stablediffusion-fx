package sdfx.ui.view

import javafx.collections.FXCollections
import javafx.geometry.{Insets, Pos}
import javafx.scene.control.*
import javafx.scene.input.{Clipboard, ClipboardContent}
import javafx.scene.layout.{HBox, Priority, VBox}
import sdfx.ui.vm.{Outcome, ProgressState, ProgressVM}

/** Bottom bar: Generate / Interrupt, Queue, progress of both passes, job status and the collapsible log. */
final class StatusBar(progress: ProgressVM, generate: () => Unit, enqueue: () => Unit, interrupt: () => Unit,
    clearQueue: () => Unit, protection: javafx.scene.Node):
  val generateBtn = new Button("▶  Generate")
  generateBtn.getStyleClass.add("generate-button")
  generateBtn.setDefaultButton(false)
  generateBtn.setOnAction(_ => if progress.get.running then interrupt() else generate())
  W.tip(generateBtn, "Generate (Ctrl+Enter). While running: interrupt (Esc)")

  val queueBtn = new Button("＋ Queue")
  queueBtn.getStyleClass.add("queue-button")
  queueBtn.setOnAction(_ => enqueue())
  W.tip(queueBtn, "Add another job with the current settings to the queue (Ctrl+Shift+Enter)")

  private val clearBtn = W.iconButton("⨯", "Remove all pending jobs from the queue")(clearQueue())

  private val baseBar = new ProgressBar(0)
  private val hiresBar = new ProgressBar(0)
  baseBar.getStyleClass.add("base-bar")
  hiresBar.getStyleClass.add("hires-bar")
  Seq(baseBar, hiresBar).foreach { b =>
    b.setMaxWidth(Double.MaxValue)
    b.setPrefHeight(8)
  }
  W.tip(baseBar, "Sampling progress (first pass)")
  W.tip(hiresBar, "Hires. fix progress (second pass)")
  private val stepText = W.label("", "step-text")
  private val statusText = W.label("Idle", "status-text")

  val logToggle = new ToggleButton("☰ Log")
  logToggle.getStyleClass.add("log-toggle")
  W.tip(logToggle, "Show the engine and job log")

  // ---- log ----------------------------------------------------------------------------------------------
  private val lines = FXCollections.observableArrayList[String]()
  private val logList = new ListView[String](lines)
  logList.getStyleClass.add("log-list")
  logList.setPrefHeight(170)
  private val logCopy = W.button("Copy", "Copy the whole log")({
    val c = new ClipboardContent()
    c.putString(lines.toArray(Array.empty[String]).mkString("\n"))
    Clipboard.getSystemClipboard.setContent(c)
  })
  private val logClear = W.button("Clear", "Clear the log")(lines.clear())
  private val logPane = new VBox(4, W.hbox(6, W.label("Log", "pane-title"), W.spacer(), logCopy, logClear), logList)
  logPane.getStyleClass.add("log-pane")
  logPane.setPadding(new Insets(6, 10, 6, 10))
  logPane.visibleProperty.bind(logToggle.selectedProperty)
  logPane.managedProperty.bind(logToggle.selectedProperty)

  /** Appends a log line (FX thread). Keeps the last 3000 lines. */
  def log(line: String): Unit =
    lines.add(line)
    if lines.size > 3000 then lines.remove(0, lines.size - 3000)
    if logToggle.isSelected then logList.scrollTo(lines.size - 1)

  private def render(s: ProgressState): Unit =
    if s.running then
      generateBtn.setText("■  Interrupt")
      if !generateBtn.getStyleClass.contains("running") then generateBtn.getStyleClass.add("running")
    else
      generateBtn.setText("▶  Generate")
      generateBtn.getStyleClass.remove("running")
    queueBtn.setText(if s.pendingCount > 0 then s"＋ Queue (${s.pendingCount})" else "＋ Queue")
    clearBtn.setVisible(s.pendingCount > 0)
    clearBtn.setManaged(s.pendingCount > 0)
    baseBar.setProgress(if s.running || s.outcome.contains(Outcome.Finished) then s.base.fraction else 0)
    hiresBar.setVisible(s.hiresSeen)
    hiresBar.setManaged(s.hiresSeen)
    hiresBar.setProgress(s.hires.fraction)
    stepText.setText(s.stepText)
    val st = Seq(s.statusText, if s.running && s.statusText.isEmpty then s.message else if !s.running then s.message else "").filter(_.nonEmpty)
    statusText.setText(st.mkString("  ·  "))
    statusText.getStyleClass.removeAll("status-error", "status-ok")
    s.outcome.filter(_ => !s.running).foreach {
      case Outcome.Failed   => statusText.getStyleClass.add("status-error")
      case Outcome.Finished => statusText.getStyleClass.add("status-ok")
      case _                => ()
    }

  progress.state.addListener((_, _, s) => render(s))
  render(progress.get)

  private val bars = new VBox(4, baseBar, hiresBar)
  bars.setAlignment(Pos.CENTER)
  private val texts = new VBox(2, stepText, statusText)
  texts.setAlignment(Pos.CENTER_LEFT)
  texts.setMinWidth(260)
  private val bar = new HBox(10, generateBtn, queueBtn, clearBtn, bars, texts, protection, logToggle)
  HBox.setHgrow(bars, Priority.ALWAYS)
  bar.setAlignment(Pos.CENTER_LEFT)
  bar.getStyleClass.add("status-bar")
  bar.setPadding(new Insets(8, 12, 8, 12))

  val node: VBox = new VBox(logPane, bar)
