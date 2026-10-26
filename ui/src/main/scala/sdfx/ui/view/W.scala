package sdfx.ui.view

import javafx.beans.property.*
import javafx.collections.ObservableList
import javafx.geometry.{Insets, Pos}
import javafx.scene.Node
import javafx.scene.control.*
import javafx.scene.layout.*
import javafx.util.{Duration, StringConverter}
import sdfx.core.plugins.api.Values
import sdfx.ui.vm.NumberSync

/** Small widget factory so every form looks and behaves the same. */
object W:
  def tip[N <: Node](n: N, text: String): N =
    if text.nonEmpty then
      val t = new Tooltip(text)
      t.setShowDelay(Duration.millis(450))
      t.setShowDuration(Duration.seconds(20))
      t.setWrapText(true)
      t.setMaxWidth(420)
      n match
        case c: Control => c.setTooltip(t)
        case _          => Tooltip.install(n, t)
    n

  def label(text: String, classes: String*): Label =
    val l = new Label(text)
    l.getStyleClass.addAll(classes*)
    l

  def button(text: String, tooltip: String = "", classes: Seq[String] = Nil)(action: => Unit): Button =
    val b = new Button(text)
    b.getStyleClass.addAll(classes*)
    b.setOnAction(_ => action)
    tip(b, tooltip)

  def iconButton(symbol: String, tooltip: String)(action: => Unit): Button =
    button(symbol, tooltip, Seq("icon-button"))(action)

  def hgrow[N <: Node](n: N): N =
    HBox.setHgrow(n, Priority.ALWAYS)
    n match
      case r: Region => r.setMaxWidth(Double.MaxValue)
      case _         => ()
    n

  def vgrow[N <: Node](n: N): N =
    VBox.setVgrow(n, Priority.ALWAYS)
    n

  def spacer(): Region = hgrow(new Region())

  def hbox(spacing: Double, nodes: Node*): HBox =
    val h = new HBox(spacing, nodes*)
    h.setAlignment(Pos.CENTER_LEFT)
    h

  def vbox(spacing: Double, nodes: Node*): VBox = new VBox(spacing, nodes*)

  // ---- spinners ------------------------------------------------------------------------------------------

  /** Commits the text typed into an editable spinner (invalid text reverts). */
  def commitSpinner[T](s: Spinner[T]): Unit =
    val vf = s.getValueFactory
    if vf != null && s.isEditable then
      try
        val v = vf.getConverter.fromString(s.getEditor.getText)
        if v != null then vf.setValue(v) else s.getEditor.setText(vf.getConverter.toString(vf.getValue))
      catch case _: Exception => s.getEditor.setText(vf.getConverter.toString(vf.getValue))

  /** Commits the spinner that contains `n` (e.g. the focus owner) before an action reads the model. */
  def commitEnclosingSpinner(n: Node): Unit =
    var cur = n
    while cur != null && !cur.isInstanceOf[Spinner[?]] do cur = cur.getParent
    cur match
      case s: Spinner[?] => commitSpinner(s)
      case _             => ()

  private def commitOnFocusLost[T](s: Spinner[T]): Unit =
    s.focusedProperty.addListener((_, _, focused) => if !focused then commitSpinner(s))

  def intSpinner(p: IntegerProperty, min: Int, max: Int, step: Int = 1, width: Double = 86): Spinner[Integer] =
    val f = new SpinnerValueFactory.IntegerSpinnerValueFactory(min, max, p.get.max(min).min(max), step)
    val s = new Spinner[Integer](f)
    s.setEditable(true)
    s.setPrefWidth(width)
    s.setMinWidth(width)
    NumberSync.bindIntFactory(p, f.valueProperty)
    commitOnFocusLost(s)
    s

  def doubleSpinner(p: DoubleProperty, min: Double, max: Double, step: Double, width: Double = 86): Spinner[java.lang.Double] =
    val f = new SpinnerValueFactory.DoubleSpinnerValueFactory(min, max, p.get.max(min).min(max), step)
    f.setConverter(numberConverter(step))
    val s = new Spinner[java.lang.Double](f)
    s.setEditable(true)
    s.setPrefWidth(width)
    s.setMinWidth(width)
    NumberSync.bindDoubleFactory(p, f.valueProperty)
    commitOnFocusLost(s)
    s

  private def decimalsFor(step: Double): Int =
    val s = Values.fmt(step)
    val i = s.indexOf('.')
    if i < 0 then 0 else s.length - i - 1

  def numberConverter(step: Double): StringConverter[java.lang.Double] =
    val decimals = decimalsFor(step).max(1).min(4)
    new StringConverter[java.lang.Double]:
      def toString(d: java.lang.Double): String =
        if d == null then "" else BigDecimal(d.doubleValue).setScale(decimals, BigDecimal.RoundingMode.HALF_UP).bigDecimal.stripTrailingZeros.toPlainString match
          case s if !s.contains('.') && decimals > 0 => s + ".0"
          case s                                    => s
      def fromString(s: String): java.lang.Double =
        Option(s).map(_.trim.replace(',', '.')).flatMap(_.toDoubleOption).map(java.lang.Double.valueOf).orNull

  // ---- slider + spinner ----------------------------------------------------------------------------------

  def doubleSlider(p: DoubleProperty, min: Double, max: Double, step: Double): HBox =
    val sl = new Slider(min, max, p.get.max(min).min(max))
    sl.setBlockIncrement(step)
    NumberSync.bindDoubleSlider(p, sl.valueProperty, min, max, step)
    val sp = doubleSpinner(p, min, max, step)
    val box = hbox(8, hgrow(sl), sp)
    box.getStyleClass.add("slider-row")
    box

  def intSlider(p: IntegerProperty, min: Int, max: Int, step: Int = 1, spinnerMax: Int = -1): HBox =
    val sl = new Slider(min, max, p.get.max(min).min(max))
    sl.setBlockIncrement(step)
    NumberSync.bindIntSlider(p, sl.valueProperty, min, max, step)
    val sp = intSpinner(p, min, if spinnerMax > 0 then spinnerMax else max, step)
    val box = hbox(8, hgrow(sl), sp)
    box.getStyleClass.add("slider-row")
    box

  // ---- choices -------------------------------------------------------------------------------------------

  def combo(items: ObservableList[String], p: StringProperty): ComboBox[String] =
    val c = new ComboBox[String](items)
    c.setValue(p.get)
    c.valueProperty.addListener((_, _, v) => if v != null && v != p.get then p.set(v))
    p.addListener((_, _, v) => if v != c.getValue then c.setValue(v))
    // keep the selection visible when the items are replaced (catalog refresh)
    items.addListener((_: javafx.collections.ListChangeListener.Change[? <: String]) => if c.getValue != p.get then c.setValue(p.get))
    c.setMaxWidth(Double.MaxValue)
    c

  /** Index-valued choice (resize mode, masked content...). */
  def indexCombo(options: Seq[String], p: IntegerProperty): ComboBox[String] =
    val c = new ComboBox[String](javafx.collections.FXCollections.observableArrayList(options*))
    c.getSelectionModel.select(p.get.max(0).min(options.size - 1))
    c.getSelectionModel.selectedIndexProperty.addListener((_, _, i) => if i.intValue >= 0 && i.intValue != p.get then p.set(i.intValue))
    p.addListener((_, _, v) => if v.intValue != c.getSelectionModel.getSelectedIndex then c.getSelectionModel.select(v.intValue))
    c.setMaxWidth(Double.MaxValue)
    c

  def check(text: String, p: BooleanProperty, tooltip: String = ""): CheckBox =
    val c = new CheckBox(text)
    c.selectedProperty.bindBidirectional(p)
    tip(c, tooltip)

  def textField(p: StringProperty, prompt: String = ""): TextField =
    val t = new TextField()
    t.textProperty.bindBidirectional(p)
    t.setPromptText(prompt)
    t

  /** Text field editing a Long property (seeds). Invalid input reverts on focus loss. */
  def longField(p: LongProperty, width: Double = 140): TextField =
    val t = new TextField(p.get.toString)
    t.setPrefWidth(width)
    t.textProperty.addListener((_, _, v) => v.trim.toLongOption.foreach(l => if l != p.get then p.set(l)))
    p.addListener((_, _, v) => if t.getText.trim.toLongOption.forall(_ != v.longValue) then t.setText(v.toString))
    t.focusedProperty.addListener((_, _, f) => if !f && t.getText.trim.toLongOption.isEmpty then t.setText(p.get.toString))
    t

  // ---- forms ---------------------------------------------------------------------------------------------

  /** Two column label/control grid. */
  final class Form:
    val grid = new GridPane()
    grid.getStyleClass.add("form")
    grid.setHgap(10)
    grid.setVgap(7)
    private val c0 = new ColumnConstraints()
    // the label column grows to the widest inline label; longer labels go above their control (see add)
    c0.setMinWidth(Region.USE_PREF_SIZE)
    c0.setPrefWidth(Region.USE_COMPUTED_SIZE)
    c0.setHalignment(javafx.geometry.HPos.LEFT)
    private val c1 = new ColumnConstraints()
    c1.setHgrow(Priority.ALWAYS)
    c1.setFillWidth(true)
    grid.getColumnConstraints.addAll(c0, c1)
    // keeps the label column at least 88px so stacked cards line up; longer inline labels widen it
    private val labelSpacer = new Region()
    labelSpacer.setMinWidth(88)
    labelSpacer.setPrefWidth(88)
    labelSpacer.setMaxHeight(0)
    labelSpacer.setMouseTransparent(true)
    grid.add(labelSpacer, 0, 0)
    private var row = 0

    def add(label: String, node: Node, tooltip: String = ""): Form =
      val l = new Label(label)
      l.disableProperty.bind(node.disableProperty) // a greyed-out control greys its label too
      l.getStyleClass.add("form-label")
      l.setMinWidth(Region.USE_PREF_SIZE)
      if tooltip.nonEmpty then
        tip(l, tooltip)
        node match
          case c: Control if c.getTooltip == null => tip(c, tooltip)
          case _                                  => ()
      node match
        case r: Region => r.setMaxWidth(Double.MaxValue)
        case _         => ()
      if label.length > Form.MaxInlineLabel then
        l.setWrapText(true)
        l.setMinWidth(0)
        l.getStyleClass.add("form-label-stacked")
        val box = new VBox(3, l, node)
        box.setFillWidth(true)
        grid.add(box, 0, row, 2, 1)
        GridPane.setHgrow(box, Priority.ALWAYS)
      else
        grid.add(l, 0, row)
        grid.add(node, 1, row)
        GridPane.setHgrow(node, Priority.ALWAYS)
      row += 1
      this

    def full(node: Node): Form =
      grid.add(node, 0, row, 2, 1)
      GridPane.setHgrow(node, Priority.ALWAYS)
      row += 1
      this

  object Form:
    /** Labels longer than this are put above their control instead of in the label column. */
    val MaxInlineLabel = 18

  def form(): Form = new Form

  /** Collapsible section; its expanded state is persisted under `key`. */
  def section(title: String, content: Node, expanded: Boolean, tooltip: String = ""): TitledPane =
    val t = new TitledPane(title, content)
    t.setExpanded(expanded)
    t.setAnimated(false)
    t.getStyleClass.add("section")
    tip(t, tooltip)

  def card(title: String, content: Node*): VBox =
    val header = label(title, "card-title")
    val v = new VBox(8)
    v.getChildren.add(header)
    v.getChildren.addAll(content*)
    v.getStyleClass.add("card")
    v.setPadding(new Insets(10, 12, 12, 12))
    v
