package sdfx.ui.vm

import javafx.beans.property.{DoubleProperty, IntegerProperty, ObjectProperty}

/** Two-way syncing between model properties and slider / spinner value properties (no toolkit needed).
  * A slider only *shows* the clamped value: model values outside its range (e.g. a typed 3000 with a
  * 2048 slider) are kept, because slider changes caused by the sync itself are ignored.
  */
object NumberSync:
  def roundTo(v: Double, step: Double, min: Double): Double =
    if step <= 0 then v else BigDecimal(min + math.round((v - min) / step) * step).setScale(6, BigDecimal.RoundingMode.HALF_UP).toDouble

  def bindIntSlider(model: IntegerProperty, slider: DoubleProperty, min: Int, max: Int, step: Int): Unit =
    var syncing = false
    def push(v: Int): Unit =
      syncing = true
      try slider.set(v.max(min).min(max).toDouble)
      finally syncing = false
    push(model.get)
    slider.addListener((_, _, v) =>
      if !syncing then
        val r = roundTo(v.doubleValue, step, min).round.toInt
        if r != model.get then model.set(r)
    )
    model.addListener((_, _, v) =>
      val target = v.intValue.max(min).min(max)
      if roundTo(slider.get, step, min).round.toInt != target then push(v.intValue)
    )

  def bindDoubleSlider(model: DoubleProperty, slider: DoubleProperty, min: Double, max: Double, step: Double): Unit =
    var syncing = false
    def push(v: Double): Unit =
      syncing = true
      try slider.set(v.max(min).min(max))
      finally syncing = false
    push(model.get)
    slider.addListener((_, _, v) =>
      if !syncing then
        val r = roundTo(v.doubleValue, step, min)
        if math.abs(r - model.get) > 1e-9 then model.set(r)
    )
    model.addListener((_, _, v) =>
      val target = v.doubleValue.max(min).min(max)
      if math.abs(roundTo(slider.get, step, min) - target) > 1e-9 then push(v.doubleValue)
    )

  /** Spinner value factory <-> model. The factory clamps to its own range (and that clamp is written back). */
  def bindIntFactory(model: IntegerProperty, value: ObjectProperty[Integer]): Unit =
    value.addListener((_, _, v) => if v != null && v.intValue != model.get then model.set(v.intValue))
    model.addListener((_, _, v) => if value.get == null || value.get.intValue != v.intValue then value.set(v.intValue))

  def bindDoubleFactory(model: DoubleProperty, value: ObjectProperty[java.lang.Double]): Unit =
    value.addListener((_, _, v) => if v != null && math.abs(v.doubleValue - model.get) > 1e-12 then model.set(v.doubleValue))
    model.addListener((_, _, v) => if value.get == null || math.abs(value.get.doubleValue - v.doubleValue) > 1e-12 then value.set(v.doubleValue))
