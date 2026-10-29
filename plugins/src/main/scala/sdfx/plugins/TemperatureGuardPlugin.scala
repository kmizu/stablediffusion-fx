package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

class TemperatureGuardPlugin extends HookPlugin:
  def id = "gpu_temperature_guard"
  def hook = id
  def name = "GPU temperature guard"
  override def description = "Pause sampling above the sleep temperature, resume below the wake temperature."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Check("print_readings", "Log temperature readings while paused"),
    Choice("provider", "Temperature provider", Seq("NVIDIA", "ROCm", "OpenHardwareMonitor"), "NVIDIA"),
    Text("sensor_id", "OpenHardwareMonitor sensor ID", placeholder = "/amdgpu/0/temperature/0"),
    Note("sensor", "NVIDIA/ROCm use the physical device index. Windows AMD requires a running OpenHardwareMonitor with WMI enabled and its exact GPU sensor ID. No monitor is installed automatically."),
    IntField("device", "GPU device index", 0, 31, 0),
    Slider("sleep_temperature", "Sleep temperature (°C)", 30, 100, 1, 80),
    Slider("wake_temperature", "Wake temperature (°C)", 20, 100, 1, 70),
    Slider("interval", "Minimum check interval (seconds)", 0, 60, 0.1, 1),
    Slider("sleep_time", "Recheck interval (seconds)", 0.1, 60, 0.1, 1),
    IntField("max_sleep", "Maximum pause (seconds; 0 unlimited)", 0, 3600, 0)
  )
  def hookFields(v: Values): ujson.Obj = ujson.Obj(
    "provider" -> v.str("provider", "NVIDIA"),
    "sensor_id" -> v.str("sensor_id"),
    "print_readings" -> v.bool("print_readings"),
    "device" -> v.int("device", 0), "sleep_temperature" -> v.double("sleep_temperature", 80),
    "wake_temperature" -> v.double("wake_temperature", 70), "interval" -> v.double("interval", 1),
    "sleep_time" -> v.double("sleep_time", 1), "max_sleep" -> v.int("max_sleep", 0))
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] =
    o.value.toSeq.map((key, value) => s"GPU temperature $key" -> Hooks.num(value))
