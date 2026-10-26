package sdfx.plugins

import sdfx.core.plugins.api.*
import sdfx.core.plugins.api.Control.*

/** Native recorder inspired by AlUlkesh/sd_save_intermediate_images (Unlicense). */
class SaveIntermediatePlugin extends HookPlugin:
  def id = "save_intermediate"
  def name = "Save intermediate images"
  def hook = id
  override def description = "Keep the denoising journey as PNG frames or a video, separately from final images."
  override def controls: Seq[Control] = Seq(
    Check("enabled", "Enabled"),
    Choice("image_type", "Frame subject", Seq("Denoised", "Noisy", "Preview"), "Denoised"),
    Note("subject_help", "Denoised estimates the clean image; Noisy keeps the current sampling noise. Preview uses your preview decoder. Full VAE frames cost extra time."),
    IntField("every_n", "Save every N steps", 1, 1000, 5),
    IntField("start_at", "Start at step (0 = beginning)", 0, 10000, 0),
    IntField("stop_at", "Stop at step (0 = no limit)", 0, 10000, 0),
    Note("clock_help", "Base and hires steps share one clock, restarted per batch iteration. Start/stop round down to multiples of N. Stop limits saving; generation continues."),
    Check("final_save", "Also copy the final image"),
    Choice("mode", "Output mode", Seq("Standard", "Suffix", "Video"), "Standard"),
    Note("location", "Frames, videos and a JSON manifest are saved in the output folder's intermediates directory. Your usual preview and saved control values remain available."),
    Section("video_options", "Video export", expanded = false),
    Note("video_help", "Video mode needs ffmpeg on PATH. Each image has its own video. Frames of different sizes fit a shared canvas. PNG frames are retained."),
    Choice("format", "Video format", Seq("mp4", "webm", "gif"), "mp4"),
    Choice("codec", "MP4 codec", Seq("h264", "h265/hevc", "av1"), "h264"),
    Slider("fps", "Frames per second", 1, 60, 1, 2),
    IntField("add_first_frames", "Hold final image at beginning", 0, 10000, 0),
    IntField("add_last_frames", "Hold final image at end", 0, 10000, 0),
    Check("smooth", "Interpolate motion"),
    Slider("seconds", "Duration (0 = automatic)", 0, 120, 0.5, 0),
    Note("timing_help", "Interpolation needs FPS greater than 1. Its source frames run at 1 FPS or slower: duration is at least the number of selected frames in seconds, including holds. Ordinary mode defaults to frame count / FPS."),
    Check("lores", "Include base pass", true),
    Check("hires", "Include hires pass", true),
    Check("ffmpeg_bat", "Write separate base/hires ffmpeg recipes"),
    Check("bat_only", "Recipes only, skip encoding")
  )
  override def enabledBy: Map[String, String] = controls.filter(c => c.hasValue && c.key != "enabled")
    .map(_.key -> "enabled").toMap
  def hookFields(v: Values): ujson.Obj =
    val o = ujson.Obj()
    Seq("image_type", "mode", "format", "codec").foreach(k => o(k) = ujson.Str(v.str(k, Values.defaults(controls).str(k))))
    Seq("every_n", "start_at", "stop_at", "add_first_frames", "add_last_frames").foreach(k =>
      o(k) = ujson.Num(v.int(k, Values.defaults(controls).int(k))))
    Seq("fps", "seconds").foreach(k => o(k) = ujson.Num(v.double(k, Values.defaults(controls).double(k))))
    Seq("final_save", "smooth", "lores", "hires", "ffmpeg_bat", "bat_only").foreach(k =>
      o(k) = ujson.Bool(v.bool(k, Values.defaults(controls).bool(k))))
    o
  def infotextKeys(o: ujson.Obj): Seq[(String, String)] = Seq("Save intermediate settings" -> o.render())
