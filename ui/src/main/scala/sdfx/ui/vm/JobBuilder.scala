package sdfx.ui.vm

import sdfx.engine.{CaptureOptions, Job}
import sdfx.core.payload.SubmittedPayloadContext

/** Builds queue jobs from a tab's form and plugin state. */
object JobBuilder:
  def build(vm: GenerationVM, plugins: PluginsVM, captureOptions: CaptureOptions = CaptureOptions(),
      imageSecret: Option[sdfx.core.png.ImageScrambleSecret] = None): Job =
    Job(Job.newId(), vm.toParams, plugins.scriptJob, plugins.alwaysOnJob, origin = "ui",
      submittedContext = Some(SubmittedPayloadContext("ui", Some(vm.mode.key), Some(plugins.payloadConfigurationJson()),
        Some(vm.variationEnabled.get))),
      captureOptions = Some(captureOptions), imageSecret = imageSecret)
