package sdfx.ui.vm

import sdfx.engine.{CaptureOptions, Job}
import sdfx.core.payload.SubmittedPayloadContext

/** Builds queue jobs from a tab's form and plugin state. */
object JobBuilder:
  def build(vm: GenerationVM, plugins: PluginsVM, captureOptions: CaptureOptions = CaptureOptions()): Job =
    Job(Job.newId(), vm.toParams, plugins.scriptJob, plugins.alwaysOnJob, origin = "ui",
      submittedContext = Some(SubmittedPayloadContext("ui", Some(vm.mode.key), Some(plugins.payloadConfigurationJson()))),
      captureOptions = Some(captureOptions))
