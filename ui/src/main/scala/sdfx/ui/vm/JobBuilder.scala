package sdfx.ui.vm

import sdfx.engine.Job

/** Builds queue jobs from a tab's form and plugin state. */
object JobBuilder:
  def build(vm: GenerationVM, plugins: PluginsVM): Job =
    Job(Job.newId(), vm.toParams, plugins.scriptJob, plugins.alwaysOnJob, origin = "ui")
