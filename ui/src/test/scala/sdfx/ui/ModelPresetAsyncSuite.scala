package sdfx.ui

import sdfx.ui.presets.ModelPresetAsync
import scala.collection.mutable.ArrayBuffer

class ModelPresetAsyncSuite extends munit.FunSuite:
  test("superseded operation cannot accept a stale result and cancellation is checked before work") {
    val jobs = ArrayBuffer.empty[() => Unit]
    val updates = ArrayBuffer.empty[String]
    val async = new ModelPresetAsync(jobs += _, f => f(), _ => (), e => updates += e.getMessage)
    async.run(check => { check(); "old" })(updates += _)
    async.run(check => { check(); "new" })(updates += _)
    jobs.foreach(_())
    assertEquals(updates.toVector, Vector("new"))
    jobs.clear()
    var wrote = false
    async.run(check => { check(); wrote = true })(_ => ())
    async.cancel()
    jobs.foreach(_())
    assert(!wrote)
    assert(updates.last.toLowerCase.contains("cancel"))
  }
  test("dispose suppresses late completions") {
    val jobs = ArrayBuffer.empty[() => Unit]
    var accepted = false
    val async = new ModelPresetAsync(jobs += _, f => f(), _ => (), _ => ())
    async.run(_ => 1)(_ => accepted = true)
    async.dispose()
    jobs.foreach(_())
    assert(!accepted)
  }

  test("a late cancel after a completed publication does not falsely report it rolled back") {
    val jobs = ArrayBuffer.empty[() => Unit]
    var accepted = false
    val errors = ArrayBuffer.empty[String]
    val async = new ModelPresetAsync(jobs += _, f => f(), _ => (), e => errors += e.getMessage)
    async.run(check => { check(); async.cancel(); "committed" })(_ => accepted = true)
    jobs.foreach(_())
    assert(accepted)
    assert(errors.isEmpty)
  }
