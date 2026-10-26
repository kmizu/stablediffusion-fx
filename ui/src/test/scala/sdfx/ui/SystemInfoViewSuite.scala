package sdfx.ui

import sdfx.ui.view.SystemInfoTab

class SystemInfoViewSuite extends munit.FunSuite:
  test("benchmark scores lead the visible table while configuration remains available"):
    val record = ujson.Obj("timestamp" -> "now", "config" -> ujson.Obj("width" -> 512),
      "results" -> ujson.Arr(ujson.Obj("batch" -> 2, "status" -> "complete", "iterations_per_second" -> 25.319,
        "seconds" -> .3949, "megapixels_per_second" -> .1582)))
    val rows = SystemInfoTab.benchmarkRows(record)
    assertEquals(rows.head._1, "Batch 2")
    assert(rows.head._2.contains("25.32 it/s"))
    assert(rows.head._2.contains("0.39 s"))
    assert(rows.exists(_._1 == "config.width"))
    assert(!rows.exists(_._1.startsWith("results[")))

  test("failed benchmark batch remains visible with its error"):
    val record = ujson.Obj("results" -> ujson.Arr(ujson.Obj("batch" -> 4, "status" -> "oom", "error" -> "out of memory")))
    assertEquals(SystemInfoTab.benchmarkRows(record).head, "Batch 4" -> "oom: out of memory")
