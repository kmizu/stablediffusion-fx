package sdfx.engine

class VramEstimatorServiceSuite extends munit.FunSuite:
  def profile(path: String, state: String = "complete"): ujson.Value = ujson.Obj(
    "version" -> 1, "status" -> state, "fingerprint" -> ("a" * 24), "identity" -> ujson.Obj("model" -> "fixture"),
    "profile_path" -> path, "rows" -> ujson.Arr())

  test("partial newer run does not destroy completed profile of same hardware"):
    val complete = profile("old")
    val partial = profile("new", "interrupted")
    assertEquals(VramEstimatorHistory.merge(Seq(complete), partial).size, 2)
    val updated = VramEstimatorHistory.merge(Seq(complete, partial), profile("new"))
    assertEquals(updated.map(_("profile_path").str), Seq("old", "new"))
    assertEquals(updated.last("status").str, "complete")

  test("history retains newest records under limits and owns immutable snapshots"):
    val value = profile("new")
    val merged = VramEstimatorHistory.merge(Seq(profile("one"), profile("two")), value, maxRecords = 2)
    value("status") = "interrupted"
    assertEquals(merged.map(_("profile_path").str), Seq("two", "new"))
    assertEquals(merged.last("status").str, "complete")
    intercept[IllegalArgumentException](VramEstimatorHistory.merge(Nil, ujson.Obj()))
    intercept[IllegalArgumentException](VramEstimatorHistory.merge(Nil, profile("large"), maxBytes = 16))

  test("late completion cannot release a newer benchmark owner"):
    val session = new VramEstimatorSession
    val first = session.start().get
    assertEquals(session.start(), None)
    session.finish(first)
    val second = session.start().get
    session.finish(first)
    assertEquals(session.current, Some(second))
    session.finish(second)
    assertEquals(session.current, None)

  test("empty interrupted calibration does not hide older compatible measurements"):
    val measured = profile("measured")
    measured("rows") = ujson.Arr(ujson.Obj("status" -> "complete"))
    val emptyPartial = profile("empty", "interrupted")
    val foreign = profile("foreign")
    foreign("fingerprint") = "b" * 24
    foreign("rows") = measured("rows")
    val records = Seq(measured, emptyPartial, foreign)
    assertEquals(VramEstimatorHistory.newestMeasured(records, "a" * 24).map(_("profile_path").str), Some("measured"))
    assertEquals(VramEstimatorHistory.newestMeasured(records, "c" * 24), None)
