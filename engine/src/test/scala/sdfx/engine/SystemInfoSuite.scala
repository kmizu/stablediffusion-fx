package sdfx.engine

class SystemInfoSuite extends munit.FunSuite:
  test("benchmark history replaces an identical system fingerprint and preserves changed systems"):
    val a = ujson.Obj("fingerprint" -> "one", "timestamp" -> "old")
    val b = ujson.Obj("fingerprint" -> "two", "timestamp" -> "other")
    val updated = ujson.Obj("fingerprint" -> "one", "timestamp" -> "new")
    assertEquals(SystemInfoService.merge(Seq(a, b), updated), Seq(updated, b))
    val third = ujson.Obj("fingerprint" -> "three")
    assertEquals(SystemInfoService.merge(Seq(a, b), third), Seq(a, b, third))
    intercept[IllegalArgumentException](SystemInfoService.merge(Nil, ujson.Obj()))
