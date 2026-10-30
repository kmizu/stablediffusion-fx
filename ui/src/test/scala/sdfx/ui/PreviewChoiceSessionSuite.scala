package sdfx.ui

import sdfx.ui.vm.PreviewChoiceSession

class PreviewChoiceSessionSuite extends munit.FunSuite:
  test("new selection invalidates older decode and closing rejects every late callback"):
    val session = new PreviewChoiceSession(Map("a" -> "a.jpg", "b" -> "b.jpg"))
    val a = session.begin("a").get
    val b = session.begin("b").get
    assert(!session.accept(a._1))
    assert(session.accept(b._1))
    session.close()
    assert(!session.accept(b._1))
    assertEquals(session.begin("a"), None)

  test("no-preview selection invalidates loading and retry gets a fresh lease"):
    val session = new PreviewChoiceSession(Map("a" -> "a.jpg"))
    val a = session.begin("a").get
    assertEquals(session.begin("Not set"), None)
    assert(!session.accept(a._1))
    val retry = session.begin("a").get
    assert(retry._1 > a._1 && session.accept(retry._1))

  test("filter uses labels and case-insensitive Unicode without changing source order"):
    assertEquals(PreviewChoiceSession.filter(Seq("Zebra", "École", "éclair", "cat"), " É "), Seq("École", "éclair"))
