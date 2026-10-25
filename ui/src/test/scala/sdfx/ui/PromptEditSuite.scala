package sdfx.ui

import sdfx.ui.vm.PromptEdit
import sdfx.ui.vm.PromptEdit.Edit

class PromptEditSuite extends munit.FunSuite:
  private def adj(text: String, s: Int, e: Int, d: Int) = PromptEdit.adjustWeight(text, s, e, d)

  test("bare word under the caret gets wrapped with weight 1.1") {
    // caret inside "cat"
    val r = adj("a cat, dog", 3, 3, 1)
    assertEquals(r.text, "a (cat:1.1), dog")
    assertEquals(r.text.substring(r.selStart, r.selEnd), "cat")
  }

  test("selection is wrapped, trailing spaces excluded") {
    val r = adj("red hair  , blue eyes", 0, 10, 1)
    assertEquals(r.text, "(red hair:1.1)  , blue eyes")
    assertEquals(r.text.substring(r.selStart, r.selEnd), "red hair")
  }

  test("decrease a bare word") {
    assertEquals(adj("cat", 1, 1, -1).text, "(cat:0.9)")
  }

  test("existing weight is adjusted with the caret inside the parentheses") {
    val r = adj("a (cat:1.1), dog", 5, 5, 1)
    assertEquals(r.text, "a (cat:1.2), dog")
    assertEquals(r.text.substring(r.selStart, r.selEnd), "cat")
    assertEquals(adj("a (cat:1.1), dog", 5, 5, -1).text, "a (cat:1.0), dog".replace("(cat:1.0)", "cat"))
  }

  test("adjusting a selected weighted word keeps precision tidy") {
    val r1 = adj("(cat:1.2)", 1, 4, 1)
    assertEquals(r1.text, "(cat:1.3)")
    val r2 = adj("(cat:0.7)", 1, 4, -1)
    assertEquals(r2.text, "(cat:0.6)")
  }

  test("reaching 1.0 removes the parentheses") {
    val r = adj("x, (cat:0.9), y", 5, 5, 1)
    assertEquals(r.text, "x, cat, y")
    assertEquals(r.text.substring(r.selStart, r.selEnd), "cat")
  }

  test("old style (((emphasis))) is converted") {
    val r = adj("((cat))", 2, 5, 1)
    assertEquals(r.text, "(cat:1.3)")
    val sq = adj("[cat]", 1, 4, -1)
    assertEquals(sq.text, "(cat:0.8)")
  }

  test("lora weights step by 0.05") {
    val r = adj("<lora:detail:0.8>", 3, 3, 1)
    assertEquals(r.text, "<lora:detail:0.85>")
    assertEquals(adj("<lora:detail:1>", 3, 3, -1).text, "<lora:detail:0.95>")
  }

  test("multi-word parenthesised block") {
    assertEquals(adj("(red hair:1.3), x", 4, 4, 1).text, "(red hair:1.4), x")
  }

  test("nothing to edit returns the input unchanged") {
    assertEquals(adj("", 0, 0, 1), Edit("", 0, 0))
    assertEquals(adj("a,  ,b", 3, 3, 1), Edit("a,  ,b", 3, 3))
  }

  test("weight formatting") {
    assertEquals(PromptEdit.fmt(1.0), "1.0")
    assertEquals(PromptEdit.fmt(1.1 + 0.1), "1.2")
    assertEquals(PromptEdit.fmt(0.85), "0.85")
  }

  test("token counter and insert") {
    assertEquals(PromptEdit.tokenCounter(""), "0/75")
    assertEquals(PromptEdit.approxTokens("a cat, <lora:x:1>"), 3)
    assertEquals(PromptEdit.tokenCounter((1 to 80).map(_ => "w").mkString(" ")), "80/150")
    assertEquals(PromptEdit.insertAt("a cat", 5, "<lora:x:1.0>").text, "a cat, <lora:x:1.0>")
    assertEquals(PromptEdit.insertAt("", 0, "<lora:x:1.0>").text, "<lora:x:1.0>")
    assertEquals(PromptEdit.insertAt("a cat, ", 7, "t").text, "a cat, t")
    val e = PromptEdit.insertAt("a cat,", 6, "t")
    assertEquals(e.text, "a cat, t")
    assertEquals(e.selStart, 8)
  }
