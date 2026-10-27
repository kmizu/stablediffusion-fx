package sdfx.ui

import munit.FunSuite
import sdfx.ui.vm.TagCompletion

class TagCompletionSuite extends FunSuite:
  private def replace(text: String, caret: Int, tag: String): String =
    val t = TagCompletion.token(text, caret, caret, caret).get
    text.take(t.start) + tag + text.drop(t.end)

  test("middle token replaced with comma spacing and Japanese context untouched") {
    val text = "桜の風景,  blue ha  , evening"
    assertEquals(replace(text, text.indexOf("ha") + 2, "blue hair"), "桜の風景,  blue hair  , evening")
  }
  test("weighted token retains brackets and weight") {
    assertEquals(replace("(blue ha:1.25), red", 8, "blue hair"), "(blue hair:1.25), red")
    assertEquals(TagCompletion.token("(blue:1.25)", 9, 9, 9), None)
  }
  test("completion replaces token suffix after caret as well") {
    assertEquals(replace("red, blue hxxx, green", 11, "blue hair"), "red, blue hair, green")
  }
  test("cross-token selection and extra networks are suppressed") {
    assertEquals(TagCompletion.token("red, blue", 9, 0, 9), None)
    assertEquals(TagCompletion.token("<lora:blue:1>", 10, 10, 10), None)
    assert(TagCompletion.token("blue hair", 4, 0, 4).nonEmpty)
  }
  test("CSV aliases quoted commas BOM and population ordering") {
    val tags = TagCompletion.parse(Iterator("\ufeffblue_hair,0,12,\"azure_hair,bluehair\"", "blue_eyes,0,42,", "cat,0"))
    assertEquals(TagCompletion.suggestions(tags, "blue").map(_.name), Vector("blue_eyes", "blue_hair"))
    assertEquals(TagCompletion.suggestions(tags, "azure h").map(_.name), Vector("blue_hair"))
    assertEquals(tags.last.insertion, "cat")
    assertEquals(TagCompletion.csvRow("\"a\"\"b\",0"), Vector("a\"b", "0"))
  }
  test("escaped parentheses and newline boundaries") {
    assertEquals(replace("日本語\nblue ha\nnext", 10, "blue hair"), "日本語\nblue hair\nnext")
    val tag = sdfx.ui.vm.CompletionTag("character_(series)", 1, Vector.empty)
    assertEquals(tag.insertion, "character \\(series\\)")
  }
  test("invalid CSV fails with no partial dictionary") {
    intercept[IllegalArgumentException](TagCompletion.parse(Iterator("tag,0", "\"bad,0")))
    intercept[IllegalArgumentException](TagCompletion.parse(Iterator.empty))
  }
  test("header CSV supports WD name/category/count and optional tag_id") {
    val tags = TagCompletion.parse(Iterator("tag_id,name,category,count", "1,blue_hair,0,20", "2,blue_eyes,0,30"))
    assertEquals(tags.map(_.name), Vector("blue_eyes", "blue_hair"))
    assertEquals(TagCompletion.parse(Iterator("name,type,count,aliases", "blue_hair,0,1,azure")).head.aliases, Vector("azure"))
  }
