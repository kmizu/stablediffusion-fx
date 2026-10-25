package sdfx.core

import sdfx.core.model.*

import java.nio.file.Files

class CsvSuite extends munit.FunSuite:

  test("simple rows, CRLF and LF") {
    assertEquals(Csv.parse("a,b,c\r\n1,2,3\n"), Vector(Vector("a", "b", "c"), Vector("1", "2", "3")))
    assertEquals(Csv.parse("a,b"), Vector(Vector("a", "b")))
  }

  test("quoted fields with commas, quotes and newlines") {
    val rows = Csv.parse("name,prompt\n\"x, y\",\"say \"\"hi\"\"\nline2\"\n")
    assertEquals(rows, Vector(Vector("name", "prompt"), Vector("x, y", "say \"hi\"\nline2")))
  }

  test("empty fields and trailing empty field") {
    assertEquals(Csv.parse("a,,c\n,,\n1,2,\n"), Vector(Vector("a", "", "c"), Vector("", "", ""), Vector("1", "2", "")))
  }

  test("blank lines are skipped") {
    assertEquals(Csv.parse("\n\na,b\n\n\nc,d\n\n"), Vector(Vector("a", "b"), Vector("c", "d")))
    assertEquals(Csv.parse(""), Vector.empty)
  }

  test("quote inside an unquoted field is literal (Python csv)") {
    assertEquals(Csv.parse("a \"b\" c,d\n"), Vector(Vector("a \"b\" c", "d")))
  }

  test("quoted empty field") {
    assertEquals(Csv.parse("\"\",x\n"), Vector(Vector("", "x")))
  }

  test("escape") {
    assertEquals(Csv.escape("plain"), "plain")
    assertEquals(Csv.escape(""), "")
    assertEquals(Csv.escape("a,b"), "\"a,b\"")
    assertEquals(Csv.escape("a\"b"), "\"a\"\"b\"")
    assertEquals(Csv.escape("a\nb"), "\"a\nb\"")
    assertEquals(Csv.escape("a\rb"), "\"a\rb\"")
  }

  test("escape -> parse round trip") {
    val row = Vector("x, y", "q\"uo\"te", "multi\r\nline", "", "日本語, テスト", "(masterpiece:1.2)")
    assertEquals(Csv.parse(row.map(Csv.escape).mkString(",") + "\r\n"), Vector(row))
  }

class StylesSuite extends munit.FunSuite with TempDir:

  // As A1111 writes it (csv.DictWriter, QUOTE_MINIMAL, \r\n), plus a hand-edited row.
  val a1111Csv: String =
    "name,prompt,negative_prompt\r\n" +
      "None,,\r\n" +
      "Anime,\"anime style, {prompt}, vibrant colors\",\"photo, realistic\"\r\n" +
      "Quality,\"masterpiece, best quality\",\"lowres, bad anatomy\"\r\n" +
      "Multi,\"line one,\nline two\",\r\n" +
      "Quoted,\"say \"\"cheese\"\", smile\",\r\n" +
      ",orphan prompt,\r\n" +
      "日本語スタイル,\"浮世絵, {prompt}\",低品質\r\n"

  test("parse A1111 styles.csv") {
    val s = Styles.parseCsv(a1111Csv)
    assertEquals(s.map(_.name), Vector("None", "Anime", "Quality", "Multi", "Quoted", "日本語スタイル"))
    assertEquals(s(1), PromptStyle("Anime", "anime style, {prompt}, vibrant colors", "photo, realistic"))
    assertEquals(s(3).prompt, "line one,\nline two")
    assertEquals(s(3).negativePrompt, "")
    assertEquals(s(4).prompt, "say \"cheese\", smile")
    assertEquals(s(5).negativePrompt, "低品質")
  }

  test("parse: BOM (utf-8-sig), column order, extra columns, legacy text column") {
    val bom = "﻿name,prompt,negative_prompt\nA,p,n\n"
    assertEquals(Styles.parseCsv(bom), Vector(PromptStyle("A", "p", "n")))
    val reordered = "negative_prompt, Prompt ,NAME,description\nn,p,A,desc\n"
    assertEquals(Styles.parseCsv(reordered), Vector(PromptStyle("A", "p", "n")))
    val legacy = "name,text\nOld,old style\n"
    assertEquals(Styles.parseCsv(legacy), Vector(PromptStyle("Old", "old style", "")))
    val shortRow = "name,prompt,negative_prompt\nA,p\n"
    assertEquals(Styles.parseCsv(shortRow), Vector(PromptStyle("A", "p", "")))
  }

  test("parse: empty / header-only / no name column") {
    assertEquals(Styles.parseCsv(""), Vector.empty)
    assertEquals(Styles.parseCsv("name,prompt,negative_prompt\r\n"), Vector.empty)
    assertEquals(Styles.parseCsv("foo,bar\n1,2\n"), Vector.empty)
  }

  test("toCsv -> parseCsv round trip") {
    val s = Styles.parseCsv(a1111Csv)
    val csv = Styles.toCsv(s)
    assert(csv.startsWith("name,prompt,negative_prompt\r\n"))
    assert(csv.endsWith("\r\n"))
    assertEquals(Styles.parseCsv(csv), s)
  }

  test("toCsv matches A1111 QUOTE_MINIMAL output") {
    val s = Seq(PromptStyle("Anime", "anime style, {prompt}", "photo"), PromptStyle("Plain", "flat", ""))
    assertEquals(Styles.toCsv(s), "name,prompt,negative_prompt\r\nAnime,\"anime style, {prompt}\",photo\r\nPlain,flat,\r\n")
  }

  tmp.test("load: missing file -> empty, existing file parsed (UTF-8)") { dir =>
    assertEquals(Styles.load(dir.resolve("nope.csv")), Vector.empty)
    val f = dir.resolve("styles.csv")
    Files.writeString(f, a1111Csv)
    assertEquals(Styles.load(f).size, 6)
    assertEquals(Styles.load(f).last.name, "日本語スタイル")
  }

  test("{prompt} placement") {
    val st = PromptStyle("s", "anime style, {prompt}, vibrant", "photo, {prompt}")
    assertEquals(st.applyTo("1girl", "lowres"), ("anime style, 1girl, vibrant", "photo, lowres"))
    // empty user prompt still substitutes
    assertEquals(st.applyTo("", ""), ("anime style, , vibrant", "photo, "))
    // multiple placeholders all replaced
    assertEquals(PromptStyle("d", "{prompt} | {prompt}", "").applyTo("x", "")._1, "x | x")
  }

  test("no {prompt}: appended with ', ' (A1111 merge_prompts strips both sides)") {
    val st = PromptStyle("q", " masterpiece, best quality ", "lowres")
    assertEquals(st.applyTo("1girl  ", ""), ("1girl, masterpiece, best quality", "lowres"))
    assertEquals(st.applyTo("", "bad"), ("masterpiece, best quality", "bad, lowres"))
  }

  test("empty style parts leave text alone (trimmed)") {
    assertEquals(PromptStyle("e", "", "").applyTo("a", "b"), ("a", "b"))
    assertEquals(PromptStyle("e", "", "").applyTo(" a ", ""), ("a", ""))
  }

  test("apply multiple styles in the given order, unknown names ignored") {
    val styles = Styles.parseCsv(a1111Csv)
    val (p, n) = Styles.apply(styles, Seq("Anime", "missing", "Quality"), "1girl", "")
    assertEquals(p, "anime style, 1girl, vibrant colors, masterpiece, best quality")
    assertEquals(n, "photo, realistic, lowres, bad anatomy")
    val (p2, _) = Styles.apply(styles, Seq("Quality", "Anime"), "1girl", "")
    assertEquals(p2, "anime style, 1girl, masterpiece, best quality, vibrant colors")
  }

  test("apply with no names is identity") {
    assertEquals(Styles.apply(Styles.parseCsv(a1111Csv), Nil, " p ", "n"), (" p ", "n"))
  }

  test("apply Japanese style") {
    val styles = Styles.parseCsv(a1111Csv)
    assertEquals(Styles.apply(styles, Seq("日本語スタイル"), "富士山", "ぼやけ"), ("浮世絵, 富士山", "ぼやけ, 低品質"))
  }
