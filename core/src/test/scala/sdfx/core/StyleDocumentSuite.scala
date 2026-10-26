package sdfx.core

import sdfx.core.model.*
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.Files

class StyleDocumentSuite extends munit.FunSuite with TempDir:
  tmp.test("editing one style preserves UTF8 BOM, legacy field order, unknown columns and untouched record bytes") { dir =>
    val path = dir.resolve("styles.csv")
    val source = "\uFEFFnegative_prompt,Name,text,private\r\n\"n,1\",A,\"hello {prompt}\",\"keep \"\"quotes\"\"\"\r\n\r\n\"bad\nline\",B,\"unchanged, quoted\",opaque\r\n"
    Files.writeString(path, source)
    val doc = StyleDocument.read(path)
    val a = doc.entries.find(e => doc.value(e, "name") == "A").get
    val edited = doc.edit(a.id, Map("prompt" -> "桜, \"new\"\nline"))
    val out = new String(edited.bytes, UTF_8)
    assert(out.startsWith("\uFEFFnegative_prompt,Name,text,private\r\n"))
    assert(out.endsWith("\r\n\r\n\"bad\nline\",B,\"unchanged, quoted\",opaque\r\n"))
    assert(out.contains("keep \"\"quotes\"\""))
    assertEquals(Styles.parseCsv(out).head.prompt, "桜, \"new\"\nline")
    assertEquals(new String(doc.bytes, UTF_8), source)
  }

  tmp.test("duplicate names append x without merging, reorder and deletion are identity-based") { dir =>
    val doc = StyleDocument.read(dir.resolve("styles.csv")).append(Map("name" -> "A", "prompt" -> "one"))
      .append(Map("name" -> "A", "prompt" -> "two"))
    assertEquals(doc.promptStyles.map(_.name), Vector("A", "Ax"))
    val dup = doc.duplicate(Seq(doc.entries.head.id))
    assertEquals(dup.promptStyles.map(_.name), Vector("A", "Ax", "Axx"))
    val order = dup.entries.reverse.map(_.id)
    assertEquals(dup.reorder(order).promptStyles.map(_.prompt), Vector("one", "two", "one"))
    assertEquals(dup.delete(Seq(dup.entries(1).id)).promptStyles.map(_.name), Vector("A", "Axx"))
  }

  tmp.test("all-column substring/insensitive/regex filters, invalid regex and literal replace preserve hidden rows") { dir =>
    val doc = StyleDocument.read(dir.resolve("styles.csv")).append(Map("name" -> "A", "prompt" -> "Red, {prompt}", "notes" -> "Japan"))
      .append(Map("name" -> "B", "prompt" -> "blue", "notes" -> "other"))
    assertEquals(doc.filtered("Jap", "Exact match").toOption.get.size, 1)
    assertEquals(doc.filtered("RED", "Case insensitive").toOption.get.size, 1)
    assertEquals(doc.filtered("^blue$", "regex").toOption.get.size, 1)
    assert(doc.filtered("[", "regex").isLeft)
    val replaced = doc.replace("Red", "$literal\\")
    assertEquals(replaced.promptStyles.head.prompt, "$literal\\, {prompt}")
    assertEquals(replaced.promptStyles.last.prompt, "blue")
  }

  tmp.test("malformed UTF8 and unterminated quoted CSV are rejected without touching source bytes") { dir =>
    val p = dir.resolve("bad.csv")
    val bad = "name,prompt,negative_prompt\nA,\"unterminated,n\n"
    Files.writeString(p, bad)
    intercept[IllegalArgumentException](StyleDocument.read(p))
    assertEquals(Files.readString(p), bad)
    Files.write(p, Array[Byte](0xc3.toByte, 0x28.toByte))
    intercept[Exception](StyleDocument.read(p))
  }

  tmp.test("numeric sort is stable and is not applied when any sort value is invalid") { dir =>
    val doc = StyleDocument.read(dir.resolve("styles.csv")).append(Map("name" -> "A", "sort" -> "2"))
      .append(Map("name" -> "B", "sort" -> "1"))
    assertEquals(doc.numericSort.promptStyles.map(_.name), Vector("B", "A"))
    assertEquals(doc.edit(doc.entries.head.id, Map("sort" -> "oops")).numericSort.promptStyles.map(_.name), Vector("A", "B"))
  }

  tmp.test("ambiguous duplicate column names are rejected for editing without discarding source values") { dir =>
    val file = dir.resolve("styles.csv")
    val raw = "name,prompt,private, PRIVATE \nA,p,one,two\n"
    Files.writeString(file, raw)
    intercept[IllegalArgumentException](StyleDocument.read(file))
    assertEquals(Files.readString(file), raw)
  }
