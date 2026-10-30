package sdfx.core

import sdfx.core.model.HiresPromptModes

class HiresPromptModesSuite extends munit.FunSuite:
  test("all pinned upstream reference cases preserve exact strings"):
    val input = getClass.getResourceAsStream("/hires-prompt-reference.json")
    val data = try ujson.read(new String(input.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8)) finally input.close()
    data("cases").arr.zipWithIndex.foreach { (row, index) =>
      val result = HiresPromptModes.transform(row("base").str, row("hires").str, row("mode").str,
        row("remove").bool, row("separator").str, row("marker").str)
      assertEquals(result, (row("expected")(0).str, row("expected")(1).str), s"reference case $index")
    }

  test("default chooses explicit hires text before optional raw network removal"):
    assertEquals(HiresPromptModes.transform("cat <lora:x:1> embedding", "", "Default", removeNetworks = true),
      ("cat <lora:x:1> embedding", "cat  embedding"))
    assertEquals(HiresPromptModes.transform("cat <lora:x:1>", "dog", "Default", removeNetworks = true),
      ("cat <lora:x:1>", "dog"))

  test("append and prepend preserve configured separator and blank behavior"):
    assertEquals(HiresPromptModes.transform("cat", "dog", "Append"), ("cat", "cat\ndog"))
    assertEquals(HiresPromptModes.transform("cat", "dog", "Prepend", separator = ", "), ("cat", "dog, cat"))
    assertEquals(HiresPromptModes.transform("cat", "  ", "Append"), ("cat", "  "))

  test("search replace instructions remove insertion markers only from first pass"):
    assertEquals(HiresPromptModes.transform("cat @slot@ red", "@slot@blue\n@red@green", "Prompt S/R"),
      ("cat  red", "cat blue\n green"))

  test("ordered replacements, empty replacement and doubled markers are literal"):
    assertEquals(HiresPromptModes.transform("red red", "@red@blue\n@blue@", "Prompt S/R"), ("red red", "\n \n"))
    assertEquals(HiresPromptModes.transform("a@b", "@a@@b@x@@y", "Prompt S/R"), ("a@b", "x@y"))
    assertEquals(HiresPromptModes.transform("cat #slot#", "#slot#dog", "Prompt S/R", marker = "#"), ("cat ", "cat dog"))

  test("malformed mode and marker fail instead of silently changing text"):
    intercept[IllegalArgumentException](HiresPromptModes.transform("x", "y", "unknown"))
    Seq("", "ab", "a", "_", " ").foreach(marker =>
      intercept[IllegalArgumentException](HiresPromptModes.transform("x", "y", "Prompt S/R", marker = marker)))
