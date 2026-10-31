package sdfx.plugins

import java.nio.file.Files
import sdfx.plugins.prompt.DynamicPrompt

class WildcardPathSafetySuite extends munit.FunSuite:
  test("wildcards remain inside declared real roots and retain provenance") {
    val parent = Files.createTempDirectory("wildcard-safety-")
    val root = Files.createDirectory(parent.resolve("root"))
    val child = Files.createDirectory(root.resolve("nested"))
    Files.writeString(child.resolve("safe.txt"), "allowed\n")
    val outside = parent.resolve("secret.txt"); Files.writeString(outside, "private\n")
    val w = new DynamicPrompt.Wildcards(Seq(root), trackSources = true)
    assertEquals(w.lines("nested/safe"), Vector("allowed"))
    assert(w.sourceHashes.contains("wildcards_file:nested/safe"))
    for name <- Seq("../secret", "nested/../../secret", outside.toString.stripSuffix(".txt"), "\\\\server\\share\\file", "C:\\private\\file") do
      intercept[IllegalArgumentException](w.lines(name))
    val link = root.resolve("leak.txt")
    try
      Files.createSymbolicLink(link, outside)
      intercept[IllegalArgumentException](w.lines("leak"))
      assert(!w.names.contains("leak"))
    catch case _: UnsupportedOperationException | _: java.nio.file.FileSystemException | _: SecurityException => ()
  }
