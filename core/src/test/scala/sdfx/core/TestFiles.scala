package sdfx.core

import java.awt.image.BufferedImage
import java.nio.file.{Files, Path}
import java.util.Comparator

/** Temp directory fixture deleted after each test. */
trait TempDir:
  self: munit.FunSuite =>

  val tmp: FunFixture[Path] = FunFixture[Path](
    setup = t => Files.createTempDirectory("sdfx-test-"),
    teardown = dir => TestFiles.deleteRecursively(dir)
  )

object TestFiles:
  def deleteRecursively(dir: Path): Unit =
    if Files.exists(dir) then
      val s = Files.walk(dir)
      try s.sorted(Comparator.reverseOrder[Path]()).forEach(p => Files.deleteIfExists(p))
      finally s.close()

  def image(w: Int, h: Int, rgb: Int = 0x336699, tpe: Int = BufferedImage.TYPE_INT_RGB): BufferedImage =
    val img = new BufferedImage(w, h, tpe)
    for y <- 0 until h; x <- 0 until w do img.setRGB(x, y, rgb | 0xff000000)
    img

  /** Raw PNG chunk types in file order. */
  def chunkTypes(file: Path): Seq[String] =
    val b = Files.readAllBytes(file)
    val out = Seq.newBuilder[String]
    var i = 8
    while i + 8 <= b.length do
      val len = ((b(i) & 0xff) << 24) | ((b(i + 1) & 0xff) << 16) | ((b(i + 2) & 0xff) << 8) | (b(i + 3) & 0xff)
      out += new String(b, i + 4, 4, "US-ASCII")
      i += 12 + len
    out.result()

  def resource(name: String): Path =
    val url = getClass.getResource("/fixtures/" + name)
    assert(url != null, s"missing fixture $name")
    Path.of(url.toURI)
