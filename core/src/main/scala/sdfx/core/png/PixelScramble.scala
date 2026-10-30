package sdfx.core.png

import java.awt.image.BufferedImage
import java.awt.color.ColorSpace
import java.awt.image.IndexColorModel
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/** Compatibility with sd-encrypt-image pixel permutations. This is not cryptographic encryption. */
object PixelScramble:
  private def digest(s: String): String =
    MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.UTF_8)).map(b => f"${b & 255}%02x").mkString

  def passwordMarker(password: String): String = digest(digest(password) + "Encrypt")
  def verifyPassword(password: String, marker: Option[String]): Option[Boolean] = marker.map { m =>
    m.matches("[0-9a-f]{64}") && MessageDigest.isEqual(passwordMarker(password).getBytes(StandardCharsets.US_ASCII), m.getBytes(StandardCharsets.US_ASCII))
  }

  private def indices(length: Int, key: String, version: Int): Array[Int] =
    val seed = digest(key)
    val doubled = seed + seed
    val arr = Array.tabulate(length)(identity)
    for i <- arr.indices do
      val offset = i % seed.length
      val target = (java.lang.Long.parseLong(doubled.substring(offset, offset + 8), 16) % (length - i)).toInt
      val source = if version == 3 then length - i - 1 else i
      val tmp = arr(source); arr(source) = arr(target); arr(target) = tmp
    arr

  private def transform(image: BufferedImage, password: String, version: Int, reverse: Boolean): BufferedImage =
    require(image != null && password != null && password.nonEmpty, "image and nonempty password required")
    require(version >= 1 && version <= 3, "unsupported pixel scramble version")
    val color = image.getColorModel
    val rgb = !color.isInstanceOf[IndexColorModel] && color.getColorSpace.getType == ColorSpace.TYPE_RGB &&
      color.getNumColorComponents == 3 && (color.getNumComponents == 3 || color.getNumComponents == 4) &&
      color.getComponentSize.forall(bits => bits == 8 || bits == 16)
    val legacySingleChannel = reverse && version == 1 && image.getRaster.getNumBands == 1 &&
      (color.isInstanceOf[IndexColorModel] || (color.getColorSpace.getType == ColorSpace.TYPE_GRAY &&
        color.getNumComponents == 1 && color.getComponentSize.forall(bits => bits == 8 || bits == 16)))
    val grayAlpha = !color.isInstanceOf[IndexColorModel] && color.getColorSpace.getType == ColorSpace.TYPE_GRAY &&
      color.getNumColorComponents == 1 && color.hasAlpha && color.getNumComponents == 2 &&
      color.getComponentSize.forall(_ == 8)
    require(rgb || grayAlpha || legacySingleChannel,
      "pixel scrambling requires RGB, RGBA or 8-bit gray-alpha; legacy v1 decoding also supports grayscale and indexed images")
    val w = image.getWidth; val h = image.getHeight
    val key = digest(password)
    val xs = indices(w, key, version); val ys = indices(h, digest(key), version)
    // Permute indices instead of using getRGB, which converts to 8-bit sRGB.
    val data = Array.tabulate(w * h)(identity)
    def swap(a: Int, b: Int): Unit =
      val value = data(a); data(a) = data(b); data(b) = value
    def xsOrder = if reverse then (w - 1 to 0 by -1) else (0 until w)
    def ysOrder = if reverse then (h - 1 to 0 by -1) else (0 until h)
    version match
      case 1 =>
        for x <- xsOrder; y <- ysOrder do swap(y * w + x, ys(y) * w + xs(x))
      case 2 =>
        def rows(): Unit = for y <- ysOrder; x <- 0 until w do swap(y * w + x, ys(y) * w + x)
        def columns(): Unit = for x <- xsOrder; y <- 0 until h do swap(y * w + x, y * w + xs(x))
        if reverse then { columns(); rows() } else { rows(); columns() }
      case 3 =>
        val original = data.clone()
        for y <- 0 until h; x <- 0 until w do
          if reverse then data(ys(y) * w + xs(x)) = original(y * w + x)
          else data(y * w + x) = original(ys(y) * w + xs(x))
    val input = image.getRaster
    val output = input.createCompatibleWritableRaster(w, h)
    var element: Object = null
    for y <- 0 until h; x <- 0 until w do
      val source = data(y * w + x)
      element = input.getDataElements(source % w, source / w, element)
      output.setDataElements(x, y, element)
    new BufferedImage(color, output, color.isAlphaPremultiplied, null)

  def encode(image: BufferedImage, password: String): BufferedImage = transform(image, password, 3, false)
  def decode(image: BufferedImage, password: String, version: Int): BufferedImage = transform(image, password, version, true)
