package sdfx.core.png

import java.awt.image.BufferedImage
import java.nio.file.Path

/** Read/write upstream-compatible scrambled PNGs; password absence is never silently ignored. */
object ScrambledPng:
  final case class Decoded(image: BufferedImage, passwordVerified: Option[Boolean], scrambled: Boolean)
  private val versions = Map("pixel_shuffle" -> 1, "pixel_shuffle_2" -> 2, "pixel_shuffle_3" -> 3)

  def read(path: Path, password: Option[String]): Decoded =
    val texts = PngMetadata.readTexts(path)
    texts.get("Encrypt") match
      case None => Decoded(PngMetadata.readImage(path), None, false)
      case Some(tag) =>
        val version = versions.getOrElse(tag, throw new IllegalArgumentException("Unsupported image scramble marker"))
        val pass = password.filter(_.nonEmpty).getOrElse(throw new IllegalArgumentException("Image password required"))
        val verified = PixelScramble.verifyPassword(pass, texts.get("EncryptPwdSha"))
        require(!verified.contains(false), "Incorrect image password or invalid password marker")
        Decoded(PixelScramble.decode(PngMetadata.readImage(path), pass, version), verified, true)

  def write(image: BufferedImage, path: Path, texts: Seq[(String, String)], password: Option[String], preserveFrom: Option[Path] = None): Unit =
    require(!texts.exists((key, _) => key == "Encrypt" || key == "EncryptPwdSha"), "Reserved image scramble metadata")
    password match
      case None => PngMetadata.write(image, path, texts, preserveFrom)
      case Some(pass) =>
        val encoded = PixelScramble.encode(image, pass)
        PngMetadata.write(encoded, path, texts ++ Seq("Encrypt" -> "pixel_shuffle_3", "EncryptPwdSha" -> PixelScramble.passwordMarker(pass)), preserveFrom)
