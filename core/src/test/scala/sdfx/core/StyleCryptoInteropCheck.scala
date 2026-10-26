package sdfx.core

import sdfx.core.model.AesCryptV2
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets.UTF_8

/** Parent-run CPU helper: write native encrypted files for independent pyAesCrypt decryption.
  * core/Test/runMain sdfx.core.StyleCryptoInteropCheck outputs/styles-crypto-interop
  */
object StyleCryptoInteropCheck:
  def main(args: Array[String]): Unit =
    val out = Paths.get(args.headOption.getOrElse("outputs/styles-crypto-interop"))
    Files.createDirectories(out)
    val cases = Seq(0, 1, 15, 16, 17, 32, 1031).map(n => s"bytes-$n" -> Array.tabulate[Byte](n)(i => (i % 251).toByte)) :+
      ("unicode-csv" -> "name,prompt,negative_prompt\r\n桜,浮世絵,lowres\r\n".getBytes(UTF_8))
    cases.foreach { (name, plain) =>
      Files.write(out.resolve(s"$name.plain"), plain)
      Files.write(out.resolve(s"$name.aes"), AesCryptV2.encrypt(plain, "桜-päss"))
    }
    println(s"Native AES-Crypt v2 candidates: ${out.toAbsolutePath}; verify with styles_crypto_reference_check.py")
