package sdfx.core

import sdfx.core.model.AesCryptV2
import java.nio.file.{Files, Paths}

class StyleBackupCipherSuite extends munit.FunSuite:
  test("decrypts an independently generated external pyAesCrypt v2 Unicode-password fixture") {
    val root = Paths.get("core/src/test/resources/styles")
    val encrypted = Files.readAllBytes(root.resolve("pyaescrypt-v2.csv.aes"))
    val plain = Files.readAllBytes(root.resolve("pyaescrypt-v2.csv"))
    assertEquals(AesCryptV2.decrypt(encrypted, "桜-päss").toSeq, plain.toSeq)
    intercept[IllegalArgumentException](AesCryptV2.decrypt(encrypted, "wrong"))
    val corrupted = encrypted.clone()
    corrupted(corrupted.length - 40) = (corrupted(corrupted.length - 40) ^ 1).toByte
    intercept[IllegalArgumentException](AesCryptV2.decrypt(corrupted, "桜-päss"))
  }

  test("v2 backup encryption handles zero/block/partial lengths, is randomized and rejects truncation") {
    Seq(0, 1, 15, 16, 17, 32, 1031).foreach { n =>
      val plain = Array.tabulate[Byte](n)(i => (i % 251).toByte)
      val a = AesCryptV2.encrypt(plain, "pass")
      val b = AesCryptV2.encrypt(plain, "pass")
      assert(a.take(5).sameElements(Array[Byte](65, 69, 83, 2, 0)))
      assert(!a.sameElements(b))
      assertEquals(AesCryptV2.decrypt(a, "pass").toSeq, plain.toSeq)
      intercept[IllegalArgumentException](AesCryptV2.decrypt(a.take(15), "pass"))
    }
  }
