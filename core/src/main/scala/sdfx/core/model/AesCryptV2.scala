package sdfx.core.model

import java.nio.charset.StandardCharsets
import java.security.{MessageDigest, SecureRandom}
import javax.crypto.{Cipher, Mac}
import javax.crypto.spec.{IvParameterSpec, SecretKeySpec}

/** AES Crypt stream format v2, the format used by upstream's optional pyAesCrypt backups.
  * Protocol: https://www.aescrypt.com/aes_stream_format.html (v2); password stretching is described by the
  * primary pyAesCrypt implementation. Uses JDK primitives; no Python dependency or copied reference code.
  * Like the v2 format itself, the final length byte is not authenticated. Ciphertext/key HMACs are verified before
  * releasing plaintext; partial-block padding is also checked. Unsupported stream versions fail explicitly.
  */
object AesCryptV2:
  private val random = new SecureRandom()
  private def fresh(n: Int): Array[Byte] =
    val b = new Array[Byte](n)
    random.nextBytes(b)
    b
  private def key(password: String, iv: Array[Byte]): Array[Byte] =
    require(password.length <= 1024, "AES Crypt password is too long")
    val p = password.getBytes(StandardCharsets.UTF_16LE)
    val sha = MessageDigest.getInstance("SHA-256")
    var digest = iv ++ new Array[Byte](16)
    (0 until 8192).foreach { _ =>
      sha.update(digest)
      digest = sha.digest(p)
    }
    java.util.Arrays.fill(p, 0.toByte)
    digest
  private def aes(mode: Int, data: Array[Byte], key: Array[Byte], iv: Array[Byte]): Array[Byte] =
    val c = Cipher.getInstance("AES/CBC/NoPadding")
    c.init(mode, new SecretKeySpec(key, "AES"), new IvParameterSpec(iv))
    c.doFinal(data)
  private def mac(data: Array[Byte], key: Array[Byte]): Array[Byte] =
    val m = Mac.getInstance("HmacSHA256")
    m.init(new SecretKeySpec(key, "HmacSHA256"))
    m.doFinal(data)
  def encrypt(plain: Array[Byte], password: String): Array[Byte] =
    val outerIv = fresh(16)
    val outerKey = key(password, outerIv)
    val innerIv = fresh(16)
    val innerKey = fresh(32)
    val wrapped = aes(Cipher.ENCRYPT_MODE, innerIv ++ innerKey, outerKey, outerIv)
    val padding = (16 - plain.length % 16) % 16
    val ciphertext = aes(Cipher.ENCRYPT_MODE, plain ++ Array.fill[Byte](padding)(padding.toByte), innerKey, innerIv)
    // The reserved 128-byte extension container is recommended by the v2 format.
    Array[Byte](65, 69, 83, 2, 0, 0, -128) ++ new Array[Byte](128) ++ Array[Byte](0, 0) ++
      outerIv ++ wrapped ++ mac(wrapped, outerKey) ++ ciphertext ++ Array((plain.length % 16).toByte) ++ mac(ciphertext, innerKey)
  def decrypt(data: Array[Byte], password: String): Array[Byte] =
    require(data.length >= 136 && data.take(3).sameElements(Array[Byte](65, 69, 83)), "Not an AES Crypt backup")
    require(data(3) == 2, "Only AES Crypt v2 backups (.aes, pyAesCrypt compatible) are supported")
    var pos = 5
    var done = false
    while !done do
      require(pos + 2 <= data.length, "Truncated AES Crypt extension header")
      val count = (data(pos) & 255) * 256 + (data(pos + 1) & 255)
      pos += 2
      if count == 0 then done = true
      else
        require(count <= data.length - pos, "Truncated AES Crypt extension")
        pos += count
    require(data.length - pos >= 129, "Truncated AES Crypt backup")
    val iv = data.slice(pos, pos + 16)
    val outerKey = key(password, iv)
    val wrapped = data.slice(pos + 16, pos + 64)
    require(MessageDigest.isEqual(mac(wrapped, outerKey), data.slice(pos + 64, pos + 96)), "Wrong backup password or corrupted key")
    val session = aes(Cipher.DECRYPT_MODE, wrapped, outerKey, iv)
    val innerKey = session.drop(16)
    val cipher = data.slice(pos + 96, data.length - 33)
    require(cipher.length % 16 == 0, "Truncated AES Crypt ciphertext")
    require(MessageDigest.isEqual(mac(cipher, innerKey), data.takeRight(32)), "Corrupted AES Crypt backup")
    val lengthMod = data(data.length - 33) & 255
    require(lengthMod < 16, "Invalid AES Crypt length")
    val plain = aes(Cipher.DECRYPT_MODE, cipher, innerKey, session.take(16))
    val padding = (16 - lengthMod) % 16
    require(padding <= plain.length && (padding == 0 || plain.takeRight(padding).forall(_ == padding.toByte)), "Invalid AES Crypt padding")
    plain.dropRight(padding)
