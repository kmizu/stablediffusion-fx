package sdfx.core.modelpresets

import java.net.URI

class CivitaiRemotePolicySuite extends munit.FunSuite:
  test("official image redirect is allowed without trusting arbitrary hosts or schemes") {
    CivitaiPresetClient.requireRemote(URI.create("https://blobs-b2.civitai.com/file/blobs-managed-public/example"))
    for url <- Vector("http://blobs-b2.civitai.com/file/x", "https://other.civitai.com/x",
        "https://blobs-b2.civitai.com.evil.test/x", "https://user@blobs-b2.civitai.com/x",
        "https://blobs-b2.civitai.com:444/x", "https://127.0.0.1/x") do
      intercept[IllegalArgumentException](CivitaiPresetClient.requireRemote(URI.create(url)))
  }
