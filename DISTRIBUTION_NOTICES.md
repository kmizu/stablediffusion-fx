# JVM binary distribution notices

The packaged application includes the following JVM libraries. The corresponding upstream license and notice texts are preserved as downloaded in `licenses/`; the table records each original source and the SHA-256 of its bytes. `THIRD_PARTY_NOTICES.md` covers the separately listed plugins. The bundled runtime has its own supplied legal files in the runtime image.

| Shipped artifacts | Version | Pinned upstream source |
| --- | --- | --- |
| `org.openjfx:javafx-base`, `javafx-graphics`, `javafx-controls`, `javafx-swing` (Windows) | 21.0.6 | [OpenJFX 21.0.6+3](https://github.com/openjdk/jfx21u/tree/5ad442749972f73caf4aa320106d9a25e866addc) |
| `org.scala-lang:scala3-library_3` | 3.3.6 | [Scala 3](https://github.com/scala/scala3/tree/7c79db9bb70b1049b7dcf9a799d6fc90a357ce3d) |
| `org.scala-lang:scala-library` | 2.13.16 | [Scala 2](https://github.com/scala/scala/tree/3f6bdaeafde17d790023cc3f299b81eaaf876ca3) |
| `com.lihaoyi:upickle_3`, `ujson_3`, `upack_3`, `upickle-implicits_3`, `upickle-core_3` | 4.1.0 | [uPickle](https://github.com/com-lihaoyi/upickle/tree/77d6f785fdf3d0054d5db06ab2714c67f02d77c3) |
| `com.lihaoyi:geny_3` | 1.1.1 | [geny](https://github.com/com-lihaoyi/geny/tree/9b290357503e5601a650bc469a072bb9a2266efb) |

The OpenJFX source contains additional module legal texts for graphics, included below; the corresponding base, controls and swing source directories contain no `src/main/legal` files at this revision. Media and web are not shipped.

## Original texts and SHA-256

| Project | Local file | Original pinned URL | SHA-256 |
| --- | --- | --- | --- |
| OpenJFX | [`JVM__openjfx_LICENSE.txt`](licenses/JVM__openjfx_LICENSE.txt) | [original](https://raw.githubusercontent.com/openjdk/jfx21u/5ad442749972f73caf4aa320106d9a25e866addc/LICENSE) | `4b9abebc4338048a7c2dc184e9f800deb349366bdf28eb23c2677a77b4c87726` |
| OpenJFX | [`JVM__openjfx_ADDITIONAL_LICENSE_INFO.txt`](licenses/JVM__openjfx_ADDITIONAL_LICENSE_INFO.txt) | [original](https://raw.githubusercontent.com/openjdk/jfx21u/5ad442749972f73caf4aa320106d9a25e866addc/ADDITIONAL_LICENSE_INFO) | `a69bce275ba7a3570af6579cb0f55682cd75fedfcd49e0e8e9022270c447c916` |
| OpenJFX | [`JVM__openjfx_ASSEMBLY_EXCEPTION.txt`](licenses/JVM__openjfx_ASSEMBLY_EXCEPTION.txt) | [original](https://raw.githubusercontent.com/openjdk/jfx21u/5ad442749972f73caf4aa320106d9a25e866addc/ASSEMBLY_EXCEPTION) | `a44eb7b5caf5534c6ef536b21edb40b4d6babf91bf97d9d45596868618b2c6fb` |
| OpenJFX | [`JVM__openjfx_jpeg_fx.txt`](licenses/JVM__openjfx_jpeg_fx.txt) | [original](https://raw.githubusercontent.com/openjdk/jfx21u/5ad442749972f73caf4aa320106d9a25e866addc/modules/javafx.graphics/src/main/legal/jpeg_fx.md) | `a7bb58d9ed7f118480aaad20c318d1e5d4693ba6a4db5101903065ec5552c8b5` |
| OpenJFX | [`JVM__openjfx_mesa3d.txt`](licenses/JVM__openjfx_mesa3d.txt) | [original](https://raw.githubusercontent.com/openjdk/jfx21u/5ad442749972f73caf4aa320106d9a25e866addc/modules/javafx.graphics/src/main/legal/mesa3d.md) | `63f4e6f75caebbccb95d903fb43e46ac7111b3624d0a34f146b276d7d9e7b152` |
| Scala 3 | [`JVM__scala3_LICENSE.txt`](licenses/JVM__scala3_LICENSE.txt) | [original](https://raw.githubusercontent.com/scala/scala3/7c79db9bb70b1049b7dcf9a799d6fc90a357ce3d/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| Scala 2 | [`JVM__scala2_LICENSE.txt`](licenses/JVM__scala2_LICENSE.txt) | [original](https://raw.githubusercontent.com/scala/scala/3f6bdaeafde17d790023cc3f299b81eaaf876ca3/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| Scala 2 | [`JVM__scala2_NOTICE.txt`](licenses/JVM__scala2_NOTICE.txt) | [original](https://raw.githubusercontent.com/scala/scala/3f6bdaeafde17d790023cc3f299b81eaaf876ca3/NOTICE) | `50ce10d50898ba07e632c8913b34098521416d6eed480d44edca93bd92344a45` |
| uPickle | [`JVM__upickle_LICENSE.txt`](licenses/JVM__upickle_LICENSE.txt) | [original](https://raw.githubusercontent.com/com-lihaoyi/upickle/77d6f785fdf3d0054d5db06ab2714c67f02d77c3/LICENSE) | `fc06ee462a6d140906b364eb86b1f0c86fc9134827f7605df49b84190d1e2120` |
| geny | [`JVM__geny_LICENSE.txt`](licenses/JVM__geny_LICENSE.txt) | [original](https://raw.githubusercontent.com/com-lihaoyi/geny/9b290357503e5601a650bc469a072bb9a2266efb/LICENSE) | `f8a443667d027fd656ab15ad601b7a12489f3f8bbe0ee528b558a0b7fd1863db` |
