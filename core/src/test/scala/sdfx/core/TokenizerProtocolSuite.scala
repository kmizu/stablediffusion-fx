package sdfx.core

import sdfx.core.protocol.*

class TokenizerProtocolSuite extends munit.FunSuite:
  test("token inspection decodes snake-case results including SDXL streams and UTF-8 groups"):
    val json = ujson.read("""{"input_mode":"ids","streams":[{"name":"tokenizer_2","source":"clip","count":2,"ids":[3,4],"content_capacity":75,"bos_id":10,"eos_id":11,"pad_id":0,"decoded":"é","tokens":[{"index":1,"id":3,"piece":"Ã","text":"�","bytes_hex":"c3","special":false}],"groups":[{"ids":[3,4],"text":"é"}]}]}""")
    val result = Json.fromJs[TokenizeResult](json)
    assertEquals(result.inputMode, "ids")
    assertEquals(result.streams.head.groups.head.ids, Seq(3, 4))
    assertEquals(result.streams.head.tokens.head.bytesHex, "c3")
    assertEquals(Json.fromJs[TokenizeResult](Json.toJs(result)), result)
