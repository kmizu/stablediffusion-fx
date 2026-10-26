package sdfx.core

import sdfx.core.protocol.*

class InterruptionProtocolSuite extends munit.FunSuite:
  test("reader state and blocked decisions are typed; restart decisions may omit affected") {
    val state = Json.fromJs[InterruptionProtection](ujson.Obj("enabled" -> true, "revision" -> 3))
    assertEquals(state, InterruptionProtection(enabled = true, revision = 3))
    val decision = Json.fromJs[InterruptionDecision](ujson.Obj("accepted" -> false, "blocked" -> true,
      "protection" -> Json.toJs(state), "message" -> "Interruptions are blocked"))
    assert(!decision.accepted && decision.blocked && !decision.affected)
    assertEquals(decision.message, Some("Interruptions are blocked"))
    assertEquals(Json.fromJs[InterruptionDecision](Json.toJs(decision)), decision)
  }
  test("untyped old interrupt response cannot accidentally authorize a destructive action") {
    intercept[Exception](Json.fromJs[InterruptionDecision](ujson.Obj()))
  }
