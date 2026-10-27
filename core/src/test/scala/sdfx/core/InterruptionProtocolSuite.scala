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

  test("publication too-late admission is distinct from protection") {
    val raw = ujson.Obj("accepted" -> false, "blocked" -> false, "too_late" -> true,
      "affected" -> false, "protection" -> ujson.Obj("enabled" -> false, "revision" -> 0),
      "message" -> "Output publication is in progress")
    val decision = Json.fromJs[InterruptionDecision](raw)
    assert(decision.tooLate && !decision.blocked && !decision.accepted && !decision.affected)
    assertEquals(Json.fromJs[InterruptionDecision](Json.toJs(decision)), decision)
  }
