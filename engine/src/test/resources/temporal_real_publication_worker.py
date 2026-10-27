"""Test-only real worker with a held Temporal Lab publication."""
from pathlib import Path
import sys
import time

sys.path.insert(0, sys.argv[1])
from sdfx.base import BaseEngine
from sdfx.protocol import Emitter
from sdfx.server import Server
from sdfx.temporal_lab_io import OutputTransaction
import sdfx.temporal_lab  # Load NumPy/Pillow before the worker advertises readiness.

ready, hold = Path(sys.argv[2]), Path(sys.argv[3])
trace = ready.with_name("trace")
def mark(value):
    with trace.open("a", encoding="utf-8") as f:
        f.write(value + "\n")

commit = OutputTransaction.commit

def held_commit(transaction):
    ready.write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 20
    while hold.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    if hold.exists():
        raise RuntimeError("publication hold timed out")
    return commit(transaction)

OutputTransaction.commit = held_commit
class FixtureEngine(BaseEngine):
    def temporal_lab(self, params, ctx):
        mark("temporal_lab entered")
        try:
            return super().temporal_lab(params, ctx)
        except BaseException as exc:
            mark(f"temporal_lab exception: {type(exc).__name__}: {exc}")
            raise

class FixtureServer(Server):
    def handle_line(self, line):
        mark("handle_line " + str(line)[:200])
        return super().handle_line(line)

FixtureServer(FixtureEngine(), Emitter(sys.stdout.buffer)).run(sys.stdin.buffer)
