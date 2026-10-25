"""python -m sdfx [--dry-run] [--log-level LEVEL] [--dry-run-delay-ms N]"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="sdfx", description="stablediffusion-fx inference engine (JSON Lines on stdin/stdout)")
    ap.add_argument("--dry-run", action="store_true", help="fake engine: no torch/diffusers, writes placeholder images")
    ap.add_argument("--log-level", default=os.environ.get("SDFX_LOG_LEVEL", "INFO"),
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "debug", "info", "warning", "error"])
    ap.add_argument("--dry-run-delay-ms", type=int, default=int(os.environ.get("SDFX_DRY_RUN_DELAY_MS", "0")),
                    help="sleep per fake sampling step (dry run only)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    os.environ.setdefault("TQDM_DISABLE", "1")
    from .server import main as server_main
    return server_main(dry_run=args.dry_run, delay_ms=args.dry_run_delay_ms)


if __name__ == "__main__":
    sys.exit(main())
