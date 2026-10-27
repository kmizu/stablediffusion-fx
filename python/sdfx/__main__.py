"""python -m sdfx [--dry-run] [--log-level LEVEL] [--dry-run-delay-ms N]"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings


def _reset_inherited_windows_dll_directory() -> None:
    if sys.platform != "win32":
        return
    # jpackage's launcher can set a DLL directory that Windows inherits into
    # this child process. Restore the default before importing torch or server.
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_directory = kernel32.SetDllDirectoryW
    set_directory.argtypes = [ctypes.c_wchar_p]
    set_directory.restype = ctypes.c_int
    if not set_directory(None):
        raise ctypes.WinError(ctypes.get_last_error())


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
    _reset_inherited_windows_dll_directory()
    from .server import main as server_main
    return server_main(dry_run=args.dry_run, delay_ms=args.dry_run_delay_ms)


if __name__ == "__main__":
    sys.exit(main())
