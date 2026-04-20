"""
xlog.py — dead-simple print-based logger for the DirectX .x addon.

Why not Python's logging module?
  Blender patches sys.stdout and sys.stderr at startup.  The standard
  logging.StreamHandler captures those patched objects and its output
  disappears into Blender's internal buffers.  sys.__stdout__ is the
  *original* file descriptor that was open before Blender touched anything
  — writing to it always reaches the terminal you launched Blender from.

Usage (from any module in this addon):
    from .xlog import XLog
    log = XLog("importer")          # one instance per file
    log.info("Vertices: %d", n)
    log.warn("Missing texture: %s", path)
    log.debug("Matrix: %s", mat)    # only prints when verbose=True
    log.error("Crash in %s", func)
    XLog.set_verbose(True/False)    # called once from the operator
"""

import sys
import time


class XLog:
    # Class-level verbose flag shared across all instances
    _verbose: bool = False

    def __init__(self, tag: str):
        self._tag = tag

    @classmethod
    def set_verbose(cls, verbose: bool):
        cls._verbose = verbose

    # ── write to sys.__stdout__ unconditionally ──
    @staticmethod
    def _write(line: str):
        # sys.__stdout__ is the original fd before Blender patches sys.stdout.
        # flush=True ensures every line appears immediately in the terminal.
        try:
            print(line, file=sys.__stdout__, flush=True)
        except Exception:
            # Last-ditch fallback: try stderr too
            try:
                print(line, file=sys.__stderr__, flush=True)
            except Exception:
                pass

    # ── public API ──
    def info(self, msg: str, *args):
        self._write(f"[DX.x][INFO]  {self._tag}: {msg % args if args else msg}")

    def warn(self, msg: str, *args):
        self._write(f"[DX.x][WARN]  {self._tag}: {msg % args if args else msg}")

    def error(self, msg: str, *args):
        self._write(f"[DX.x][ERROR] {self._tag}: {msg % args if args else msg}")

    def debug(self, msg: str, *args):
        if self._verbose:
            self._write(f"[DX.x][DEBUG] {self._tag}: {msg % args if args else msg}")

    def section(self, title: str):
        """Print a visible separator — always shown regardless of verbose."""
        self._write(f"[DX.x] {'─' * 10} {title} {'─' * 10}")
