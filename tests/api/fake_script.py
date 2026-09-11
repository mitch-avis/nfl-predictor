"""A stand-in for a project CLI, used to exercise the job runner.

It writes lines in the project's log format with the same ANSI colors ``coloredlogs`` emits, can
report walk-forward style progress, can stall so a cancel has something to interrupt, and can
ignore ``SIGTERM`` so the escalation to ``SIGKILL`` is observable.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from types import FrameType

GREEN = "\x1b[32m"
RESET = "\x1b[0m"


def emit(level: str, message: str) -> None:
    """Write one colored line in the project's log format."""
    sys.stdout.write(
        f"{GREEN}[2026-09-10 12:00:00.123][{level}][fake_script:emit:1] {message}{RESET}\n"
    )
    sys.stdout.flush()


def _ignore(_signum: int, _frame: FrameType | None) -> None:
    """Swallow a termination signal so the runner has to escalate."""
    emit("WARNING", "ignoring SIGTERM")


def main(argv: list[str] | None = None) -> int:
    """Emit the requested output and exit with the requested code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=2)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--stderr", action="store_true")
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--ignore-sigterm", action="store_true")
    args = parser.parse_args(argv)

    if args.ignore_sigterm:
        signal.signal(signal.SIGTERM, _ignore)
    for index in range(args.lines):
        emit("INFO", f"line {index + 1}")
    for index in range(args.progress):
        emit("INFO", f"WF candidate {index + 1}/{args.progress} done in 0m1.0s")
    if args.plain:
        sys.stdout.write("a line with no log prefix\n")
        sys.stdout.flush()
    if args.stderr:
        sys.stderr.write("[2026-09-10 12:00:00.123][ERROR][fake_script:main:1] from stderr\n")
        sys.stderr.flush()
    if args.sleep:
        time.sleep(args.sleep)
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
