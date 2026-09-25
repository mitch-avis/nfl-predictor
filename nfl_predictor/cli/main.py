"""The ``nfl-predictor`` front door: one command with a subcommand for each task.

``nfl-predictor <command> [options]`` runs the command with its own options, exactly as its
module would; ``nfl-predictor <command> --help`` shows them. ``python -m nfl_predictor`` is the
same front door, and the per-module forms (``python -m nfl_predictor.data_collection`` and the
others) keep working. Commands are imported only when they run, so ``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

PROG = "nfl-predictor"


@dataclass(frozen=True)
class Command:
    """One subcommand: where its ``main`` lives and how it takes its arguments."""

    name: str
    group: str
    summary: str
    module: str
    takes_argv: bool
    function: str = "main"
    requires: str | None = None


COMMANDS: tuple[Command, ...] = (
    Command(
        "weekly",
        "weekly",
        "Refresh data, select, train, predict and write the week's reports.",
        "nfl_predictor.weekly_run.pipeline",
        takes_argv=False,
    ),
    Command(
        "backtest",
        "research",
        "Walk-forward evaluation (the benchmark).",
        "nfl_predictor.cli.backtest",
        takes_argv=False,
    ),
    Command(
        "sweep",
        "research",
        "Walk-forward sweep of calibration and market-probability settings.",
        "nfl_predictor.cli.sweep",
        takes_argv=False,
    ),
    Command(
        "explain",
        "research",
        "SHAP feature attribution for a saved model.",
        "nfl_predictor.cli.explain",
        takes_argv=False,
    ),
    Command(
        "data",
        "data",
        "Run the ETL: collect, transform and write the datasets.",
        "nfl_predictor.data_collection",
        takes_argv=True,
    ),
    Command(
        "validate",
        "data",
        "Validate data/all_data.csv (offline checks, or --live against the schedule).",
        "nfl_predictor.cli.validate",
        takes_argv=True,
    ),
    Command(
        "leakage-audit",
        "data",
        "Audit an ML dataset for features that leak the outcome.",
        "nfl_predictor.cli.leakage_audit",
        takes_argv=False,
    ),
    Command(
        "lines",
        "data",
        "Refresh the betting lines.",
        "nfl_predictor.lines_refresh",
        takes_argv=True,
    ),
    Command(
        "build-week",
        "data",
        "Build a week's games-to-predict file.",
        "nfl_predictor.week_builder",
        takes_argv=True,
    ),
    Command(
        "train",
        "models",
        "Train a model (and optionally predict a week).",
        "nfl_predictor.cli.train",
        takes_argv=False,
    ),
    Command(
        "predict",
        "models",
        "Predict a week with a saved model (--model-in; the same options as train).",
        "nfl_predictor.cli.train",
        takes_argv=False,
        requires="--model-in",
    ),
    Command(
        "rankings",
        "models",
        "Power rankings and projected standings.",
        "nfl_predictor.cli.rankings",
        takes_argv=True,
    ),
    Command(
        "web",
        "web",
        "Serve the web UI and its API.",
        "nfl_predictor.api.__main__",
        takes_argv=True,
    ),
    Command(
        "users",
        "web",
        "Manage web UI users.",
        "nfl_predictor.api.auth.cli",
        takes_argv=True,
    ),
)
COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}
GROUP_TITLES = {
    "weekly": "weekly",
    "research": "research",
    "data": "data",
    "models": "models by hand",
    "web": "web",
}


def _command_list() -> str:
    """Return the grouped command table shown by ``nfl-predictor --help``."""
    width = max(len(command.name) for command in COMMANDS)
    blocks = []
    for group, title in GROUP_TITLES.items():
        rows = [
            f"  {command.name:<{width}}  {command.summary}"
            for command in COMMANDS
            if command.group == group
        ]
        blocks.append(f"{title}:\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def _build_parser() -> argparse.ArgumentParser:
    """Build the front-door parser: a command name, then that command's own options."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="NFL predictions for pick'em and confidence pools.",
        epilog=_command_list() + f"\n\nRun '{PROG} <command> --help' for a command's options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=sorted(COMMANDS_BY_NAME), metavar="command")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def _resolve(command: Command) -> Callable[..., Any]:
    """Import the command's module and return its entry function."""
    return getattr(importlib.import_module(command.module), command.function)


def _missing_requirement(command: Command, args: Sequence[str]) -> bool:
    """Return whether a required option is absent (help requests never are)."""
    if command.requires is None or {"-h", "--help"} & set(args):
        return False
    return not any(
        arg == command.requires or arg.startswith(f"{command.requires}=") for arg in args
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``nfl-predictor <command> [options]`` and return the command's exit code."""
    parser = _build_parser()
    namespace = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    command = COMMANDS_BY_NAME[namespace.command]
    args = list(namespace.args)
    if _missing_requirement(command, args):
        parser.error(f"{command.name} needs {command.requires}")

    entry = _resolve(command)
    saved_argv = sys.argv
    # Each command's parser names itself after argv[0], so help reads "nfl-predictor <name>".
    sys.argv = [f"{PROG} {command.name}", *args]
    try:
        result = entry(args) if command.takes_argv else entry()
    finally:
        sys.argv = saved_argv
    return int(result or 0)
