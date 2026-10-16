"""Inventory `models/` for the 2026-10-16 cleanup: what is kept, what is deleted, and why.

Usage: models_inventory.py MODELS_DIR OUT_JSON

Read-only. A top-level directory is kept when the repository's code or config names it
(`src/`, `config/`), when it is on the kept list approved by the user on 2026-10-16, or when the
API's run indexer (`api.runs.indexer.summarize_run`, what the web app lists) classifies it as a
run of season 2026 of any kind (the power page compares a run with the previous week's).
A fold-checkpoint directory under `wf_checkpoints/` is kept when any
`metadata.json` inside a kept directory names it (`config.checkpoint.dir`), or when it was
modified in the last 24 hours (a run in progress writes its folds before its metadata). Every
other entry is listed for deletion with its size.
"""

import json
import re
import sys
import time
from pathlib import Path

from nfl_predictor.api.runs.indexer import summarize_run

REPO = Path("/home/mitch/workspace/nfl-predictor")
KEPT_BY_DECISION = frozenset({"qb_elos_measure", "qb_adj_no_lines", "qb_wt_measure", "step6"})
CHECKPOINTS = "wf_checkpoints"
RECENT_SECONDS = 24 * 3600
NAMED = re.compile(r"models/([A-Za-z0-9_.-]+)")
CURRENT_SEASON = 2026
MIB = 1024 * 1024

Entry = dict[str, object]


def size_of(path: Path) -> int:
    """Return the bytes under a path."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def newest_mtime(path: Path) -> float:
    """Return the newest modification time under a path."""
    times = [path.stat().st_mtime]
    times.extend(item.stat().st_mtime for item in path.rglob("*"))
    return max(times)


def names_in_code() -> set[str]:
    """Return every top-level `models/` entry named in src/ or config/."""
    names: set[str] = set()
    for root in (REPO / "src", REPO / "config"):
        for path in root.rglob("*"):
            if path.suffix in {".py", ".yaml", ".yml", ".toml", ".json"} and path.is_file():
                names.update(NAMED.findall(path.read_text(errors="ignore")))
    return names


def script_references(models: Path, kept: list[str]) -> dict[str, set[str]]:
    """Return, per top-level entry, the paths under it that a script still names.

    Scripts are the `.py` and `.sh` files under `.agents/`, `scripts/`, `src/`, `config/`,
    `tests/` and the kept measurement directories; a path is `models/<entry>/<rest>`.
    """
    roots = [REPO / ".agents", REPO / "scripts", REPO / "src", REPO / "config", REPO / "tests"]
    roots.extend(models / name for name in kept)
    pattern = re.compile(r"models/([A-Za-z0-9_.-]+)/([A-Za-z0-9_./-]*)")
    found: dict[str, set[str]] = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix in {".py", ".sh"} and path.is_file():
                for entry, rest in pattern.findall(path.read_text(errors="ignore")):
                    found.setdefault(entry, set()).add(rest.rstrip("/."))
    return found


def checkpoint_dirs(kept: Path) -> set[str]:
    """Return the checkpoint fingerprints the metadata files under a kept entry name."""
    found: set[str] = set()
    for meta in kept.rglob("metadata.json"):
        try:
            data = json.loads(meta.read_text())
        except OSError, ValueError:
            continue
        config = data.get("config") if isinstance(data, dict) else None
        checkpoint = config.get("checkpoint") if isinstance(config, dict) else None
        directory = checkpoint.get("dir") if isinstance(checkpoint, dict) else None
        if isinstance(directory, str) and directory:
            found.add(Path(directory).name)
    return found


def top_level_entries(models: Path) -> tuple[list[Entry], list[Path]]:
    """Return one entry per top-level item under ``models`` and the directories kept so far."""
    in_code = names_in_code()
    entries: list[Entry] = []
    kept_dirs: list[Path] = []
    for child in sorted(models.iterdir()):
        if child.name == CHECKPOINTS:
            continue
        summary = summarize_run(child) if child.is_dir() else None
        kind = summary.kind if summary else None
        season = summary.season if summary else None
        reasons: list[str] = []
        if child.name in in_code:
            reasons.append("named in src/ or config/")
        if child.name in KEPT_BY_DECISION:
            reasons.append("kept by the user's decision of 2026-10-16")
        if season == CURRENT_SEASON:
            reasons.append(f"a {kind} run of season 2026 (API indexer; the power page's movement")
            reasons[-1] += " reads the previous week's run)"
        if reasons and child.is_dir():
            kept_dirs.append(child)
        entries.append(
            {
                "name": child.name,
                "bytes": size_of(child),
                "kind": kind,
                "season": season,
                "keep": bool(reasons),
                "reasons": reasons,
            }
        )
    return entries, kept_dirs


def apply_script_references(models: Path, entries: list[Entry], kept_dirs: list[Path]) -> None:
    """Keep, whole or in part, the entries a script still names (updates both lists in place)."""
    references = script_references(models, [path.name for path in kept_dirs])
    for entry in entries:
        name = str(entry["name"])
        if entry["keep"] or name not in references:
            continue
        rests = {rest for rest in references[name] if rest}
        whole = not rests or any((models / name / rest).is_dir() for rest in rests)
        if whole:
            entry["keep"] = True
            entry["reasons"] = ["a script names a directory in it"]
            kept_dirs.append(models / name)
        else:
            entry["keep_only"] = sorted(rests)
            entry["reasons"] = [f"partial: a script names {sorted(rests)}; the rest is deleted"]
            entry["bytes"] = int(str(entry["bytes"])) - sum(
                size_of(models / name / rest) for rest in rests if (models / name / rest).exists()
            )


def checkpoint_entries(models: Path, referenced: set[str]) -> list[Entry]:
    """Return one entry per fold-checkpoint directory, kept when referenced or recent."""
    now = time.time()
    entries: list[Entry] = []
    for child in sorted((models / CHECKPOINTS).iterdir()):
        reasons = []
        if child.name in referenced:
            reasons.append("named by a kept run's metadata")
        if now - newest_mtime(child) < RECENT_SECONDS:
            reasons.append("modified in the last 24 hours")
        entries.append(
            {
                "name": f"{CHECKPOINTS}/{child.name}",
                "bytes": size_of(child),
                "kind": "checkpoints",
                "season": None,
                "keep": bool(reasons),
                "reasons": reasons,
            }
        )
    return entries


def summary_lines(entries: list[Entry], missing: list[str]) -> list[str]:
    """Return the printed summary: totals, then every kept and deleted entry."""
    keep = [e for e in entries if e["keep"]]
    drop = [e for e in entries if not e["keep"]]
    lines = [
        f"keep {len(keep)} entries, {sum(int(str(e['bytes'])) for e in keep) / MIB:.0f} MiB",
        f"delete {len(drop)} entries, {sum(int(str(e['bytes'])) for e in drop) / MIB:.0f} MiB",
    ]
    lines.extend(f"  KEEP {e['name']} ({e['kind']}, {e['season']}): {e['reasons']}" for e in keep)
    for entry in drop:
        only = f" keeping only {entry['keep_only']}" if entry.get("keep_only") else ""
        size = int(str(entry["bytes"])) / MIB
        lines.append(
            f"  DEL  {entry['name']} ({entry['kind']}, {entry['season']}) {size:.1f} MiB{only}"
        )
    lines.append(f"referenced checkpoints missing: {missing}")
    return lines


def main() -> int:
    """Write the inventory and print a summary."""
    models, out = Path(sys.argv[1]), Path(sys.argv[2])
    entries, kept_dirs = top_level_entries(models)
    apply_script_references(models, entries, kept_dirs)
    referenced: set[str] = set()
    for kept in kept_dirs:
        referenced |= checkpoint_dirs(kept)
    entries.extend(checkpoint_entries(models, referenced))
    missing = sorted(referenced - {p.name for p in (models / CHECKPOINTS).iterdir()})
    report = {"entries": entries, "referenced_checkpoints_missing": missing}
    out.write_text(json.dumps(report, indent=2) + "\n")
    sys.stdout.write("\n".join(summary_lines(entries, missing)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
