#!/usr/bin/env bash
# The one validation gate. CI runs this script after syncing the environment, so a local run and
# CI run the same checks. It uses the repo's own .venv (never `uv run`, which re-syncs the
# environment first) and runs every selected step before exiting, so one run reports everything
# that is wrong rather than only the first failure.
#
# Usage:
#   scripts/gate.sh          lock and sync checks, ruff format, ruff, ty, pyright, pytest, the
#                            src/ coverage bar, every console script's --help, shellcheck,
#                            markdownlint
#   scripts/gate.sh --quick  skip pytest and the coverage bar (static checks only; for
#                            iteration, never the report)
#
# A project's own steps live in scripts/gate.hooks.sh, which this script sources when present, so
# this file stays the same in every project; --help prints the hooks file's header comment too.
# The hooks file may define:
#   gate_option <arg>   Handles an option this script doesn't know; returns non-zero for an
#                       option the project doesn't know either.
#   gate_sync_args      Prints extra `uv sync --check` arguments, one per line.
#   gate_extra_steps    Runs after the standard steps: call `run_step <name> <command...>`, and
#                       skip slow steps when RUN_TESTS is 0 (--quick).
#
# Exit status is 0 only when every selected step passed.

set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 2

HOOKS_FILE="scripts/gate.hooks.sh"
gate_option() { return 1; }
gate_sync_args() { :; }
gate_extra_steps() { :; }
if [[ -f "$HOOKS_FILE" ]]; then
	# shellcheck source=/dev/null
	source "$HOOKS_FILE"
fi

RUN_TESTS=1
for arg in "$@"; do
	case "$arg" in
	--quick) RUN_TESTS=0 ;;
	-h | --help)
		# This header, then the hooks file's, each up to its first blank line.
		sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
		if [[ -f "$HOOKS_FILE" ]]; then
			sed -n '2,/^$/p' "$HOOKS_FILE" | sed 's/^# \{0,1\}//'
		fi
		exit 0
		;;
	*)
		if ! gate_option "$arg"; then
			echo "gate: unknown option: $arg" >&2
			exit 2
		fi
		;;
	esac
done

if [[ ! -x .venv/bin/python ]]; then
	echo "gate: .venv/bin/python not found; create it with 'uv sync'" >&2
	exit 2
fi

declare -a NAMES=()
declare -a STATUSES=()
FAILED=0

run_step() {
	# run_step <name> <command...>
	local name="$1"
	shift
	echo
	echo "==> ${name}"
	NAMES+=("$name")
	if "$@"; then
		STATUSES+=("ok")
	else
		STATUSES+=("FAIL")
		FAILED=1
	fi
}

console_scripts_step() {
	# Every console script in pyproject.toml must print its help, so a new one is covered without
	# editing this script. A project without console scripts passes.
	local -a scripts
	local script
	mapfile -t scripts < <(
		.venv/bin/python -c 'import tomllib
with open("pyproject.toml", "rb") as file:
    print("\n".join(tomllib.load(file)["project"].get("scripts", {})))'
	) || return 1
	for script in "${scripts[@]}"; do
		[[ -n "$script" ]] || continue
		if ! ".venv/bin/$script" --help >/dev/null; then
			echo "$script --help failed" >&2
			return 1
		fi
	done
}

shellcheck_step() {
	local -a files
	if ! command -v shellcheck >/dev/null 2>&1; then
		echo "shellcheck is not installed (apt install shellcheck)" >&2
		return 1
	fi
	# Tracked and new files alike, but nothing .gitignore excludes (the venv, run directories).
	mapfile -t files < <(git ls-files --cached --others --exclude-standard -- '*.sh' | sort -u)
	if [[ ${#files[@]} -gt 0 ]]; then
		shellcheck "${files[@]}"
	fi
}

markdownlint_step() {
	# .markdownlint-cli2.yaml holds the rules and keeps .gitignore'd paths such as .venv/ out.
	if ! command -v markdownlint-cli2 >/dev/null 2>&1; then
		echo "markdownlint-cli2 is not installed (npm install --global markdownlint-cli2)" >&2
		return 1
	fi
	markdownlint-cli2 "**/*.md"
}

declare -a SYNC_ARGS=()
mapfile -t SYNC_ARGS < <(gate_sync_args)

run_step "uv lock --check" uv lock --check
run_step "uv sync --check" uv sync --check "${SYNC_ARGS[@]}"
run_step "ruff format --check" .venv/bin/ruff format --check
run_step "ruff check" .venv/bin/ruff check
run_step "ty check" .venv/bin/ty check
run_step "pyright" .venv/bin/pyright
if [[ "$RUN_TESTS" -eq 1 ]]; then
	run_step "pytest" .venv/bin/pytest -q -p no:cacheprovider
	# The total in pyproject.toml counts tests/ as well, which would let src/ slip under the bar;
	# `coverage report` applies the same fail_under to src/ alone.
	run_step "src coverage" .venv/bin/coverage report --include="src/*" --format=total
fi
run_step "console scripts --help" console_scripts_step
run_step "shellcheck" shellcheck_step
run_step "markdownlint" markdownlint_step
gate_extra_steps

echo
echo "==> gate summary"
for i in "${!NAMES[@]}"; do
	printf '  %-32s %s\n' "${NAMES[$i]}" "${STATUSES[$i]}"
done
if [[ "$FAILED" -ne 0 ]]; then
	echo "gate: FAILED"
	exit 1
fi
echo "gate: all steps passed"
