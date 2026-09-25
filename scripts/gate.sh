#!/usr/bin/env bash
# The one validation gate. Runs every check that .github/workflows/validation.yml runs, in the
# same order, using the repo's own .venv, and reports every step before exiting non-zero, so a
# single run shows everything that is wrong rather than the first failure.
#
# Usage:
#   scripts/gate.sh          Python gate: lock, sync, ruff format, ruff, ty, pyright, pytest,
#                            markdownlint, CLI help smoke checks.
#   scripts/gate.sh --web    Also run the frontend gate in web/ (npm ci, lint, typecheck,
#                            vitest, build); needs nvm and the Node version CI uses.
#   scripts/gate.sh --quick  Skip pytest and the frontend gate (static checks only).
#
# Exit status is 0 only when every selected step passed. No chunk of work is "done" until this
# script exits 0 on the final tree (AGENTS.md, "Delegation guardrails").

set -u
set -o pipefail

cd "$(dirname "$0")/.." || exit 2

RUN_WEB=0
RUN_TESTS=1
for arg in "$@"; do
    case "$arg" in
        --web) RUN_WEB=1 ;;
        --quick) RUN_TESTS=0 ;;
        -h | --help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

if [ ! -x .venv/bin/python ]; then
    echo "gate: .venv/bin/python not found; run 'uv sync' first" >&2
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
    if "$@"; then
        NAMES+=("$name")
        STATUSES+=("ok")
    else
        NAMES+=("$name")
        STATUSES+=("FAIL")
        FAILED=1
    fi
}

markdownlint_step() {
    # CI runs `markdownlint .` (markdownlint-cli). Locally either binary is accepted; the
    # cli2 form mirrors .markdownlintignore explicitly because symlinked trees escape it.
    if command -v markdownlint >/dev/null 2>&1; then
        markdownlint .
    elif command -v markdownlint-cli2 >/dev/null 2>&1; then
        markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings" "#.agents/skills" \
            "#web/node_modules" "#web/dist" "#.agents/*transcript*.md"
    else
        echo "neither markdownlint nor markdownlint-cli2 is installed" >&2
        return 1
    fi
}

cli_smoke_step() {
    # Every front-door command must print its help. The list comes from the front door itself,
    # so a new command is checked without editing this script.
    local commands command
    commands="$(.venv/bin/python -c 'from nfl_predictor.cli.main import COMMANDS
print("\n".join(command.name for command in COMMANDS))')" || return 1
    .venv/bin/nfl-predictor --help >/dev/null || return 1
    while read -r command; do
        .venv/bin/nfl-predictor "$command" --help >/dev/null || {
            echo "nfl-predictor $command --help failed" >&2
            return 1
        }
    done <<<"$commands"
}

web_step() {
    # shellcheck disable=SC1091
    [ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh"
    (
        cd web &&
            npm ci --no-audit --no-fund &&
            npm run lint &&
            npm run typecheck &&
            npx vitest run &&
            npm run build
    )
}

run_step "uv lock --check" uv lock --check
# On a machine with a CUDA toolkit, LightGBM is a local CUDA build of the locked version (see
# nfl_predictor/lightgbm_cuda.py); the sync check is told the same build flags, so it stays
# strict. The helper prints nothing on CPU-only machines and in CI.
LIGHTGBM_CUDA_ARGS=()
if [[ -x .venv/bin/nfl-lightgbm-cuda-install ]]; then
    mapfile -t LIGHTGBM_CUDA_ARGS < <(.venv/bin/nfl-lightgbm-cuda-install uv-args 2>/dev/null)
fi
run_step "uv sync --check --active" uv sync --check --active "${LIGHTGBM_CUDA_ARGS[@]}"
run_step "ruff format --check" .venv/bin/ruff format --check .
run_step "ruff check" .venv/bin/ruff check .
run_step "ty check" .venv/bin/ty check .
run_step "pyright" .venv/bin/pyright .
if [ "$RUN_TESTS" -eq 1 ]; then
    run_step "pytest" .venv/bin/python -m pytest -q -p no:cacheprovider
fi
run_step "markdownlint" markdownlint_step
run_step "CLI help smoke checks" cli_smoke_step
if [ "$RUN_WEB" -eq 1 ] && [ "$RUN_TESTS" -eq 1 ]; then
    run_step "web gate" web_step
fi

echo
echo "==> gate summary"
for i in "${!NAMES[@]}"; do
    printf '  %-40s %s\n' "${NAMES[$i]}" "${STATUSES[$i]}"
done
if [ "$FAILED" -ne 0 ]; then
    echo "gate: FAILED"
    exit 1
fi
echo "gate: all steps passed"
