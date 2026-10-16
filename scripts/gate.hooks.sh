# shellcheck shell=bash
# This project's steps for scripts/gate.sh, which sources this file: the LightGBM CUDA build flags
# for the sync check, every front-door command's --help, and the frontend gate. scripts/gate.sh
# lists what a hooks file may define.
#
# Extra option:
#   scripts/gate.sh --web    Also run the frontend gate in web/ (pnpm install --frozen-lockfile,
#                            lint, typecheck, vitest, build); needs nvm, the Node version CI uses,
#                            and pnpm on PATH. --quick skips it.

RUN_WEB=0

gate_option() {
	case "$1" in
	--web) RUN_WEB=1 ;;
	*) return 1 ;;
	esac
}

# On a machine with a CUDA toolkit, LightGBM is a local CUDA build of the locked version (see
# src/nfl_predictor/lightgbm_cuda.py); the sync check is told the same build flags, so it stays
# strict. The helper prints nothing on CPU-only machines and in CI.
gate_sync_args() {
	if [[ -x .venv/bin/nfl-lightgbm-cuda-install ]]; then
		.venv/bin/nfl-lightgbm-cuda-install uv-args 2>/dev/null
	fi
}

gate_extra_steps() {
	run_step "nfl-predictor commands --help" cli_commands_step
	if [[ "$RUN_WEB" -eq 1 && "$RUN_TESTS" -eq 1 ]]; then
		run_step "web gate" web_step
	fi
}

cli_commands_step() {
	# Every front-door command must print its help. The list comes from the front door itself,
	# so a new command is checked without editing this file.
	local commands command
	commands="$(.venv/bin/python -c 'from nfl_predictor.cli.main import COMMANDS
print("\n".join(command.name for command in COMMANDS))')" || return 1
	while read -r command; do
		.venv/bin/nfl-predictor "$command" --help >/dev/null || {
			echo "nfl-predictor $command --help failed" >&2
			return 1
		}
	done <<<"$commands"
}

web_step() {
	# nvm.sh lives outside the repo, so shellcheck can't follow it.
	# shellcheck disable=SC1091
	[[ -s "$HOME/.nvm/nvm.sh" ]] && . "$HOME/.nvm/nvm.sh"
	command -v pnpm >/dev/null || {
		echo "gate: pnpm not found on PATH; the frontend uses pnpm (web/pnpm-lock.yaml)" >&2
		return 1
	}
	(
		cd web &&
			pnpm install --frozen-lockfile &&
			pnpm run lint &&
			pnpm run typecheck &&
			pnpm exec vitest run &&
			pnpm run build
	)
}
