#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# Runs from the repo root regardless of where it's invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PATH="$REPO_ROOT/.venv"
PYTHON_REQUEST=""
cd "$REPO_ROOT"

# uv is treated as an external tool on PATH rather than as a project dependency inside .venv.

usage() {
	cat <<'EOF'
Usage: ./update_requirements.sh [--python <version-request>]

Updates the project's uv lockfile and syncs the active .venv.
With a local CUDA toolkit, it then rebuilds LightGBM with CUDA if the synced build lacks it.

If compatibility requirements files are present, they are refreshed from uv.lock:
- requirements.txt -> runtime dependencies only
- requirements-<group>.txt -> only that dependency group

Examples:
	./update_requirements.sh
	./update_requirements.sh --python 3.14

When no --python argument is provided, uv chooses the newest compatible Python
according to the project's configuration and local uv installation.
EOF
}

info() {
	echo "==> $*"
}

die() {
	echo "Error: $*" >&2
	exit 1
}

canonical_path() {
	cd "$1" >/dev/null 2>&1 && pwd -P
}

venv_python() {
	echo "$VENV_PATH/bin/python"
}

python_stdlib_smoke_test() {
	local python_bin="$1"
	"$python_bin" -c "import dis; import json" >/dev/null 2>&1
}

confirm() {
	local prompt="$1"
	local reply
	read -r -p "$prompt [y/N] " reply
	[[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
		-p | --python)
			[[ $# -ge 2 ]] || die "Missing value for $1."
			PYTHON_REQUEST="$2"
			shift 2
			;;
		-h | --help)
			usage
			exit 0
			;;
		--)
			shift
			break
			;;
		*)
			die "Unknown option: $1"
			;;
		esac
	done

	[[ $# -eq 0 ]] || die "Unexpected positional arguments: $*"
}

ensure_uv() {
	if ! command -v uv >/dev/null 2>&1; then
		die "'uv' is not installed or not on PATH. Install it from https://astral.sh/uv/."
	fi
}

ensure_pyproject_exists() {
	[[ -f "$REPO_ROOT/pyproject.toml" ]] || die "A pyproject.toml file is required at the repo root."
}

create_venv() {
	local -a command=(uv venv .venv)

	if [[ -n "$PYTHON_REQUEST" ]]; then
		command+=(--python "$PYTHON_REQUEST")
	fi

	"${command[@]}"
}

# The installed project and its dependencies are hardlinks into uv's cache, so a damaged cache
# entry survives a plain reinstall; this import catches it before the tools fail one by one.
PACKAGE_SMOKE_IMPORTS="import numpy, pandas, polars, scipy, sklearn, xgboost"

venv_python_request() {
	if [[ -n "$PYTHON_REQUEST" ]]; then
		echo "$PYTHON_REQUEST"
	elif [[ -f "$VENV_PATH/pyvenv.cfg" ]]; then
		sed -n 's/^version_info = //p' "$VENV_PATH/pyvenv.cfg"
	fi
}

repair_venv_python() {
	local version
	local -a python_args=()

	version="$(venv_python_request)"
	if [[ -n "$version" ]]; then
		python_args=(--python "$version")
	fi

	info "Reinstalling the uv-managed Python ${version} and recreating .venv with it"
	uv python install --reinstall ${version:+"$version"}
	uv venv .venv --clear --managed-python "${python_args[@]}"
	if python_stdlib_smoke_test "$(venv_python)"; then
		return
	fi

	info "The reinstalled managed Python still fails; recreating .venv with a system Python"
	uv venv .venv --clear --no-managed-python "${python_args[@]}"
	if ! python_stdlib_smoke_test "$(venv_python)"; then
		die "The recreated .venv still failed stdlib imports. Repair or replace your local Python install, then rerun this script."
	fi
}

ensure_venv_python_is_healthy() {
	local python_bin

	python_bin="$(venv_python)"
	if python_stdlib_smoke_test "$python_bin"; then
		return
	fi

	info "The current .venv interpreter failed a stdlib smoke test"
	echo "Its standard library is damaged, for example rewritten in place by a tool run outside a repo." >&2
	if ! confirm "Reinstall its uv-managed Python and recreate .venv?"; then
		die "Reinstall with 'uv python install --reinstall', recreate .venv with 'uv venv .venv --clear', and rerun this script."
	fi

	repair_venv_python
}

ensure_venv_packages_import() {
	if "$(venv_python)" -c "$PACKAGE_SMOKE_IMPORTS" >/dev/null 2>&1; then
		return
	fi

	echo "Error: the synced .venv cannot import its core packages (${PACKAGE_SMOKE_IMPORTS#import })." >&2
	echo "Installed files are hardlinks into uv's cache, so a damaged cache entry survives a" >&2
	echo "plain reinstall. Repair with:" >&2
	echo "  uv cache clean && uv sync --active --reinstall" >&2
	exit 1
}

ensure_venv_exists() {
	local create_command="uv venv .venv"

	if [[ -n "$PYTHON_REQUEST" ]]; then
		create_command+=" --python $PYTHON_REQUEST"
	fi

	if [[ -d "$VENV_PATH" ]]; then
		return
	fi

	info "No .venv directory was found."
	if ! confirm "Create one now with '$create_command'?"; then
		die "A project virtual environment is required to continue."
	fi

	create_venv
	ensure_venv_python_is_healthy
	info "Created .venv. Activate it with: source .venv/bin/activate"
	info "Then rerun ./update_requirements.sh"
	exit 0
}

ensure_venv_is_active() {
	local active_venv=""

	if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -d "${VIRTUAL_ENV}" ]]; then
		active_venv="$(canonical_path "$VIRTUAL_ENV")"
	fi

	if [[ "$active_venv" == "$VENV_PATH" ]]; then
		return
	fi

	info "The project virtual environment exists but is not active."
	echo "Activate it in your current shell with:" >&2
	echo "  source .venv/bin/activate" >&2
	echo "Then rerun this script:" >&2
	echo "  ./update_requirements.sh" >&2
	exit 1
}

export_runtime_requirements() {
	info "Refreshing requirements.txt from uv.lock"
	uv export \
		--format requirements.txt \
		--no-default-groups \
		--no-emit-project \
		--no-hashes \
		--output-file requirements.txt
}

export_group_requirements() {
	local group_name="$1"
	local output_file="requirements-${group_name}.txt"

	info "Refreshing ${output_file} from dependency group '${group_name}'"
	uv export \
		--format requirements.txt \
		--only-group "$group_name" \
		--no-emit-project \
		--no-hashes \
		--output-file "$output_file"
}

refresh_compatibility_requirements() {
	local file group_name
	declare -A seen_groups=()

	if [[ -f "$REPO_ROOT/requirements.in" || -f "$REPO_ROOT/requirements.txt" ]]; then
		export_runtime_requirements
	fi

	for file in requirements-*.in requirements-*.txt; do
		[[ -e "$file" ]] || continue
		group_name="${file#requirements-}"
		group_name="${group_name%.in}"
		group_name="${group_name%.txt}"

		if [[ -z "$group_name" || -n "${seen_groups[$group_name]:-}" ]]; then
			continue
		fi

		seen_groups["$group_name"]=1
		export_group_requirements "$group_name"
	done
}

main() {
	parse_args "$@"
	ensure_uv
	ensure_pyproject_exists
	ensure_venv_exists
	ensure_venv_is_active
	ensure_venv_python_is_healthy

	# If supported, keep uv itself up to date (no-op on older uv builds).
	uv self update >/dev/null 2>&1 || true

	info "Locking project dependencies (upgrade all)"
	uv lock --upgrade

	# With a CUDA toolkit, sync with the LightGBM CUDA build flags so an unchanged LightGBM
	# keeps its CUDA build and a new version is built with CUDA (see lightgbm_cuda.py).
	local -a lightgbm_cuda_args=()
	if [[ -x .venv/bin/nfl-lightgbm-cuda-install ]]; then
		mapfile -t lightgbm_cuda_args < <(.venv/bin/nfl-lightgbm-cuda-install uv-args 2>/dev/null)
	fi

	info "Syncing the active virtual environment from uv.lock"
	uv sync --active "${lightgbm_cuda_args[@]}"

	info "Making LightGBM a CUDA build if this machine can build one (no-op when it already is)"
	.venv/bin/nfl-lightgbm-cuda-install install

	info "Checking that the core packages import"
	ensure_venv_packages_import

	refresh_compatibility_requirements

	info "Done."
}

# Sourcing the script (as the tests do) defines its functions without running it.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	main "$@"
fi
