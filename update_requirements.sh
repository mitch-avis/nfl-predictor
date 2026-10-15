#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

# Runs from the repo root regardless of where it's invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PATH="$REPO_ROOT/.venv"
PYTHON_REQUEST=""
cd "$REPO_ROOT"

# uv is treated as an external tool on PATH rather than as a project dependency inside .venv.

# A project's own steps live in update_requirements.hooks.sh beside this script, so this file
# stays the same in every project. The hooks file may define:
#   hook_smoke_imports  Prints Python statements the synced .venv must run, such as
#                       "import numpy, pandas"; printing nothing (the default) skips the check.
#   hook_sync_args      Prints extra `uv sync` arguments, one per line.
#   hook_after_sync     Runs after the sync and before the import check.
HOOKS_FILE_NAME="update_requirements.hooks.sh"
hook_smoke_imports() { :; }
hook_sync_args() { :; }
hook_after_sync() { :; }

usage() {
	cat <<'EOF'
Usage: ./update_requirements.sh [--python <version-request>]

Upgrades every dependency in the project's uv lockfile, syncs .venv from it (creating .venv
after a prompt when it is missing), refreshes the pre-commit hook versions when the project
uses pre-commit, and runs the steps in update_requirements.hooks.sh when the project has one.
The venv does not need to be activated.

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

report_failed_command() {
	echo "Error: '$2' failed (update_requirements.sh line $1)." >&2
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

load_hooks() {
	local hooks_file="$REPO_ROOT/$HOOKS_FILE_NAME"
	if [[ -f "$hooks_file" ]]; then
		# shellcheck source=/dev/null
		source "$hooks_file"
	fi
}

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

# The installed project and its dependencies are hardlinks into uv's cache, so a damaged cache
# entry survives a plain reinstall; this import catches it before the tools fail one by one.
ensure_venv_packages_import() {
	local imports

	imports="$(hook_smoke_imports)"
	if [[ -z "$imports" ]] || "$(venv_python)" -c "$imports" >/dev/null 2>&1; then
		return
	fi

	echo "Error: the synced .venv cannot run: $imports" >&2
	echo "Installed files are hardlinks into uv's cache, so a damaged cache entry survives a" >&2
	echo "plain reinstall. Repair with:" >&2
	echo "  uv cache clean && uv sync --reinstall" >&2
	exit 1
}

ensure_venv_exists() {
	local -a command=(uv venv .venv)

	if [[ -d "$VENV_PATH" ]]; then
		return
	fi

	if [[ -n "$PYTHON_REQUEST" ]]; then
		command+=(--python "$PYTHON_REQUEST")
	fi

	info "No .venv directory was found."
	if ! confirm "Create one now with '${command[*]}'?"; then
		die "A project virtual environment is required to continue."
	fi

	"${command[@]}"
}

# `uv self update` only works for uv's standalone installer; other installs keep their version.
update_uv() {
	local before after

	before="$(uv --version)"
	if ! uv self update >/dev/null 2>&1; then
		info "'uv self update' failed or is unavailable for this install; keeping ${before}"
		return
	fi

	after="$(uv --version)"
	if [[ "$before" == "$after" ]]; then
		info "uv is up to date (${after})"
	else
		info "Updated ${before} -> ${after}"
	fi
}

refresh_pre_commit_hooks() {
	if [[ ! -f .pre-commit-config.yaml || ! -x .venv/bin/pre-commit ]]; then
		return
	fi

	info "Updating the pre-commit hook versions in .pre-commit-config.yaml"
	.venv/bin/pre-commit autoupdate
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

	# nullglob drops a pattern that matches nothing.
	for file in requirements-*.in requirements-*.txt; do
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
	local -a sync_args=()

	trap 'report_failed_command "$LINENO" "$BASH_COMMAND"' ERR
	parse_args "$@"
	ensure_uv
	ensure_pyproject_exists
	load_hooks
	ensure_venv_exists
	ensure_venv_python_is_healthy
	update_uv

	info "Locking project dependencies (upgrade all)"
	uv lock --upgrade

	mapfile -t sync_args < <(hook_sync_args)
	info "Syncing .venv from uv.lock"
	uv sync "${sync_args[@]}"
	hook_after_sync

	ensure_venv_packages_import

	refresh_pre_commit_hooks
	refresh_compatibility_requirements

	info "Done. Run the project's gate before committing the refreshed lockfile."
}

# Sourcing the script (as the tests do) defines its functions without running it.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	main "$@"
fi
