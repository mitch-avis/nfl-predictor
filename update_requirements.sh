#!/usr/bin/env bash
set -euo pipefail

# Runs from the repo root regardless of where it's invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_PATH="$REPO_ROOT/.venv"
cd "$REPO_ROOT"

# uv is treated as an external tool on PATH rather than as a project dependency inside .venv.

info() {
	echo "==> $*"
}

die() {
	echo "Error: $*" >&2
	exit 1
}

confirm() {
	local prompt="$1"
	local reply
	read -r -p "$prompt [y/N] " reply
	[[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

ensure_uv() {
	if ! command -v uv >/dev/null 2>&1; then
		die "'uv' is not installed or not on PATH. Install it from https://astral.sh/uv/."
	fi
}

ensure_venv_exists() {
	if [[ -d "$VENV_PATH" ]]; then
		return
	fi

	info "No .venv directory was found."
	if ! confirm "Create one now with 'uv venv .venv'?"; then
		die "A project virtual environment is required to continue."
	fi

	uv venv .venv
	info "Created .venv. Activate it with: source .venv/bin/activate"
	info "Then rerun ./update_requirements.sh"
	exit 0
}

ensure_venv_is_active() {
	if [[ "${VIRTUAL_ENV:-}" == "$VENV_PATH" ]]; then
		return
	fi

	info "The project virtual environment exists but is not active."
	echo "Activate it in your current shell with:" >&2
	echo "  source .venv/bin/activate" >&2
	echo "Then rerun this script:" >&2
	echo "  ./update_requirements.sh" >&2
	exit 1
}

main() {
	ensure_uv
	ensure_venv_exists
	ensure_venv_is_active

	# If supported, keep uv itself up to date (no-op on older uv builds).
	uv self update >/dev/null 2>&1 || true

	info "Locking project dependencies (upgrade all)"
	uv lock --upgrade

	info "Syncing the active virtual environment from uv.lock"
	uv sync --active

	info "Done."
}

main "$@"
