# shellcheck shell=bash
# This project's steps for update_requirements.sh, which sources this file: the LightGBM CUDA
# build and the import check of the core packages. update_requirements.sh lists what a hooks file
# may define.

hook_smoke_imports() {
	echo "import numpy, pandas, polars, scipy, sklearn, xgboost"
}

# With a CUDA toolkit, sync with the LightGBM CUDA build flags so an unchanged LightGBM keeps its
# CUDA build and a new version is built with CUDA (see src/nfl_predictor/lightgbm_cuda.py).
hook_sync_args() {
	if [[ -x .venv/bin/nfl-lightgbm-cuda-install ]]; then
		.venv/bin/nfl-lightgbm-cuda-install uv-args 2>/dev/null
	fi
}

hook_after_sync() {
	info "Making LightGBM a CUDA build if this machine can build one (no-op when it already is)"
	install_lightgbm_cuda_build
}

# The CUDA build of LightGBM is best effort: a failed build leaves the CPU build in place, so the
# rest of the refresh (the import check, the requirements export) still runs.
install_lightgbm_cuda_build() {
	if .venv/bin/nfl-lightgbm-cuda-install install; then
		return
	fi

	echo "Warning: LightGBM could not be made a CUDA build; the CPU build stays installed." >&2
	echo "Check it later with: .venv/bin/nfl-lightgbm-cuda-install status" >&2
}
