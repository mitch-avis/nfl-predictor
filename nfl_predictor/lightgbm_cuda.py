"""Build LightGBM with CUDA support in the project venv, only when it is missing.

The lockfile installs the CPU-only LightGBM wheel. On a machine with a CUDA toolkit and an NCCL
built for the same CUDA major version, the locked LightGBM can instead be built from source with
``USE_CUDA=ON`` by passing ``uv sync`` the flags that ``nfl-lightgbm-cuda-install uv-args``
prints. uv records those build settings, so a later ``uv sync`` or ``uv sync --check`` given the
same flags treats the CUDA build as up to date, and uv's build cache means a rebuild happens only
when the locked LightGBM version changes. On CPU-only machines (CI included) ``uv-args`` prints
nothing and every command behaves exactly as without this module.

Commands: ``install`` (the default) does nothing when LightGBM already trains on the GPU, and
otherwise reinstalls the CUDA build (from uv's cache when possible, from source when not);
``status`` reports the current state; ``uv-args`` prints the flags, one per line.

A CUDA build links NCCL, and NCCL built for another CUDA major version leaves an unresolved
symbol that stops LightGBM from loading, so the flags are withheld until NVIDIA's ``libnccl2``
and ``libnccl-dev`` packages tagged ``+cudaXX.Y`` for the toolkit are installed.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nfl_predictor.utils.logger import log

# Files a previous shim-based installer wrote into site-packages; removed on install.
_LEGACY_FILES = (
    "nfl_predictor_lightgbm_cuda_compat.pth",
    "liblightgbm_cuda_compat.so",
    "_lightgbm_cuda_compat.c",
)
_CPU_ONLY_FRAGMENTS = (
    "CUDA Tree Learner was not enabled in this build",
    "recompile with CMake option -DUSE_CUDA",
)
_SMOKE_TEST = """
import lightgbm as lgb
import numpy as np

X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, 1.0]], dtype=np.float32)
y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
params = {"objective": "regression", "device": "cuda", "verbosity": -1,
          "num_leaves": 7, "min_data_in_leaf": 1, "num_threads": 1}
lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=2)
""".strip()


@dataclass(frozen=True)
class Runtime:
    """The interpreter, LightGBM install and CUDA toolkit the installer works with."""

    python_executable: Path
    venv: Path
    purelib: Path
    lightgbm_version: str
    nvcc: Path | None
    cuda_version: str | None
    nccl_package_version: str | None
    has_dpkg: bool


@dataclass(frozen=True)
class InstallResult:
    """What the installer found or did: a status word and whether it rebuilt LightGBM."""

    status: str
    rebuilt: bool


def _lightgbm_version() -> str:
    """Return the installed LightGBM version, or an empty string when it is absent."""
    try:
        return importlib.metadata.version("lightgbm")
    except importlib.metadata.PackageNotFoundError:
        return ""


def _find_nvcc() -> Path | None:
    """Return the CUDA compiler, symlinks resolved, from ``CUDA_HOME``, ``PATH`` or a default.

    The path is resolved so the build flag, which uv records, is the same however ``PATH``
    reaches the toolkit (``/usr/local/cuda``, ``/usr/local/cuda-13``, ...).
    """
    candidates: list[Path] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    on_path = shutil.which("nvcc")
    if on_path:
        candidates.append(Path(on_path))
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    found = next((path for path in candidates if path.exists()), None)
    return found.resolve() if found else None


def _run(
    command: list[str], *, capture: bool = True, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a local tool with fixed arguments and return the completed process."""
    return subprocess.run(  # noqa: S603 - fixed tool names and repo-controlled arguments
        command, check=False, capture_output=capture, text=True, env=env
    )


def _cuda_version(nvcc: Path | None) -> str | None:
    """Return the toolkit's ``major.minor`` release from ``nvcc --version``."""
    if nvcc is None:
        return None
    match = re.search(r"release (\d+\.\d+)", _run([str(nvcc), "--version"]).stdout)
    return match.group(1) if match else None


def _nccl_package_version() -> str | None:
    """Return the installed ``libnccl-dev`` package version, or None without it."""
    if shutil.which("dpkg-query") is None:
        return None
    completed = _run(["dpkg-query", "-W", "-f=${Version}", "libnccl-dev"])
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def detect_runtime() -> Runtime:
    """Describe the active interpreter, its LightGBM and the local CUDA toolkit."""
    nvcc = _find_nvcc()
    return Runtime(
        python_executable=Path(sys.executable),
        venv=Path(sys.prefix),
        purelib=Path(sysconfig.get_paths()["purelib"]),
        lightgbm_version=_lightgbm_version(),
        nvcc=nvcc,
        cuda_version=_cuda_version(nvcc),
        nccl_package_version=_nccl_package_version() if nvcc else None,
        has_dpkg=shutil.which("dpkg-query") is not None,
    )


def nccl_matches_toolkit(runtime: Runtime) -> bool:
    """Return whether the installed NCCL was built for the toolkit's CUDA major version.

    NVIDIA's packages carry the CUDA release in the version (``2.31.2-1+cuda13.3``). A version
    without that tag (Ubuntu's own ``libnccl-dev``) or with another major version does not match.
    Without dpkg the check cannot be made and is treated as a match; with dpkg, a missing
    ``libnccl-dev`` does not match, because the CUDA build requires NCCL.
    """
    if not runtime.has_dpkg:
        return True
    if runtime.nccl_package_version is None or runtime.cuda_version is None:
        return False
    major = runtime.cuda_version.split(".", 1)[0]
    return re.search(rf"\+cuda{major}\.", runtime.nccl_package_version) is not None


def uv_args(runtime: Runtime) -> list[str]:
    """Return the ``uv sync`` flags that build the locked LightGBM with CUDA, or none.

    Empty without a CUDA toolkit or with a mismatched NCCL, so those machines keep the wheel.
    """
    if runtime.nvcc is None or not nccl_matches_toolkit(runtime):
        return []
    return [
        "--no-binary-package",
        "lightgbm",
        "--config-settings-package",
        "lightgbm:cmake.define.USE_CUDA=ON",
        "--config-settings-package",
        f"lightgbm:cmake.define.CMAKE_CUDA_COMPILER={runtime.nvcc}",
    ]


def sync_command(runtime: Runtime, *, rebuild: bool = False) -> list[str]:
    """Return the ``uv sync`` command that reinstalls the locked LightGBM with CUDA.

    ``--reinstall-package`` is needed because uv does not treat an installed registry wheel as
    stale when ``--no-binary-package`` is passed. uv's build cache then supplies an earlier CUDA
    build of the same version in seconds; ``rebuild`` bypasses the cache and compiles from
    source, for when the cached build no longer loads (an NCCL change, which the cache key
    does not see).
    """
    command = ["uv", "sync", "--active", *uv_args(runtime), "--reinstall-package", "lightgbm"]
    if rebuild:
        command.append("--no-cache")
    return command


def cuda_training_works(runtime: Runtime) -> tuple[bool, str]:
    """Train two rounds on the GPU in a child interpreter; return success and its output."""
    completed = _run([str(runtime.python_executable), "-c", _SMOKE_TEST])
    return completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}".strip()


def remove_legacy_files(runtime: Runtime) -> list[Path]:
    """Delete the shim files an earlier installer left in site-packages; return them."""
    removed = []
    for name in _LEGACY_FILES:
        path = runtime.purelib / name
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def _sync(runtime: Runtime, *, rebuild: bool) -> None:
    """Run the reinstall against this venv; raise when uv fails."""
    env = {**os.environ, "VIRTUAL_ENV": str(runtime.venv)}
    completed = _run(sync_command(runtime, rebuild=rebuild), capture=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"uv sync with the LightGBM CUDA build failed ({completed.returncode}).")


def install(*, require_cuda: bool = False, runtime: Runtime | None = None) -> InstallResult:
    """Make the venv's LightGBM a CUDA build when this machine can build one.

    Does nothing when LightGBM already trains on the GPU. Otherwise reinstalls it with the CUDA
    flags (a cache hit, in seconds, when uv built this version before) and, only if that build
    still cannot train on the GPU, compiles it again from source.
    """
    runtime = runtime or detect_runtime()

    def skip(status: str, message: str) -> InstallResult:
        """Log why nothing was done, or raise when CUDA support is required."""
        if require_cuda:
            raise RuntimeError(message)
        log.info(message)
        return InstallResult(status=status, rebuilt=False)

    if not runtime.lightgbm_version:
        return skip("skipped-no-lightgbm", "LightGBM is not installed; run uv sync first.")
    if runtime.nvcc is None:
        return skip("skipped-no-cuda", "No CUDA toolkit found; LightGBM stays CPU-only.")
    if not nccl_matches_toolkit(runtime):
        return skip(
            "skipped-nccl-mismatch",
            f"libnccl-dev {runtime.nccl_package_version} was not built for CUDA "
            f"{runtime.cuda_version}; LightGBM stays CPU-only. Install NVIDIA's libnccl2 and "
            f"libnccl-dev packages tagged +cuda{runtime.cuda_version} "
            "(developer.download.nvidia.com/compute/cuda/repos) to enable the CUDA build.",
        )

    for path in remove_legacy_files(runtime):
        log.info("Removed legacy CUDA shim file %s", path)

    if cuda_training_works(runtime)[0]:
        log.info("LightGBM %s already trains on CUDA; nothing to do.", runtime.lightgbm_version)
        return InstallResult(status="cuda", rebuilt=False)

    log.info("Installing the CUDA build of LightGBM %s.", runtime.lightgbm_version)
    _sync(runtime, rebuild=False)
    if cuda_training_works(runtime)[0]:
        log.info("LightGBM %s trains on CUDA.", runtime.lightgbm_version)
        return InstallResult(status="installed", rebuilt=False)

    log.info("That build cannot train on CUDA; rebuilding from source without the cache.")
    _sync(runtime, rebuild=True)
    works, output = cuda_training_works(runtime)
    if not works:
        raise RuntimeError(f"LightGBM was rebuilt but still cannot train on CUDA:\n{output}")
    log.info("LightGBM %s rebuilt with CUDA support.", runtime.lightgbm_version)
    return InstallResult(status="rebuilt", rebuilt=True)


def status(runtime: Runtime | None = None) -> str:
    """Return ``cuda``, ``cpu-only``, ``broken``, ``no-cuda-toolkit`` or ``missing``."""
    runtime = runtime or detect_runtime()
    if not runtime.lightgbm_version:
        return "missing"
    if runtime.nvcc is None:
        return "no-cuda-toolkit"
    works, output = cuda_training_works(runtime)
    if works:
        return "cuda"
    return "cpu-only" if any(f in output for f in _CPU_ONLY_FRAGMENTS) else "broken"


USAGE = """Usage: nfl-lightgbm-cuda-install [install [--require-cuda] | status | uv-args]

install   Make LightGBM a CUDA build when this machine has a CUDA toolkit and a matching
          NCCL (the default command). Does nothing if it already is one; compiles only when
          uv has no usable cached build of the locked version.
status    Print cuda, cpu-only, broken, no-cuda-toolkit or missing; change nothing.
uv-args   Print the uv sync flags for the CUDA build, one per line (nothing on CPU-only
          machines), for scripts that run uv sync or uv sync --check themselves."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``nfl-lightgbm-cuda-install`` command line."""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args and not args[0].startswith("-") else "install"
    if command in {"-h", "--help", "help"} or "--help" in args or "-h" in args:
        sys.stdout.write(USAGE + "\n")
        return 0
    if command in {"status", "uv-args"} and not args:
        runtime = detect_runtime()
        lines = [status(runtime)] if command == "status" else uv_args(runtime)
        sys.stdout.write("".join(f"{line}\n" for line in lines))
        return 0
    if command != "install" or any(arg != "--require-cuda" for arg in args):
        sys.stderr.write(USAGE + "\n")
        return 2
    try:
        install(require_cuda="--require-cuda" in args)
    except RuntimeError as error:
        log.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
