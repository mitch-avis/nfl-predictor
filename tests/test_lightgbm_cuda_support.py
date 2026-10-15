"""Tests for the as-needed LightGBM CUDA installer and its hooks."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any

import pytest

from nfl_predictor import lightgbm_cuda

REPO_ROOT = Path(__file__).resolve().parents[1]


def _runtime(tmp_path: Path, **overrides: Any) -> lightgbm_cuda.Runtime:
    """Return a runtime with a CUDA 13.3 toolkit and a matching NCCL, overridable per test."""
    base = lightgbm_cuda.Runtime(
        python_executable=tmp_path / "bin" / "python",
        venv=tmp_path,
        purelib=tmp_path / "site-packages",
        lightgbm_version="4.7.0",
        nvcc=tmp_path / "cuda" / "bin" / "nvcc",
        cuda_version="13.3",
        nccl_package_version="2.31.2-1+cuda13.3",
        has_dpkg=True,
    )
    return dataclasses.replace(base, **overrides)


def _record_calls(
    monkeypatch: pytest.MonkeyPatch, smoke_results: list[bool], build_code: int = 0
) -> list[str]:
    """Replace the smoke test and the build with fakes; return the ordered call log."""
    calls: list[str] = []
    results = iter(smoke_results)

    def fake_smoke(_runtime: lightgbm_cuda.Runtime) -> tuple[bool, str]:
        """Return the next scripted smoke-test outcome."""
        calls.append("smoke")
        return next(results), "CUDA Tree Learner was not enabled in this build"

    def fake_run(
        command: list[str], *, capture: bool = True, env: dict[str, str] | None = None
    ) -> object:
        """Record a cached reinstall or a no-cache rebuild and return its scripted exit code."""
        calls.append("rebuild" if "--no-cache" in command else "reinstall")
        return type("Completed", (), {"returncode": build_code, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(lightgbm_cuda, "cuda_training_works", fake_smoke)
    monkeypatch.setattr(lightgbm_cuda, "_run", fake_run)
    return calls


def test_install_does_nothing_when_cuda_already_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that already trains on the GPU is left alone: one smoke test, no uv call."""
    # Arrange
    calls = _record_calls(monkeypatch, [True])

    # Act
    result = lightgbm_cuda.install(runtime=_runtime(tmp_path))

    # Assert
    assert result == lightgbm_cuda.InstallResult(status="cuda", rebuilt=False)
    assert calls == ["smoke"]


def test_install_reinstalls_the_cached_cuda_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CPU wheel is replaced from uv's cache first, without compiling."""
    # Arrange
    calls = _record_calls(monkeypatch, [False, True])

    # Act
    result = lightgbm_cuda.install(runtime=_runtime(tmp_path))

    # Assert
    assert result == lightgbm_cuda.InstallResult(status="installed", rebuilt=False)
    assert calls == ["smoke", "reinstall", "smoke"]


def test_install_rebuilds_without_the_cache_when_the_cached_build_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only when the cached build still cannot train on the GPU is it compiled again."""
    # Arrange
    calls = _record_calls(monkeypatch, [False, False, True])

    # Act
    result = lightgbm_cuda.install(runtime=_runtime(tmp_path))

    # Assert
    assert result == lightgbm_cuda.InstallResult(status="rebuilt", rebuilt=True)
    assert calls == ["smoke", "reinstall", "smoke", "rebuild", "smoke"]


def test_install_skips_without_a_cuda_toolkit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without nvcc nothing runs."""
    # Arrange
    calls = _record_calls(monkeypatch, [])
    runtime = _runtime(tmp_path, nvcc=None, cuda_version=None)

    # Act
    result = lightgbm_cuda.install(runtime=runtime)

    # Assert
    assert result.status == "skipped-no-cuda"
    assert calls == []


def test_require_cuda_turns_a_missing_toolkit_into_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--require-cuda`` turns the skip without nvcc into an error."""
    # Arrange
    _record_calls(monkeypatch, [])
    runtime = _runtime(tmp_path, nvcc=None, cuda_version=None)

    # Act & Assert
    with pytest.raises(RuntimeError, match="No CUDA toolkit"):
        lightgbm_cuda.install(require_cuda=True, runtime=runtime)


def test_install_skips_an_nccl_built_for_another_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ubuntu's untagged NCCL stops the install before any build."""
    # Arrange
    calls = _record_calls(monkeypatch, [])
    runtime = _runtime(tmp_path, nccl_package_version="2.22.3-1-1")

    # Act
    result = lightgbm_cuda.install(runtime=runtime)

    # Assert
    assert result.status == "skipped-nccl-mismatch"
    assert calls == []


def test_require_cuda_names_the_nccl_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``--require-cuda`` the NCCL mismatch is an error that names the matching package."""
    # Arrange
    _record_calls(monkeypatch, [])
    runtime = _runtime(tmp_path, nccl_package_version="2.22.3-1-1")

    # Act & Assert
    with pytest.raises(RuntimeError, match=r"\+cuda13\.3"):
        lightgbm_cuda.install(require_cuda=True, runtime=runtime)


def test_install_reports_a_failed_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero uv exit is an error, not a silent success."""
    # Arrange
    _record_calls(monkeypatch, [False], build_code=1)

    # Act & Assert
    with pytest.raises(RuntimeError, match="failed"):
        lightgbm_cuda.install(runtime=_runtime(tmp_path))


@pytest.mark.parametrize(
    ("nccl", "has_dpkg", "expected"),
    [
        ("2.31.2-1+cuda13.3", True, True),
        ("2.31.2-1+cuda13.4", True, True),
        ("2.31.2-1+cuda12.9", True, False),
        ("2.22.3-1-1", True, False),
        (None, True, False),
        (None, False, True),
    ],
)
def test_nccl_matches_toolkit(
    tmp_path: Path, nccl: str | None, *, has_dpkg: bool, expected: bool
) -> None:
    """NCCL matches when its package names the toolkit's CUDA major version."""
    # Arrange
    runtime = _runtime(tmp_path, nccl_package_version=nccl, has_dpkg=has_dpkg)

    # Act
    matches = lightgbm_cuda.nccl_matches_toolkit(runtime)

    # Assert
    assert matches is expected


def test_uv_args_name_the_cuda_build(tmp_path: Path) -> None:
    """The flags build LightGBM from source with CUDA."""
    # Act
    args = lightgbm_cuda.uv_args(_runtime(tmp_path))

    # Assert
    assert args[:2] == ["--no-binary-package", "lightgbm"]
    assert "lightgbm:cmake.define.USE_CUDA=ON" in args


@pytest.mark.parametrize(
    "overrides",
    [{"nvcc": None}, {"nccl_package_version": "2.22.3-1-1"}],
    ids=["cpu-only", "nccl-mismatch"],
)
def test_uv_args_are_empty_where_the_cuda_build_cannot_work(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    """CPU-only or mismatched machines get no flags."""
    # Act
    args = lightgbm_cuda.uv_args(_runtime(tmp_path, **overrides))

    # Assert
    assert args == []


def test_sync_command_reinstalls_lightgbm_from_the_cache(tmp_path: Path) -> None:
    """The plain command forces the LightGBM reinstall and keeps uv's cache."""
    # Act
    cached = lightgbm_cuda.sync_command(_runtime(tmp_path))

    # Assert
    assert cached[:3] == ["uv", "sync", "--active"]
    assert cached[-2:] == ["--reinstall-package", "lightgbm"]
    assert "--no-cache" not in cached


def test_sync_command_bypasses_the_cache_only_for_a_rebuild(tmp_path: Path) -> None:
    """The rebuild is the cached command with the cache turned off."""
    # Arrange
    runtime = _runtime(tmp_path)
    cached = lightgbm_cuda.sync_command(runtime)

    # Act
    rebuild = lightgbm_cuda.sync_command(runtime, rebuild=True)

    # Assert
    assert rebuild == [*cached, "--no-cache"]


def test_remove_legacy_files_deletes_only_the_old_shim(tmp_path: Path) -> None:
    """The old shim and its startup hook go; other site-packages files stay."""
    # Arrange
    runtime = _runtime(tmp_path)
    runtime.purelib.mkdir(parents=True)
    for name in ("nfl_predictor_lightgbm_cuda_compat.pth", "liblightgbm_cuda_compat.so"):
        (runtime.purelib / name).write_text("x", encoding="utf-8")
    keep = runtime.purelib / "other.pth"
    keep.write_text("x", encoding="utf-8")

    # Act
    removed = lightgbm_cuda.remove_legacy_files(runtime)

    # Assert
    assert sorted(path.name for path in removed) == [
        "liblightgbm_cuda_compat.so",
        "nfl_predictor_lightgbm_cuda_compat.pth",
    ]
    assert keep.exists()


@pytest.mark.parametrize("argv", [["frobnicate"], ["install", "--force"]], ids=["command", "flag"])
def test_main_rejects_unknown_arguments(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    """An unknown command or flag prints the usage and exits 2."""
    # Act
    status = lightgbm_cuda.main(argv)

    # Assert
    assert status == 2
    assert "Usage:" in capsys.readouterr().err


def test_pyproject_exposes_the_installer_entrypoint() -> None:
    """The installer is a console script of the project."""
    # Act
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    # Assert
    assert config["project"]["scripts"]["nfl-lightgbm-cuda-install"] == (
        "nfl_predictor.lightgbm_cuda:main"
    )


def test_gate_checks_the_environment_without_changing_it() -> None:
    """The gate neither syncs nor rebuilds; its strict sync check gets the CUDA build flags."""
    # Act
    gate = (REPO_ROOT / "scripts" / "gate.sh").read_text(encoding="utf-8")

    # Assert
    assert "nfl-lightgbm-cuda-install install" not in gate
    assert 'run_step "uv sync --active"' not in gate
    assert "nfl-lightgbm-cuda-install uv-args" in gate
    assert 'uv sync --check --active "${LIGHTGBM_CUDA_ARGS[@]}"' in gate


def test_update_requirements_runs_the_installer_after_sync() -> None:
    """Refreshing the environment syncs with the CUDA flags, then fixes LightGBM if needed."""
    # Act
    script = (REPO_ROOT / "update_requirements.sh").read_text(encoding="utf-8")

    # Assert
    main = script[script.index("main() {") :]
    sync = main.index('uv sync --active "${lightgbm_cuda_args[@]}"')
    assert main.index("nfl-lightgbm-cuda-install uv-args") < sync
    assert sync < main.index("install_lightgbm_cuda_build")
    builder = script[script.index("install_lightgbm_cuda_build() {") :]
    assert "nfl-lightgbm-cuda-install install" in builder[: builder.index("\n}")]
