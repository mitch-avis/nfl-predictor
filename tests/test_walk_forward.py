"""Tests for walk-forward split correctness, determinism, and calibration time-awareness."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from scipy.stats import norm

from nfl_predictor import constants, ml_model
from nfl_predictor.ml import ml_model_xgb_utils, walk_forward
from nfl_predictor.ml.feature_spec import resolve_feature_group_columns
from nfl_predictor.utils.logger import log
from tests.fakes import returning


def _fixture_df() -> pd.DataFrame:
    """Create a tiny deterministic dataset spanning multiple seasons/weeks."""
    rows = [
        {
            "season": season,
            "week": week,
            "game_type": "REG",
            "game_id": f"{season}_{week}_{game_idx}",
            "feat1": float(season % 2000) + week + game_idx,
            "feat2": float(season % 2000) - week + game_idx,
            "away_score": 17 + week + (game_idx * 3),
            "home_score": 24 + week - (game_idx * 5),
            "home_moneyline": -110,
        }
        for season, week, game_idx in itertools.product((2022, 2023), (1, 2, 3), (0, 1))
    ]
    df = pd.DataFrame(rows)
    return df[
        [
            "season",
            "week",
            "game_type",
            "game_id",
            "feat1",
            "feat2",
            "home_moneyline",
            "away_score",
            "home_score",
        ]
    ]


def _base_config() -> walk_forward.WalkForwardConfig:
    """Return a walk-forward config suitable for unit tests."""
    return walk_forward.WalkForwardConfig(
        eval_seasons=[2023],
        eval_last_n_seasons=1,
        wf_start_week=2,
        calibration="none",
        random_seed=7,
        include_market=False,
        market_anchor=False,
        feature_start="feat1",
        feature_end="feat2",
        xgb_params_overrides={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_jobs": 1,
            "verbosity": 0,
        },
    )


def _capture_info_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route ``walk_forward.log.info`` into a list of formatted messages and return it."""
    messages: list[str] = []

    def capture(message: str, *args: object) -> None:
        messages.append(message % args if args else message)

    monkeypatch.setattr(log, "info", capture)
    return messages


def test_walk_forward_split_excludes_eval_week() -> None:
    """Train set must exclude any games from the predicted eval week."""
    # Arrange
    df = _fixture_df()

    # Act
    folds = walk_forward.build_walk_forward_folds(df, [2023], start_week=2)

    # Assert
    assert folds
    for fold in folds:
        same_season = fold.train_df[fold.train_df["season"] == fold.season]
        assert (same_season["week"] < fold.week).all()


def test_summarize_eval_window_flags_incomplete_regular_season() -> None:
    """Summarize eval window reports incomplete regular seasons when data is partial."""
    # Arrange
    df = _fixture_df()

    # Act
    summary = walk_forward.summarize_eval_window(
        df,
        eval_seasons=[2023],
        start_week=2,
        include_postseason=False,
    )

    # Assert
    assert summary["include_postseason"] is False
    assert 2023 in summary["incomplete_seasons"]
    season_summary = summary["seasons"]["2023"]
    assert season_summary["eval_start_week"] == 2
    assert season_summary["incomplete_regular_season"] is True


def test_walk_forward_deterministic_outputs() -> None:
    """Fixed seeds yield identical per-fold outputs."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    result_a = walk_forward.run_walk_forward_backtest(df, config)

    # Act
    result_b = walk_forward.run_walk_forward_backtest(df, config)

    # Assert
    pdt.assert_frame_equal(result_a["predictions"], result_b["predictions"])
    assert result_a["per_week"] == result_b["per_week"]


def test_run_walk_forward_backtest_logs_each_finished_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every finished fold logs its position and timing, so a long run shows progress."""
    # Arrange
    messages: list[str] = []

    def _capture(msg: str, *args: object, **_kwargs: object) -> None:
        """Record the formatted message."""
        messages.append(msg % args if args else msg)

    monkeypatch.setattr(log, "info", _capture)

    # Act
    walk_forward.run_walk_forward_backtest(_fixture_df(), _base_config())

    # Assert
    progress = [message for message in messages if message.startswith("Walk-forward fold")]
    assert len(progress) == 2
    assert progress[0].startswith("Walk-forward fold 1/2 done: season 2023 week 2 (2 games")
    assert progress[1].startswith("Walk-forward fold 2/2 done: season 2023 week 3 (2 games")
    assert all("elapsed" in message and "remaining" in message for message in progress)


class _RunStoppedError(Exception):
    """Raised by a fold callback to simulate a run stopped partway through."""


def _run_recording_folds(
    df: pd.DataFrame,
    config: walk_forward.WalkForwardConfig,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> tuple[dict[str, object], list[tuple[int, int]]]:
    """Run the backtest and return its result plus the (season, week) of each computed fold."""
    computed: list[tuple[int, int]] = []

    def _record(_metrics: dict[str, object], fold: walk_forward.WalkForwardFold) -> None:
        """Record a fold that was trained rather than restored."""
        computed.append((fold.season, fold.week))

    result = walk_forward.run_walk_forward_backtest(
        df,
        config,
        fold_callback=_record,
        checkpoints=(
            None
            if checkpoint_dir is None
            else walk_forward.FoldCheckpoints(checkpoint_dir, resume=resume)
        ),
    )
    return result, computed


def _assert_same_results(actual: dict[str, object], expected: dict[str, object]) -> None:
    """Assert two backtest results carry identical per-week metrics and predictions."""
    assert actual["per_week"] == expected["per_week"]
    assert actual["feature_list"] == expected["feature_list"]
    predictions = actual["predictions"]
    expected_predictions = expected["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    assert isinstance(expected_predictions, pd.DataFrame)
    pdt.assert_frame_equal(predictions, expected_predictions, check_exact=True)


def test_resumed_run_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    """A run stopped after its first week resumes from the checkpoint and ends identical."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    clean = walk_forward.run_walk_forward_backtest(df, config)

    def _stop(_metrics: dict[str, object], _fold: walk_forward.WalkForwardFold) -> None:
        """Stop the run as soon as the first week is finished."""
        raise _RunStoppedError

    with pytest.raises(_RunStoppedError):
        walk_forward.run_walk_forward_backtest(
            df, config, fold_callback=_stop, checkpoints=walk_forward.FoldCheckpoints(tmp_path)
        )

    # Act
    resumed, computed = _run_recording_folds(df, config, checkpoint_dir=tmp_path)

    # Assert
    assert computed == [(2023, 3)]
    assert resumed["checkpoint"] == {
        "dir": str(tmp_path / walk_forward.fold_checkpoint_fingerprint(df, config)[:20]),
        "restored_folds": 1,
        "computed_folds": 1,
    }
    _assert_same_results(resumed, clean)


def test_finished_run_restores_every_week(tmp_path: Path) -> None:
    """Re-running a finished run trains nothing and reproduces the same results."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    first, first_computed = _run_recording_folds(df, config, checkpoint_dir=tmp_path)

    # Act
    second, second_computed = _run_recording_folds(df, config, checkpoint_dir=tmp_path)

    # Assert
    assert first_computed == [(2023, 2), (2023, 3)]
    assert second_computed == []
    _assert_same_results(second, first)


def test_checkpoints_from_another_config_are_not_reused(tmp_path: Path) -> None:
    """A different config fingerprints differently, so every week is trained again."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    changed = replace(
        config,
        xgb_params_overrides={**(config.xgb_params_overrides or {}), "n_estimators": 11},
    )
    _run_recording_folds(df, config, checkpoint_dir=tmp_path)
    expected = walk_forward.run_walk_forward_backtest(df, changed)

    # Act
    result, computed = _run_recording_folds(df, changed, checkpoint_dir=tmp_path)

    # Assert
    assert computed == [(2023, 2), (2023, 3)]
    _assert_same_results(result, expected)


def test_corrupt_checkpoint_is_recomputed(tmp_path: Path) -> None:
    """An unreadable checkpoint is ignored and that week is trained again."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    clean, _ = _run_recording_folds(df, config, checkpoint_dir=tmp_path)
    (fold_file,) = tmp_path.rglob("fold_2023_w02.joblib")
    fold_file.write_bytes(b"not a checkpoint")

    # Act
    result, computed = _run_recording_folds(df, config, checkpoint_dir=tmp_path)

    # Assert
    assert computed == [(2023, 2)]
    _assert_same_results(result, clean)


def test_resume_disabled_recomputes_every_week(tmp_path: Path) -> None:
    """With resume off, existing checkpoints are overwritten rather than read."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    _run_recording_folds(df, config, checkpoint_dir=tmp_path)

    # Act
    _, computed = _run_recording_folds(df, config, checkpoint_dir=tmp_path, resume=False)

    # Assert
    assert computed == [(2023, 2), (2023, 3)]


def test_run_without_a_checkpoint_dir_reports_no_checkpoint() -> None:
    """Checkpointing is opt-in for library callers."""
    # Act
    result = walk_forward.run_walk_forward_backtest(_fixture_df(), _base_config())

    # Assert
    assert result["checkpoint"] is None


def test_fold_checkpoint_fingerprint_is_stable_for_identical_inputs() -> None:
    """The fingerprint is stable for identical inputs."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    baseline = walk_forward.fold_checkpoint_fingerprint(df, config)

    # Act
    result = walk_forward.fold_checkpoint_fingerprint(df.copy(), config)

    # Assert
    assert result == baseline


def test_fold_checkpoint_fingerprint_moves_with_the_data() -> None:
    """The fingerprint moves when the data changes."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    baseline = walk_forward.fold_checkpoint_fingerprint(df, config)
    changed_df = df.copy()
    changed_df.loc[0, "feat1"] = 99.0

    # Act
    result = walk_forward.fold_checkpoint_fingerprint(changed_df, config)

    # Assert
    assert result != baseline


def test_fold_checkpoint_fingerprint_moves_with_the_config() -> None:
    """The fingerprint moves when the config changes."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    baseline = walk_forward.fold_checkpoint_fingerprint(df, config)

    # Act
    result = walk_forward.fold_checkpoint_fingerprint(df, replace(config, wf_start_week=3))

    # Assert
    assert result != baseline


def test_fold_checkpoint_fingerprint_sources_are_the_package_modelling_files() -> None:
    """The fingerprint reads the package's ``ml`` modules, ``constants`` and ``ml_model``."""
    # Arrange
    package_dir = Path(walk_forward.__file__).resolve().parents[1]

    # Act
    sources = walk_forward._modelling_source_files()

    # Assert
    assert [path.name for path in sources] == [
        *sorted(path.name for path in (package_dir / "ml").glob("*.py")),
        "constants.py",
        "ml_model.py",
    ]
    assert all(path.resolve().is_relative_to(package_dir) for path in sources)


def _copy_modelling_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Copy the modelling source files elsewhere and point the fingerprint at the copies."""
    sources = walk_forward._modelling_source_files()
    copies: list[Path] = []
    for index, path in enumerate(sources):
        target = tmp_path / "elsewhere" / str(index) / path.name
        target.parent.mkdir(parents=True)
        target.write_bytes(path.read_bytes())
        copies.append(target)
    monkeypatch.setattr(walk_forward, "_modelling_source_files", lambda: copies)
    return copies


def test_fold_checkpoint_fingerprint_hashes_source_names_and_bytes_not_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical source files in another directory give the same fingerprint."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    baseline = walk_forward.fold_checkpoint_fingerprint(df, config)
    _copy_modelling_sources(tmp_path, monkeypatch)

    # Act
    result = walk_forward.fold_checkpoint_fingerprint(df, config)

    # Assert
    assert result == baseline


def test_fold_checkpoint_fingerprint_moves_with_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing a copied source file's bytes changes the fingerprint."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    baseline = walk_forward.fold_checkpoint_fingerprint(df, config)
    copies = _copy_modelling_sources(tmp_path, monkeypatch)
    copies[0].write_bytes(copies[0].read_bytes() + b"\n")

    # Act
    result = walk_forward.fold_checkpoint_fingerprint(df, config)

    # Assert
    assert result != baseline


def _with_device(
    config: walk_forward.WalkForwardConfig, device: str | None
) -> walk_forward.WalkForwardConfig:
    """Return ``config`` with its XGBoost device override set, or removed for ``None``."""
    overrides = {k: v for k, v in (config.xgb_params_overrides or {}).items() if k != "device"}
    if device is not None:
        overrides["device"] = device
    return replace(config, xgb_params_overrides=overrides)


@pytest.mark.parametrize("requested", [None, "auto", "cuda"])
def test_with_resolved_xgb_device_names_the_concrete_device(
    monkeypatch: pytest.MonkeyPatch, requested: str | None
) -> None:
    """``auto`` or no device becomes the device the run trains on; other overrides stay."""
    # Arrange
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    config = _base_config()

    # Act
    resolved = walk_forward.with_resolved_xgb_device(_with_device(config, requested))

    # Assert
    assert resolved.xgb_params_overrides == {
        **(config.xgb_params_overrides or {}),
        "device": "cuda",
    }


def test_with_resolved_xgb_device_without_overrides_names_the_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config with no overrides gains only the resolved device."""
    # Arrange
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    config = _base_config()

    # Act
    no_overrides = walk_forward.with_resolved_xgb_device(replace(config, xgb_params_overrides=None))

    # Assert
    assert no_overrides.xgb_params_overrides == {"device": "cuda"}


@pytest.mark.parametrize("requested", ["auto", None, "gpu"])
def test_fold_checkpoint_fingerprint_hashes_the_resolved_device(
    monkeypatch: pytest.MonkeyPatch, requested: str | None
) -> None:
    """An ``auto`` run shares checkpoints with an explicit run on the same device only."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    gpu = walk_forward.fold_checkpoint_fingerprint(df, _with_device(config, "cuda"))

    # Act
    result = walk_forward.fold_checkpoint_fingerprint(df, _with_device(config, requested))

    # Assert
    assert result == gpu


def test_fold_checkpoint_fingerprint_differs_between_cpu_and_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CPU run does not share checkpoints with a GPU run."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    gpu = walk_forward.fold_checkpoint_fingerprint(df, _with_device(config, "cuda"))

    # Act
    cpu = walk_forward.fold_checkpoint_fingerprint(df, _with_device(config, "cpu"))

    # Assert
    assert cpu != gpu


def test_fold_checkpoint_fingerprint_auto_without_cuda_matches_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``auto`` run on a machine without CUDA shares checkpoints with an explicit CPU run."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: True)
    cpu = walk_forward.fold_checkpoint_fingerprint(df, _with_device(config, "cpu"))
    monkeypatch.setattr(ml_model_xgb_utils, "xgb_cuda_usable", lambda: False)

    # Act
    cpu_auto = walk_forward.fold_checkpoint_fingerprint(df, _with_device(config, "auto"))

    # Assert
    assert cpu_auto == cpu


def test_run_records_the_resolved_xgb_device(tmp_path: Path) -> None:
    """The result and the checkpoint manifest name the concrete device, never ``auto``."""
    # Arrange
    config = _with_device(_base_config(), "auto")

    # Act
    result = walk_forward.run_walk_forward_backtest(
        _fixture_df(), config, checkpoints=walk_forward.FoldCheckpoints(tmp_path)
    )

    # Assert
    assert result["resolved_settings"]["xgb_device"] == "cpu"
    (manifest_path,) = tmp_path.rglob("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["xgb_params_overrides"]["device"] == "cpu"


def test_walk_forward_probabilities_in_bounds() -> None:
    """Home win probabilities are always in [0, 1]."""
    # Arrange
    df = _fixture_df()
    config = _base_config()

    # Act
    result = walk_forward.run_walk_forward_backtest(df, config)

    # Assert
    probs = result["predictions"]["home_win_prob"]
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_walk_forward_quantile_intervals_monotonic() -> None:
    """Walk-forward outputs include monotonic quantile intervals for margin/total."""
    # Arrange
    df = _fixture_df()
    config = _base_config()

    # Act
    result = walk_forward.run_walk_forward_backtest(df, config)

    # Assert
    preds = result["predictions"]
    required = {
        "predicted_margin_p10",
        "predicted_margin_p50",
        "predicted_margin_p90",
        "predicted_total_p10",
        "predicted_total_p50",
        "predicted_total_p90",
    }
    assert required.issubset(preds.columns)

    assert (preds["predicted_margin_p10"] <= preds["predicted_margin_p50"]).all()
    assert (preds["predicted_margin_p50"] <= preds["predicted_margin_p90"]).all()
    assert (preds["predicted_total_p10"] <= preds["predicted_total_p50"]).all()
    assert (preds["predicted_total_p50"] <= preds["predicted_total_p90"]).all()


def test_walk_forward_can_disable_quantiles() -> None:
    """Walk-forward can skip quantile model training for faster comparisons."""
    # Arrange
    df = _fixture_df()
    config = _base_config()
    config = replace(
        config,
        eval_seasons=[2023],
        include_quantiles=False,
    )

    # Act
    result = walk_forward.run_walk_forward_backtest(df, config)

    # Assert
    preds = result["predictions"]
    assert "predicted_margin" in preds.columns
    assert "predicted_total" in preds.columns
    assert "predicted_margin_p10" not in preds.columns
    assert "predicted_total_p90" not in preds.columns


def test_dataset_fingerprint_matches_sha256(tmp_path: Path) -> None:
    """Computes SHA-256 fingerprint of file contents."""
    # Arrange
    path = Path(tmp_path) / "data.bin"
    payload = b"abc\x00def"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    # Act
    result = walk_forward.dataset_fingerprint(path)

    # Assert
    assert result == expected


def test_generate_run_id_is_deterministic_under_fixed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds a stable run_id when datetime is fixed."""

    # Arrange
    class _FixedDatetime:
        @staticmethod
        def now(_tz: object) -> _FixedDatetime:
            """Return a fixed datetime for testing."""
            return _FixedDatetime()

        def strftime(self, _fmt: str) -> str:
            """Return a fixed timestamp string for testing."""
            return "20260110_000000"

    monkeypatch.setattr(walk_forward, "datetime", _FixedDatetime)

    cfg = walk_forward.WalkForwardConfig(
        eval_seasons=[2024],
        eval_last_n_seasons=1,
        wf_start_week=3,
        calibration="none",
        include_market=False,
        market_anchor=False,
        include_quantiles=False,
    )
    run_id_a = walk_forward.generate_run_id("deadbeef", cfg)

    # Act
    run_id_b = walk_forward.generate_run_id("deadbeef", cfg)

    # Assert
    assert run_id_a == run_id_b
    assert run_id_a.startswith("wf_20260110_000000_")
    assert len(run_id_a.split("_")[-1]) == 8


def test_build_metrics_report_shape() -> None:
    """Builds a JSON-serializable metrics report envelope."""
    # Act
    report = walk_forward.build_metrics_report(
        run_id="wf_test",
        created_at="2026-01-10T00:00:00Z",
        config_payload={"foo": "bar"},
        results={
            "per_week": [
                {
                    "week": 3,
                    "games": 1,
                    "deterministic_brier": 0.2,
                    "deterministic_log_loss": 0.6,
                    "market_brier": 0.21,
                    "market_log_loss": 0.61,
                }
            ],
            "per_season": [{"season": 2024, "games": 1}],
            "overall": {
                "games": 1,
                "deterministic_brier": 0.2,
                "deterministic_log_loss": 0.6,
                "market_brier": 0.21,
                "market_log_loss": 0.61,
            },
            "probability_windows": [
                {
                    "window": "all_weeks",
                    "label": "all weeks",
                    "games": 1,
                    "deterministic_brier_vs_market": -0.01,
                    "deterministic_brier_vs_market_ci_low": -0.03,
                    "deterministic_brier_vs_market_ci_high": 0.01,
                    "deterministic_log_loss_vs_market": -0.02,
                    "deterministic_log_loss_vs_market_ci_low": -0.04,
                    "deterministic_log_loss_vs_market_ci_high": 0.0,
                }
            ],
            "reliability": [{"bin_lower": 0.0, "bin_upper": 0.1, "count": 1}],
            "eval_window": {"include_postseason": False, "seasons": {}},
        },
    )

    # Assert
    assert report["run_id"] == "wf_test"
    assert report["metrics"]["overall"]["games"] == 1
    assert report["metrics"]["fold_summary"]["folds"] == 1
    assert isinstance(report["metrics"]["summary_table"], list)
    assert report["metric_strategy"]["primary"][0]["metric"] == "brier"
    assert any(row["metric"] == "deterministic_brier" for row in report["metrics"]["summary_table"])
    assert any(row["metric"] == "market_brier" for row in report["metrics"]["summary_table"])
    assert any(
        row.get("metric") == "deterministic_brier_vs_market" and row.get("window") == "all_weeks"
        for row in report["metrics"]["summary_table"]
    )
    assert report["calibration"]["bin_count"] == walk_forward.RELIABILITY_BINS
    assert report["splits"]["eval_window"]["include_postseason"] is False
    assert report["splits"]["excluded_incomplete_seasons"] == []


@pytest.mark.parametrize(
    ("regular_season_weeks", "expect_kept"),
    [(4, False), (3, True)],
    ids=["seasons_short_of_the_schedule_dropped", "complete_seasons_kept"],
)
def test_filter_incomplete_eval_seasons_tracks_drops(
    monkeypatch: pytest.MonkeyPatch, regular_season_weeks: int, *, expect_kept: bool
) -> None:
    """Filtering incomplete seasons returns kept + dropped lists."""
    # Arrange
    df = _fixture_df()
    eval_seasons = [2022, 2023]
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(regular_season_weeks))

    # Act
    kept, dropped = walk_forward.filter_incomplete_eval_seasons(df, eval_seasons)

    # Assert
    if expect_kept:
        assert kept == eval_seasons
        assert dropped == []
    else:
        assert kept == []
        assert dropped == eval_seasons


def test_aggregate_metrics_includes_market_residuals_and_interval_coverage() -> None:
    """Computes optional market residual MAE and interval coverage diagnostics."""
    # Arrange
    frame = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "actual_margin": [3.0, -7.0],
            "predicted_margin": [2.0, -6.0],
            "actual_total": [41.0, 38.0],
            "predicted_total": [40.0, 39.0],
            "actual_home_win": [1, 0],
            "home_win_prob": [0.8, 0.4],
            "deterministic_home_win_prob": [0.7, 0.3],
            "market_home_win_prob": [0.65, 0.35],
            "expected_points": [1.5, 2.0],
            "actual_points": [1.0, 2.0],
            "pick_correct": [True, True],
            "market_baseline_margin": [1.0, -5.0],
            "market_baseline_total": [39.0, 37.0],
            "predicted_margin_p10": [0.0, -9.0],
            "predicted_margin_p90": [5.0, -3.0],
            "predicted_total_p10": [35.0, 33.0],
            "predicted_total_p90": [47.0, 45.0],
        }
    )

    # Act
    metrics = walk_forward._aggregate_metrics(frame, market_anchor=True)

    # Assert
    assert metrics["season"] == 2024
    assert metrics["games"] == 2
    assert "market_margin_resid_mae" in metrics
    assert "market_total_resid_mae" in metrics
    assert "margin_p10_p90_coverage" in metrics
    assert "total_p10_p90_coverage" in metrics
    assert "reliability_ece" in metrics
    assert metrics["deterministic_brier"] == pytest.approx(0.09)
    assert metrics["deterministic_log_loss"] == pytest.approx(0.3566749439)
    assert metrics["deterministic_pick_accuracy"] == pytest.approx(1.0)
    assert metrics["market_brier"] == pytest.approx(0.1225)
    assert metrics["market_log_loss"] == pytest.approx(0.4307829161)
    assert metrics["market_pick_accuracy"] == pytest.approx(1.0)
    assert metrics["deterministic_brier_vs_market"] == pytest.approx(-0.0325)
    assert metrics["deterministic_log_loss_vs_market"] == pytest.approx(-0.0741079722)


def test_season_win_totals_summary() -> None:
    """Summarize expected vs actual season win totals."""
    # Arrange
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 4],
            "away_abbr": ["AAA", "BBB"],
            "home_abbr": ["BBB", "AAA"],
            "home_win_prob": [0.7, 0.4],
            "actual_margin": [3.0, -7.0],
        }
    )

    # Act
    totals = walk_forward._season_win_totals(predictions)

    # Assert
    per_team = {row["team"]: row for row in totals["per_team"]}
    assert totals["overall"]["teams"] == 2
    assert per_team["AAA"]["expected_wins"] == pytest.approx(0.7)
    assert per_team["AAA"]["actual_wins"] == pytest.approx(0.0)
    assert per_team["BBB"]["expected_wins"] == pytest.approx(1.3)
    assert per_team["BBB"]["actual_wins"] == pytest.approx(2.0)
    assert totals["overall"]["mean_abs_error"] == pytest.approx(0.7)


def test_calibration_drift_summary() -> None:
    """Summarize calibration drift by season/week."""
    # Arrange
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 3],
            "home_win_prob": [0.8, 0.2],
            "actual_home_win": [1, 0],
        }
    )

    # Act
    drift = walk_forward._calibration_drift(predictions)

    # Assert
    assert len(drift["per_week"]) == 1
    row = drift["per_week"][0]
    assert row["season"] == 2024
    assert row["week"] == 3
    assert row["avg_pred"] == pytest.approx(0.5)
    assert row["avg_actual"] == pytest.approx(0.5)
    assert row["bias"] == pytest.approx(0.0)
    assert row["brier"] == pytest.approx(0.04)


def test_probability_window_rows_include_bootstrap_intervals() -> None:
    """Window summaries include deterministic-versus-market intervals for each standard window."""
    # Arrange
    predictions = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4],
            "actual_margin": [7.0, -3.0, 10.0, -6.0],
            "predicted_margin": [6.0, -2.0, 8.0, -5.0],
            "actual_total": [45.0, 41.0, 48.0, 39.0],
            "predicted_total": [44.0, 40.0, 47.0, 38.0],
            "actual_home_win": [1, 0, 1, 0],
            "home_win_prob": [0.76, 0.41, 0.81, 0.36],
            "deterministic_home_win_prob": [0.74, 0.38, 0.79, 0.33],
            "market_home_win_prob": [0.71, 0.42, 0.76, 0.39],
            "expected_points": [1.0, 2.0, 3.0, 4.0],
            "actual_points": [1.0, 2.0, 3.0, 4.0],
            "pick_correct": [True, True, True, True],
        }
    )

    # Act
    rows = walk_forward._probability_window_rows(predictions, seed=7)

    # Assert
    by_window = {row["window"]: row for row in rows}
    assert set(by_window) == {"week_1_only", "week_2_only", "weeks_3_18", "all_weeks"}
    assert by_window["all_weeks"]["deterministic_brier_vs_market"] < 0
    assert "deterministic_brier_vs_market_ci_low" in by_window["all_weeks"]
    assert "deterministic_brier_vs_market_ci_high" in by_window["all_weeks"]
    assert "deterministic_log_loss_vs_market_ci_low" in by_window["all_weeks"]
    assert "deterministic_log_loss_vs_market_ci_high" in by_window["all_weeks"]
    assert by_window["weeks_3_18"]["games"] == 2


def test_git_commit_hash_returns_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns None when git command fails or returns non-zero."""

    # Arrange
    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(subprocess, "run", returning(_Result()))

    # Act
    result = walk_forward._git_commit_hash()

    # Assert
    assert result is None


def test_library_versions_handles_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Records None version when a dependency import fails."""

    # Arrange
    def _fake_import(name: str):
        if name == "optuna":
            msg = "missing"
            raise ImportError(msg)
        return type("M", (), {"__version__": "1.0.0"})

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    # Act
    versions = walk_forward._library_versions()

    # Assert
    assert versions["optuna"] is None
    assert versions["numpy"] == "1.0.0"


def _write_games_csv(tmp_path: Path) -> Path:
    """Write a two-game CSV with one regular-season and one postseason game."""
    path = tmp_path / "games.csv"
    path.write_text("season,week,game_type\n2024,1,REG\n2024,2,WC\n", encoding="utf-8")
    return path


def test_load_games_reads_the_csv_and_logs_its_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Game loading reads every row and logs the source path."""
    # Arrange
    path = _write_games_csv(tmp_path)
    messages = _capture_info_messages(monkeypatch)

    # Act
    loaded = walk_forward.load_games(path)

    # Assert
    assert len(loaded) == 2
    assert any(str(path) in message for message in messages)


def test_filter_regular_season_keeps_only_regular_season_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Season filtering keeps the regular-season games and logs the filter."""
    # Arrange
    path = _write_games_csv(tmp_path)
    messages = _capture_info_messages(monkeypatch)
    loaded = walk_forward.load_games(path)

    # Act
    filtered = walk_forward.filter_regular_season(loaded)

    # Assert
    assert len(filtered) == 1
    assert filtered["game_type"].iloc[0] == "REG"
    assert any("Filtered to regular-season games" in message for message in messages)


def test_filter_regular_season_with_postseason_passes_the_frame_through(
    tmp_path: Path,
) -> None:
    """Including the postseason returns the input frame unchanged."""
    # Arrange
    loaded = walk_forward.load_games(_write_games_csv(tmp_path))

    # Act
    result = walk_forward.filter_regular_season(loaded, include_postseason=True)

    # Assert
    assert result is loaded


def test_filter_regular_season_without_game_type_passes_the_frame_through(
    tmp_path: Path,
) -> None:
    """A frame without a game_type column is returned unchanged."""
    # Arrange
    loaded = walk_forward.load_games(_write_games_csv(tmp_path))
    no_game_type = loaded.drop(columns=["game_type"])

    # Act
    result = walk_forward.filter_regular_season(no_game_type)

    # Assert
    assert result is no_game_type


def test_resolve_eval_seasons_rejects_an_empty_frame() -> None:
    """A frame with no seasons cannot resolve evaluation seasons."""
    # Arrange
    empty_df = pd.DataFrame(columns=["season", "week"])

    # Act & Assert
    with pytest.raises(ValueError, match="No seasons available"):
        walk_forward.resolve_eval_seasons(empty_df, None, 1)


def test_resolve_eval_seasons_drops_missing_requested_seasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested seasons absent from the data are dropped with a log message."""
    # Arrange
    df = _fixture_df()
    messages = _capture_info_messages(monkeypatch)

    # Act
    result = walk_forward.resolve_eval_seasons(df, [2021, 2023], 1)

    # Assert
    assert result == [2023]
    assert any("Dropping missing eval seasons" in message for message in messages)


@pytest.mark.parametrize(
    ("requested", "last_n", "match"),
    [
        ([1999], 1, "None of the requested eval seasons"),
        (None, 0, "eval_last_n_seasons must be positive"),
    ],
)
def test_resolve_eval_seasons_rejects_invalid_requests(
    requested: list[int] | None, last_n: int, match: str
) -> None:
    """Requests naming no available season or a non-positive window are errors."""
    # Arrange
    df = _fixture_df()

    # Act & Assert
    with pytest.raises(ValueError, match=match):
        walk_forward.resolve_eval_seasons(df, requested, last_n)


def test_resolve_eval_seasons_caps_the_last_n_window_at_the_available_seasons() -> None:
    """Asking for more trailing seasons than exist returns every season."""
    # Arrange
    df = _fixture_df()

    # Act
    result = walk_forward.resolve_eval_seasons(df, None, 5)

    # Assert
    assert result == [2022, 2023]


def test_build_walk_forward_folds_requires_season_and_week_columns() -> None:
    """Fold building fails without a week column."""
    # Act & Assert
    with pytest.raises(ValueError, match="season and week columns are required"):
        walk_forward.build_walk_forward_folds(pd.DataFrame({"season": [2024]}), [2024], 1)


def test_build_walk_forward_folds_skips_weeks_without_training_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A week with no earlier games yields no fold and logs the skip."""
    # Arrange
    messages = _capture_info_messages(monkeypatch)
    single_week_df = pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "game_type": ["REG"],
            "away_score": [17],
            "home_score": [24],
        }
    )

    # Act
    result = walk_forward.build_walk_forward_folds(single_week_df, [2024], 1)

    # Assert
    assert result == []
    assert any("no training data" in message for message in messages)


def test_build_walk_forward_folds_includes_postseason_weeks() -> None:
    """With the postseason included, a postseason week becomes a fold."""
    # Arrange
    postseason_df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [18, 20],
            "game_type": ["REG", "SB"],
            "away_score": [17, 20],
            "home_score": [24, 27],
        }
    )

    # Act
    postseason_folds = walk_forward.build_walk_forward_folds(
        postseason_df,
        [2024],
        20,
        include_postseason=True,
    )

    # Assert
    assert len(postseason_folds) == 1
    assert postseason_folds[0].week == 20


def test_summarize_eval_window_with_postseason_ends_at_the_last_postseason_week() -> None:
    """The eval window extends into the postseason when it is included."""
    # Act
    postseason_summary = walk_forward.summarize_eval_window(
        pd.DataFrame({"season": [2024], "week": [20]}),
        eval_seasons=[2024],
        start_week=19,
        include_postseason=True,
    )

    # Assert
    assert postseason_summary["seasons"]["2024"]["eval_end_week"] == 20


def test_resolve_market_settings_without_market_columns_disables_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing market columns turn off the market features and anchor, with log messages."""
    # Arrange
    messages = _capture_info_messages(monkeypatch)
    no_market_df = pd.DataFrame({"season": [2024], "week": [1]})

    # Act
    result = walk_forward.resolve_market_settings(
        no_market_df, include_market=True, market_transform=None, market_anchor=True
    )

    # Assert
    assert result == (
        False,
        False,
        False,
    )
    assert any("Market columns missing" in message for message in messages)
    assert any("Market anchor requested" in message for message in messages)


def test_resolve_market_settings_anchors_without_market_features() -> None:
    """Market columns allow the anchor even when market features are off."""
    # Arrange
    market_df = pd.DataFrame({"home_spread": [-3.0], "total_line": [44.5]})

    # Act
    result = walk_forward.resolve_market_settings(
        market_df, include_market=False, market_transform=None, market_anchor=True
    )

    # Assert
    assert result == (
        False,
        True,
        True,
    )


def test_xgb_params_for_config_passes_overrides_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config's overrides and random seed reach the XGBoost parameter resolver."""
    # Arrange
    captured: dict[str, object] = {}

    def fake_resolve(_defaults: dict[str, Any], overrides: dict[str, Any]) -> object:
        return captured.setdefault("overrides", dict(overrides)) or dict(overrides)

    monkeypatch.setattr(ml_model, "resolve_xgb_params", fake_resolve)
    config = replace(_base_config(), xgb_params_overrides={"max_depth": 4}, random_seed=99)

    # Act
    walk_forward.xgb_params_for_config(config)

    # Assert
    assert captured["overrides"] == {"max_depth": 4, "random_state": 99}


def test_walk_forward_backtest_covers_market_anchor_uncertainty_and_callback() -> None:
    """A market-aware walk-forward run should exercise quantiles and callbacks."""
    # Arrange
    df = _fixture_df().copy()
    df["home_spread"] = -3.0
    df["total_line"] = 44.5
    df["away_moneyline"] = 100

    config = replace(
        _base_config(),
        include_market=True,
        market_anchor=True,
        include_quantiles=True,
    )

    folds: list[tuple[int, int, str]] = []

    def _capture(metrics: dict[str, object], fold: walk_forward.WalkForwardFold) -> None:
        """Record the fold callback payload for assertions."""
        folds.append((fold.season, fold.week, str(metrics["calibration_method"])))

    # Act
    result = walk_forward.run_walk_forward_backtest(df, config, fold_callback=_capture)

    # Assert
    assert len(folds) == 2
    assert result["resolved_settings"]["market_anchor"] is True
    assert "market_baseline_margin" in result["predictions"].columns
    assert "market_baseline_total" in result["predictions"].columns
    assert "predicted_margin_p10" in result["predictions"].columns
    assert result["excluded_incomplete_seasons"] == []
    assert result["eval_window"]["include_postseason"] is False
    assert set(result["predictions"]["calibration_method"].unique()) == {"none"}


def test_walk_forward_fails_cleanly_when_every_eval_season_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete-season filtering that removes every eval season is an error, not a no-op."""
    # Arrange
    df = _fixture_df()
    monkeypatch.setattr(constants, "get_regular_season_weeks", returning(99))
    incomplete_config = replace(_base_config(), exclude_incomplete_seasons=True)

    # Act & Assert
    with pytest.raises(ValueError, match="No complete seasons available"):
        walk_forward.run_walk_forward_backtest(df, incomplete_config)


def test_walk_forward_auto_calibration_uses_the_deterministic_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The submitted probability is the deterministic floor of each predicted margin."""
    # Act
    result = walk_forward.run_walk_forward_backtest(
        _fixture_df(),
        replace(_base_config(), calibration="auto", include_quantiles=False),
    )

    # Assert
    predictions = result["predictions"]
    margins: npt.NDArray[np.float64] = predictions["predicted_margin"].to_numpy()
    np.testing.assert_allclose(
        predictions["home_win_prob"].to_numpy(),
        norm.cdf(margins / constants.SCORE_DIFF_STD_DEV),
    )

    assert set(result["predictions"]["calibration_method"].unique()) == {"none"}


def test_walk_forward_disables_small_window_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk-forward fits should no longer use per-fold early stopping windows."""

    # Arrange
    class _DummyBooster:
        def num_boosted_rounds(self) -> int:
            return 12

    class _DummyModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.n_estimators = 12

        def get_booster(self) -> _DummyBooster:
            return _DummyBooster()

        def get_params(self) -> dict[str, object]:
            return {"device": "cpu"}

    margin_rounds: list[object] = []
    quantile_rounds: list[object] = []

    def fake_fit_margin_total_models(
        *_args: object, **kwargs: object
    ) -> tuple[_DummyModel, _DummyModel]:
        margin_rounds.append(kwargs.get("early_stopping_rounds"))
        return _DummyModel("margin"), _DummyModel("total")

    def fake_fit_quantile_models(*_args: object, **kwargs: object) -> dict[float, _DummyModel]:
        quantile_rounds.append(kwargs.get("early_stopping_rounds"))
        return {}

    def fake_predict_xgb(model: _DummyModel, x: np.ndarray) -> np.ndarray:
        if model.name == "margin":
            return np.full(x.shape[0], 3.0, dtype=float)
        return np.full(x.shape[0], 44.0, dtype=float)

    monkeypatch.setattr(
        ml_model,
        "fit_margin_total_models",
        fake_fit_margin_total_models,
    )
    monkeypatch.setattr(ml_model, "fit_quantile_models", fake_fit_quantile_models)
    monkeypatch.setattr(ml_model, "predict_xgb", fake_predict_xgb)

    # Act
    result = walk_forward.run_walk_forward_backtest(
        _fixture_df(),
        replace(_base_config(), include_quantiles=True),
    )

    # Assert
    assert margin_rounds == [None, None]
    assert quantile_rounds == [None, None, None, None]
    assert result["resolved_settings"]["in_season_early_stopping"] is False
    first_week = result["per_week"][0]
    assert first_week["margin_model.best_iteration"] == 11
    assert first_week["total_model.best_iteration"] == 11
    assert first_week["margin_model.early_stopped"] is False
    # Running the full budget is the configured behavior, not a warning condition.
    assert "iteration_warnings" not in first_week


def test_iteration_details_warns_only_on_real_early_stopping_outcomes() -> None:
    """Cap and below-10 flags describe early stopping; a full-budget fit is not flagged."""

    # Arrange
    class _Booster:
        def __init__(self, rounds: int) -> None:
            self._rounds = rounds

        def num_boosted_rounds(self) -> int:
            return self._rounds

    class _FullBudget:
        n_estimators = 12

        def get_booster(self) -> _Booster:
            return _Booster(12)

    class _StoppedAtCap:
        n_estimators = 12
        best_iteration = 11

        def get_booster(self) -> _Booster:
            return _Booster(12)

    class _StoppedEarly:
        n_estimators = 12
        best_iteration = 3

        def get_booster(self) -> _Booster:
            return _Booster(4)

    # Act
    details = walk_forward._iteration_details(
        {"full": _FullBudget(), "capped": _StoppedAtCap(), "tiny": _StoppedEarly()}
    )

    # Assert
    assert details["full.best_iteration"] == 11
    assert details["full.early_stopped"] is False
    assert "full.warning" not in details
    assert details["capped.best_iteration"] == 11
    assert details["capped.early_stopped"] is True
    assert details["capped.warning"] == "at_cap"
    assert details["tiny.best_iteration"] == 3
    assert details["tiny.warning"] == "below_10"
    assert details["iteration_warnings"] == ["capped:at_cap", "tiny:below_10"]


def test_resolve_feature_group_columns_matches_configured_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with non-empty markers resolves to the columns whose names contain a marker."""
    # Arrange
    monkeypatch.setattr(
        constants,
        "FEATURE_GROUP_COLUMN_MARKERS",
        {"pbp": ("epa",), "other": ("success_rate",)},
    )
    columns = ["away_epa_per_play", "home_success_rate", "week", "season"]

    # Act
    resolved = resolve_feature_group_columns(columns, ["pbp"])

    # Assert
    assert resolved == ["away_epa_per_play"]


def test_resolve_feature_group_columns_empty_markers_matches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty marker tuple for a group resolves to zero columns, dropping nothing."""
    # Arrange
    monkeypatch.setattr(constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ()})
    columns = ["away_epa_per_play", "home_epa_per_play", "epa_per_play_diff"]

    # Act
    resolved = resolve_feature_group_columns(columns, ["pbp"])

    # Assert
    assert resolved == []


def test_resolve_feature_group_columns_substring_matches_prefix_and_suffix_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single marker catches away_/home_/opponent_ prefixed and _diff suffixed columns."""
    # Arrange
    monkeypatch.setattr(constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa_per_play",)})
    columns = [
        "away_epa_per_play",
        "home_epa_per_play",
        "opponent_epa_per_play",
        "epa_per_play_diff",
        "unrelated_column",
    ]

    # Act
    resolved = resolve_feature_group_columns(columns, ["pbp"])

    # Assert
    assert resolved == sorted(
        [
            "away_epa_per_play",
            "home_epa_per_play",
            "opponent_epa_per_play",
            "epa_per_play_diff",
        ]
    )


def test_resolve_feature_group_columns_unknown_group_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown group name raises a ValueError naming the bad group."""
    # Arrange
    monkeypatch.setattr(constants, "FEATURE_GROUP_COLUMN_MARKERS", {"pbp": ("epa",)})

    # Act & Assert
    with pytest.raises(ValueError, match="not_a_real_group"):
        resolve_feature_group_columns(["away_epa"], ["not_a_real_group"])


def test_walk_forward_config_disabled_feature_groups_serializes_to_list() -> None:
    """`disabled_feature_groups` round-trips through `to_dict` as a list."""
    # Arrange
    configured = replace(_base_config(), disabled_feature_groups=("pbp",))

    # Act
    result = configured.to_dict()

    # Assert
    assert result["disabled_feature_groups"] == ["pbp"]


def test_walk_forward_config_disabled_feature_groups_defaults_to_empty_list() -> None:
    """`disabled_feature_groups` serializes to an empty list by default."""
    # Arrange
    default_config = _base_config()

    # Act
    result = default_config.to_dict()

    # Assert
    assert result["disabled_feature_groups"] == []


def test_run_walk_forward_backtest_drops_disabled_feature_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config alone is enough to ablate a group, without the caller pre-filtering.

    The CLI scripts drop the columns before building the config, so this guards a direct
    caller of `run_walk_forward_backtest` against silently getting no ablation at all.
    """
    # Arrange
    monkeypatch.setattr(
        constants,
        "FEATURE_GROUP_COLUMN_MARKERS",
        {"demo": ("widget",)},
    )

    captured: dict[str, list[str]] = {}

    def fake_filter(df: pd.DataFrame, *, include_postseason: bool = False) -> pd.DataFrame:
        captured["columns"] = list(df.columns)
        msg = "stop after the drop"
        raise RuntimeError(msg)

    monkeypatch.setattr(walk_forward, "filter_regular_season", fake_filter)

    df = pd.DataFrame(
        {
            "season": [2023],
            "week": [3],
            "away_widget_rate": [1.0],
            "home_widget_rate": [2.0],
            "widget_rate_diff": [-1.0],
            "away_rest": [7],
        }
    )
    config = walk_forward.WalkForwardConfig(disabled_feature_groups=("demo",))

    # Act & Assert
    with pytest.raises(RuntimeError, match="stop after the drop"):
        walk_forward.run_walk_forward_backtest(df, config)

    assert captured["columns"] == ["season", "week", "away_rest"]
