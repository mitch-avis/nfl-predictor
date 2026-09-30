"""Tests for the one model-kind vocabulary shared by the training and ranking commands.

``margin_total`` is the only model kind. ``blend`` (and its old name ``blended_margin_total``)
was retired: asking for it fails with a message that says so, rather than a bare invalid
choice.
"""

from __future__ import annotations

import sys

import pytest

from nfl_predictor.cli import options, rankings, train

RANKINGS_ARGV = ["--model-in", "model.joblib", "--season", "2026", "--through-week", "2"]


def test_the_vocabulary_is_margin_total() -> None:
    """One model kind."""
    assert options.MODEL_KINDS == ("margin_total",)
    assert options.model_kind("margin_total") == "margin_total"


def test_train_and_rankings_default_to_margin_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--model-kind`` both commands use the one kind."""
    monkeypatch.setattr(sys, "argv", ["train"])
    assert train._parse_args().model_kind == "margin_total"
    assert rankings._parse_args(RANKINGS_ARGV).model_kind == "margin_total"


@pytest.mark.parametrize("given", ["blend", "blended_margin_total"])
@pytest.mark.parametrize("module", [train, rankings])
def test_the_retired_blend_kind_is_rejected_with_a_reason(
    module: object,
    given: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--model-kind blend`` fails in argparse and names the retirement."""
    argv = ["--model-kind", given]
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(SystemExit):
        _parse_args(module, argv)
    assert "retired" in capsys.readouterr().err


@pytest.mark.parametrize("module", [train, rankings])
def test_an_unknown_model_kind_is_rejected(module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything outside the vocabulary fails in argparse."""
    argv = ["--model-kind", "score"]
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(SystemExit):
        _parse_args(module, argv)


def _parse_args(module: object, argv: list[str]) -> None:
    """Parse ``argv`` with the module's own parser (train reads it from ``sys.argv``)."""
    if module is train:
        train._parse_args()
    else:
        rankings._parse_args([*RANKINGS_ARGV, *argv])
