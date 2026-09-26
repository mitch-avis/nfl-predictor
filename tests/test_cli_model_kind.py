"""Tests for the one model-kind vocabulary shared by the training and ranking commands.

The canonical names are ``margin_total`` and ``blend`` (what training records in a run's
metadata); ``blended_margin_total`` is accepted everywhere as an old name for ``blend``.
"""

from __future__ import annotations

import sys

import pytest

from nfl_predictor.cli import options, rankings, train

RANKINGS_ARGV = ["--model-in", "model.joblib", "--season", "2026", "--through-week", "2"]


def test_the_vocabulary_is_margin_total_and_blend() -> None:
    """Two canonical kinds, one alias."""
    assert options.MODEL_KINDS == ("margin_total", "blend")
    assert options.model_kind("blended_margin_total") == "blend"
    assert options.model_kind("blend") == "blend"
    assert options.model_kind("margin_total") == "margin_total"


@pytest.mark.parametrize(
    ("given", "expected"),
    [("margin_total", "margin_total"), ("blend", "blend"), ("blended_margin_total", "blend")],
)
def test_train_accepts_both_names_of_the_blend(
    monkeypatch: pytest.MonkeyPatch, given: str, expected: str
) -> None:
    """``train --model-kind`` takes the canonical name and the old one."""
    monkeypatch.setattr(sys, "argv", ["train", "--model-kind", given])
    assert train._parse_args().model_kind == expected


@pytest.mark.parametrize(
    ("given", "expected"),
    [("margin_total", "margin_total"), ("blend", "blend"), ("blended_margin_total", "blend")],
)
def test_rankings_accepts_both_names_of_the_blend(given: str, expected: str) -> None:
    """``rankings --model-kind`` takes the name training records and the old one."""
    args = rankings._parse_args([*RANKINGS_ARGV, "--model-kind", given])
    assert args.model_kind == expected


@pytest.mark.parametrize("module", [train, rankings])
def test_an_unknown_model_kind_is_rejected(module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything outside the vocabulary fails in argparse."""
    argv = ["--model-kind", "score"]
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    with pytest.raises(SystemExit):
        if module is train:
            train._parse_args()
        else:
            rankings._parse_args([*RANKINGS_ARGV, *argv])
