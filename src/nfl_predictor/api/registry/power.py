"""Column metadata for power rankings and projected standings."""

from __future__ import annotations

from nfl_predictor.api.registry.columns import ColumnMeta, register

RANKING = "Ranking"
RECORD = "Record"
PROJECTION = "Projection"

register(
    ColumnMeta(
        "rank",
        "Rank",
        "Position by power rating (1 = strongest).",
        RANKING,
        "int",
        "lower",
        sticky=True,
    ),
    ColumnMeta("team_abbr", "Team", "Team.", RANKING, "team", sticky=True),
    ColumnMeta("division", "Division", "Division.", RANKING),
    ColumnMeta("conference", "Conf.", "Conference.", RANKING),
    ColumnMeta(
        "rating_raw",
        "Rating",
        "Bradley-Terry strength on the log-odds scale; the difference between two teams is the log-odds of the stronger one winning on a neutral field.",
        RANKING,
        "float",
        "higher",
        heatmap=True,
        decimals=2,
    ),
    ColumnMeta(
        "power_rating_1_10",
        "Power (1-10)",
        "Rating mapped to 1..10 through a logistic curve. Absolute, so comparable across weeks.",
        RANKING,
        "float",
        "higher",
        heatmap=True,
        decimals=2,
    ),
    ColumnMeta(
        "power_rating_0_10",
        "Power (0-10)",
        "Rating mapped to 0..10 (10 times the win probability against an average team).",
        RANKING,
        "float",
        "higher",
        decimals=2,
    ),
    ColumnMeta(
        "rank_change",
        "Δ rank",
        "Change in rank since the previous week's rankings (positive = climbed).",
        RANKING,
        "int",
        "higher",
        heatmap=True,
    ),
    ColumnMeta(
        "previous_rank",
        "Prev. rank",
        "Rank in the previous week's rankings.",
        RANKING,
        "int",
        "lower",
    ),
    ColumnMeta("wins", "W", "Wins to date.", RECORD, "int", "higher"),
    ColumnMeta("losses", "L", "Losses to date.", RECORD, "int", "lower"),
    ColumnMeta("ties", "T", "Ties to date.", RECORD, "int"),
    ColumnMeta("games_played", "GP", "Games played.", RECORD, "int"),
    ColumnMeta("record", "Record", "W-L(-T) to date.", RECORD, "text"),
    ColumnMeta(
        "home_advantage_prob",
        "HFA %",
        "Win probability of two equal teams for the home side in this fit.",
        RANKING,
        "prob",
        decimals=1,
    ),
    ColumnMeta(
        "home_advantage_logit",
        "HFA logit",
        "Home-field advantage on the log-odds scale.",
        RANKING,
        "float",
        decimals=3,
    ),
    ColumnMeta(
        "exp_wins",
        "Exp. wins left",
        "Expected wins over the remaining schedule, summing model win probabilities.",
        PROJECTION,
        "float",
        "higher",
        heatmap=True,
        decimals=1,
    ),
    ColumnMeta("games_remaining", "Left", "Games remaining.", PROJECTION, "int"),
    ColumnMeta(
        "projected_wins",
        "Proj. W",
        "Current wins plus expected remaining wins.",
        PROJECTION,
        "float",
        "higher",
        heatmap=True,
        decimals=1,
    ),
    ColumnMeta(
        "projected_losses",
        "Proj. L",
        "Current losses plus expected remaining losses.",
        PROJECTION,
        "float",
        "lower",
        decimals=1,
    ),
    ColumnMeta(
        "projected_win_pct",
        "Proj. win %",
        "Projected wins divided by scheduled games.",
        PROJECTION,
        "prob",
        "higher",
        heatmap=True,
        decimals=1,
    ),
    ColumnMeta(
        "projected_division_rank",
        "Proj. div. rank",
        "Projected finish within the division.",
        PROJECTION,
        "int",
        "lower",
    ),
    ColumnMeta("through_week", "Through", "Last week of results included.", RANKING, "int"),
)

POWER_COLUMNS: list[str] = [
    "rank", "team_abbr", "rank_change", "previous_rank", "division", "conference", "record",
    "power_rating_1_10", "rating_raw", "power_rating_0_10", "wins", "losses", "ties", "games_played",
    "home_advantage_prob",
]  # fmt: skip

STANDINGS_COLUMNS: list[str] = [
    "team_abbr", "division", "conference", "record", "wins", "losses", "ties", "games_played",
    "games_remaining", "exp_wins", "projected_wins", "projected_losses", "projected_win_pct",
    "projected_division_rank",
]  # fmt: skip
