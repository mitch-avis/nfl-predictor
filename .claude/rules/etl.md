---
paths:
  - "src/nfl_predictor/data_collection.py"
  - "src/nfl_predictor/utils/**"
  - "src/nfl_predictor/week_builder.py"
  - "src/nfl_predictor/lines_refresh.py"
---

# ETL and data rules

- Use the Polars load/save helpers in `data_collection.py`; pandas CSV helpers are legacy, so do
  not add new ones.
- ETL emits an invariant schema for every run and never silently drops columns. Sources missing
  for a season become nulls or defined defaults, and the model path reports how many rows use
  fallbacks per feature group.
- Normalize team identifiers through `constants.ALIAS_TO_CANONICAL` and
  `normalize_team_column(df, col)`; for play-by-play, normalize `posteam`, `defteam`,
  `home_team` and `away_team` and filter to the regular season for feature inputs.
- Features apply to every matchup: carry counts and sums through season-to-date aggregation and
  compute rates afterward. New season-to-date families inherit the prior blend when they flow
  through `team_stats_df` (the conventions are in `.agents/modeling_spec.md`, "Season and week
  conventions").
- Any new nflreadpy source follows the cache-then-degrade pattern: selected columns cached per
  season as Parquet under `data/cache/nflreadpy/`, current-season refresh, non-fatal failure.
