---
paths:
  - "src/nfl_predictor/constants.py"
---

# constants.py

- The canonical reference for file paths, feature column names and lists, team mapping tables,
  season and week rules, and the default parameters used in data collection. Other modules
  import from here; nothing hard-codes a column list. Keep it organized into clear sections and
  remove unused constants.
- Before adding a stat to `EXCLUDE_FROM_OPPONENT_STATS`, grep `_compute_derived_metrics` for its
  `opponent_` mirror: a derived metric that reads it needs the stat in
  `OPPONENT_MIRROR_INTERMEDIATES` too, or the derived columns go null at the next rebuild.
- Any edit here changes every walk-forward checkpoint fingerprint; batch such edits with other
  fingerprint-changing work so they share one new reference.
