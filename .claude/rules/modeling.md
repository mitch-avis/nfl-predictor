---
paths:
  - "src/nfl_predictor/ml/**"
  - "src/nfl_predictor/ml_model.py"
  - "src/nfl_predictor/weekly_run/**"
  - "src/nfl_predictor/reporting/**"
  - "config/**"
---

# Model, weekly-run and reporting code

- Read `.agents/modeling_spec.md` before changing prediction, calibration, market, pool,
  evaluation, artifact, leakage-audit or feature code; it is the specification these modules
  implement.
- `src/nfl_predictor/ml/*.py`, `constants.py` and `ml_model.py` are fingerprinted: an edit
  invalidates every walk-forward checkpoint, so a rerun retrains from scratch. Plan runs after the
  code is stable and group such edits.
- Training and evaluation defaults are explicit about whether postseason rows are included and,
  if so, how they are weighted.
- Any setting that differs between the reference walk-forward configuration and the production
  weekly run (`config/weekly_run.yaml` and the weekly stage-1 selection) is a defect until the
  user approves it (guardrail 11).
