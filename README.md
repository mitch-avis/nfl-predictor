# nfl-predictor

## Testing

Run the test suite with:

```bash
pytest
```

## Validation

Offline validation against the latest dataset:

```bash
python scripts/validate_offline.py
```

Live validation against the latest schedule (may require network access):

```bash
python scripts/validate_live.py
```
