# Experiment log

One row per run. Change one thing at a time. Metrics are **test at the dev-EER threshold** unless noted;
selection decisions must be justified from **dev** numbers.

| ID | Date | Config / overrides | Seed | Dev EER | Test ACER | Test AUC | Params M | Size MB | p50 ms | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| E00 | | default.yaml | 42 | | | | | | | baseline |

## How to record a run
1. `python -m liveness.train --config configs/default.yaml --override ... output.dir=runs/<ID>`
2. `python -m liveness.evaluate --checkpoint runs/<ID>/best.pt --run-dir runs/<ID>`
3. `python -m liveness.export --checkpoint runs/<ID>/best.pt --out-dir models/<ID>` (for cost columns)
4. Copy numbers from `runs/<ID>/metrics_*.json` and `models/<ID>/model_card.json` into the table.
5. Write one line of interpretation under "Findings".

## Findings
_(none yet)_
