# Dataset

The company supplied no data, so one must be chosen and documented. **Status: not yet chosen.**

## Selection criteria
1. Image-based or easily converted to frames (the task says "input images").
2. Both live and spoof classes, ideally several attack types (print, screen replay).
3. Predefined splits, or subject IDs so splits can be made subject-disjoint.
4. Small enough to train on a 4 GB GPU in reasonable time.
5. Obtainable without a lengthy licence process, so the work can start now.

## Candidates (from a quick search; verify each before relying on it)

| Dataset | Notes | Fit |
|---|---|---|
| LCC-FASD | Crowd-collected, released by Timoshenko et al.; a recent paper describes the public release as about 18.8K files with training / development / evaluation splits. Exact download location not confirmed. | **Proposed primary**: image-based, predefined splits, small |
| CASIA-FASD | 600 videos from 50 subjects (20 train / 30 test subjects), print and replay attacks. Research licence; videos need frame extraction. | Good backup / cross-dataset test |
| Replay-Attack (Idiap) | Print and replay attacks, 50 subjects. Licence agreement. | Good cross-dataset test |
| CASIA-SURF | 492,522 images, RGB + depth + IR. Far larger than needed; access through a challenge site. | Too big for this task |
| Vendor samples on Hugging Face (TrainingDataPro, Axon) | Sample subsets of commercial datasets; the full versions are sold. | Not suitable as the main training set |

## Decision
_To be filled in: chosen dataset, download URL, licence, citation, date downloaded._

## Expected layout after preparation
```
data/processed/
  train/{live,spoof}/   dev/{live,spoof}/   test/{live,spoof}/
  manifest.csv          # path, split, label, group (subject id when known)
```

## Preparing
Predefined splits (rename keys to match the raw folder names):
```bash
python scripts/prepare_data.py --src RAW --dst data/processed \
  --split-map training=train,development=dev,evaluation=test
```
No splits provided - subject-disjoint random split; `--group-regex` must capture the subject id:
```bash
python scripts/prepare_data.py --src RAW --dst data/processed \
  --auto-split --group-regex "(?:^|/)(\d+)_"
```
Class folders are recognised by name (`live/real/genuine/bonafide` vs `spoof/fake/attack/imposter`);
extend with `--live-names` / `--spoof-names`. The script prints per-split counts and fails if any
subject appears in more than one split.

## Rules
- Never split randomly by image: frames of the same person in train and test inflate every metric.
- Test split is used once per finished model; model and threshold choices use dev only.
- Record final counts below.

| Split | Live | Spoof | Subjects |
|---|---|---|---|
| train | TBD | TBD | TBD |
| dev | TBD | TBD | TBD |
| test | TBD | TBD | TBD |
