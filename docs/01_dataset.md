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
**LCC-FASD**, Kaggle copy `aleksandrpikul222/lcc-fasd` (downloaded with `scripts/dataset_download.py`).
Licence and citation: _to be recorded from the dataset's Kaggle page and original paper before submission._

What the download contains (checked locally):
- one flat folder `LCC_FASD/` with 18,827 PNG files (roughly 650-680 px square) and 4 list files
- `CLIENT_TRAIN.txt` (1,553), `CLIENT_TEST.txt` (389), `IMPOSTER_TRAIN.txt` (13,508),
  `IMPOSTER_TEST.txt` (3,377); the four add up to exactly the number of PNGs
- label convention assumed from biometrics terminology: **client = live, imposter = spoof**.
  Confirm it visually with the preview sheets written by the prepare script before training.
- file names such as `YOUTUBE_id43_s0_105.png` or `AA5742_id154_s0_112.png` carry an `idNN` token
  (subject or video id, used as the split group); some test files are plain `spoof_1793.png`
- no dev split is provided, so dev is carved out of the official train list by group
- strong class imbalance (about 90% spoof), so accuracy is not reported as a headline metric

## Findings from the first preparation run
- Label convention confirmed visually (preview sheets): client = live, imposter = spoof.
- 7,580 of 18,827 files (about 40%) have no `idNN` in the name (`real_N.png`, `spoof_N.png`); they are
  grouped by blocks of 200 consecutive indices, which is an approximation, not a subject id.
- The official train and test lists are **not subject-disjoint**: 12 ids occur in both (for example
  `id117`, `id122`, `id123` have live images on both sides; `id42`, `id43`, `id48` have both classes
  on both sides). Whether an id is one person or just a reused number is not documented.
- First dev carve was unbalanced (117 live vs 2,345 spoof); the carve now selects whole groups until
  live and spoof each reach the target share, so the dev threshold is not set by a handful of images.
- Handling: the official test split is used as given (comparable with other work). The overlap and the
  approximate grouping are stated as limitations in the report. Planned extra check: also report test
  metrics restricted to ids that never occur in train/dev, to show how much the overlap matters.

## Expected layout after preparation
```
data/processed/
  train/{live,spoof}/   dev/{live,spoof}/   test/{live,spoof}/
  manifest.csv          # path, split, label, group (subject id when known)
```

## Preparing
LCC-FASD (this project):
```powershell
python scripts/prepare_lcc_fasd.py `
  --src "C:\Users\anura\.cache\kagglehub\datasets\aleksandrpikul222\lcc-fasd\versions\1\LCC_FASD" `
  --dst data/processed --resize 256
```
- `--resize 256` shrinks every image to 256x256 while copying (faster training, about 2 GB on disk);
  `--resize 0` keeps the originals. Training at 224 or below still works from the 256 copies.
- The script prints per-split counts, writes `manifest.csv` and `dataset_summary.json`, and creates
  `data/processed/_preview/*.png` contact sheets. **Open them** and check that the live sheets show
  genuine faces and the spoof sheets show attacks (screens, prints) before training.
- It also reports how many subject ids and id-less index blocks occur in both train/dev and the
  official test list.
- Re-running is safe and fast: files already present are moved between split folders, not recopied.
  If you change `--resize`, delete `data/processed` first.

Other datasets with real/spoof folders: `scripts/prepare_data.py` (see its `--help`).
Predefined splits: `--split-map training=train,development=dev,evaluation=test`.
Flat data: `--auto-split --group-regex "(?:^|/)(\d+)_"` (the regex must capture the subject id).

## Data integrity
The first training attempt stopped with `UnidentifiedImageError` on one prepared image
(`train/live/YOUTUBE_id20_s0_15.png`), so a damaged file had been written during preparation.
- `python scripts/check_images.py --root data/processed --size 256` scans every prepared image
  (full decode, optional size check). `--delete` removes the bad ones.
- Re-running `scripts/prepare_lcc_fasd.py` with the same arguments recreates the missing files from the
  original download. Source images that are themselves unreadable are skipped and listed in
  `data/processed/skipped_unreadable.txt`; they are also excluded from `manifest.csv` and the counts.
- Run the scanner after every (re)preparation and before long training runs.

## Rules
- Never split randomly by image: frames of the same person in train and test inflate every metric.
- Test split is used once per finished model; model and threshold choices use dev only.
- Record final counts below.

| Split | Live | Spoof | Subjects |
|---|---|---|---|
| train | TBD | TBD | TBD |
| dev | TBD | TBD | TBD |
| test | TBD | TBD | TBD |
