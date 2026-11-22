# liveness-lite

Lightweight deep-learning classifier that decides whether an input image is **genuine (live)** or
**spoofed** (printed photo, screen replay, ...), built to stay small and fast: ImageNet-pretrained
MobileNet backbone, single-logit head, exported to ONNX and quantised to INT8.

> Context: take-home task for Deep Algorithms Solutions - *"Develop a lightweight deep learning model
> to classify input images as genuine or spoofed. High accuracy, low computational cost and memory."*

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo scaffold, training / evaluation / export code, tests, CI | done (code written; Done✅) |
| 1 | Dataset chosen, downloaded, prepared, leakage-checked | **next** - see [docs/01_dataset.md](docs/01_dataset.md) |
| 2 | Baseline training + evaluation | todo |
| 3 | Ablations (image size, backbone, augmentation) | todo |
| 4 | INT8 compression + budget gate | todo (code ready) |
| 5 | Demo deployment + CD | todo |
| 6 | Report and ZIP deliverable | todo |

Full plan with acceptance criteria: [docs/PLAN.md](docs/PLAN.md).

## Layout

```
configs/        default.yaml (real runs), smoke.yaml (CI, synthetic data)
src/liveness/   data.py  model.py  metrics.py  train.py  evaluate.py  export.py  utils.py
scripts/        prepare_data.py (raw dataset -> train/dev/test), make_synthetic_data.py
tests/          metric maths (numpy) + model / data / end-to-end pipeline (torch, onnx)
docs/           PLAN, dataset notes, methodology, experiment log, design decisions
.github/        CI: lint, tests, synthetic-data smoke run, budget gate
```

## Setup

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
macOS / Linux: `python -m venv .venv && source .venv/bin/activate`

**GPU PyTorch.** `pip install torch` from the default index gives a CPU-only build on Windows
(`torch.cuda.is_available()` prints `False`). For NVIDIA GPUs, get the exact command from
https://pytorch.org/get-started/locally/ (OS: Windows, Package: Pip, then a CUDA option your
driver supports; `nvidia-smi` shows the driver's CUDA version). Index URLs for old CUDA versions
(for example `cu121`) can lack builds for current Python/torch versions, which shows up as
"No matching distribution found". If a CPU build is already installed, remove it first:
```powershell
pip uninstall -y torch torchvision
# run the command copied from pytorch.org here
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # want a +cuXXX suffix and True
```
Then the rest of the dependencies and the package itself:
```powershell
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```
No usable GPU? Training also runs on a free Kaggle or Colab GPU notebook; the code is unchanged.

## Run

```bash
# 1. Prepare data (details and options: docs/01_dataset.md)
python scripts/prepare_data.py --src <RAW_DIR> --dst data/processed --split-map training=train,development=dev,evaluation=test

# 2. Train (best epoch chosen by dev EER)
python -m liveness.train --config configs/default.yaml

# 3. Evaluate: threshold is picked on dev, reported on test
python -m liveness.evaluate --config configs/default.yaml --checkpoint runs/baseline/best.pt

# 4. Export ONNX + INT8, verify parity, benchmark, check budget
python -m liveness.export --config configs/default.yaml --checkpoint runs/baseline/best.pt --out-dir models --enforce-budget
python -m liveness.evaluate --config configs/default.yaml --onnx models/model_int8.onnx --tag int8
```

Override any config value without editing files, e.g.
`python -m liveness.train --override data.img_size=160 model.name=mobilenet_v2 output.dir=runs/mnv2_160`.

Pipeline check without any real data (this is what CI runs):

```bash
python scripts/make_synthetic_data.py --dst data/synthetic
python -m liveness.train --config configs/smoke.yaml
```

## Results

Filled in after real runs - nothing below is measured yet.

| Model | Img | Params (M) | GMACs | Size MB | CPU p50 ms | Test ACER | Test AUC |
|---|---|---|---|---|---|---|---|
| MobileNetV3-Small FP32 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNetV3-Small INT8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Documentation

- [docs/PLAN.md](docs/PLAN.md) - phases, checklists, acceptance criteria
- [docs/01_dataset.md](docs/01_dataset.md) - candidate datasets, selection, layout, leakage rules
- [docs/02_methodology.md](docs/02_methodology.md) - task definition, metrics, protocol, model, compression
- [docs/03_experiments.md](docs/03_experiments.md) - experiment log (one row per run)
- [docs/DECISIONS.md](docs/DECISIONS.md) - why things are built the way they are

## Deliverable ZIP

```bash
git archive --format=zip --output=submission.zip HEAD   # tracked files only
# then add the trained models (models/*.onnx), report PDF and any result plots, and send via Reply All
```
