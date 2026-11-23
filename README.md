# liveness-lite

A lightweight deep-learning classifier that decides whether an input image is a **genuine (live) face**
or a **spoof** (printed photo or screen replay). Built for the Deep Algorithms Solutions take-home:
*"classify input images as genuine or spoofed, with high accuracy and low computational cost and
memory"*. Python/PyTorch for training, ONNX for deployment, and a browser demo that runs the model
client-side.

**Live demo:** _add the Vercel URL here after deploying (see docs/DEMO.md)_
&nbsp;|&nbsp; **Report:** `report.pdf`

## Results

Dataset: LCC-FASD, official test split (389 live, 3,377 spoof; about 90% spoof, so accuracy is not the
headline). Threshold chosen on the dev split. Cost: ONNX Runtime, one CPU thread, batch 1.

| Model | Backbone | Params | GMACs | ONNX size | CPU p50 | Test ACER | APCER | BPCER | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Lite** | MobileNetV3-Small | 1.52M | 0.056 | 6.1 MB | about 2 ms | 4.27% | 3.91% | 4.63% | 0.9941 |
| **Accurate** | MobileNetV3-Large | 4.20M | 0.215 | 16.8 MB | about 6 ms | 3.02% | 2.96% | 3.08% | 0.9968 |

APCER = attacks accepted as live, BPCER = live images rejected, ACER = their mean. Read
[docs/03_experiments.md](docs/03_experiments.md) before quoting these numbers: single seed, the official
split is not subject-disjoint, the dev set is small, the Accurate model was chosen with the test results
in view, and INT8 quantisation was tried and failed (not shipped). All disclosed there and in `report.pdf`.

## Quick start: classify an image

```powershell
pip install -r requirements.txt
pip install -e .
python -m liveness.predict path\to\cropped_face.png --model models/lite/model_fp32.onnx
```
Prints LIVE or SPOOF with the spoof score and the stored decision threshold. Inputs should be tightly
cropped faces like the training data; other images give unreliable scores.

## Reproduce everything

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
# GPU users: install the CUDA build of PyTorch first (command from pytorch.org matching your driver)
pip install -r requirements.txt -r requirements-dev.txt ; pip install -e .

python scripts/dataset_download.py                      # Kaggle: aleksandrpikul222/lcc-fasd
python scripts/prepare_lcc_fasd.py --src <LCC_FASD folder> --dst data/processed --resize 256
python scripts/check_images.py --root data/processed --size 256

python -m liveness.train --config configs/default.yaml --override output.dir=runs/E04_aug_basic train.epochs=8 train.patience=99 data.augment=basic
python -m liveness.evaluate --config configs/default.yaml --checkpoint runs/E04_aug_basic/best.pt --run-dir runs/E04_aug_basic --tag fp32
python -m liveness.export --config configs/default.yaml --checkpoint runs/E04_aug_basic/best.pt --out-dir models/lite --no-quant
python scripts/run_experiments.py                       # the four ablations, evaluated
python scripts/summarize_runs.py                        # docs/results_table.md
python -m pytest -q                                     # unit and end-to-end tests
```
More commands: [docs/USAGE.md](docs/USAGE.md). Training from a spinning hard drive is very slow; keep
the processed data on an SSD.

## Repository layout

```
src/liveness/   data, model, metrics, train, evaluate, export, predict
scripts/        dataset preparation and checks, experiment runner, report and demo builders
configs/        default.yaml (real runs), smoke.yaml (CI)
tests/          metrics, data, model, ONNX export, predictor, preparation scripts
demo/           static browser demo (onnxruntime-web); models are copied in by scripts/make_demo_assets.py
docs/           plan, dataset notes, methodology, experiment log, decisions, demo guide
.github/        CI: lint, tests, synthetic end-to-end run, size budget, demo preprocessing test
models/         exported ONNX models and model cards (lite, accurate)
```

## Documentation

- [docs/01_dataset.md](docs/01_dataset.md): dataset, preparation, leakage findings, data integrity
- [docs/02_methodology.md](docs/02_methodology.md): task definition, metrics, protocol, model
- [docs/03_experiments.md](docs/03_experiments.md): experiment log, findings, model selection disclosure
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and why
- [docs/DEMO.md](docs/DEMO.md): browser demo, local preview, Vercel deployment
- [docs/PLAN.md](docs/PLAN.md): phases and status

## Limits, briefly

Within-dataset results only (no cross-dataset test); 12 subject ids appear in both official train/dev and
test; single seed per run; the model expects cropped faces; latency is from one laptop. The dataset images
are photos of real people and are not redistributed here.
