# liveness-lite

A lightweight deep-learning model that decides whether an input image is a **genuine (live) face** or a
**spoof** (a printed photo or a screen showing a face). Built for the Deep Algorithms Solutions take-home:
*"classify input images as genuine or spoofed, with high accuracy and low computational cost and memory."*

- **Live demo:** https://liveness-lite-demo.vercel.app/ (runs in your browser; your image never leaves your device)
- **Report:** [`report.pdf`](report.pdf), a short plain-language write-up of the dataset, model, results and limits
- **Author:** Anurag Tiwari

## What it is

- **Dataset:** LCC-FASD, 18,827 face images labelled live or spoof. The official training and test lists are
  used; part of the training list is set aside as a validation set.
- **Model:** MobileNetV3 (a small convolutional neural network) that was already trained on ImageNet and then
  fine-tuned on these images. Two sizes: **Lite** (MobileNetV3-Small) and **Accurate** (MobileNetV3-Large).
- **Output:** one score between 0 and 1 (how likely the image is a spoof). A score above a cut-off, chosen
  on the validation images, means spoof.

## Results (on the 3,766 test images)

| Model | Accuracy | ROC-AUC | Live images wrongly flagged | Spoofs wrongly accepted | File size | Speed |
|---|---|---|---|---|---|---|
| **Lite** | 96.0% | 0.994 | 4.6% | 3.9% | 6 MB | about 2 ms |
| **Accurate** | 97.0% | 0.997 | 3.1% | 3.0% | 17 MB | about 6 ms |

About 90% of the images are spoofs, so accuracy alone can mislead (always answering "spoof" would score
89.7%); that is why ROC-AUC and the two error rates are shown. Speed is one image on one laptop CPU core.
The Accurate model was chosen with the test results in view, so its numbers may be slightly optimistic.
Details and limits: `report.pdf` and the technical notes in `docs/`.

## Try it

**In the browser:** open the demo, pick Lite or Accurate, and upload a tightly cropped face image. It shows
LIVE or SPOOF, the score, and how long the model took.

**From the command line:**

```powershell
pip install -r requirements.txt
pip install -e .
python -m liveness.predict path\to\cropped_face.png --model models/lite/model_fp32.onnx
```

The model expects a tightly cropped face like the training images; whole photos give unreliable scores.

## Reproduce

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
```

More commands: [docs/USAGE.md](docs/USAGE.md). Keep the processed data on an SSD; training from a spinning
hard drive is many times slower.

## Tests and automatic checks

```powershell
python -m pytest -q          # 28 automated tests
node demo/preprocess.test.js # tests for the demo's image preparation
```

GitHub Actions runs these checks, plus a small end-to-end training run on synthetic data, on every push. The
demo is hosted on Vercel and updates automatically when `main` changes.

## Repository layout

```
src/liveness/   data, model, metrics, train, evaluate, export, predict
scripts/        dataset preparation and checks, experiment runner, report, demo and submission builders
configs/        default.yaml (real runs), smoke.yaml (CI)
tests/          automated tests
demo/           static browser demo; models are copied in by scripts/make_demo_assets.py
docs/           technical notes: dataset, methodology, experiment log, design decisions, demo guide
models/         exported ONNX models and model cards (lite, accurate)
.github/        automatic checks
```

The `docs/` folder holds the detailed technical notes (including the experiment log); the report and this
README are the short versions.

## Limits

Tested only on this dataset, so other cameras or kinds of attack may do worse. The official training and test
lists share 12 person IDs, so results may be slightly better than on completely new people. The model expects
cropped faces. Each setting was trained once. Dataset images are photos of real people and are not
redistributed in this repository or the demo.

## Credits

Dataset: LCC-FASD (Timoshenko et al.), from the Kaggle copy `aleksandrpikul222/lcc-fasd`; see that page for
its licence and citation. Pretrained ImageNet weights come from torchvision. Inference uses ONNX Runtime and
onnxruntime-web.
