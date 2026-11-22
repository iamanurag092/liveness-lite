# Plan

Rule: a phase is done only when its acceptance criteria are met and the docs are updated.

## Phase 0 - Scaffold (done)
- [x] Package layout, configs, training / evaluation / export code
- [x] Metric unit tests (run locally); model / data / end-to-end tests (run in CI)
- [x] CI: lint, tests, synthetic smoke run, budget gate
- [✅] First green CI run on GitHub (confirms the torch/ONNX code, which was not executed during authoring)

## Phase 1 - Data
- [ ] Choose dataset (docs/01_dataset.md) and record license / citation there
- [ ] Download and run `scripts/prepare_data.py`
- [ ] Sanity checks: class balance, image sizes, a visual grid of live vs spoof, 0 shared subjects
- **Accepted when:** `data/processed/{train,dev,test}/{live,spoof}` exists, manifest audited, numbers recorded in docs/01_dataset.md

## Phase 2 - Baseline
- [ ] Train `configs/default.yaml` (MobileNetV3-Small, 224px)
- [ ] Evaluate on test at the dev-EER threshold
- [ ] Log run in docs/03_experiments.md
- **Accepted when:** metrics JSON + plots exist and the README results row is filled

## Phase 3 - Ablations (one change per run)
- [ ] Image size: 112 / 160 / 224
- [ ] Backbone: MobileNetV3-Small vs MobileNetV2 vs MobileNetV3-Large
- [ ] Augmentation: with / without JPEG + blur; resize vs crop effect on high-frequency cues
- [ ] Input crop: full frame vs face crop with margin (needs a face detector, only if the dataset is not pre-cropped)
- **Accepted when:** a table in docs/03_experiments.md supports the final model choice

## Phase 4 - Compression
- [ ] ONNX FP32 parity check (< 1e-3) and INT8 static quantisation
- [ ] Re-evaluate INT8 on test; accept only if ACER loss is small (decide threshold with data)
- [ ] `--enforce-budget` passes with the real budget in configs/default.yaml
- **Accepted when:** model card JSON + FP32 vs INT8 comparison table in the README

## Phase 5 - Demo and CD
- [ ] Browser demo: static page using `onnxruntime-web` with the INT8 model, hosted on Vercel
      (inference runs client-side; no GPU/PyTorch server needed, ~MB-sized model)
- [ ] Optional: Gradio / FastAPI + ONNX Runtime container on Hugging Face Spaces or Render
- [ ] CD: GitHub Actions deploys on merge to `main` after CI passes
- **Accepted when:** public URL classifies an uploaded image and matches `evaluate.py` output on the same image

## Phase 6 - Report and submission
- [ ] 2-4 page report: data, method, metrics, cost table, limitations
- [ ] ZIP: source, docs, models, report, plots; demo + repo links in the README and email body
- [ ] Send with **Reply All**
