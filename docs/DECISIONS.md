# Design decisions

Short records of choices and why. Add a new entry when a choice changes.

1. **Label 1 = spoof, score = P(spoof).** Matches PAD metric definitions (APCER/BPCER), so formulas
   in `metrics.py` read like the literature.
2. **Normalisation inside the model.** One preprocessing contract (resize, /255) for training,
   ONNX, and any browser client; removes a common train/serve mismatch bug.
3. **Single logit + BCE** rather than 2-class softmax: simpler head, direct probability, `pos_weight`
   handles imbalance.
4. **torchvision MobileNets, ImageNet-pretrained.** Few dependencies, small (about 1.5M params for V3-Small),
   good transfer; no timm needed.
5. **Dev-set threshold, test reported once.** Avoids tuning the operating point on test data.
6. **ONNX as the interchange format; static INT8 (QDQ).** Runs on CPU via ONNX Runtime and in the
   browser via onnxruntime-web; one artefact serves benchmark, CI and demo.
7. **Subject-disjoint splits enforced by tooling.** `prepare_data.py` fails on identity leakage.
8. **Budget gate in CI.** Size / latency limits are checked automatically so "lightweight" is a
   tested property, not a claim.
9. **No runtime AI agents in the product.** They would add cost and complexity without improving the
   classifier; automation effort goes into CI/CD and reproducibility instead.
10. **Browser inference for the demo.** Vercel is a poor fit for Python ML servers (size limits), so
    the plan is a static page running the INT8 ONNX model client-side.
11. **BatchNorm momentum 0.1 instead of torchvision's 0.01.** First smoke run (2 epochs, no pretrained
    weights, 12 steps) gave dev AUC exactly 0.5 while the training loss fell: running statistics had barely
    moved, so eval-mode outputs collapsed to a constant. A higher momentum lets short fine-tuning runs
    adapt; `train.py` also warns when dev scores are near-constant.
