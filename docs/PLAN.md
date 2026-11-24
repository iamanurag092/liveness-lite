# Plan and status

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold, training / evaluation / export code, tests, CI | done, CI green |
| 1 | Dataset: LCC-FASD prepared, leakage and integrity checks | done (docs/01_dataset.md) |
| 2 | Baseline training and evaluation (E00) | done |
| 3 | Ablations: epochs, resolution, augmentation, backbone (E01 to E05) | done (docs/03_experiments.md) |
| 4 | Compression: ONNX FP32 verified; INT8 attempted | FP32 shipped; INT8 failed and is documented |
| 5 | Browser demo and continuous deployment | done: https://liveness-lite-demo.vercel.app/ |
| 6 | Report and submission ZIP | report.pdf built; `scripts/make_submission.py` builds the ZIP |

## Submission checklist
- [x] CI green on main (six consecutive green runs)
- [x] Demo deployed and the URL in README.md and report.pdf
- [ ] `python scripts/make_submission.py`; open the ZIP and check contents and size
- [ ] Email with **Reply All**: ZIP attached, GitHub and demo links in the body
