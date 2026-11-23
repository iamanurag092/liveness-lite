# Plan and status

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold, training / evaluation / export code, tests, CI | done, CI green |
| 1 | Dataset: LCC-FASD prepared, leakage and integrity checks | done (docs/01_dataset.md) |
| 2 | Baseline training and evaluation (E00) | done |
| 3 | Ablations: epochs, resolution, augmentation, backbone (E01 to E05) | done (docs/03_experiments.md) |
| 4 | Compression: ONNX FP32 verified; INT8 attempted | FP32 shipped; INT8 failed, documented |
| 5 | Browser demo and continuous deployment | demo built; Vercel deployment (steps in docs/DEMO.md) |
| 6 | Report and submission ZIP | report.pdf built; `scripts/make_submission.py` builds the ZIP |

## Submission checklist
- [ ] CI green on main
- [ ] Demo URL put into README.md (and `python scripts/build_report.py --demo-url ...`)
- [ ] `python scripts/make_submission.py`, open the ZIP and check the contents and size
- [ ] Email with **Reply All**: ZIP attached, GitHub and demo links in the body
