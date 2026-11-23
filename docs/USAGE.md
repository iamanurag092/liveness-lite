# Usage cheat sheet

## Classify your own images
```powershell
python -m liveness.predict path\to\face1.jpg path\to\face2.png --model models/lite/model_fp32.onnx
python -m liveness.predict face.jpg --json            # machine-readable
python -m liveness.predict face.jpg --no-center-crop  # image is already a square face crop
```
The score is the spoof probability; the verdict uses the threshold stored in the `model_card.json` next to the model.
Inputs should be tightly cropped faces like the training data. Full photos with backgrounds are
out of distribution and give unreliable scores.

## Quantisation sweep (dev split only)
```powershell
python scripts/quant_sweep.py --config configs/default.yaml --onnx models/lite/model_fp32.onnx
```
Percentile/entropy calibration keeps every layer output in RAM, so it uses tiny calibration batches
(`--hist-batch`); close other programs and watch Task Manager. Prints size, CPU latency and dev EER/AUC for several INT8 recipes and recommends one, or says to
keep FP32. To apply a recommendation, re-export with its flags, for example:
`python -m liveness.export --checkpoint runs/E00_baseline/best.pt --out-dir models --quant-method percentile --conv-only`

## Ablations, unattended
```powershell
python scripts/run_experiments.py              # 4 trainings + evaluation + summary table
python scripts/run_experiments.py --with-sweep # also the INT8 sweep first (needs spare RAM)
python scripts/run_experiments.py --only E03_img160
python scripts/summarize_runs.py               # rebuild docs/results_table.md from runs/
```
Each run: 8 epochs, full cosine schedule. Choose the final model by **dev** EER and cost, then
report its test numbers.
