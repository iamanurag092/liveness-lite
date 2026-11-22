# Methodology

## Task
Binary classification of an input image: **live (genuine)** vs **spoof**. Treated as face
presentation attack detection (PAD). Constraints: high accuracy, low compute, low memory.

## Conventions
- Label 1 = spoof (attack), label 0 = live. Model output = probability of spoof.
- Predict spoof when `score >= threshold`.

## Metrics
| Metric | Meaning |
|---|---|
| APCER | share of attacks accepted as live (security failure) |
| BPCER | share of live samples rejected (usability failure) |
| ACER | (APCER + BPCER) / 2; equals HTER under these definitions |
| EER | error rate where APCER = BPCER |
| AUC | threshold-free ranking quality |
| Accuracy | reported for reference only; misleading when classes are imbalanced |

## Protocol
1. Train on `train`; pick the best epoch by **dev EER**.
2. Choose the operating threshold as the **dev EER threshold**.
3. Report `test` metrics at that fixed threshold (also at 0.5 for reference).
4. Evaluate FP32 and INT8 models the same way; the INT8 model must be re-evaluated, never assumed.

## Model
- Backbone: torchvision MobileNetV3-Small (ImageNet pretrained), final layer replaced by one logit.
  Alternatives implemented for ablation: MobileNetV3-Large, MobileNetV2.
- Preprocessing contract: RGB, resize to `img_size`, scale to [0, 1]. Mean/std normalisation is inside
  the model, so ONNX / browser inputs need only resize + /255.
- Loss: BCE-with-logits, `pos_weight = n_live / n_spoof`, light label smoothing.
- Optimiser: AdamW, cosine schedule, mixed precision on CUDA, early stopping on dev EER.
- Augmentation: random resized crop, flip, colour jitter, blur, random JPEG re-encode.
  Caveat: resizing can destroy fine texture cues (moire, print dots); this is an ablation item.

## Compression and cost reporting
- ONNX export (opset 17) with sigmoid in the graph; parity with PyTorch checked to 1e-3.
- Static INT8 quantisation (QDQ, per-channel weights), calibrated on dev images only.
- Reported: parameters, GMACs (PyTorch FLOP counter / 2), file size, single-thread CPU latency
  (batch 1, p50 / p95).
- Budget limits live in `configs/default.yaml` and can fail CI via `--enforce-budget`.

## Known limitations (to discuss in the report)
- PAD models often generalise poorly across datasets, cameras and attack types; a within-dataset
  result is optimistic. If time allows, add a cross-dataset test.
- Quantisation can cost accuracy on hard-swish networks; measured, not assumed.
- Latency numbers are for the benchmarking machine, not a target device.
