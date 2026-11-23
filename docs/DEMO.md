# Browser demo

A static page (`demo/`) that classifies an uploaded image **entirely in the browser** with
`onnxruntime-web` (WebAssembly). No server, no GPU, and the image never leaves the user's device.

## What it does
1. The user picks or drops an image.
2. The page centre-crops it to a square, resizes to 256 px and then to the model size (the same steps
   as `src/liveness/predict.py`), scales to [0, 1] and runs the ONNX model.
3. It shows LIVE or SPOOF, the spoof score against the stored decision threshold, and the inference time.
4. A switch selects the **Lite** or the **Accurate** model (see docs/03_experiments.md).

## Build the assets (after exporting the final models)
```powershell
python -m liveness.export --config configs/default.yaml --checkpoint runs/E04_aug_basic/best.pt --out-dir models/lite --no-quant
python -m liveness.export --config configs/default.yaml --checkpoint runs/E05_mnv3_large/best.pt --out-dir models/accurate --no-quant
python scripts/make_demo_assets.py --lite models/lite --accurate models/accurate
```
This copies `model.onnx` and a slim `card.json` into `demo/models/<tier>/`.

## Try it locally
```powershell
python scripts/make_demo_assets.py --lite models/lite --accurate models/accurate --with-samples
python -m http.server 8000 --directory demo
```
Open http://localhost:8000. `--with-samples` adds a few test images with their Python reference
scores so you can see that the browser reproduces them (differences of about 0.01 or less come from
the browser's image resampling). The page loads the WebAssembly runtime from a CDN, so it needs internet.

## Privacy rule for sample images
`demo/samples/` is git-ignored. LCC-FASD images are photos of real people and may not be redistributable.
Never commit them or deploy them publicly. The deployed demo works with the visitor's own uploads only.

## Deploy on Vercel (also gives continuous deployment)
1. Commit `demo/models/` (the `.onnx` files are 6 to 20 MB, fine for GitHub) and push.
2. vercel.com, sign in with GitHub, **Add New, Project**, import the repository.
3. **Root Directory:** `demo`. **Framework Preset:** Other. Leave build command and output directory empty.
4. Deploy. Every later push to `main` redeploys automatically.
5. Put the resulting URL in the README and in the submission email.

## Limits to state honestly
- The model expects tightly cropped faces like the training data. Whole-scene photos, other cameras or
  attack types not in LCC-FASD can give unreliable scores.
- Resampling in the browser is not bit-identical to the Python pipeline.
