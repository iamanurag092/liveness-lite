"use strict";

const TIERS = [
  { id: "lite", label: "Lite" },
  { id: "accurate", label: "Accurate" },
];
const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = 1; // single thread: no cross-origin-isolation headers required

const $ = (id) => document.getElementById(id);
const state = { cards: {}, sessions: {}, tier: null, last: null };

function setStatus(text) {
  $("status").textContent = text || "";
}

function describe(card) {
  return (
    `${card.backbone}, ${card.params_m} M parameters, ${card.gmacs} GMACs, ` +
    `${card.fp32_onnx_mb} MB, ${card.img_size} px input, decision threshold ${card.threshold}`
  );
}

async function loadBitmap(blob) {
  try {
    return await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch (error) {
    return await createImageBitmap(blob);
  }
}

// Same steps as src/liveness/predict.py: centre-crop, 256 px, model size.
function prepare(bitmap, size) {
  const { sx, sy, side } = Preprocess.squareCrop(bitmap.width, bitmap.height);
  const first = document.createElement("canvas");
  first.width = first.height = Preprocess.PREP_SIZE;
  const g1 = first.getContext("2d", { willReadFrequently: true });
  g1.imageSmoothingQuality = "high";
  g1.drawImage(bitmap, sx, sy, side, side, 0, 0, Preprocess.PREP_SIZE, Preprocess.PREP_SIZE);

  const second = document.createElement("canvas");
  second.width = second.height = size;
  const g2 = second.getContext("2d", { willReadFrequently: true });
  g2.imageSmoothingQuality = "high";
  g2.drawImage(first, 0, 0, Preprocess.PREP_SIZE, Preprocess.PREP_SIZE, 0, 0, size, size);
  return second;
}

async function getSession(tier) {
  if (!state.sessions[tier]) {
    const card = state.cards[tier];
    setStatus(`Loading the ${card.label} model (${card.fp32_onnx_mb} MB)...`);
    state.sessions[tier] = ort.InferenceSession.create(`models/${tier}/model.onnx`, {
      executionProviders: ["wasm"],
    }).catch((error) => {
      delete state.sessions[tier];
      throw error;
    });
  }
  return state.sessions[tier];
}

async function classify(bitmap, sample) {
  state.last = { bitmap, sample };
  const tier = state.tier;
  const card = state.cards[tier];
  const size = card.img_size;

  const canvas = prepare(bitmap, size);
  const preview = $("preview");
  preview.getContext("2d").drawImage(canvas, 0, 0, preview.width, preview.height);
  const rgba = canvas.getContext("2d").getImageData(0, 0, size, size).data;
  const input = new ort.Tensor("float32", Preprocess.rgbaToChw(rgba, size), [1, 3, size, size]);

  const session = await getSession(tier);
  setStatus("Running the model...");
  const started = performance.now();
  const output = await session.run({ [session.inputNames[0]]: input });
  const elapsed = performance.now() - started;
  const score = Number(output[session.outputNames[0]].data[0]);
  setStatus("");
  render(score, card, elapsed, sample);
}

function render(score, card, elapsed, sample) {
  const result = Preprocess.verdict(score, card.threshold);
  $("result").hidden = false;
  const badge = $("badge");
  badge.textContent = result === "live" ? "LIVE" : "SPOOF";
  badge.className = `badge ${result}`;
  $("fill").style.left = `${Math.min(Math.max(score, 0), 1) * 100}%`;
  $("tick").style.left = `${card.threshold * 100}%`;
  $("detail").textContent =
    `Spoof score ${score.toFixed(3)} (spoof if at least ${card.threshold}). ` +
    `Inference ${elapsed.toFixed(0)} ms in WebAssembly on this device.`;
  const ref = sample && sample.scores && sample.scores[state.tier];
  $("reference").textContent =
    ref === undefined
      ? ""
      : `Python reference score ${ref.toFixed(4)}, difference ${Math.abs(ref - score).toFixed(4)}. ` +
        `True label: ${sample.label}.`;
}

async function handleBlob(blob, sample) {
  try {
    await classify(await loadBitmap(blob), sample);
  } catch (error) {
    console.error(error);
    setStatus(`Could not process this image: ${error.message || error}`);
  }
}

function setupTiers() {
  const available = TIERS.filter((t) => state.cards[t.id]);
  const box = $("tiers");
  for (const tier of available) {
    state.cards[tier.id].label = tier.label;
    const label = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "tier";
    radio.value = tier.id;
    radio.addEventListener("change", () => {
      state.tier = tier.id;
      $("model-info").textContent = describe(state.cards[tier.id]);
      if (state.last) handleBlobFromLast();
    });
    label.append(radio, tier.label);
    box.append(label);
  }
  state.tier = available[available.length - 1].id; // default: the most accurate model
  box.querySelector(`input[value="${state.tier}"]`).checked = true;
  $("model-info").textContent = describe(state.cards[state.tier]);
}

async function handleBlobFromLast() {
  try {
    await classify(state.last.bitmap, state.last.sample);
  } catch (error) {
    setStatus(`Could not run the model: ${error.message || error}`);
  }
}

async function loadSamples() {
  try {
    const response = await fetch("samples/samples.json");
    if (!response.ok) return;
    const samples = await response.json();
    const box = $("samples");
    for (const sample of samples) {
      const button = document.createElement("button");
      button.title = `Sample (${sample.label})`;
      const img = document.createElement("img");
      img.src = `samples/${sample.file}`;
      img.alt = `Sample, true label ${sample.label}`;
      button.append(img);
      button.addEventListener("click", async () => {
        const blob = await (await fetch(img.src)).blob();
        handleBlob(blob, sample);
      });
      box.append(button);
    }
    box.hidden = samples.length === 0;
  } catch (error) {
    /* samples are optional */
  }
}

function setupInput() {
  const drop = $("drop");
  const file = $("file");
  file.addEventListener("change", () => file.files[0] && handleBlob(file.files[0]));
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      file.click();
    }
  });
  ["dragenter", "dragover"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("over");
    })
  );
  drop.addEventListener("drop", (event) => {
    const dropped = event.dataTransfer.files[0];
    if (dropped) handleBlob(dropped);
  });
}

async function init() {
  setupInput();
  for (const tier of TIERS) {
    try {
      const response = await fetch(`models/${tier.id}/card.json`);
      if (response.ok) state.cards[tier.id] = await response.json();
    } catch (error) {
      /* tier not available */
    }
  }
  if (!Object.keys(state.cards).length) {
    $("model-info").textContent = "No model found. Run scripts/make_demo_assets.py first.";
    return;
  }
  setupTiers();
  await loadSamples();
}

init();
