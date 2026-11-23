/* Pure helpers shared by the page and the Node test. Mirrors src/liveness/predict.py:
   centre-crop to a square, resize to 256 px, then to the model size, scale to [0, 1], CHW. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Preprocess = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const PREP_SIZE = 256;

  function squareCrop(width, height) {
    const side = Math.min(width, height);
    return { sx: Math.floor((width - side) / 2), sy: Math.floor((height - side) / 2), side };
  }

  // rgba: Uint8ClampedArray from canvas getImageData, size x size pixels -> Float32 [3, size, size]
  function rgbaToChw(rgba, size) {
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      out[i] = rgba[4 * i] / 255;
      out[plane + i] = rgba[4 * i + 1] / 255;
      out[2 * plane + i] = rgba[4 * i + 2] / 255;
    }
    return out;
  }

  function verdict(score, threshold) {
    return score >= threshold ? "spoof" : "live";
  }

  return { PREP_SIZE, squareCrop, rgbaToChw, verdict };
});
