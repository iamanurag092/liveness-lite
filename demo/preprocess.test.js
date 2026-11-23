// Run with: node demo/preprocess.test.js
const assert = require("assert");
const P = require("./preprocess.js");

assert.deepStrictEqual(P.squareCrop(200, 100), { sx: 50, sy: 0, side: 100 });
assert.deepStrictEqual(P.squareCrop(100, 201), { sx: 0, sy: 50, side: 100 });
assert.deepStrictEqual(P.squareCrop(64, 64), { sx: 0, sy: 0, side: 64 });

// 2x2 image: pixels (r,g,b,a) -> channel planes
const rgba = new Uint8ClampedArray([
  255, 0, 0, 255, 0, 255, 0, 255,
  0, 0, 255, 255, 255, 255, 255, 255,
]);
const chw = P.rgbaToChw(rgba, 2);
assert.strictEqual(chw.length, 12);
assert.deepStrictEqual(Array.from(chw.slice(0, 4)), [1, 0, 0, 1]); // red plane
assert.deepStrictEqual(Array.from(chw.slice(4, 8)), [0, 1, 0, 1]); // green plane
assert.deepStrictEqual(Array.from(chw.slice(8, 12)), [0, 0, 1, 1]); // blue plane

assert.strictEqual(P.verdict(0.2, 0.18), "spoof");
assert.strictEqual(P.verdict(0.18, 0.18), "spoof"); // score >= threshold means spoof
assert.strictEqual(P.verdict(0.1, 0.18), "live");
console.log("preprocess tests passed");
