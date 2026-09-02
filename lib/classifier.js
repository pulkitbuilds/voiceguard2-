// Lightweight feed-forward classifier: standardize -> Linear -> ReLU -> Linear -> Sigmoid.
// Runs pure-JS inference (no ONNX/TF runtime needed) so it's fast and cold-start-friendly
// on Vercel serverless/edge functions.
//
// Weight format (see lib/model_weights.json) is produced by scripts/train_classifier.py.
// The shipped weights.json is a DEMO model — see README.md "Training a real model" for how
// to retrain on an actual bona-fide/spoof corpus (ASVspoof 2019/2021, WaveFake, In-the-Wild, etc.)
// and drop the exported JSON in to replace it, with zero code changes.

const weights = require('./model_weights.json');

function relu(x) {
  return x > 0 ? x : 0;
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function standardize(vector, mean, std) {
  return vector.map((v, i) => (v - mean[i]) / (std[i] || 1e-8));
}

function matVec(W, x) {
  // W: [outDim][inDim], x: [inDim] -> [outDim]
  const out = new Array(W.length).fill(0);
  for (let o = 0; o < W.length; o++) {
    let s = 0;
    const row = W[o];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function addBias(v, b) {
  return v.map((x, i) => x + b[i]);
}

/**
 * @param {number[]} featureVector - ordered per lib/audioFeatures.js FEATURE_NAMES
 * @returns {{ spoofProbability: number, hiddenActivations: number[] }}
 */
function classify(featureVector) {
  const { mean, std, W1, b1, W2, b2 } = weights;

  if (featureVector.length !== mean.length) {
    throw new Error(
      `classifier: feature vector length ${featureVector.length} does not match model input dim ${mean.length}`
    );
  }

  const x = standardize(featureVector, mean, std);
  const h = addBias(matVec(W1, x), b1).map(relu);
  const outRaw = addBias(matVec(W2, h), b2);
  const spoofProbability = sigmoid(outRaw[0]);

  return { spoofProbability, hiddenActivations: h };
}

module.exports = { classify, weights };
