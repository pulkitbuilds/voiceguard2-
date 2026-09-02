// Frame-based acoustic feature extraction for voice spoof / clone detection.
//
// Designed to run isomorphically: it is called client-side in the browser
// right after Web Audio API decodes the recorded/uploaded clip (so we never
// need a native audio codec on the server), but it has zero DOM dependencies
// so it can equally be required from a Node script (see scripts/ for the
// offline-training parity implementation in Python).
//
// The exact same ordered feature vector (see FEATURE_NAMES) is what the
// classifier in lib/classifier.js expects, and what scripts/train_classifier.py
// must reproduce when training on a labeled bona-fide/spoof dataset.

const { magnitudeSpectrum } = require('./fft');

const FRAME_SIZE = 1024; // ~64ms @ 16kHz
const HOP_SIZE = 512; // 50% overlap
const N_MFCC = 13;
const N_MEL = 26;
const SILENCE_RMS = 0.008;

const FEATURE_NAMES = [
  'spectralCentroidMean', 'spectralCentroidStd',
  'spectralFlatnessMean', 'spectralFlatnessStd',
  'spectralRolloffMean', 'spectralRolloffStd',
  'spectralBandwidthMean', 'spectralBandwidthStd',
  'zcrMean', 'zcrStd',
  'rmsMean', 'rmsStd',
  ...Array.from({ length: N_MFCC }, (_, i) => `mfcc${i + 1}Mean`),
  ...Array.from({ length: N_MFCC }, (_, i) => `mfcc${i + 1}Std`),
  'pitchMean', 'pitchStd', 'pitchVoicedRatio',
  'jitter', 'shimmer',
  'hfEnergyRatio', 'silenceRatio',
];

function hammingWindow(size) {
  const w = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (size - 1));
  }
  return w;
}

function frameSignal(samples, frameSize, hopSize) {
  const frames = [];
  for (let start = 0; start + frameSize <= samples.length; start += hopSize) {
    frames.push(samples.subarray(start, start + frameSize));
  }
  if (frames.length === 0 && samples.length > 0) {
    // pad a single short clip up to one frame
    const padded = new Float32Array(frameSize);
    padded.set(samples);
    frames.push(padded);
  }
  return frames;
}

function mean(arr) {
  if (!arr.length) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

function std(arr, m) {
  if (arr.length < 2) return 0;
  const mm = m === undefined ? mean(arr) : m;
  let s = 0;
  for (const v of arr) s += (v - mm) * (v - mm);
  return Math.sqrt(s / arr.length);
}

// --- Mel filterbank for MFCCs -------------------------------------------------

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}
function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelFilterbank(nMel, nFftBins, sampleRate) {
  const fMin = 0;
  const fMax = sampleRate / 2;
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = Array.from({ length: nMel + 2 }, (_, i) => melMin + ((melMax - melMin) * i) / (nMel + 1));
  const hzPoints = melPoints.map(melToHz);
  const bin = hzPoints.map((hz) => Math.floor(((nFftBins * 2 - 1) * hz) / sampleRate));

  const filters = [];
  for (let m = 1; m <= nMel; m++) {
    const f = new Float64Array(nFftBins);
    const left = bin[m - 1];
    const center = bin[m];
    const right = bin[m + 1];
    for (let k = left; k < center; k++) {
      if (k >= 0 && k < nFftBins && center > left) f[k] = (k - left) / (center - left);
    }
    for (let k = center; k < right; k++) {
      if (k >= 0 && k < nFftBins && right > center) f[k] = (right - k) / (right - center);
    }
    filters.push(f);
  }
  return filters;
}

function dct(input, nOut) {
  const N = input.length;
  const out = new Float64Array(nOut);
  for (let k = 0; k < nOut; k++) {
    let s = 0;
    for (let n = 0; n < N; n++) {
      s += input[n] * Math.cos((Math.PI / N) * (n + 0.5) * k);
    }
    out[k] = s;
  }
  return out;
}

// --- Pitch (F0) via autocorrelation ------------------------------------------

function estimatePitch(frame, sampleRate, minHz = 70, maxHz = 400) {
  const n = frame.length;
  const maxLag = Math.floor(sampleRate / minHz);
  const minLag = Math.floor(sampleRate / maxHz);
  let bestLag = -1;
  let bestVal = 0;

  // normalize
  let energy = 0;
  for (let i = 0; i < n; i++) energy += frame[i] * frame[i];
  if (energy < 1e-9) return null;

  for (let lag = minLag; lag <= Math.min(maxLag, n - 1); lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += frame[i] * frame[i + lag];
    const norm = sum / (n - lag);
    if (norm > bestVal) {
      bestVal = norm;
      bestLag = lag;
    }
  }
  if (bestLag <= 0) return null;
  const confidence = bestVal / (energy / n);
  if (confidence < 0.3) return null; // unvoiced / noisy frame
  return sampleRate / bestLag;
}

/**
 * Extract the full ordered feature vector from a mono Float32Array of PCM
 * samples. `sampleRate` should match the AudioBuffer's sampleRate (features
 * are resilient to typical telephony/VoIP rates: 8k-48kHz).
 */
function extractFeatures(samples, sampleRate) {
  const window = hammingWindow(FRAME_SIZE);
  const frames = frameSignal(samples, FRAME_SIZE, HOP_SIZE);

  const nFftBins = 512; // magnitudeSpectrum() returns nextPow2(1024)/2 = 512
  const melFilters = buildMelFilterbank(N_MEL, nFftBins, sampleRate);

  const centroids = [];
  const flatness = [];
  const rolloffs = [];
  const bandwidths = [];
  const zcrs = [];
  const rmss = [];
  const mfccFrames = [];
  const pitches = [];
  let voicedFrames = 0;
  let silentFrames = 0;
  let hfEnergyTotal = 0;
  let totalEnergy = 0;

  const nyquist = sampleRate / 2;
  const binHz = nyquist / nFftBins;
  const hfCutoffBin = Math.floor(4000 / binHz); // above 4kHz = artifact-prone band

  for (const rawFrame of frames) {
    // RMS + ZCR (time domain, unwindowed)
    let rms = 0;
    let zc = 0;
    for (let i = 0; i < rawFrame.length; i++) {
      rms += rawFrame[i] * rawFrame[i];
      if (i > 0 && ((rawFrame[i - 1] >= 0) !== (rawFrame[i] >= 0))) zc++;
    }
    rms = Math.sqrt(rms / rawFrame.length);
    zc = zc / rawFrame.length;
    rmss.push(rms);
    zcrs.push(zc);

    if (rms < SILENCE_RMS) {
      silentFrames++;
      continue; // skip spectral/pitch stats for silence
    }

    // windowed frame for spectral analysis
    const windowed = new Float64Array(rawFrame.length);
    for (let i = 0; i < rawFrame.length; i++) windowed[i] = rawFrame[i] * window[i];

    const mag = magnitudeSpectrum(windowed); // length 512
    let sumMag = 0;
    let sumFreqMag = 0;
    let logSum = 0;
    let cumMag = [];
    let running = 0;

    for (let k = 0; k < mag.length; k++) {
      const freq = k * binHz;
      sumMag += mag[k];
      sumFreqMag += freq * mag[k];
      logSum += Math.log(mag[k] + 1e-12);
      running += mag[k];
      cumMag.push(running);
      if (k >= hfCutoffBin) hfEnergyTotal += mag[k];
      totalEnergy += mag[k];
    }

    const centroid = sumMag > 0 ? sumFreqMag / sumMag : 0;
    centroids.push(centroid);

    const geoMean = Math.exp(logSum / mag.length);
    const arithMean = sumMag / mag.length;
    flatness.push(arithMean > 0 ? geoMean / arithMean : 0);

    const rolloffThresh = 0.85 * running;
    let rolloffBin = mag.length - 1;
    for (let k = 0; k < cumMag.length; k++) {
      if (cumMag[k] >= rolloffThresh) {
        rolloffBin = k;
        break;
      }
    }
    rolloffs.push(rolloffBin * binHz);

    let varFreq = 0;
    for (let k = 0; k < mag.length; k++) {
      const freq = k * binHz;
      varFreq += mag[k] * (freq - centroid) * (freq - centroid);
    }
    bandwidths.push(sumMag > 0 ? Math.sqrt(varFreq / sumMag) : 0);

    // MFCCs
    const melEnergies = new Float64Array(N_MEL);
    for (let m = 0; m < N_MEL; m++) {
      let s = 0;
      const filt = melFilters[m];
      for (let k = 0; k < mag.length; k++) s += mag[k] * filt[k];
      melEnergies[m] = Math.log(s + 1e-8);
    }
    const mfcc = dct(melEnergies, N_MFCC);
    mfccFrames.push(mfcc);

    // Pitch
    const f0 = estimatePitch(rawFrame, sampleRate);
    if (f0) {
      pitches.push(f0);
      voicedFrames++;
    }
  }

  const nVoicedFrames = Math.max(1, centroids.length);

  // aggregate MFCCs
  const mfccMeans = new Array(N_MFCC).fill(0);
  const mfccStds = new Array(N_MFCC).fill(0);
  for (let c = 0; c < N_MFCC; c++) {
    const col = mfccFrames.map((f) => f[c]);
    mfccMeans[c] = mean(col);
    mfccStds[c] = std(col, mfccMeans[c]);
  }

  // jitter: mean absolute relative change between consecutive voiced-frame periods
  let jitter = 0;
  if (pitches.length > 1) {
    let sum = 0;
    for (let i = 1; i < pitches.length; i++) {
      sum += Math.abs(1 / pitches[i] - 1 / pitches[i - 1]) / (1 / pitches[i - 1]);
    }
    jitter = sum / (pitches.length - 1);
  }

  // shimmer: mean absolute relative change between consecutive frame RMS values
  let shimmer = 0;
  if (rmss.length > 1) {
    let sum = 0;
    let count = 0;
    for (let i = 1; i < rmss.length; i++) {
      if (rmss[i - 1] > SILENCE_RMS) {
        sum += Math.abs(rmss[i] - rmss[i - 1]) / rmss[i - 1];
        count++;
      }
    }
    shimmer = count > 0 ? sum / count : 0;
  }

  const totalFrames = Math.max(1, frames.length);

  const featureMap = {
    spectralCentroidMean: mean(centroids),
    spectralCentroidStd: std(centroids),
    spectralFlatnessMean: mean(flatness),
    spectralFlatnessStd: std(flatness),
    spectralRolloffMean: mean(rolloffs),
    spectralRolloffStd: std(rolloffs),
    spectralBandwidthMean: mean(bandwidths),
    spectralBandwidthStd: std(bandwidths),
    zcrMean: mean(zcrs),
    zcrStd: std(zcrs),
    rmsMean: mean(rmss),
    rmsStd: std(rmss),
    pitchMean: mean(pitches),
    pitchStd: std(pitches),
    pitchVoicedRatio: voicedFrames / nVoicedFrames,
    jitter,
    shimmer,
    hfEnergyRatio: totalEnergy > 0 ? hfEnergyTotal / totalEnergy : 0,
    silenceRatio: silentFrames / totalFrames,
  };
  mfccMeans.forEach((v, i) => { featureMap[`mfcc${i + 1}Mean`] = v; });
  mfccStds.forEach((v, i) => { featureMap[`mfcc${i + 1}Std`] = v; });

  const vector = FEATURE_NAMES.map((name) => {
    const v = featureMap[name];
    return Number.isFinite(v) ? v : 0;
  });

  return { vector, featureMap, meta: { frames: frames.length, voicedFrames, durationSec: samples.length / sampleRate } };
}

module.exports = { extractFeatures, FEATURE_NAMES, FRAME_SIZE, HOP_SIZE };
