// Minimal iterative radix-2 Cooley-Tukey FFT. No native dependencies so it
// runs identically in the browser (client-side feature extraction) and in
// Node (server-side / offline tooling).
//
// Input: real-valued Float64Array/Array of length N (N must be a power of 2).
// Output: { re: Float64Array, im: Float64Array } of length N.

function nextPow2(n) {
  return 1 << Math.ceil(Math.log2(n));
}

function fft(reIn, imIn) {
  const n = reIn.length;
  if (n <= 1) return { re: Float64Array.from(reIn), im: Float64Array.from(imIn || new Float64Array(n)) };
  if (n & (n - 1)) throw new Error('fft: length must be a power of 2');

  const re = Float64Array.from(reIn);
  const im = imIn ? Float64Array.from(imIn) : new Float64Array(n);

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlRe = Math.cos(ang);
    const wlIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * wRe - im[i + k + len / 2] * wIm;
        const vIm = re[i + k + len / 2] * wIm + im[i + k + len / 2] * wRe;

        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;

        const nWRe = wRe * wlRe - wIm * wlIm;
        const nWIm = wRe * wlIm + wIm * wlRe;
        wRe = nWRe;
        wIm = nWIm;
      }
    }
  }

  return { re, im };
}

// Real-input magnitude spectrum, zero-padded to the next power of 2.
function magnitudeSpectrum(frame) {
  const n = nextPow2(frame.length);
  const re = new Float64Array(n);
  re.set(frame);
  const { re: fre, im: fim } = fft(re, new Float64Array(n));
  const half = n / 2;
  const mag = new Float64Array(half);
  for (let i = 0; i < half; i++) {
    mag[i] = Math.sqrt(fre[i] * fre[i] + fim[i] * fim[i]);
  }
  return mag;
}

module.exports = { fft, nextPow2, magnitudeSpectrum };
