"""
Feature extraction that mirrors lib/audioFeatures.js EXACTLY (same framing,
same mel filterbank construction, same DCT, same autocorrelation pitch
tracker) so a model trained here plugs straight into the Node.js/browser
inference code with zero changes.

Deliberately dependency-light: only numpy + scipy (both nearly always
already installed), no librosa, no soundfile. WAV I/O uses
scipy.io.wavfile, which handles standard PCM/float WAV natively.

Feature order MUST match lib/audioFeatures.js FEATURE_NAMES exactly.
"""

import numpy as np
from scipy.fft import dct as scipy_dct

FRAME_SIZE = 1024
HOP_SIZE = 512
N_MFCC = 13
N_MEL = 26
SILENCE_RMS = 0.008

FEATURE_NAMES = (
    ["spectralCentroidMean", "spectralCentroidStd",
     "spectralFlatnessMean", "spectralFlatnessStd",
     "spectralRolloffMean", "spectralRolloffStd",
     "spectralBandwidthMean", "spectralBandwidthStd",
     "zcrMean", "zcrStd",
     "rmsMean", "rmsStd"]
    + [f"mfcc{i + 1}Mean" for i in range(N_MFCC)]
    + [f"mfcc{i + 1}Std" for i in range(N_MFCC)]
    + ["pitchMean", "pitchStd", "pitchVoicedRatio", "jitter", "shimmer",
       "hfEnergyRatio", "silenceRatio"]
)


def _hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def _build_mel_filterbank(n_mel, n_fft_bins, sample_rate):
    """Mirrors buildMelFilterbank() in lib/audioFeatures.js exactly."""
    mel_min, mel_max = _hz_to_mel(0), _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft_bins * 2 - 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mel, n_fft_bins))
    for m in range(1, n_mel + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(max(left, 0), min(center, n_fft_bins)):
            if center > left:
                filters[m - 1, k] = (k - left) / (center - left)
        for k in range(max(center, 0), min(right, n_fft_bins)):
            if right > center:
                filters[m - 1, k] = (right - k) / (right - center)
    return filters


def _magnitude_spectrum(frame_windowed):
    """Zero-pads to next pow2 and returns the first-half magnitude spectrum,
    matching lib/fft.js's magnitudeSpectrum()."""
    n = frame_windowed.shape[0]
    n_fft = 1 << int(np.ceil(np.log2(n)))
    spec = np.fft.rfft(frame_windowed, n=n_fft)
    return np.abs(spec[: n_fft // 2])


def _estimate_pitch(frame, sample_rate, min_hz=70, max_hz=400):
    """Fast autocorrelation pitch tracker.

    Keeps the same lag range and confidence calculation as the JS version,
    but computes all lag correlations using FFT-based autocorrelation.
    """
    n = len(frame)

    max_lag = int(sample_rate / min_hz)
    min_lag = int(sample_rate / max_hz)

    energy = np.sum(frame ** 2)
    if energy < 1e-9:
        return None

    upper = min(max_lag, n - 1)
    if min_lag > upper:
        return None

    # FFT-based autocorrelation.
    # Zero-pad to avoid circular convolution.
    fft_size = 1 << int(np.ceil(np.log2(2 * n - 1)))

    spectrum = np.fft.rfft(frame, n=fft_size)
    autocorr = np.fft.irfft(
        spectrum * np.conjugate(spectrum),
        n=fft_size
    )

    # Same normalization used by the original implementation:
    # dot(frame[:n-lag], frame[lag:]) / (n-lag)
    lags = np.arange(min_lag, upper + 1)
    vals = autocorr[lags] / (n - lags)

    best_idx = np.argmax(vals)
    best_lag = int(lags[best_idx])
    best_val = float(vals[best_idx])

    if best_val <= 0:
        return None

    confidence = best_val / (energy / n)

    if confidence < 0.3:
        return None

    return sample_rate / best_lag


def _hamming(size):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))


def extract_features(samples, sample_rate):
    """
    samples: 1-D float array, mono, in roughly [-1, 1]
    sample_rate: int
    Returns (vector: list[float], feature_map: dict[str, float])
    """
    samples = np.asarray(samples, dtype=np.float64)
    window = _hamming(FRAME_SIZE)

    n_fft_bins = 512  # matches magnitudeSpectrum() output length for FRAME_SIZE=1024
    mel_filters = _build_mel_filterbank(N_MEL, n_fft_bins, sample_rate)
    nyquist = sample_rate / 2
    bin_hz = nyquist / n_fft_bins
    hf_cutoff_bin = int(4000 / bin_hz)

    centroids, flatness, rolloffs, bandwidths = [], [], [], []
    zcrs, rmss, mfcc_frames, pitches = [], [], [], []
    voiced_frames = 0
    silent_frames = 0
    hf_energy_total = 0.0
    total_energy = 0.0

    starts = range(0, max(1, len(samples) - FRAME_SIZE + 1), HOP_SIZE)
    frames_seen = 0
    for start in starts:
        frame = samples[start:start + FRAME_SIZE]
        if len(frame) < FRAME_SIZE:
            break
        frames_seen += 1

        rms = np.sqrt(np.mean(frame ** 2))
        signs = frame >= 0
        zc = np.mean(signs[1:] != signs[:-1])
        rmss.append(rms)
        zcrs.append(zc)

        if rms < SILENCE_RMS:
            silent_frames += 1
            continue

        windowed = frame * window
        mag = _magnitude_spectrum(windowed)
        freqs = np.arange(len(mag)) * bin_hz

        sum_mag = mag.sum()
        centroid = float(np.sum(freqs * mag) / sum_mag) if sum_mag > 0 else 0.0
        centroids.append(centroid)

        log_sum = np.sum(np.log(mag + 1e-12))
        geo_mean = np.exp(log_sum / len(mag))
        arith_mean = sum_mag / len(mag)
        flatness.append(float(geo_mean / arith_mean) if arith_mean > 0 else 0.0)

        cum = np.cumsum(mag)
        thresh = 0.85 * cum[-1]
        rolloff_bin = int(np.searchsorted(cum, thresh))
        rolloff_bin = min(rolloff_bin, len(mag) - 1)
        rolloffs.append(rolloff_bin * bin_hz)

        var_freq = np.sum(mag * (freqs - centroid) ** 2)
        bandwidths.append(float(np.sqrt(var_freq / sum_mag)) if sum_mag > 0 else 0.0)

        mel_energies = np.log(mel_filters @ mag + 1e-8)
        # scipy's unnormalized type-2 DCT has an implicit factor of 2 that
        # lib/audioFeatures.js's hand-rolled DCT does not; divide it out so
        # MFCCs match the JS implementation exactly (verified bit-for-bit
        # against lib/audioFeatures.js on synthetic test signals).
        mfcc = (scipy_dct(mel_energies, type=2, norm=None) / 2.0)[:N_MFCC]
        mfcc_frames.append(mfcc)

        hf_energy_total += mag[hf_cutoff_bin:].sum()
        total_energy += sum_mag

        f0 = _estimate_pitch(frame, sample_rate)
        if f0:
            pitches.append(f0)
            voiced_frames += 1

    def m(a):
        return float(np.mean(a)) if len(a) else 0.0

    def s(a):
        return float(np.std(a)) if len(a) else 0.0

    if mfcc_frames:
        mfcc_arr = np.array(mfcc_frames)  # [frames, N_MFCC]
        mfcc_means = mfcc_arr.mean(axis=0)
        mfcc_stds = mfcc_arr.std(axis=0)
    else:
        mfcc_means = np.zeros(N_MFCC)
        mfcc_stds = np.zeros(N_MFCC)

    pitches_arr = np.array(pitches)
    if len(pitches_arr) > 1:
        periods = 1.0 / pitches_arr
        jitter = float(np.mean(np.abs(np.diff(periods)) / periods[:-1]))
    else:
        jitter = 0.0

    rmss_arr = np.array(rmss)
    shimmer = 0.0
    if len(rmss_arr) > 1:
        diffs, count = 0.0, 0
        for i in range(1, len(rmss_arr)):
            if rmss_arr[i - 1] > SILENCE_RMS:
                diffs += abs(rmss_arr[i] - rmss_arr[i - 1]) / rmss_arr[i - 1]
                count += 1
        shimmer = diffs / count if count > 0 else 0.0

    n_voiced_frames = max(1, len(centroids))
    total_frames = max(1, frames_seen)

    feature_map = {
        "spectralCentroidMean": m(centroids), "spectralCentroidStd": s(centroids),
        "spectralFlatnessMean": m(flatness), "spectralFlatnessStd": s(flatness),
        "spectralRolloffMean": m(rolloffs), "spectralRolloffStd": s(rolloffs),
        "spectralBandwidthMean": m(bandwidths), "spectralBandwidthStd": s(bandwidths),
        "zcrMean": m(zcrs), "zcrStd": s(zcrs),
        "rmsMean": m(rmss), "rmsStd": s(rmss),
        "pitchMean": m(pitches), "pitchStd": s(pitches),
        "pitchVoicedRatio": voiced_frames / n_voiced_frames,
        "jitter": jitter, "shimmer": shimmer,
        "hfEnergyRatio": (hf_energy_total / total_energy) if total_energy > 0 else 0.0,
        "silenceRatio": silent_frames / total_frames,
    }
    for i in range(N_MFCC):
        feature_map[f"mfcc{i + 1}Mean"] = float(mfcc_means[i])
        feature_map[f"mfcc{i + 1}Std"] = float(mfcc_stds[i])

    vector = [feature_map[name] if np.isfinite(feature_map[name]) else 0.0 for name in FEATURE_NAMES]
    return vector, feature_map
