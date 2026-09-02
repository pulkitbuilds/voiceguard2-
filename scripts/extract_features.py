# scripts/extract_features.py
#
# Python training-time implementation of lib/audioFeatures.js.
# IMPORTANT: Keep this implementation in sync with the JS extractor.

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.fftpack import dct as scipy_dct


FRAME_SIZE = 1024
HOP_SIZE = 512
N_MFCC = 13
N_MEL = 26
SILENCE_RMS = 0.008


FEATURE_NAMES = [
    "spectralCentroidMean",
    "spectralCentroidStd",
    "spectralFlatnessMean",
    "spectralFlatnessStd",
    "spectralRolloffMean",
    "spectralRolloffStd",
    "spectralBandwidthMean",
    "spectralBandwidthStd",
    "zcrMean",
    "zcrStd",
    "rmsMean",
    "rmsStd",
    *[f"mfcc{i + 1}Mean" for i in range(N_MFCC)],
    *[f"mfcc{i + 1}Std" for i in range(N_MFCC)],
    "pitchMean",
    "pitchStd",
    "pitchVoicedRatio",
    "jitter",
    "shimmer",
    "hfEnergyRatio",
    "silenceRatio",
]


def hamming_window(size):
    return 0.54 - 0.46 * np.cos(
        (2 * np.pi * np.arange(size)) / (size - 1)
    )


def frame_signal(samples):
    frames = []

    for start in range(
        0,
        max(1, len(samples) - FRAME_SIZE + 1),
        HOP_SIZE,
    ):
        if start + FRAME_SIZE <= len(samples):
            frames.append(samples[start:start + FRAME_SIZE])

    # IMPORTANT:
    # Match JS behavior: pad a short clip to one frame.
    if len(frames) == 0 and len(samples) > 0:
        padded = np.zeros(FRAME_SIZE, dtype=np.float64)
        padded[:len(samples)] = samples
        frames.append(padded)

    return frames


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(n_mel, n_fft_bins, sample_rate):
    f_min = 0.0
    f_max = sample_rate / 2.0

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mel + 2
    )

    hz_points = mel_to_hz(mel_points)

    # Match JS:
    # floor(((nFftBins * 2 - 1) * hz) / sampleRate)
    bins = np.floor(
        ((n_fft_bins * 2 - 1) * hz_points) / sample_rate
    ).astype(int)

    filters = []

    for m in range(1, n_mel + 1):
        filt = np.zeros(n_fft_bins, dtype=np.float64)

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center > left:
            for k in range(left, center):
                if 0 <= k < n_fft_bins:
                    filt[k] = (k - left) / (center - left)

        if right > center:
            for k in range(center, right):
                if 0 <= k < n_fft_bins:
                    filt[k] = (right - k) / (right - center)

        filters.append(filt)

    return filters


def estimate_pitch(frame, sample_rate, min_hz=70, max_hz=400):
    n = len(frame)

    max_lag = int(sample_rate / min_hz)
    min_lag = int(sample_rate / max_hz)

    energy = np.sum(frame * frame)

    if energy < 1e-9:
        return None

    best_lag = -1
    best_val = 0.0

    upper = min(max_lag, n - 1)

    for lag in range(min_lag, upper + 1):
        s = np.dot(
            frame[:n - lag],
            frame[lag:]
        )

        norm = s / (n - lag)

        if norm > best_val:
            best_val = norm
            best_lag = lag

    if best_lag <= 0:
        return None

    confidence = best_val / (energy / n)

    if confidence < 0.3:
        return None

    return sample_rate / best_lag


def magnitude_spectrum(windowed):
    """
    Match the JS magnitude spectrum behavior:
    1024-point FFT -> 512 positive-frequency bins.
    """
    fft_result = rfft(windowed)

    magnitude = np.abs(fft_result)

    # JS implementation uses 512 bins.
    return magnitude[:512]


def mean(arr):
    if len(arr) == 0:
        return 0.0

    return float(np.mean(arr))


def std(arr, m=None):
    if len(arr) < 2:
        return 0.0

    if m is None:
        m = mean(arr)

    return float(
        np.sqrt(
            np.mean(
                (np.asarray(arr) - m) ** 2
            )
        )
    )


def dct_type_ii(input_array, n_out):
    """
    Match JS DCT-II implementation:

    sum(input[n] *
        cos((pi / N) * (n + 0.5) * k))
    """
    x = np.asarray(input_array, dtype=np.float64)
    N = len(x)

    output = np.zeros(n_out, dtype=np.float64)

    for k in range(n_out):
        n = np.arange(N)

        output[k] = np.sum(
            x * np.cos(
                (np.pi / N) * (n + 0.5) * k
            )
        )

    return output


def extract_features(samples, sample_rate):
    samples = np.asarray(samples, dtype=np.float64)

    if len(samples) == 0:
        return {
            "vector": [0.0] * len(FEATURE_NAMES),
            "featureMap": {},
            "meta": {
                "frames": 0,
                "voicedFrames": 0,
                "durationSec": 0.0,
            },
        }

    frames = frame_signal(samples)

    window = hamming_window(FRAME_SIZE)

    n_fft_bins = 512

    mel_filters = build_mel_filterbank(
        N_MEL,
        n_fft_bins,
        sample_rate,
    )

    centroids = []
    flatness = []
    rolloffs = []
    bandwidths = []
    zcrs = []
    rmss = []
    mfcc_frames = []
    pitches = []

    voiced_frames = 0
    silent_frames = 0

    hf_energy_total = 0.0
    total_energy = 0.0

    nyquist = sample_rate / 2.0
    bin_hz = nyquist / n_fft_bins

    hf_cutoff_bin = int(
        np.floor(4000.0 / bin_hz)
    )

    for raw_frame in frames:

        # -------------------------
        # RMS + ZCR
        # -------------------------

        rms = np.sqrt(
            np.mean(raw_frame * raw_frame)
        )

        zc = np.sum(
            (
                raw_frame[:-1] >= 0
            ) != (
                raw_frame[1:] >= 0
            )
        )

        zc = zc / len(raw_frame)

        rmss.append(float(rms))
        zcrs.append(float(zc))

        # Silence
        if rms < SILENCE_RMS:
            silent_frames += 1
            continue

        # -------------------------
        # Window
        # -------------------------

        windowed = raw_frame * window

        mag = magnitude_spectrum(windowed)

        sum_mag = 0.0
        sum_freq_mag = 0.0
        log_sum = 0.0

        cumulative = []

        running = 0.0

        for k in range(len(mag)):

            freq = k * bin_hz

            value = float(mag[k])

            sum_mag += value

            sum_freq_mag += freq * value

            log_sum += np.log(
                value + 1e-12
            )

            running += value

            cumulative.append(running)

            if k >= hf_cutoff_bin:
                hf_energy_total += value

            total_energy += value

        # -------------------------
        # Spectral centroid
        # -------------------------

        if sum_mag > 0:
            centroid = (
                sum_freq_mag / sum_mag
            )
        else:
            centroid = 0.0

        centroids.append(centroid)

        # -------------------------
        # Spectral flatness
        # -------------------------

        geo_mean = np.exp(
            log_sum / len(mag)
        )

        arith_mean = (
            sum_mag / len(mag)
        )

        if arith_mean > 0:
            flatness_value = (
                geo_mean / arith_mean
            )
        else:
            flatness_value = 0.0

        flatness.append(
            flatness_value
        )

        # -------------------------
        # Spectral rolloff
        # -------------------------

        rolloff_threshold = (
            0.85 * running
        )

        rolloff_bin = len(mag) - 1

        for k, value in enumerate(cumulative):
            if value >= rolloff_threshold:
                rolloff_bin = k
                break

        rolloffs.append(
            rolloff_bin * bin_hz
        )

        # -------------------------
        # Spectral bandwidth
        # -------------------------

        var_freq = 0.0

        for k in range(len(mag)):
            freq = k * bin_hz

            var_freq += (
                mag[k]
                * (freq - centroid)
                * (freq - centroid)
            )

        if sum_mag > 0:
            bandwidth = np.sqrt(
                var_freq / sum_mag
            )
        else:
            bandwidth = 0.0

        bandwidths.append(
            bandwidth
        )

        # -------------------------
        # MFCC
        # -------------------------

        mel_energies = np.zeros(
            N_MEL,
            dtype=np.float64
        )

        for m in range(N_MEL):
            mel_energies[m] = np.log(
                np.sum(
                    mag * mel_filters[m]
                ) + 1e-8
            )

        mfcc = dct_type_ii(
            mel_energies,
            N_MFCC
        )

        mfcc_frames.append(mfcc)

        # -------------------------
        # Pitch
        # -------------------------

        f0 = estimate_pitch(
            raw_frame,
            sample_rate
        )

        if f0 is not None:
            pitches.append(f0)
            voiced_frames += 1

    # Match JS behavior
    n_voiced_frames = max(
        1,
        len(centroids)
    )

    # -------------------------
    # MFCC aggregate
    # -------------------------

    mfcc_means = np.zeros(
        N_MFCC
    )

    mfcc_stds = np.zeros(
        N_MFCC
    )

    if len(mfcc_frames) > 0:

        mfcc_matrix = np.asarray(
            mfcc_frames
        )

        for c in range(N_MFCC):

            col = mfcc_matrix[:, c]

            mfcc_means[c] = mean(col)

            mfcc_stds[c] = std(
                col,
                mfcc_means[c]
            )

    # -------------------------
    # Jitter
    # -------------------------

    jitter = 0.0

    if len(pitches) > 1:

        periods = (
            1.0 /
            np.asarray(pitches)
        )

        changes = (
            np.abs(
                periods[1:]
                - periods[:-1]
            )
            / periods[:-1]
        )

        jitter = float(
            np.mean(changes)
        )

    # -------------------------
    # Shimmer
    # -------------------------

    shimmer = 0.0

    if len(rmss) > 1:

        total = 0.0
        count = 0

        for i in range(1, len(rmss)):

            if rmss[i - 1] > SILENCE_RMS:

                total += (
                    abs(
                        rmss[i]
                        - rmss[i - 1]
                    )
                    / rmss[i - 1]
                )

                count += 1

        if count > 0:
            shimmer = total / count

    total_frames = max(
        1,
        len(frames)
    )

    # -------------------------
    # Feature map
    # -------------------------

    feature_map = {

        "spectralCentroidMean":
            mean(centroids),

        "spectralCentroidStd":
            std(centroids),

        "spectralFlatnessMean":
            mean(flatness),

        "spectralFlatnessStd":
            std(flatness),

        "spectralRolloffMean":
            mean(rolloffs),

        "spectralRolloffStd":
            std(rolloffs),

        "spectralBandwidthMean":
            mean(bandwidths),

        "spectralBandwidthStd":
            std(bandwidths),

        "zcrMean":
            mean(zcrs),

        "zcrStd":
            std(zcrs),

        "rmsMean":
            mean(rmss),

        "rmsStd":
            std(rmss),

        "pitchMean":
            mean(pitches),

        "pitchStd":
            std(pitches),

        "pitchVoicedRatio":
            voiced_frames / n_voiced_frames,

        "jitter":
            jitter,

        "shimmer":
            shimmer,

        "hfEnergyRatio":
            (
                hf_energy_total /
                total_energy
                if total_energy > 0
                else 0.0
            ),

        "silenceRatio":
            silent_frames /
            total_frames,
    }

    for i in range(N_MFCC):

        feature_map[
            f"mfcc{i + 1}Mean"
        ] = float(
            mfcc_means[i]
        )

        feature_map[
            f"mfcc{i + 1}Std"
        ] = float(
            mfcc_stds[i]
        )

    # -------------------------
    # Ordered 45-D vector
    # -------------------------

    vector = []

    for name in FEATURE_NAMES:

        value = feature_map.get(
            name,
            0.0
        )

        if np.isfinite(value):
            vector.append(
                float(value)
            )
        else:
            vector.append(0.0)

    return {
        "vector": vector,
        "featureMap": feature_map,
        "meta": {
            "frames": len(frames),
            "voicedFrames": voiced_frames,
            "durationSec":
                len(samples) /
                sample_rate,
        },
    }