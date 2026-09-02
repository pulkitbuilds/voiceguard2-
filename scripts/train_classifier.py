"""
Train the VoiceGuard spoof/clone classifier on a real labeled dataset and
export weights that drop straight into lib/model_weights.json for the
Node.js/browser inference code (lib/classifier.js) -- no code changes needed
on the app side, only this JSON file.

Expected dataset layout (matches common ASVspoof-style releases):

    dataset/
      bonafide/*.wav      # genuine human speech
      spoof/*.wav         # TTS / voice-conversion / replayed / cloned speech

Good public sources to point this at: ASVspoof 2019/2021 (LA/DF partitions),
WaveFake, In-the-Wild, or your own recorded + cloned pairs.

Usage:
    pip install -r requirements.txt   # numpy, scipy, scikit-learn
    python scripts/train_classifier.py --data_dir dataset/ --out ../lib/model_weights.json

The script:
  1. Extracts the same feature vector as lib/audioFeatures.js (via
     extract_features.py) for every clip.
  2. Standardizes features (mean/std saved into the export -- inference
     re-applies the identical transform).
  3. Trains a small 2-layer MLP (matching lib/classifier.js's architecture:
     Linear -> ReLU -> Linear -> Sigmoid) with scikit-learn.
  4. Exports weights/biases + normalization stats as JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_dataset(data_dir: Path):
    from scipy.io import wavfile

    from extract_features import extract_features, FEATURE_NAMES  # noqa: F401

    X, y_labels = [], []
    for label, folder in (("bonafide", 0), ("spoof", 1)):
        for wav_path in sorted((data_dir / label).glob("*.wav")):
            sr, audio = wavfile.read(wav_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # normalize to [-1, 1] based on the source dtype's full-scale range
            if np.issubdtype(audio.dtype, np.integer):
                full_scale = float(np.iinfo(audio.dtype).max)
                audio = audio.astype(np.float64) / full_scale
            else:
                audio = audio.astype(np.float64)
            vector, _ = extract_features(audio, sr)
            X.append(vector)
            y_labels.append(folder)
    if not X:
        raise SystemExit(
            f"No .wav files found under {data_dir}/bonafide or {data_dir}/spoof. "
            "See the docstring at the top of this file for the expected layout."
        )
    return np.array(X, dtype=np.float64), np.array(y_labels, dtype=np.int64)


def train(X, y, hidden_units=12, seed=42):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_units,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=2000,
        random_state=seed,
    )
    clf.fit(X_train_s, y_train)

    preds = clf.predict_proba(X_test_s)[:, 1]
    print(f"Held-out AUC:      {roc_auc_score(y_test, preds):.4f}")
    print(f"Held-out accuracy: {accuracy_score(y_test, preds > 0.5):.4f}")

    return clf, scaler


def export_weights(clf, scaler, feature_names, out_path: Path, version="trained-v1"):
    # sklearn MLPClassifier stores coefs_[i] as [inDim, outDim]; our JS
    # inference expects W as [outDim][inDim] (row = output neuron).
    W1 = clf.coefs_[0].T.tolist()
    b1 = clf.intercepts_[0].tolist()
    W2 = [clf.coefs_[1].T[0].tolist()]
    b2 = [float(clf.intercepts_[1][0])]

    payload = {
        "featureNames": feature_names,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "hiddenUnitNames": [f"h{i}" for i in range(len(b1))],
        "version": version,
        "note": "Trained on a labeled bona-fide/spoof corpus via scripts/train_classifier.py.",
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} ({len(feature_names)} input dims, {len(b1)} hidden units)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True, help="Folder with bonafide/ and spoof/ subfolders of .wav files")
    parser.add_argument("--out", type=Path, default=Path("../lib/model_weights.json"))
    parser.add_argument("--hidden_units", type=int, default=12)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from extract_features import FEATURE_NAMES

    print(f"Loading dataset from {args.data_dir} ...")
    X, y = load_dataset(args.data_dir)
    print(f"Loaded {len(X)} clips ({int((y == 0).sum())} bonafide / {int((y == 1).sum())} spoof)")

    clf, scaler = train(X, y, hidden_units=args.hidden_units)
    export_weights(clf, scaler, FEATURE_NAMES, args.out)


if __name__ == "__main__":
    main()
