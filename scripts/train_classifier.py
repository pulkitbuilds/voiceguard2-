# scripts/train_classifier.py

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from extract_features import (
    extract_features,
    FEATURE_NAMES,
)


# ---------------------------------------------------------
# WAV loading
# ---------------------------------------------------------

def load_wav(path):

    sample_rate, audio = wavfile.read(path)

    audio = np.asarray(audio)

    # Convert stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(
            audio.astype(np.float64),
            axis=1,
        )
    else:
        audio = audio.astype(
            np.float64
        )

    # Normalize PCM
    if np.issubdtype(
        audio.dtype,
        np.integer,
    ):
        info = np.iinfo(audio.dtype)

        max_abs = max(
            abs(info.min),
            abs(info.max),
        )

        audio = audio / max_abs

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    return sample_rate, audio


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_dataset(data_dir):

    data_dir = Path(data_dir)

    classes = [
        ("bonafide", 0),
        ("spoof", 1),
    ]

    X = []
    y = []
    paths = []

    for folder, label in classes:

        class_dir = data_dir / folder

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset folder: {class_dir}"
            )

        wav_files = sorted(
            class_dir.rglob("*.wav")
        )

        print(
            f"{folder}: "
            f"{len(wav_files)} files"
        )

        for wav_path in wav_files:

            try:

                sample_rate, audio = load_wav(
                    wav_path
                )

                result = extract_features(
                    audio,
                    sample_rate,
                )

                vector = result["vector"]

                if len(vector) != len(
                    FEATURE_NAMES
                ):
                    raise ValueError(
                        f"Expected "
                        f"{len(FEATURE_NAMES)} "
                        f"features, got "
                        f"{len(vector)}"
                    )

                X.append(vector)
                y.append(label)
                paths.append(
                    str(wav_path)
                )

            except Exception as e:

                print(
                    f"Skipping {wav_path}: {e}"
                )

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    print()
    print(
        f"Loaded {len(X)} clips "
        f"({np.sum(y == 0)} bonafide / "
        f"{np.sum(y == 1)} spoof)"
    )

    return X, y, paths


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
    )

    parser.add_argument(
        "--out",
        default="../lib/model_weights.json",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    X, y, paths = load_dataset(
        args.data_dir
    )

    if len(X) < 20:
        raise RuntimeError(
            "Dataset is too small."
        )

    # -----------------------------------------------------
    # Train/test split
    #
    # IMPORTANT:
    # This is still a clip-level split.
    # If your dataset has speaker IDs, we should
    # replace this with GroupShuffleSplit.
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )
    )

    print()
    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Test samples:     {len(X_test)}"
    )

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    clf = MLPClassifier(
        hidden_layer_sizes=(12,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        learning_rate_init=1e-3,
    )

    clf.fit(
        X_train_scaled,
        y_train,
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    probabilities = clf.predict_proba(
        X_test_scaled
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"AUC      : {auc:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    print()
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "bonafide",
                "spoof",
            ],
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Detect suspicious perfect performance
    # -----------------------------------------------------

    if accuracy >= 0.999 and auc >= 0.999:

        print()
        print("⚠️ WARNING")
        print(
            "Accuracy and AUC are almost perfect."
        )

        print(
            "This may indicate dataset leakage "
            "or source-specific artifacts."
        )

        print(
            "Do NOT assume this means "
            "real-world performance is perfect."
        )

    # -----------------------------------------------------
    # Export weights
    # -----------------------------------------------------

    # sklearn:
    #
    # coefs_[0] = [45][12]
    #
    # JS expects:
    #
    # W1 = [12][45]
    #

    W1 = (
        clf.coefs_[0]
        .T
        .tolist()
    )

    b1 = (
        clf.intercepts_[0]
        .tolist()
    )

    W2 = [
        clf.coefs_[1]
        .T[0]
        .tolist()
    ]

    b2 = [
        float(
            clf.intercepts_[1][0]
        )
    ]

    output = {

        "featureNames":
            FEATURE_NAMES,

        "mean":
            scaler.mean_.tolist(),

        "std":
            scaler.scale_.tolist(),

        "W1":
            W1,

        "b1":
            b1,

        "W2":
            W2,

        "b2":
            b2,

        "hiddenUnitNames": [
            f"h{i}"
            for i in range(
                len(b1)
            )
        ],

        "version":
            "trained-v2",
    }

    output_path = Path(
        args.out
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        f"Wrote {output_path}"
    )

    print(
        f"Input dimensions: "
        f"{len(FEATURE_NAMES)}"
    )

    print(
        f"Hidden units: "
        f"{len(b1)}"
    )


if __name__ == "__main__":
    main()