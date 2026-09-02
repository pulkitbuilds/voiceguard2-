# scripts/train_classifier.py

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from extract_features import (
    extract_features,
    FEATURE_NAMES,
)


# =========================================================
# CONFIG
# =========================================================

HIDDEN_UNITS = 12
LEARNING_RATE = 0.001
EPOCHS = 300
BATCH_SIZE = 64
L2 = 1e-4
RANDOM_SEED = 42


# =========================================================
# WAV LOADING
# =========================================================

def load_wav(path):

    sample_rate, audio = wavfile.read(path)

    audio = np.asarray(audio)

    # Stereo -> mono
    if audio.ndim == 2:
        audio = np.mean(
            audio.astype(np.float64),
            axis=1,
        )
    else:
        audio = audio.astype(np.float64)

    # Normalize PCM
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)

        max_abs = np.maximum(
            abs(info.min),
            abs(info.max),
        )

        audio = audio / max_abs

    audio = np.asarray(
        audio,
        dtype=np.float64,
    )

    return sample_rate, audio


# =========================================================
# DATASET LOADING
# =========================================================

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
            f"{folder}: {len(wav_files)} files"
        )

        for index, wav_path in enumerate(wav_files):

            try:

                sample_rate, audio = load_wav(
                    wav_path
                )

                result = extract_features(
                    audio,
                    sample_rate,
                )

                vector = np.asarray(
                    result["vector"],
                    dtype=np.float64,
                )

                if len(vector) != len(FEATURE_NAMES):
                    raise ValueError(
                        f"Expected "
                        f"{len(FEATURE_NAMES)} "
                        f"features, got "
                        f"{len(vector)}"
                    )

                X.append(vector)
                y.append(label)
                paths.append(str(wav_path))

                # Progress every 100 files
                if (index + 1) % 100 == 0:
                    print(
                        f"  {index + 1}/"
                        f"{len(wav_files)}"
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


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def train_test_split_manual(
    X,
    y,
    test_size=0.20,
    seed=42,
):

    rng = np.random.default_rng(seed)

    train_indices = []
    test_indices = []

    for label in [0, 1]:

        indices = np.where(y == label)[0]

        rng.shuffle(indices)

        n_test = max(
            1,
            int(len(indices) * test_size)
        )

        test_indices.extend(
            indices[:n_test]
        )

        train_indices.extend(
            indices[n_test:]
        )

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


# =========================================================
# STANDARDIZATION
# =========================================================

def fit_scaler(X):

    mean = np.mean(
        X,
        axis=0,
    )

    std = np.std(
        X,
        axis=0,
    )

    # Prevent division by zero
    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return mean, std


def standardize(
    X,
    mean,
    std,
):

    return (
        X - mean
    ) / std


# =========================================================
# ACTIVATION
# =========================================================

def relu(x):

    return np.maximum(
        0.0,
        x,
    )


def relu_derivative(x):

    return (
        x > 0
    ).astype(np.float64)


def sigmoid(x):

    # Stable sigmoid
    x = np.clip(
        x,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


# =========================================================
# MODEL INITIALIZATION
# =========================================================

def initialize_model(
    input_dim,
    hidden_dim,
    seed=42,
):

    rng = np.random.default_rng(seed)

    # He initialization
    W1 = (
        rng.normal(
            0.0,
            np.sqrt(
                2.0 / input_dim
            ),
            size=(
                hidden_dim,
                input_dim,
            ),
        )
    )

    b1 = np.zeros(
        hidden_dim,
        dtype=np.float64,
    )

    W2 = (
        rng.normal(
            0.0,
            np.sqrt(
                2.0 / hidden_dim
            ),
            size=(
                1,
                hidden_dim,
            ),
        )
    )

    b2 = np.zeros(
        1,
        dtype=np.float64,
    )

    return W1, b1, W2, b2


# =========================================================
# FORWARD PASS
# =========================================================

def forward(
    X,
    W1,
    b1,
    W2,
    b2,
):

    z1 = (
        X @ W1.T
    ) + b1

    a1 = relu(z1)

    z2 = (
        a1 @ W2.T
    ) + b2

    probabilities = sigmoid(
        z2
    ).reshape(-1)

    return (
        z1,
        a1,
        probabilities,
    )


# =========================================================
# TRAINING
# =========================================================

def train_model(
    X,
    y,
    input_dim,
    hidden_dim=12,
    learning_rate=0.001,
    epochs=300,
    batch_size=64,
    l2=1e-4,
    seed=42,
):

    rng = np.random.default_rng(seed)

    W1, b1, W2, b2 = initialize_model(
        input_dim,
        hidden_dim,
        seed,
    )

    n_samples = len(X)

    for epoch in range(epochs):

        indices = np.arange(
            n_samples
        )

        rng.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0.0

        for start in range(
            0,
            n_samples,
            batch_size,
        ):

            end = min(
                start + batch_size,
                n_samples,
            )

            xb = X_shuffled[
                start:end
            ]

            yb = y_shuffled[
                start:end
            ]

            batch_n = len(xb)

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            z1, a1, probabilities = forward(
                xb,
                W1,
                b1,
                W2,
                b2,
            )

            # -------------------------------------------------
            # Binary cross entropy
            # -------------------------------------------------

            p = np.clip(
                probabilities,
                1e-7,
                1.0 - 1e-7,
            )

            loss = -np.mean(
                yb * np.log(p)
                + (
                    1.0 - yb
                ) * np.log(
                    1.0 - p
                )
            )

            # L2 regularization
            loss += (
                l2
                * (
                    np.sum(W1 * W1)
                    + np.sum(W2 * W2)
                )
                / 2.0
            )

            epoch_loss += (
                loss * batch_n
            )

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------

            dz2 = (
                probabilities - yb
            ).reshape(
                -1,
                1,
            )

            dW2 = (
                dz2.T @ a1
            ) / batch_n

            db2 = np.mean(
                dz2,
                axis=0,
            )

            da1 = (
                dz2 @ W2
            )

            dz1 = (
                da1
                * relu_derivative(z1)
            )

            dW1 = (
                dz1.T @ xb
            ) / batch_n

            db1 = np.mean(
                dz1,
                axis=0,
            )

            # L2
            dW1 += l2 * W1
            dW2 += l2 * W2

            # -------------------------------------------------
            # Gradient update
            # -------------------------------------------------

            W1 -= (
                learning_rate
                * dW1
            )

            b1 -= (
                learning_rate
                * db1
            )

            W2 -= (
                learning_rate
                * dW2
            )

            b2 -= (
                learning_rate
                * db2
            )

        epoch_loss /= n_samples

        # Print progress
        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            _, _, train_prob = forward(
                X,
                W1,
                b1,
                W2,
                b2,
            )

            train_pred = (
                train_prob >= 0.5
            ).astype(int)

            train_acc = np.mean(
                train_pred == y
            )

            print(
                f"Epoch {epoch + 1:3d}/"
                f"{epochs} | "
                f"Loss: {epoch_loss:.6f} | "
                f"Train Acc: "
                f"{train_acc:.4f}"
            )

    return W1, b1, W2, b2


# =========================================================
# METRICS
# =========================================================

def confusion_matrix_manual(
    y_true,
    y_pred,
):

    tn = np.sum(
        (y_true == 0)
        & (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0)
        & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1)
        & (y_pred == 0)
    )

    tp = np.sum(
        (y_true == 1)
        & (y_pred == 1)
    )

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=np.int64,
    )


def binary_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    cm = confusion_matrix_manual(
        y_true,
        predictions,
    )

    tn, fp = cm[0]
    fn, tp = cm[1]

    total = (
        tn + fp + fn + tp
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm,
        predictions,
    )


def roc_auc_manual(
    y_true,
    probabilities,
):

    positives = probabilities[
        y_true == 1
    ]

    negatives = probabilities[
        y_true == 0
    ]

    if (
        len(positives) == 0
        or len(negatives) == 0
    ):
        return 0.5

    # Mann-Whitney formulation
    combined = np.concatenate(
        [
            positives,
            negatives,
        ]
    )

    order = np.argsort(
        combined
    )

    ranks = np.empty_like(
        order,
        dtype=np.float64,
    )

    ranks[order] = np.arange(
        1,
        len(combined) + 1,
    )

    positive_ranks = ranks[
        :len(positives)
    ]

    auc = (
        np.sum(
            positive_ranks
        )
        - (
            len(positives)
            * (
                len(positives)
                + 1
            )
            / 2
        )
    ) / (
        len(positives)
        * len(negatives)
    )

    return float(auc)


# =========================================================
# MAIN
# =========================================================

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
    # Load dataset
    # -----------------------------------------------------

    X, y, paths = load_dataset(
        args.data_dir
    )

    if len(X) < 20:
        raise RuntimeError(
            "Dataset is too small."
        )

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split_manual(
        X,
        y,
        test_size=0.20,
        seed=RANDOM_SEED,
    )

    print()
    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Test samples:     "
        f"{len(X_test)}"
    )

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    mean, std = fit_scaler(
        X_train
    )

    X_train_scaled = standardize(
        X_train,
        mean,
        std,
    )

    X_test_scaled = standardize(
        X_test,
        mean,
        std,
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING NUMPY MLP")
    print("=" * 60)

    W1, b1, W2, b2 = train_model(
        X_train_scaled,
        y_train,
        input_dim=len(FEATURE_NAMES),
        hidden_dim=HIDDEN_UNITS,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        l2=L2,
        seed=RANDOM_SEED,
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    _, _, probabilities = forward(
        X_test_scaled,
        W1,
        b1,
        W2,
        b2,
    )

    (
        accuracy,
        precision,
        recall,
        f1,
        cm,
        predictions,
    ) = binary_metrics(
        y_test,
        probabilities,
    )

    auc = roc_auc_manual(
        y_test,
        probabilities,
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

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

    if accuracy >= 0.999 and auc >= 0.999:

        print("⚠️ WARNING")
        print(
            "Accuracy and AUC are almost perfect."
        )

        print(
            "This may indicate dataset leakage "
            "or source-specific artifacts."
        )

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    output = {

        "featureNames":
            FEATURE_NAMES,

        "mean":
            mean.tolist(),

        "std":
            std.tolist(),

        "W1":
            W1.tolist(),

        "b1":
            b1.tolist(),

        "W2":
            W2.tolist(),

        "b2":
            b2.tolist(),

        "hiddenUnitNames": [
            f"h{i}"
            for i in range(
                len(b1)
            )
        ],

        "version":
            "trained-numpy-v3",
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