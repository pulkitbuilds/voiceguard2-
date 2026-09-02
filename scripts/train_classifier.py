# ============================================================
# VoiceGuard - Production-Oriented Audio Classifier
#
# Supports:
#   - WAV
#   - MP3
#   - 40,000 - 100,000+ files
#   - Streaming audio processing
#   - Disk-backed feature cache using NumPy memmap
#   - Low RAM usage
#   - Correct label alignment
#   - Stratified train/validation split
#   - Class-weighted BCE
#   - Early stopping
#   - Checkpointing
#   - Final JSON model export
#
# Expected dataset:
#
# data/
# ├── bonafide/
# │   ├── *.wav
# │   └── *.mp3
# │
# └── spoof/
#     ├── *.wav
#     └── *.mp3
#
# ============================================================

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("ERROR: librosa is not installed.")
    print("Run:")
    print("    pip install librosa soundfile")
    sys.exit(1)

from extract_features import (
    extract_features,
    FEATURE_NAMES,
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

TARGET_SAMPLE_RATE = 16000

HIDDEN_UNITS = 16

LEARNING_RATE = 0.001

EPOCHS = 50

BATCH_SIZE = 64

VALIDATION_SIZE = 0.20

L2 = 1e-4

EARLY_STOPPING_PATIENCE = 7

MIN_DELTA = 1e-4

CACHE_VERSION = "voiceguard-cache-v1"

MODEL_VERSION = "trained-numpy-v5-production"

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
}

PRINT_EVERY_FILES = 500

CHECKPOINT_EVERY = 1


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):
    """
    Load WAV or MP3.

    Output:
        sample_rate = 16000
        audio = mono float32
    """

    path = str(path)

    try:
        audio, sample_rate = librosa.load(
            path,
            sr=None,
            mono=True,
            dtype=np.float32,
        )

    except Exception as e:
        raise RuntimeError(
            f"Audio decode failed: {e}"
        )

    if audio is None or len(audio) == 0:
        raise ValueError(
            "Audio is empty."
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sample_rate = int(sample_rate)

    # --------------------------------------------------------
    # Resample everything to 16 kHz
    # --------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        try:

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            )

        except Exception as e:

            raise RuntimeError(
                f"Resampling failed: {e}"
            )

        sample_rate = TARGET_SAMPLE_RATE

    return (
        TARGET_SAMPLE_RATE,
        np.asarray(
            audio,
            dtype=np.float32,
        ),
    )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_dataset(data_dir):

    data_dir = Path(
        data_dir
    )

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{data_dir}"
        )

    classes = [
        ("bonafide", 0),
        ("spoof", 1),
    ]

    paths = []
    labels = []

    format_counts = {
        "bonafide": {
            ".wav": 0,
            ".mp3": 0,
        },
        "spoof": {
            ".wav": 0,
            ".mp3": 0,
        },
    }

    print()
    print("=" * 70)
    print("DISCOVERING DATASET")
    print("=" * 70)

    for folder, label in classes:

        class_dir = (
            data_dir / folder
        )

        if not class_dir.exists():

            raise FileNotFoundError(
                f"Missing dataset folder: "
                f"{class_dir}"
            )

        class_files = []

        for path in class_dir.rglob("*"):

            if not path.is_file():
                continue

            extension = (
                path.suffix.lower()
            )

            if extension in SUPPORTED_EXTENSIONS:

                class_files.append(
                    path
                )

                format_counts[
                    folder
                ][extension] += 1

        class_files.sort()

        print()
        print(
            f"{folder.upper()}"
        )

        print(
            f"  Total : {len(class_files)}"
        )

        print(
            f"  WAV   : "
            f"{format_counts[folder]['.wav']}"
        )

        print(
            f"  MP3   : "
            f"{format_counts[folder]['.mp3']}"
        )

        for path in class_files:

            paths.append(
                str(path)
            )

            labels.append(
                label
            )

    if len(paths) == 0:

        raise RuntimeError(
            "No WAV or MP3 files found."
        )

    paths = np.asarray(
        paths,
        dtype=object,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    print()
    print("-" * 70)

    print(
        f"TOTAL FILES: {len(paths):,}"
    )

    print(
        f"BONAFIDE:   "
        f"{np.sum(labels == 0):,}"
    )

    print(
        f"SPOOF:      "
        f"{np.sum(labels == 1):,}"
    )

    print()
    print("Format distribution:")

    print(
        f"  Bonafide WAV : "
        f"{format_counts['bonafide']['.wav']:,}"
    )

    print(
        f"  Bonafide MP3 : "
        f"{format_counts['bonafide']['.mp3']:,}"
    )

    print(
        f"  Spoof WAV    : "
        f"{format_counts['spoof']['.wav']:,}"
    )

    print(
        f"  Spoof MP3    : "
        f"{format_counts['spoof']['.mp3']:,}"
    )

    print("-" * 70)

    return paths, labels


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(
    paths,
    labels,
    validation_size=VALIDATION_SIZE,
    seed=RANDOM_SEED,
):

    rng = np.random.default_rng(
        seed
    )

    train_indices = []
    validation_indices = []

    for label in [0, 1]:

        class_indices = np.where(
            labels == label
        )[0]

        if len(class_indices) < 2:

            raise RuntimeError(
                f"Not enough samples for class "
                f"{label}."
            )

        rng.shuffle(
            class_indices
        )

        n_validation = max(
            1,
            int(
                len(class_indices)
                * validation_size
            ),
        )

        validation_indices.extend(
            class_indices[
                :n_validation
            ]
        )

        train_indices.extend(
            class_indices[
                n_validation:
            ]
        )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64,
    )

    validation_indices = np.asarray(
        validation_indices,
        dtype=np.int64,
    )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        validation_indices
    )

    return (
        train_indices,
        validation_indices,
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_feature_vector(path):

    sample_rate, audio = (
        load_audio(path)
    )

    result = extract_features(
        audio,
        sample_rate,
    )

    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "extract_features() "
            "did not return a dictionary."
        )

    if "vector" not in result:

        raise ValueError(
            "Feature result does not contain "
            "'vector'."
        )

    vector = np.asarray(
        result["vector"],
        dtype=np.float32,
    )

    vector = vector.reshape(
        -1
    )

    expected = len(
        FEATURE_NAMES
    )

    if len(vector) != expected:

        raise ValueError(
            f"Expected {expected} features, "
            f"got {len(vector)}."
        )

    if not np.all(
        np.isfinite(vector)
    ):

        raise ValueError(
            "Feature vector contains "
            "NaN or Inf."
        )

    return vector


# ============================================================
# CACHE PATHS
# ============================================================

def cache_paths(cache_dir):

    cache_dir = Path(
        cache_dir
    )

    return {
        "features":
            cache_dir / "features.dat",

        "labels":
            cache_dir / "labels.npy",

        "paths":
            cache_dir / "paths.json",

        "valid":
            cache_dir / "valid.npy",

        "metadata":
            cache_dir / "metadata.json",
    }


# ============================================================
# CREATE FEATURE CACHE
# ============================================================

def build_feature_cache(
    paths,
    labels,
    cache_dir,
    feature_dim,
):

    cache_dir = Path(
        cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = cache_paths(
        cache_dir
    )

    n_files = len(paths)

    # --------------------------------------------------------
    # Check existing cache
    # --------------------------------------------------------

    if (
        files["metadata"].exists()
        and files["features"].exists()
        and files["labels"].exists()
        and files["valid"].exists()
        and files["paths"].exists()
    ):

        try:

            with open(
                files["metadata"],
                "r",
                encoding="utf-8",
            ) as f:

                metadata = json.load(f)

            if (
                metadata.get(
                    "cache_version"
                )
                == CACHE_VERSION
                and metadata.get(
                    "num_files"
                )
                == n_files
                and metadata.get(
                    "feature_dim"
                )
                == feature_dim
                and metadata.get(
                    "target_sample_rate"
                )
                == TARGET_SAMPLE_RATE
            ):

                cached_paths = np.asarray(
                    json.loads(
                        files["paths"].read_text(
                            encoding="utf-8"
                        )
                    ),
                    dtype=object,
                )

                cached_labels = np.load(
                    files["labels"]
                )

                if (
                    len(cached_paths)
                    == len(paths)
                    and np.array_equal(
                        cached_labels,
                        labels,
                    )
                    and np.array_equal(
                        cached_paths,
                        paths,
                    )
                ):

                    print()
                    print("=" * 70)
                    print("FEATURE CACHE")
                    print("=" * 70)

                    print(
                        "Existing compatible "
                        "cache found."
                    )

                    print(
                        f"Cache: {cache_dir}"
                    )

                    return files

        except Exception as e:

            print(
                "Existing cache could not "
                f"be reused: {e}"
            )

    # --------------------------------------------------------
    # Create new cache
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING FEATURE CACHE")
    print("=" * 70)

    print(
        f"Files:        {n_files:,}"
    )

    print(
        f"Feature dim:  {feature_dim}"
    )

    print(
        f"Cache:        {cache_dir}"
    )

    print()
    print(
        "Audio will be decoded only once."
    )

    print(
        "Features are stored on disk, "
        "not kept in RAM."
    )

    # --------------------------------------------------------
    # Disk-backed feature matrix
    #
    # We allocate one row per discovered file.
    # Failed files are marked invalid.
    # --------------------------------------------------------

    feature_memmap = np.memmap(
        files["features"],
        dtype=np.float32,
        mode="w+",
        shape=(
            n_files,
            feature_dim,
        ),
    )

    valid = np.zeros(
        n_files,
        dtype=np.bool_,
    )

    skipped = 0

    for index in range(
        n_files
    ):

        path = paths[index]

        try:

            vector = (
                extract_feature_vector(
                    path
                )
            )

            feature_memmap[
                index
            ] = vector

            valid[index] = True

        except Exception as e:

            skipped += 1

            print()
            print(
                f"[SKIP {index + 1:,}/{n_files:,}]"
            )

            print(
                f"  {path}"
            )

            print(
                f"  Reason: {e}"
            )

            # Keep zero row.
            # `valid=False` means it will NEVER
            # be used for training/evaluation.

        if (
            (index + 1)
            % PRINT_EVERY_FILES
            == 0
            or index == n_files - 1
        ):

            valid_count = int(
                np.sum(valid)
            )

            print(
                f"Cache: "
                f"{index + 1:,}/{n_files:,} | "
                f"Valid: {valid_count:,} | "
                f"Skipped: {skipped:,}"
            )

            # Flush periodically so a crash does
            # not leave a completely unflushed cache.
            feature_memmap.flush()

    feature_memmap.flush()

    del feature_memmap

    gc.collect()

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    np.save(
        files["labels"],
        labels,
    )

    np.save(
        files["valid"],
        valid,
    )

    with open(
        files["paths"],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            paths.tolist(),
            f,
        )

    metadata = {
        "cache_version":
            CACHE_VERSION,

        "num_files":
            int(n_files),

        "feature_dim":
            int(feature_dim),

        "target_sample_rate":
            int(TARGET_SAMPLE_RATE),

        "valid_files":
            int(np.sum(valid)),

        "skipped_files":
            int(skipped),
    }

    with open(
        files["metadata"],
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        f"Valid files: "
        f"{np.sum(valid):,}"
    )

    print(
        f"Skipped files: "
        f"{skipped:,}"
    )

    return files


# ============================================================
# OPEN FEATURE CACHE
# ============================================================

def open_feature_cache(
    cache_dir,
    num_files,
    feature_dim,
):

    files = cache_paths(
        Path(cache_dir)
    )

    features = np.memmap(
        files["features"],
        dtype=np.float32,
        mode="r",
        shape=(
            num_files,
            feature_dim,
        ),
    )

    labels = np.load(
        files["labels"],
        mmap_mode="r",
    )

    valid = np.load(
        files["valid"],
        mmap_mode="r",
    )

    return (
        features,
        labels,
        valid,
    )


# ============================================================
# STREAMING SCALER FROM MEMMAP
# ============================================================

def fit_scaler(
    features,
    train_indices,
    valid_mask,
    batch_size,
):

    feature_dim = (
        features.shape[1]
    )

    count = 0

    mean = np.zeros(
        feature_dim,
        dtype=np.float64,
    )

    M2 = np.zeros(
        feature_dim,
        dtype=np.float64,
    )

    print()
    print("=" * 70)
    print("FITTING TRAINING SCALER")
    print("=" * 70)

    # We process indices in batches.
    # Only a small feature matrix is loaded at a time.

    valid_train_indices = (
        train_indices[
            np.asarray(
                valid_mask[
                    train_indices
                ],
                dtype=bool,
            )
        ]
    )

    total = len(
        valid_train_indices
    )

    if total == 0:

        raise RuntimeError(
            "No valid training features."
        )

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        batch_indices = (
            valid_train_indices[
                start:end
            ]
        )

        X = np.asarray(
            features[
                batch_indices
            ],
            dtype=np.float32,
        )

        X64 = X.astype(
            np.float64
        )

        batch_count = (
            len(X64)
        )

        batch_mean = np.mean(
            X64,
            axis=0,
        )

        centered = (
            X64 - batch_mean
        )

        batch_M2 = np.sum(
            centered * centered,
            axis=0,
        )

        if count == 0:

            count = batch_count
            mean = batch_mean
            M2 = batch_M2

        else:

            total_count = (
                count + batch_count
            )

            delta = (
                batch_mean - mean
            )

            mean = (
                mean
                + delta
                * (
                    batch_count
                    / total_count
                )
            )

            M2 = (
                M2
                + batch_M2
                + (
                    delta
                    * delta
                    * count
                    * batch_count
                    / total_count
                )
            )

            count = total_count

        if (
            end % (
                batch_size * 100
            ) == 0
            or end == total
        ):

            print(
                f"Scaler: "
                f"{end:,}/{total:,}"
            )

        del X
        del X64

    variance = (
        M2
        / max(
            count - 1,
            1,
        )
    )

    std = np.sqrt(
        np.maximum(
            variance,
            1e-8,
        )
    )

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    return (
        mean,
        std,
        total,
    )


# ============================================================
# STANDARDIZE
# ============================================================

def standardize(
    X,
    mean,
    std,
):

    return (
        (
            X - mean
        )
        / std
    ).astype(
        np.float32,
        copy=False,
    )


# ============================================================
# ACTIVATIONS
# ============================================================

def relu(x):

    return np.maximum(
        x,
        0.0,
    )


def sigmoid(x):

    x = np.clip(
        x,
        -50.0,
        50.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    ).astype(
        np.float32
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_model(
    input_dim,
    hidden_dim,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    W1 = rng.normal(
        0.0,
        np.sqrt(
            2.0 / input_dim
        ),
        size=(
            hidden_dim,
            input_dim,
        ),
    ).astype(
        np.float32
    )

    b1 = np.zeros(
        hidden_dim,
        dtype=np.float32,
    )

    W2 = rng.normal(
        0.0,
        np.sqrt(
            2.0 / hidden_dim
        ),
        size=(
            1,
            hidden_dim,
        ),
    ).astype(
        np.float32
    )

    b2 = np.zeros(
        1,
        dtype=np.float32,
    )

    return (
        W1,
        b1,
        W2,
        b2,
    )


# ============================================================
# FORWARD
# ============================================================

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

    a1 = relu(
        z1
    )

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


# ============================================================
# TRAIN ONE BATCH
# ============================================================

def train_batch(
    X,
    y,
    W1,
    b1,
    W2,
    b2,
    learning_rate,
    l2,
    positive_weight,
    negative_weight,
):

    batch_n = len(X)

    (
        z1,
        a1,
        probabilities,
    ) = forward(
        X,
        W1,
        b1,
        W2,
        b2,
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    sample_weights = np.where(
        y == 1,
        positive_weight,
        negative_weight,
    ).astype(
        np.float32
    )

    weight_sum = np.sum(
        sample_weights
    )

    p = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    per_sample_loss = -(
        y * np.log(p)
        + (
            1.0 - y
        )
        * np.log(
            1.0 - p
        )
    )

    loss = (
        np.sum(
            per_sample_loss
            * sample_weights
        )
        / weight_sum
    )

    # L2
    loss += (
        l2
        * (
            np.sum(W1 * W1)
            + np.sum(W2 * W2)
        )
        / 2.0
    )

    # --------------------------------------------------------
    # Backprop
    # --------------------------------------------------------

    dz2 = (
        (
            probabilities
            - y
        )
        * sample_weights
        / weight_sum
    ).reshape(
        -1,
        1,
    )

    dW2 = (
        dz2.T @ a1
    )

    db2 = np.sum(
        dz2,
        axis=0,
    )

    da1 = (
        dz2 @ W2
    )

    dz1 = (
        da1
        * (
            z1 > 0
        ).astype(
            np.float32
        )
    )

    dW1 = (
        dz1.T @ X
    )

    db1 = np.sum(
        dz1,
        axis=0,
    )

    # L2
    dW1 += (
        l2 * W1
    )

    dW2 += (
        l2 * W2
    )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

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

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    correct = int(
        np.sum(
            predictions == y
        )
    )

    return (
        float(loss),
        correct,
        batch_n,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    labels,
    train_indices,
    valid_mask,
):

    usable = train_indices[
        np.asarray(
            valid_mask[
                train_indices
            ],
            dtype=bool,
        )
    ]

    y = np.asarray(
        labels[usable],
        dtype=np.int64,
    )

    negatives = np.sum(
        y == 0
    )

    positives = np.sum(
        y == 1
    )

    if (
        negatives == 0
        or positives == 0
    ):

        raise RuntimeError(
            "Training set must contain "
            "both bonafide and spoof samples."
        )

    total = (
        negatives
        + positives
    )

    # Balanced class weights
    negative_weight = (
        total
        / (
            2.0
            * negatives
        )
    )

    positive_weight = (
        total
        / (
            2.0
            * positives
        )
    )

    print()
    print(
        "Class weights:"
    )

    print(
        f"  Bonafide: "
        f"{negative_weight:.4f}"
    )

    print(
        f"  Spoof:    "
        f"{positive_weight:.4f}"
    )

    return (
        float(positive_weight),
        float(negative_weight),
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    features,
    labels,
    valid_mask,
    train_indices,
    validation_indices,
    mean,
    std,
    hidden_units,
    learning_rate,
    epochs,
    batch_size,
    l2,
    seed,
    checkpoint_path,
):

    (
        W1,
        b1,
        W2,
        b2,
    ) = initialize_model(
        input_dim=len(
            FEATURE_NAMES
        ),
        hidden_dim=hidden_units,
        seed=seed,
    )

    (
        positive_weight,
        negative_weight,
    ) = calculate_class_weights(
        labels,
        train_indices,
        valid_mask,
    )

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Only valid indices participate.
    # --------------------------------------------------------

    train_indices = train_indices[
        np.asarray(
            valid_mask[
                train_indices
            ],
            dtype=bool,
        )
    ]

    validation_indices = (
        validation_indices[
            np.asarray(
                valid_mask[
                    validation_indices
                ],
                dtype=bool,
            )
        ]
    )

    print()
    print(
        f"Valid training files: "
        f"{len(train_indices):,}"
    )

    print(
        f"Valid validation files: "
        f"{len(validation_indices):,}"
    )

    best_validation_loss = (
        float("inf")
    )

    best_weights = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        shuffled = (
            train_indices.copy()
        )

        rng.shuffle(
            shuffled
        )

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        # ----------------------------------------------------
        # TRAINING BATCHES
        # ----------------------------------------------------

        for start in range(
            0,
            len(shuffled),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(shuffled),
            )

            batch_indices = (
                shuffled[
                    start:end
                ]
            )

            X = np.asarray(
                features[
                    batch_indices
                ],
                dtype=np.float32,
            )

            y = np.asarray(
                labels[
                    batch_indices
                ],
                dtype=np.float32,
            )

            X = standardize(
                X,
                mean,
                std,
            )

            (
                batch_loss,
                batch_correct,
                batch_count,
            ) = train_batch(
                X,
                y,
                W1,
                b1,
                W2,
                b2,
                learning_rate,
                l2,
                positive_weight,
                negative_weight,
            )

            epoch_loss += (
                batch_loss
                * batch_count
            )

            epoch_correct += (
                batch_correct
            )

            epoch_samples += (
                batch_count
            )

            del X
            del y

        epoch_loss /= max(
            epoch_samples,
            1,
        )

        train_accuracy = (
            epoch_correct
            / max(
                epoch_samples,
                1,
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            validation_loss,
            validation_accuracy,
            validation_metrics,
        ) = evaluate_memmap(
            features,
            labels,
            validation_indices,
            mean,
            std,
            W1,
            b1,
            W2,
            b2,
            batch_size,
            positive_weight,
            negative_weight,
        )

        print()
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Loss {epoch_loss:.6f} | "
            f"Train Acc {train_accuracy:.4f} | "
            f"Val Loss {validation_loss:.6f} | "
            f"Val Acc {validation_accuracy:.4f} | "
            f"Val F1 {validation_metrics['f1']:.4f} | "
            f"Val AUC {validation_metrics['auc']:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            validation_loss
            < (
                best_validation_loss
                - MIN_DELTA
            )
        ):

            best_validation_loss = (
                validation_loss
            )

            best_weights = (
                W1.copy(),
                b1.copy(),
                W2.copy(),
                b2.copy(),
            )

            epochs_without_improvement = 0

            print(
                "  ✓ New best validation model"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            checkpoint_path
            and (
                epoch
                % CHECKPOINT_EVERY
                == 0
            )
        ):

            save_checkpoint(
                checkpoint_path,
                W1,
                b1,
                W2,
                b2,
                mean,
                std,
                epoch,
                best_validation_loss,
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_weights is not None:

        (
            W1,
            b1,
            W2,
            b2,
        ) = best_weights

        print()
        print(
            "✓ Restored best validation model."
        )

    return (
        W1,
        b1,
        W2,
        b2,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_memmap(
    features,
    labels,
    indices,
    mean,
    std,
    W1,
    b1,
    W2,
    b2,
    batch_size,
    positive_weight,
    negative_weight,
):

    total_loss = 0.0
    total_count = 0

    all_true = []
    all_probabilities = []

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(indices),
        )

        batch_indices = (
            indices[
                start:end
            ]
        )

        X = np.asarray(
            features[
                batch_indices
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            labels[
                batch_indices
            ],
            dtype=np.float32,
        )

        X = standardize(
            X,
            mean,
            std,
        )

        (
            _,
            _,
            probabilities,
        ) = forward(
            X,
            W1,
            b1,
            W2,
            b2,
        )

        sample_weights = np.where(
            y == 1,
            positive_weight,
            negative_weight,
        ).astype(
            np.float32
        )

        weight_sum = np.sum(
            sample_weights
        )

        p = np.clip(
            probabilities,
            1e-7,
            1.0 - 1e-7,
        )

        loss = -np.sum(
            (
                y * np.log(p)
                + (
                    1.0 - y
                )
                * np.log(
                    1.0 - p
                )
            )
            * sample_weights
        ) / weight_sum

        total_loss += (
            float(loss)
            * len(y)
        )

        total_count += len(y)

        all_true.extend(
            y.astype(
                np.int64
            ).tolist()
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

        del X
        del y

    if total_count == 0:

        return (
            0.0,
            0.0,
            {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "auc": 0.0,
            },
        )

    y_true = np.asarray(
        all_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    accuracy = float(
        np.mean(
            predictions
            == y_true
        )
    )

    (
        precision,
        recall,
        f1,
    ) = calculate_precision_recall_f1(
        y_true,
        predictions,
    )

    auc = roc_auc_manual(
        y_true,
        probabilities,
    )

    return (
        float(
            total_loss
            / total_count
        ),
        accuracy,
        {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "auc":
                auc,
        },
    )


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def calculate_precision_recall_f1(
    y_true,
    predictions,
):

    tp = np.sum(
        (
            y_true == 1
        )
        & (
            predictions == 1
        )
    )

    fp = np.sum(
        (
            y_true == 0
        )
        & (
            predictions == 1
        )
    )

    fn = np.sum(
        (
            y_true == 1
        )
        & (
            predictions == 0
        )
    )

    precision = (
        tp / (tp + fp)
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (
            tp + fn
        ) > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    return (
        float(precision),
        float(recall),
        float(f1),
    )


# ============================================================
# ROC AUC
# ============================================================

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

        return 0.0

    combined = np.concatenate(
        [
            positives,
            negatives,
        ]
    )

    order = np.argsort(
        combined,
        kind="mergesort",
    )

    ranks = np.empty(
        len(combined),
        dtype=np.float64,
    )

    ranks[order] = np.arange(
        1,
        len(combined) + 1,
        dtype=np.float64,
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
            / 2.0
        )
    ) / (
        len(positives)
        * len(negatives)
    )

    return float(auc)


# ============================================================
# CONFUSION MATRIX
# ============================================================

def confusion_matrix(
    y_true,
    predictions,
):

    tn = int(
        np.sum(
            (
                y_true == 0
            )
            & (
                predictions == 0
            )
        )
    )

    fp = int(
        np.sum(
            (
                y_true == 0
            )
            & (
                predictions == 1
            )
        )
    )

    fn = int(
        np.sum(
            (
                y_true == 1
            )
            & (
                predictions == 0
            )
        )
    )

    tp = int(
        np.sum(
            (
                y_true == 1
            )
            & (
                predictions == 1
            )
        )
    )

    return np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=np.int64,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    W1,
    b1,
    W2,
    b2,
    mean,
    std,
    epoch,
    validation_loss,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch":
            int(epoch),

        "validationLoss":
            float(validation_loss),

        "W1":
            W1.tolist(),

        "b1":
            b1.tolist(),

        "W2":
            W2.tolist(),

        "b2":
            b2.tolist(),

        "mean":
            mean.tolist(),

        "std":
            std.tolist(),
    }

    temp_path = (
        path.with_suffix(
            ".tmp"
        )
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            checkpoint,
            f,
        )

    temp_path.replace(
        path
    )


# ============================================================
# EXPORT FINAL MODEL
# ============================================================

def export_model(
    output_path,
    W1,
    b1,
    W2,
    b2,
    mean,
    std,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            MODEL_VERSION,

        "sampleRate":
            TARGET_SAMPLE_RATE,

        "audioFormats":
            [
                "wav",
                "mp3",
            ],
    }

    temp_path = (
        output_path.with_suffix(
            ".tmp"
        )
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    temp_path.replace(
        output_path
    )

    print()
    print(
        f"✓ Model written to:"
    )

    print(
        f"  {output_path}"
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def final_evaluation(
    features,
    labels,
    valid_mask,
    validation_indices,
    mean,
    std,
    W1,
    b1,
    W2,
    b2,
    batch_size,
    positive_weight,
    negative_weight,
):

    usable_indices = (
        validation_indices[
            np.asarray(
                valid_mask[
                    validation_indices
                ],
                dtype=bool,
            )
        ]
    )

    y_true = []
    probabilities = []

    for start in range(
        0,
        len(usable_indices),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(usable_indices),
        )

        batch_indices = (
            usable_indices[
                start:end
            ]
        )

        X = np.asarray(
            features[
                batch_indices
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            labels[
                batch_indices
            ],
            dtype=np.int64,
        )

        X = standardize(
            X,
            mean,
            std,
        )

        (
            _,
            _,
            prob,
        ) = forward(
            X,
            W1,
            b1,
            W2,
            b2,
        )

        y_true.extend(
            y.tolist()
        )

        probabilities.extend(
            prob.tolist()
        )

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    accuracy = float(
        np.mean(
            predictions
            == y_true
        )
    )

    (
        precision,
        recall,
        f1,
    ) = calculate_precision_recall_f1(
        y_true,
        predictions,
    )

    auc = roc_auc_manual(
        y_true,
        probabilities,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Samples  : {len(y_true):,}"
    )

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
    print(
        "Confusion matrix:"
    )

    print(
        "              Predicted"
    )

    print(
        "              Bona  Spoof"
    )

    print(
        f"Actual Bona   "
        f"{cm[0, 0]:6d} "
        f"{cm[0, 1]:6d}"
    )

    print(
        f"Actual Spoof  "
        f"{cm[1, 0]:6d} "
        f"{cm[1, 1]:6d}"
    )

    return {
        "accuracy":
            accuracy,

        "auc":
            auc,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "confusionMatrix":
            cm.tolist(),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "VoiceGuard production-oriented "
            "WAV/MP3 streaming classifier."
        )
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help=(
            "Dataset containing "
            "bonafide/ and spoof/"
        ),
    )

    parser.add_argument(
        "--out",
        default="../lib/model_weights.json",
    )

    parser.add_argument(
        "--cache_dir",
        default="../cache/features",
    )

    parser.add_argument(
        "--checkpoint",
        default="../cache/checkpoint.json",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--hidden_units",
        type=int,
        default=HIDDEN_UNITS,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--validation_size",
        type=float,
        default=VALIDATION_SIZE,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if args.batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0"
        )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be > 0"
        )

    if not (
        0.0
        < args.validation_size
        < 1.0
    ):

        raise ValueError(
            "validation_size must be "
            "between 0 and 1"
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VOICEGUARD AUDIO CLASSIFIER")
    print("=" * 70)

    print(
        "Production streaming configuration"
    )

    print(
        f"Sample rate : "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print(
        f"Batch size  : "
        f"{args.batch_size}"
    )

    print(
        f"Epochs      : "
        f"{args.epochs}"
    )

    print(
        f"Features    : "
        f"{len(FEATURE_NAMES)}"
    )

    print(
        f"Hidden units: "
        f"{args.hidden_units}"
    )

    # --------------------------------------------------------
    # Discover
    # --------------------------------------------------------

    paths, labels = (
        discover_dataset(
            args.data_dir
        )
    )

    # --------------------------------------------------------
    # Split paths/indices BEFORE feature extraction
    #
    # This means the validation set never affects the
    # training scaler.
    # --------------------------------------------------------

    (
        train_indices,
        validation_indices,
    ) = stratified_split(
        paths,
        labels,
        validation_size=args.validation_size,
        seed=RANDOM_SEED,
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Training files:   "
        f"{len(train_indices):,}"
    )

    print(
        f"Validation files: "
        f"{len(validation_indices):,}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Build / reuse disk feature cache
    # --------------------------------------------------------

    cache_files = (
        build_feature_cache(
            paths,
            labels,
            args.cache_dir,
            len(FEATURE_NAMES),
        )
    )

    # --------------------------------------------------------
    # Open memory-mapped features
    # --------------------------------------------------------

    (
        features,
        cached_labels,
        valid_mask,
    ) = open_feature_cache(
        args.cache_dir,
        len(paths),
        len(FEATURE_NAMES),
    )

    # Safety check
    if not np.array_equal(
        np.asarray(cached_labels),
        labels,
    ):

        raise RuntimeError(
            "Cached labels do not match "
            "current dataset."
        )

    # --------------------------------------------------------
    # Fit scaler ONLY on training data
    # --------------------------------------------------------

    (
        mean,
        std,
        valid_train_count,
    ) = fit_scaler(
        features,
        train_indices,
        valid_mask,
        args.batch_size,
    )

    print()
    print(
        f"Scaler training samples: "
        f"{valid_train_count:,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    (
        W1,
        b1,
        W2,
        b2,
    ) = train_model(
        features=features,
        labels=cached_labels,
        valid_mask=valid_mask,
        train_indices=train_indices,
        validation_indices=validation_indices,
        mean=mean,
        std=std,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2=L2,
        seed=RANDOM_SEED,
        checkpoint_path=args.checkpoint,
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_metrics = final_evaluation(
        features,
        cached_labels,
        valid_mask,
        validation_indices,
        mean,
        std,
        W1,
        b1,
        W2,
        b2,
        args.batch_size,
        1.0,
        1.0,
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    export_model(
        args.out,
        W1,
        b1,
        W2,
        b2,
        mean,
        std,
    )

    # --------------------------------------------------------
    # Save training metadata
    # --------------------------------------------------------

    metadata_path = Path(
        args.out
    ).with_name(
        "training_metadata.json"
    )

    metadata = {
        "datasetFiles":
            int(len(paths)),

        "validFiles":
            int(np.sum(valid_mask)),

        "skippedFiles":
            int(
                len(paths)
                - np.sum(valid_mask)
            ),

        "trainingFiles":
            int(len(train_indices)),

        "validationFiles":
            int(
                len(validation_indices)
            ),

        "validTrainingFiles":
            int(
                np.sum(
                    valid_mask[
                        train_indices
                    ]
                )
            ),

        "validValidationFiles":
            int(
                np.sum(
                    valid_mask[
                        validation_indices
                    ]
                )
            ),

        "featureDimension":
            int(len(FEATURE_NAMES)),

        "sampleRate":
            TARGET_SAMPLE_RATE,

        "batchSize":
            args.batch_size,

        "epochs":
            args.epochs,

        "learningRate":
            args.learning_rate,

        "hiddenUnits":
            args.hidden_units,

        "validationSize":
            args.validation_size,

        "metrics":
            final_metrics,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model:    {args.out}"
    )

    print(
        f"Cache:    {args.cache_dir}"
    )

    print(
        f"Metadata: {metadata_path}"
    )

    print()
    print(
        "The feature cache can be reused on "
        "future training runs."
    )


if __name__ == "__main__":
    main()