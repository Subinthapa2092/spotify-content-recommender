"""
Feature engineering. Logic copied 1:1 from notebook section 4.

Produces:
  - feature_matrix: pandas DataFrame (human-readable, column names kept)
  - X: numpy float32 array (what the recommender actually uses for cosine similarity)
  - scaler: fitted StandardScaler (needed to vectorize any new/unseen song later)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

GENRE_WEIGHT = 1.2
SKEWED_FEATURES = ["speechiness", "instrumentalness", "liveness"]
STANDARDIZE_FEATURES = [
    "danceability", "energy", "acousticness", "valence",
    "loudness", "tempo", "duration_ms", "time_signature",
] + SKEWED_FEATURES
BINARY_FEATURES = ["mode", "explicit"]
CYCLIC_FEATURES = ["key_sin", "key_cos"]


def build_features(df_clean: pd.DataFrame, genre_weight: float = GENRE_WEIGHT):
    """Returns (X: np.ndarray[float32], feature_matrix: pd.DataFrame, scaler: StandardScaler)."""
    feat = df_clean.copy()
    feat["explicit"] = feat["explicit"].astype(int)

    # log1p transform for right-skewed features
    for c in SKEWED_FEATURES:
        feat[c] = np.log1p(feat[c])

    # cyclic encoding for musical key (key is circular: B is adjacent to C)
    feat["key_sin"] = np.sin(2 * np.pi * feat["key"] / 12)
    feat["key_cos"] = np.cos(2 * np.pi * feat["key"] / 12)

    # standardize continuous features (mean 0, std 1) -- required for cosine
    # similarity to actually discriminate (see notebook section 4 for why
    # min-max/non-negative features bias similarity upward)
    scaler = StandardScaler()
    feat[STANDARDIZE_FEATURES] = scaler.fit_transform(feat[STANDARDIZE_FEATURES])

    # center binary features to -0.5 / +0.5
    feat[BINARY_FEATURES] = feat[BINARY_FEATURES] - 0.5

    audio_feature_cols = STANDARDIZE_FEATURES + BINARY_FEATURES + CYCLIC_FEATURES

    # genre one-hot, weighted so it acts as one more informative feature
    # rather than overwhelming the vector
    genre_dummies = pd.get_dummies(feat["track_genre"], prefix="genre").astype(float) * genre_weight

    feature_matrix = pd.concat([feat[audio_feature_cols], genre_dummies], axis=1)
    X = feature_matrix.values.astype(np.float32)

    return X, feature_matrix, scaler


def vectorize_new_song(raw_features: dict, scaler: StandardScaler, feature_columns: list, genre_weight: float = GENRE_WEIGHT) -> np.ndarray:
    """
    Vectorize a single new song (not in the training catalog) the same way the
    training set was vectorized, using the already-fitted scaler and the exact
    column layout saved at training time. `raw_features` must contain the raw
    (untransformed) audio feature keys, e.g. danceability, energy, key, track_genre, etc.
    """
    row = {c: 0.0 for c in feature_columns}

    for c in SKEWED_FEATURES:
        raw_features[c] = np.log1p(raw_features[c])

    raw_features["key_sin"] = np.sin(2 * np.pi * raw_features["key"] / 12)
    raw_features["key_cos"] = np.cos(2 * np.pi * raw_features["key"] / 12)

    to_scale = np.array([[raw_features[c] for c in STANDARDIZE_FEATURES]])
    scaled = scaler.transform(to_scale)[0]
    for c, v in zip(STANDARDIZE_FEATURES, scaled):
        row[c] = v

    for c in BINARY_FEATURES:
        row[c] = raw_features.get(c, 0) - 0.5

    for c in CYCLIC_FEATURES:
        row[c] = raw_features[c]

    genre_col = f"genre_{raw_features.get('track_genre', '')}"
    if genre_col in row:
        row[genre_col] = genre_weight

    return np.array([row[c] for c in feature_columns], dtype=np.float32)
