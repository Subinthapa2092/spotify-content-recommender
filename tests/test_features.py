"""Tests for src/features.py."""
import numpy as np

from src.features import build_features


def test_feature_matrix_has_no_nulls_or_nans(df_clean):
    X, feature_matrix, scaler = build_features(df_clean)
    assert not np.isnan(X).any()
    assert not feature_matrix.isnull().values.any()


def test_feature_matrix_row_count_matches_input(df_clean):
    X, feature_matrix, scaler = build_features(df_clean)
    assert X.shape[0] == len(df_clean)
    assert feature_matrix.shape[0] == len(df_clean)


def test_each_song_has_exactly_one_active_genre_column(df_clean):
    _, feature_matrix, _ = build_features(df_clean)
    genre_cols = [c for c in feature_matrix.columns if c.startswith("genre_")]
    genre_values = feature_matrix[genre_cols].values
    # exactly one non-zero entry per row
    nonzero_per_row = (genre_values != 0).sum(axis=1)
    assert (nonzero_per_row == 1).all()


def test_key_sin_cos_are_bounded():
    """Cyclic encoding should always land in [-1, 1]."""
    import pandas as pd
    from src.features import build_features

    df = pd.DataFrame({
        "track_id": [str(i) for i in range(12)],
        "track_name": [f"Song {i}" for i in range(12)],
        "artists": ["X"] * 12,
        "duration_ms": [200000] * 12,
        "tempo": [120.0] * 12,
        "danceability": [0.5] * 12,
        "energy": [0.5] * 12,
        "speechiness": [0.1] * 12,
        "acousticness": [0.3] * 12,
        "instrumentalness": [0.0] * 12,
        "liveness": [0.1] * 12,
        "valence": [0.5] * 12,
        "loudness": [-8.0] * 12,
        "time_signature": [4] * 12,
        "mode": [1] * 12,
        "explicit": [False] * 12,
        "key": list(range(12)),
        "track_genre": ["pop"] * 12,
    })
    _, feature_matrix, _ = build_features(df)
    assert feature_matrix["key_sin"].between(-1, 1).all()
    assert feature_matrix["key_cos"].between(-1, 1).all()
