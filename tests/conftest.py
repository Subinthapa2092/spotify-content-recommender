"""
Shared test fixtures.

Tests run against a sample of the real dataset (not the full 89k rows) so the
suite stays fast. `session` scope means the sample is only cleaned + featurized
once, no matter how many tests use it.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.preprocessing import clean_data
from src.features import build_features

SAMPLE_SIZE = 3000
RANDOM_STATE = 42


@pytest.fixture(scope="session")
def df_clean():
    df = load_dataset("data/dataset.csv")
    df = clean_data(df)
    # Sample down for test speed. random_state keeps it deterministic.
    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    return sample.reset_index(drop=True)


@pytest.fixture(scope="session")
def features(df_clean):
    X, feature_matrix, scaler = build_features(df_clean)
    return X, feature_matrix, scaler


@pytest.fixture(scope="session")
def X(features):
    return features[0]


@pytest.fixture(scope="session")
def seed_song(df_clean):
    """A song guaranteed to exist in df_clean, for tests that need a real seed."""
    row = df_clean.iloc[0]
    return row["track_name"], row["artists"].split(";")[0]
