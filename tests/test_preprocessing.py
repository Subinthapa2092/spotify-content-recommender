"""Tests for src/preprocessing.py."""
import pandas as pd

from src.preprocessing import clean_data


def test_clean_data_removes_nulls():
    df = pd.DataFrame({
        "track_id": ["a", "b", "c"],
        "track_name": ["Song A", "Song B", None],
        "artists": ["X", "Y", "Z"],
        "duration_ms": [200000, 210000, 220000],
        "tempo": [120.0, 130.0, 140.0],
    })
    cleaned = clean_data(df)
    assert cleaned["track_name"].isnull().sum() == 0
    assert len(cleaned) == 2


def test_clean_data_removes_duplicate_track_ids():
    df = pd.DataFrame({
        "track_id": ["a", "a", "b"],
        "track_name": ["Song A", "Song A (dup)", "Song B"],
        "artists": ["X", "X", "Y"],
        "duration_ms": [200000, 200000, 210000],
        "tempo": [120.0, 120.0, 130.0],
    })
    cleaned = clean_data(df)
    assert cleaned["track_id"].is_unique
    assert len(cleaned) == 2


def test_clean_data_removes_zero_tempo_and_bad_duration():
    df = pd.DataFrame({
        "track_id": ["a", "b", "c"],
        "track_name": ["Song A", "Song B", "Song C"],
        "artists": ["X", "Y", "Z"],
        "duration_ms": [200000, 0, 210000],
        "tempo": [120.0, 130.0, 0.0],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["track_id"] == "a"


def test_clean_data_resets_index():
    df = pd.DataFrame({
        "track_id": ["a", "b", "c"],
        "track_name": ["Song A", "Song B", "Song C"],
        "artists": ["X", "Y", "Z"],
        "duration_ms": [200000, 210000, 220000],
        "tempo": [120.0, 130.0, 140.0],
    })
    cleaned = clean_data(df)
    assert list(cleaned.index) == list(range(len(cleaned)))
