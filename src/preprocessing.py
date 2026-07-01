"""
Cleans the raw dataset. Logic copied 1:1 from notebook section 2 (Data Cleaning)
so the pipeline script and the notebook always agree.
"""
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Steps (same rationale as the notebook):
    1. Drop rows with nulls.
    2. Drop duplicate track_ids, keep first occurrence.
    3. Drop any remaining exact duplicate rows.
    4. Filter degenerate rows (bad duration / zero tempo).
    5. Reset index to a clean 0..N-1 range (recommender uses positional indices).
    """
    df_clean = df.copy()
    before = df_clean.shape[0]

    df_clean = df_clean.dropna()
    df_clean = df_clean.drop_duplicates(subset="track_id", keep="first")
    df_clean = df_clean.drop_duplicates()

    # Guard against bad sensor/metadata rows
    df_clean = df_clean[
        (df_clean["duration_ms"] > 0) & (df_clean["duration_ms"] < 20 * 60 * 1000)
    ]  # < 20 min
    df_clean = df_clean[df_clean["tempo"] > 0]

    df_clean = df_clean.reset_index(drop=True)

    removed = before - df_clean.shape[0]
    print(f"Cleaning: {before} -> {df_clean.shape[0]} rows ({removed} removed)")

    assert df_clean["track_id"].is_unique
    assert df_clean.isnull().sum().sum() == 0

    return df_clean
