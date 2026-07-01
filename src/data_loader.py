"""
Loads the raw Spotify tracks dataset from disk.
"""
import pandas as pd


def load_dataset(path: str = "data/dataset.csv") -> pd.DataFrame:
    """Load the raw dataset. Mirrors notebook cell 1 (has an unnamed index col)."""
    df = pd.read_csv(path, index_col=0)
    return df
