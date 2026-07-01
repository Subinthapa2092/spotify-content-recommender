"""
Save / load trained model artifacts. Logic copied 1:1 from notebook section 7,
plus a matching loader used by the FastAPI backend at startup.
"""
import json
import pickle
from pathlib import Path

import pandas as pd


def save_artifacts(models_dir, X, scaler, feature_matrix, df_clean, metadata: dict):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "feature_matrix.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(list(feature_matrix.columns), f)

    df_clean.to_pickle(models_dir / "songs_lookup.pkl")

    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved artifacts to {models_dir}/: "
          f"feature_matrix.pkl, scaler.pkl, feature_columns.json, songs_lookup.pkl, model_metadata.json")


def load_artifacts(models_dir):
    """Used by the FastAPI app at startup. Returns a dict of everything the API needs."""
    models_dir = Path(models_dir)

    with open(models_dir / "feature_matrix.pkl", "rb") as f:
        X = pickle.load(f)
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    df_clean = pd.read_pickle(models_dir / "songs_lookup.pkl")

    with open(models_dir / "model_metadata.json") as f:
        metadata = json.load(f)

    return {
        "X": X,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "df_clean": df_clean,
        "metadata": metadata,
    }
