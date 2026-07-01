"""
Holds the model artifacts loaded once at app startup, so every request just
reuses them instead of re-reading pickles from disk each time.
"""
from src.persistence import load_artifacts

MODELS_DIR = "models"


class AppState:
    def __init__(self):
        self.X = None
        self.scaler = None
        self.feature_columns = None
        self.df_clean = None
        self.metadata = None
        self.loaded = False

    def load(self):
        artifacts = load_artifacts(MODELS_DIR)
        self.X = artifacts["X"]
        self.scaler = artifacts["scaler"]
        self.feature_columns = artifacts["feature_columns"]
        self.df_clean = artifacts["df_clean"]
        self.metadata = artifacts["metadata"]
        self.loaded = True
        print(f"Loaded {self.df_clean.shape[0]} songs, "
              f"{self.df_clean['track_genre'].nunique()} genres, "
              f"feature dim {self.X.shape[1]}")


# single shared instance, imported by router.py and main.py
state = AppState()
