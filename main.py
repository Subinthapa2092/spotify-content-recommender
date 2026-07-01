"""
Full ML pipeline runner: load -> clean -> build features -> evaluate -> persist.

This is the script version of notebook/song_recommendation.ipynb sections 1-2 and 4-7
(EDA, section 3, is exploratory-only and lives in the notebook, not here).

Usage:
    python main.py
"""
import numpy as np

from src.data_loader import load_dataset
from src.preprocessing import clean_data
from src.features import build_features, GENRE_WEIGHT
from src.recommender import recommend_songs, recommend_songs_mmr
from src.evaluation import run_eval_suite
from src.persistence import save_artifacts

RANDOM_STATE = 42
DATA_PATH = "data/dataset.csv"
MODELS_DIR = "models"
EVAL_SAMPLE_SIZE = 200  # bump to 1000 for a tighter coverage estimate, like the notebook does


def main():
    np.random.seed(RANDOM_STATE)

    print("=== 1. Loading data ===")
    df = load_dataset(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n=== 2. Cleaning ===")
    df_clean = clean_data(df)

    print("\n=== 3. Feature engineering ===")
    X, feature_matrix, scaler = build_features(df_clean, genre_weight=GENRE_WEIGHT)
    print(f"Feature matrix: {X.shape}, {X.nbytes / 1e6:.1f} MB")

    print("\n=== 4. Evaluation ===")
    sample = df_clean.sample(min(EVAL_SAMPLE_SIZE, df_clean.shape[0]), random_state=RANDOM_STATE)

    baseline_df, baseline_coverage = run_eval_suite(recommend_songs, sample, df_clean, X, k=10)
    mmr_df, mmr_coverage = run_eval_suite(recommend_songs_mmr, sample, df_clean, X, k=10, lam=0.75, pool_size=100)

    print(f"Baseline  -> genre_precision@10={baseline_df['genre_precision@k'].mean():.3f}, "
          f"ILD={baseline_df['intra_list_diversity'].mean():.3f}, coverage={baseline_coverage:.4f}")
    print(f"MMR       -> genre_precision@10={mmr_df['genre_precision@k'].mean():.3f}, "
          f"ILD={mmr_df['intra_list_diversity'].mean():.3f}, coverage={mmr_coverage:.4f}")

    print("\n=== 5. Persisting artifacts ===")
    metadata = {
        "n_songs": int(df_clean.shape[0]),
        "n_genres": int(df_clean["track_genre"].nunique()),
        "feature_dim": int(X.shape[1]),
        "genre_weight": GENRE_WEIGHT,
        "random_state": RANDOM_STATE,
        "default_recommender": "mmr",
        "mmr_params": {"lam": 0.75, "pool_size": 100},
        "eval_summary": {
            "baseline": {
                "genre_precision@10": float(baseline_df["genre_precision@k"].mean()),
                "mean_similarity@10": float(baseline_df["mean_similarity@k"].mean()),
                "intra_list_diversity": float(baseline_df["intra_list_diversity"].mean()),
                "catalog_coverage": float(baseline_coverage),
            },
            "mmr": {
                "genre_precision@10": float(mmr_df["genre_precision@k"].mean()),
                "mean_similarity@10": float(mmr_df["mean_similarity@k"].mean()),
                "intra_list_diversity": float(mmr_df["intra_list_diversity"].mean()),
                "catalog_coverage": float(mmr_coverage),
            },
        },
    }
    save_artifacts(MODELS_DIR, X, scaler, feature_matrix, df_clean, metadata)
    print("\nDone. Artifacts are in models/ -- ready for the FastAPI app to load.")


if __name__ == "__main__":
    main()
