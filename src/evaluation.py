"""
Evaluation. Logic copied 1:1 from notebook section 6.
Formal proxy metrics for content-based recommenders (no ground-truth labels exist).
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_query(track_name, df_clean, X, artist_hint=None, k=10, recommender=None, **rec_kwargs):
    recs = recommender(track_name, df=df_clean, X=X, artist_hint=artist_hint, n=k, **rec_kwargs)
    if recs is None:
        return None
    seed_row = df_clean[df_clean["track_name"].str.lower() == track_name.lower()].iloc[0]

    genre_precision = (recs["track_genre"] == seed_row["track_genre"]).mean()
    mean_sim = recs["similarity"].mean()
    pop_gap = (seed_row["popularity"] - recs["popularity"]).abs().mean()

    rec_idx = df_clean[
        df_clean["track_name"].isin(recs["track_name"]) & df_clean["artists"].isin(recs["artists"])
    ].index[:k]
    rec_vectors = X[rec_idx]
    pairwise = cosine_similarity(rec_vectors)
    ild = 1 - (pairwise.sum() - k) / (k * (k - 1))

    return {
        "genre_precision@k": genre_precision,
        "mean_similarity@k": mean_sim,
        "intra_list_diversity": ild,
        "popularity_gap": pop_gap,
        "recommended_track_ids": set(df_clean.loc[rec_idx, "track_id"]),
    }


def run_eval_suite(recommender, sample, df_clean, X, k=10, **rec_kwargs):
    rows, all_recommended = [], set()
    for _, r in sample.iterrows():
        res = evaluate_query(
            r["track_name"], df_clean, X,
            artist_hint=r["artists"].split(";")[0], k=k, recommender=recommender, **rec_kwargs,
        )
        if res is None:
            continue
        all_recommended |= res.pop("recommended_track_ids")
        res["seed_track"] = r["track_name"]
        rows.append(res)
    eval_df = pd.DataFrame(rows)
    coverage = len(all_recommended) / df_clean.shape[0]
    return eval_df, coverage
