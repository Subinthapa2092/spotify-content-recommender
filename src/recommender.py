"""
Content-based recommender. Logic copied 1:1 from notebook section 5 & 5b
(cosine similarity top-N, plus MMR diversity re-ranking).
"""
"""
Content-based recommender. Logic copied 1:1 from notebook section 5 & 5b
(cosine similarity top-N, plus MMR diversity re-ranking).
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

RESULT_COLS = ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity", "similarity"]


def _find_seed_index(df: pd.DataFrame, track_name: str, artist_hint: str = None):
    matches = df[df["track_name"].str.lower() == track_name.lower()]
    if artist_hint:
        matches = matches[matches["artists"].str.lower().str.contains(artist_hint.lower())]
    if matches.empty:
        return None
    return matches.index[0]


def recommend_songs(track_name: str, df: pd.DataFrame, X: np.ndarray, artist_hint: str = None, n: int = 10):
    """Plain top-N by cosine similarity to the seed song."""
    idx = _find_seed_index(df, track_name, artist_hint)
    if idx is None:
        return None

    sims = cosine_similarity(X[idx].reshape(1, -1), X)[0]

    result = df.copy()
    result["similarity"] = sims
    result = result[result.index != idx].sort_values("similarity", ascending=False).head(n)
    return result[RESULT_COLS].reset_index(drop=True)


def recommend_songs_mmr(track_name: str, df: pd.DataFrame, X: np.ndarray, artist_hint: str = None,
                         n: int = 10, lam: float = 0.75, pool_size: int = 100):
    """
    Diversity-aware recommender: MMR re-ranking over a top-similarity candidate pool.
    Default recommender for the app -- better catalog coverage / intra-list diversity
    at a modest genre-precision cost (see notebook section 6 for the full eval).
    """
    idx = _find_seed_index(df, track_name, artist_hint)
    if idx is None:
        return None

    sims_to_seed = cosine_similarity(X[idx].reshape(1, -1), X)[0]

    candidate_idx = np.argsort(-sims_to_seed)
    candidate_idx = candidate_idx[candidate_idx != idx][:pool_size]

    pool_relevance = sims_to_seed[candidate_idx]
    pool_sim_matrix = cosine_similarity(X[candidate_idx])

    selected_pos = []
    remaining_pos = list(range(len(candidate_idx)))

    while len(selected_pos) < n and remaining_pos:
        if not selected_pos:
            best = max(remaining_pos, key=lambda p: pool_relevance[p])
        else:
            best = max(
                remaining_pos,
                key=lambda p: lam * pool_relevance[p]
                - (1 - lam) * max(pool_sim_matrix[p, s] for s in selected_pos),
            )
        selected_pos.append(best)
        remaining_pos.remove(best)

    selected_idx = candidate_idx[selected_pos]
    result = df.loc[selected_idx].copy()
    result["similarity"] = pool_relevance[selected_pos]
    return result[RESULT_COLS].reset_index(drop=True)


RECOMMENDERS = {
    "baseline": recommend_songs,
    "mmr": recommend_songs_mmr,
}


def search_songs(df: pd.DataFrame, q: str, limit: int = 20) -> pd.DataFrame:
    """Case-insensitive substring search over track name and artist. Used by the
    /search endpoint, pulled out here so it's testable without running the API."""
    if not q:
        return df.iloc[0:0]
    q_lower = q.lower()
    mask = (
        df["track_name"].str.lower().str.contains(q_lower, na=False)
        | df["artists"].str.lower().str.contains(q_lower, na=False)
    )
    return df[mask].head(limit)


LIST_COLS = ["track_id", "track_name", "artists", "album_name", "track_genre", "popularity"]

# valence = musical positivity (sad -> happy), energy = intensity/activity.
# Both are native columns in the dataset, so moods are computed from real
# audio features, not hardcoded.
MOOD_RULES = {
    "happy":     lambda df: (df["valence"] >= 0.6) & (df["energy"] >= 0.5),
    "chill":     lambda df: (df["valence"] >= 0.4) & (df["energy"] <= 0.4),
    "sad":       lambda df: (df["valence"] <= 0.35) & (df["energy"] <= 0.5),
    "energetic": lambda df: (df["energy"] >= 0.75),
    "romantic":  lambda df: (df["valence"] >= 0.35) & (df["valence"] <= 0.7) & (df["acousticness"] >= 0.3),
    "focus":     lambda df: (df["instrumentalness"] >= 0.3) & (df["energy"] <= 0.5),
}


def popular_songs(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Most popular tracks in the catalog (real popularity column, not a model output)."""
    return df.sort_values("popularity", ascending=False).drop_duplicates("track_name").head(limit)[LIST_COLS]


def songs_by_genre(df: pd.DataFrame, genre: str, limit: int = 20) -> pd.DataFrame:
    matches = df[df["track_genre"].str.lower() == genre.lower()]
    return matches.sort_values("popularity", ascending=False).head(limit)[LIST_COLS]


def songs_by_mood(df: pd.DataFrame, mood: str, limit: int = 20) -> pd.DataFrame:
    mood = mood.lower()
    if mood not in MOOD_RULES:
        return df.iloc[0:0][LIST_COLS]
    matches = df[MOOD_RULES[mood](df)]
    return matches.sort_values("popularity", ascending=False).head(limit)[LIST_COLS]