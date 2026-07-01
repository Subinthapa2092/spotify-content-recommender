"""
Tests for src/recommender.py: recommend_songs (baseline), recommend_songs_mmr,
and search_songs.

Run with:
    pytest
from the project root (needs data/dataset.csv present, and pytest installed
via requirements-dev.txt or `pip install pytest`).
"""
import numpy as np

from src.recommender import recommend_songs, recommend_songs_mmr, search_songs


class TestRecommendSongs:
    def test_returns_requested_number_of_results(self, df_clean, X, seed_song):
        track_name, artist = seed_song
        recs = recommend_songs(track_name, df=df_clean, X=X, artist_hint=artist, n=10)
        assert recs is not None
        assert len(recs) == 10

    def test_seed_song_never_recommends_itself(self, df_clean, X, seed_song):
        track_name, artist = seed_song
        recs = recommend_songs(track_name, df=df_clean, X=X, artist_hint=artist, n=10)
        assert track_name not in recs["track_name"].values or (
            # allow same *name* if it's a genuinely different track/artist
            recs[recs["track_name"] == track_name]["artists"].str.contains(artist).sum() == 0
        )

    def test_similarity_scores_are_sorted_descending(self, df_clean, X, seed_song):
        track_name, artist = seed_song
        recs = recommend_songs(track_name, df=df_clean, X=X, artist_hint=artist, n=10)
        sims = recs["similarity"].values
        assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))

    def test_unknown_track_returns_none(self, df_clean, X):
        result = recommend_songs("this track definitely does not exist xyz123", df=df_clean, X=X, n=10)
        assert result is None

    def test_requesting_fewer_than_n_available_still_works(self, df_clean, X, seed_song):
        track_name, artist = seed_song
        recs = recommend_songs(track_name, df=df_clean, X=X, artist_hint=artist, n=3)
        assert len(recs) == 3


class TestRecommendSongsMMR:
    def test_returns_requested_number_of_results(self, df_clean, X, seed_song):
        track_name, artist = seed_song
        recs = recommend_songs_mmr(track_name, df=df_clean, X=X, artist_hint=artist, n=10)
        assert recs is not None
        assert len(recs) == 10

    def test_seed_song_never_recommended(self, df_clean, X, seed_song):
        track_name, artist = seed_song
        seed_idx = df_clean[
            (df_clean["track_name"] == track_name) & (df_clean["artists"].str.contains(artist))
        ].index[0]
        seed_track_id = df_clean.loc[seed_idx, "track_id"]

        recs = recommend_songs_mmr(track_name, df=df_clean, X=X, artist_hint=artist, n=10)
        assert seed_track_id not in recs["track_id"].values

    def test_unknown_track_returns_none(self, df_clean, X):
        result = recommend_songs_mmr("this track definitely does not exist xyz123", df=df_clean, X=X, n=10)
        assert result is None

    def test_lambda_one_matches_baseline_top_n(self, df_clean, X, seed_song):
        """lam=1 means pure relevance, no diversity penalty -- MMR should pick
        the exact same songs, in the exact same order, as the plain top-N."""
        track_name, artist = seed_song
        baseline = recommend_songs(track_name, df=df_clean, X=X, artist_hint=artist, n=10)
        mmr = recommend_songs_mmr(track_name, df=df_clean, X=X, artist_hint=artist, n=10, lam=1.0, pool_size=100)
        assert list(baseline["track_id"]) == list(mmr["track_id"])

    def test_lower_lambda_increases_intra_list_diversity(self, df_clean, X, seed_song):
        """Lower lambda should push harder for diversity, so the average
        pairwise similarity among the recommended songs should go down (or
        at least not increase) compared to lam=1."""
        from sklearn.metrics.pairwise import cosine_similarity

        track_name, artist = seed_song

        def avg_pairwise_sim(recs):
            idx = df_clean[df_clean["track_id"].isin(recs["track_id"])].index
            vectors = X[idx]
            sim_matrix = cosine_similarity(vectors)
            n = len(idx)
            return (sim_matrix.sum() - n) / (n * (n - 1))

        high_relevance = recommend_songs_mmr(track_name, df=df_clean, X=X, artist_hint=artist, n=10, lam=1.0)
        high_diversity = recommend_songs_mmr(track_name, df=df_clean, X=X, artist_hint=artist, n=10, lam=0.3)

        assert avg_pairwise_sim(high_diversity) <= avg_pairwise_sim(high_relevance)


class TestSearchSongs:
    def test_finds_known_track_by_name(self, df_clean):
        track_name = df_clean.iloc[0]["track_name"]
        results = search_songs(df_clean, track_name, limit=20)
        assert len(results) >= 1
        assert track_name in results["track_name"].values

    def test_case_insensitive(self, df_clean):
        track_name = df_clean.iloc[0]["track_name"]
        results_lower = search_songs(df_clean, track_name.lower(), limit=20)
        results_upper = search_songs(df_clean, track_name.upper(), limit=20)
        assert len(results_lower) >= 1
        assert len(results_upper) >= 1

    def test_matches_by_artist_too(self, df_clean):
        artist = df_clean.iloc[0]["artists"].split(";")[0]
        results = search_songs(df_clean, artist, limit=20)
        assert len(results) >= 1

    def test_nonsense_query_returns_empty(self, df_clean):
        results = search_songs(df_clean, "zzzqqqxxx_not_a_real_song_12345", limit=20)
        assert len(results) == 0

    def test_empty_query_returns_empty(self, df_clean):
        results = search_songs(df_clean, "", limit=20)
        assert len(results) == 0

    def test_respects_limit(self, df_clean):
        # "a" will match a huge number of tracks -- confirm limit is enforced
        results = search_songs(df_clean, "a", limit=5)
        assert len(results) <= 5
