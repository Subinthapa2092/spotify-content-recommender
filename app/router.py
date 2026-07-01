from fastapi import APIRouter, HTTPException, Query

from app.state import state
from app.schemas import (
    SearchResponse, SongOut, RecommendResponse, RecommendationOut,
    GenresResponse, HealthResponse, MoodsResponse, SongListResponse,
)
from src.recommender import RECOMMENDERS, search_songs, popular_songs, songs_by_genre, songs_by_mood, MOOD_RULES

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    if not state.loaded:
        return HealthResponse(status="not_loaded")
    return HealthResponse(
        status="ok",
        n_songs=state.metadata["n_songs"],
        n_genres=state.metadata["n_genres"],
        default_recommender=state.metadata["default_recommender"],
        eval_summary=state.metadata["eval_summary"],
    )


@router.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1, description="Track name or artist to search for"),
           limit: int = Query(20, ge=1, le=100)):
    matches = search_songs(state.df_clean, q, limit=limit)

    results = [
        SongOut(
            track_id=r.track_id, track_name=r.track_name, artists=r.artists,
            album_name=r.album_name, track_genre=r.track_genre, popularity=int(r.popularity),
        )
        for r in matches.itertuples()
    ]
    return SearchResponse(query=q, results=results)


@router.get("/recommend/{track_id}", response_model=RecommendResponse)
def recommend(track_id: str, n: int = Query(10, ge=1, le=50),
              method: str = Query("mmr", pattern="^(mmr|baseline)$")):
    df = state.df_clean
    seed_rows = df[df["track_id"] == track_id]
    if seed_rows.empty:
        raise HTTPException(status_code=404, detail=f"track_id '{track_id}' not found")
    seed_row = seed_rows.iloc[0]

    recommender_fn = RECOMMENDERS[method]
    recs = recommender_fn(
        seed_row["track_name"], df=df, X=state.X,
        artist_hint=seed_row["artists"].split(";")[0], n=n,
    )
    if recs is None:
        raise HTTPException(status_code=404, detail="Could not generate recommendations for this track")

    return RecommendResponse(
        seed=SongOut(
            track_id=seed_row.track_id, track_name=seed_row.track_name, artists=seed_row.artists,
            album_name=seed_row.album_name, track_genre=seed_row.track_genre,
            popularity=int(seed_row.popularity),
        ),
        recommender=method,
        recommendations=[
            RecommendationOut(**{**row, "popularity": int(row["popularity"]), "similarity": float(row["similarity"])})
            for row in recs.to_dict(orient="records")
        ],
    )


@router.get("/genres", response_model=GenresResponse)
def genres():
    return GenresResponse(genres=sorted(state.df_clean["track_genre"].unique().tolist()))


@router.get("/moods", response_model=MoodsResponse)
def moods():
    return MoodsResponse(moods=sorted(MOOD_RULES.keys()))


def _to_song_out(row) -> SongOut:
    return SongOut(
        track_id=row["track_id"], track_name=row["track_name"], artists=row["artists"],
        album_name=row["album_name"], track_genre=row["track_genre"], popularity=int(row["popularity"]),
    )


@router.get("/popular", response_model=SongListResponse)
def popular(limit: int = Query(20, ge=1, le=50)):
    rows = popular_songs(state.df_clean, limit=limit)
    return SongListResponse(label="Popular right now", results=[_to_song_out(r) for _, r in rows.iterrows()])


@router.get("/genre/{genre}", response_model=SongListResponse)
def by_genre(genre: str, limit: int = Query(20, ge=1, le=50)):
    rows = songs_by_genre(state.df_clean, genre, limit=limit)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"genre '{genre}' not found")
    return SongListResponse(label=genre.title(), results=[_to_song_out(r) for _, r in rows.iterrows()])


@router.get("/mood/{mood}", response_model=SongListResponse)
def by_mood(mood: str, limit: int = Query(20, ge=1, le=50)):
    if mood.lower() not in MOOD_RULES:
        raise HTTPException(status_code=404, detail=f"mood '{mood}' not recognized")
    rows = songs_by_mood(state.df_clean, mood, limit=limit)
    return SongListResponse(label=mood.title(), results=[_to_song_out(r) for _, r in rows.iterrows()])