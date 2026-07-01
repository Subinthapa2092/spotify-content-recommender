from fastapi import APIRouter, HTTPException, Query

from app.state import state
from app.schemas import (
    SearchResponse, SongOut, RecommendResponse, RecommendationOut,
    GenresResponse, HealthResponse,
)
from src.recommender import RECOMMENDERS, search_songs

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