from typing import List, Optional
from pydantic import BaseModel


class SongOut(BaseModel):
    track_id: str
    track_name: str
    artists: str
    album_name: str
    track_genre: str
    popularity: int


class SearchResponse(BaseModel):
    query: str
    results: List[SongOut]


class RecommendationOut(BaseModel):
    track_id: str
    track_name: str
    artists: str
    album_name: str
    track_genre: str
    popularity: int
    similarity: float


class RecommendResponse(BaseModel):
    seed: SongOut
    recommender: str
    recommendations: List[RecommendationOut]


class GenresResponse(BaseModel):
    genres: List[str]


class HealthResponse(BaseModel):
    status: str
    n_songs: Optional[int] = None
    n_genres: Optional[int] = None
    default_recommender: Optional[str] = None
    eval_summary: Optional[dict] = None
