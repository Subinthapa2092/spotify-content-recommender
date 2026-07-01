"""
FastAPI backend entrypoint.

Run with:
    uvicorn app.main:app --reload

Endpoints:
    GET /health                          - model/service status
    GET /search?q=...&limit=20           - search songs by name/artist
    GET /recommend/{track_id}?n=10&method=mmr  - get recommendations
    GET /genres                          - list all genres in the catalog
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.state import state
from app.router import router

app = FastAPI(
    title="Song Recommendation API",
    description="Content-based song recommender (cosine similarity + MMR re-ranking) over Spotify audio features.",
    version="1.0.0",
)

# Allow the frontend (served separately) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    state.load()


app.include_router(router)
