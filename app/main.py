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
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.state import state
from app.router import router

app = FastAPI(
    title="Song Recommendation API",
    description="Content-based song recommender (cosine similarity + MMR re-ranking) over Spotify audio features.",
    version="1.0.0",
)

# Allow the frontend to call this API even if it's ever hosted from a
# different origin (e.g. testing frontend/index.html directly from disk).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    state.load()


app.include_router(router, prefix="/api")

# Serve the frontend from the SAME service, so one deploy = frontend + API
# together, no separate hosting/CORS setup needed in production.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
