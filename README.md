# 🎵 Song Recommendation System

A **content-based music recommendation system** built using the Spotify Tracks Dataset containing **114,000+ songs across 114 genres**.

The system recommends songs similar to a selected track using **cosine similarity** over engineered audio features and genre information. To improve recommendation diversity, it also implements **Maximal Marginal Relevance (MMR)** re-ranking.

Since the dataset contains **no user interactions or ratings**, collaborative filtering is **not applicable**. A content-based approach is the appropriate solution for this problem.

---

## ✨ Features

- Content-based recommendation engine
- Cosine similarity for nearest-neighbor retrieval
- Maximal Marginal Relevance (MMR) for diverse recommendations
- FastAPI REST API
- Interactive frontend for searching songs
- Model persistence
- Evaluation metrics for recommender systems
- Unit tests with pytest
- Docker support

---

# Dataset

- **Dataset:** Spotify Tracks Dataset
- **Songs:** 114,000+
- **Genres:** 114

The dataset contains track metadata and audio features including:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration
- Popularity
- Genre
- Artist
- Track Name

---

# Recommendation Pipeline

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Feature Scaling
    │
    ▼
Cosine Similarity
    │
    ▼
Top Candidate Songs
    │
    ▼
MMR Re-ranking
    │
    ▼
Final Recommendations
```

---

# Project Structure

```
song-recommendation-system/
│
├── app/
│   ├── main.py
│   ├── router.py
│   ├── schemas.py
│   └── state.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── recommender.py
│   ├── evaluation.py
│   └── persistence.py
│
├── frontend/
│   └── index.html
│
├── notebook/
│   └── EDA.ipynb
│
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   └── test_recommender.py
│
├── models/
│
├── data/
│   └── dataset.csv
│
├── main.py
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/song-recommendation-system.git

cd song-recommendation-system
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Build the Recommendation Model

Run the ML pipeline to preprocess the dataset, engineer features, evaluate the recommender, and save all required artifacts.

```bash
python main.py
```

Generated artifacts are stored in:

```
models/

feature_matrix.pkl
scaler.pkl
feature_columns.json
songs_lookup.pkl
model_metadata.json
```

Run this step again only if the dataset or feature engineering changes.

---

# Start the API

Launch the FastAPI backend.

```bash
uvicorn app.main:app --reload
```

API Documentation

```
http://localhost:8000/docs
```

---

# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Model status and evaluation metrics |
| GET | `/search?q=love&limit=20` | Search songs by title or artist |
| GET | `/recommend/{track_id}?n=10&method=mmr` | Get recommendations |
| GET | `/genres` | List available genres |

Recommendation methods

- **baseline** → cosine similarity only
- **mmr** → cosine similarity + diversity re-ranking (default)

---

# Frontend

The project includes a lightweight HTML frontend.

Open

```
frontend/index.html
```

in your browser.

By default it connects to

```
http://localhost:8000
```

If needed, update the `API_BASE` variable inside the JavaScript section.

---

# Running Tests

Install development dependencies

```bash
pip install -r requirements-dev.txt
```

Run all tests

```bash
pytest
```

The test suite includes **24 unit tests** covering:

- Data preprocessing
- Feature engineering
- Song search
- Baseline recommender
- MMR recommender

Tests run against a **3,000-song sample of the real dataset**, providing realistic validation while maintaining fast execution.

---

# Docker

Generate the model artifacts first.

```bash
python main.py
```

Build the Docker image

```bash
docker build -t song-rec .
```

Run the container

```bash
docker run -p 8000:8000 song-rec
```

---

# Evaluation

Traditional train/test evaluation is **not applicable** because no user preference labels exist.

Instead, the recommender is evaluated using standard proxy metrics for content-based recommendation systems:

- Genre Precision
- Recommendation Diversity
- Catalog Coverage

The evaluation summary is stored in

```
models/model_metadata.json
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Pytest
- HTML
- JavaScript
- Docker

---

# Why Content-Based Filtering?

Collaborative filtering requires historical user interactions such as:

- User ratings
- Listening history
- Likes
- Play counts

The Spotify Tracks Dataset contains only **track-level metadata and audio features**, making **content-based recommendation** the correct modeling approach.

---

# Future Improvements

- Hybrid recommendation system
- Approximate nearest-neighbor search (FAISS)
- Spotify API integration
- User profiles and personalized recommendations
- Playlist generation
- Recommendation explanations
- Music embedding models

---

# License

This project is intended for educational and portfolio purposes.

---

# Author

**Subin Thapa**

Profile : 

GitHub: https://github.com/subinthapa2092

LinkedIn: https://linkedin.com/in/subinthapa
