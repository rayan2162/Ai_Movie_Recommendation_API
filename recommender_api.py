from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import traceback
import random

# --- Initialize App ---
app = FastAPI(title="Ai Movie Recommendation API")

# --- Enable CORS for Laravel frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://127.0.0.1:8000"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Dataset ---
try:
    movies = pd.read_csv("movies.csv")  # File must exist in the same directory
    print(f"Loaded {len(movies)} movies from dataset.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    movies = pd.DataFrame(columns=["movieId", "title", "genres"])

# --- Prepare TF-IDF model if dataset available ---
if not movies.empty:
    tfidf = TfidfVectorizer(stop_words='english')
    movies["genres"] = movies["genres"].fillna("")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
else:
    cosine_sim = None


# --- Helper: Get movie index ---
def get_index(title):
    """Find movie index by title (case-insensitive)"""
    matches = movies[movies["title"].str.lower() == title.lower()]
    if len(matches) > 0:
        return matches.index[0]
    return None


# --- Helper: Recommend movies ---
def recommend_movies_by_genre(liked_titles, topn=5):
    """Generate recommendations based on genre similarity"""
    if movies.empty or cosine_sim is None:
        return [{"error": "Movies dataset not loaded."}]

    idxs = [get_index(t) for t in liked_titles if get_index(t) is not None]
    if not idxs:
        return [{"error": "No matching movies found for provided titles."}]

    # Compute average similarity score for liked movies
    sim_scores = sum(cosine_sim[i] for i in idxs)
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if i[0] not in idxs][:topn]

    recommendations = movies.iloc[[i[0] for i in sim_scores]][["title", "genres"]]
    return recommendations.to_dict(orient="records")


# --- Root Endpoint ---
@app.get("/")
async def home():
    return {"message": "🎬 Movie Recommendation API is running."}


# --- Get Movie Samples ---
@app.get("/movies")
async def list_movies(limit: int = 10):
    """Return first N movies from dataset."""
    if movies.empty:
        return {"error": "Movies dataset not loaded"}
    return movies.head(limit).to_dict(orient="records")


# --- Recommend Endpoint ---
@app.post("/recommend")
async def recommend(request: Request):
    """
    Recommend movies based on liked titles.
    Expected JSON: {"liked_movies": ["Inception", "Titanic"], "n": 5}
    """
    try:
        data = await request.json()
        liked_movies = data.get("liked_movies", [])
        n = int(data.get("n", 5))

        print(f"Received request with liked movies: {liked_movies}")

        if not liked_movies:
            return JSONResponse({"recommendations": [{"error": "No liked movies provided"}]}, status_code=400)

        if movies.empty:
            return JSONResponse({"recommendations": [{"error": "Movies dataset not loaded"}]}, status_code=500)

        # Clean titles for fuzzy matching
        movies["clean_title"] = movies["title"].str.lower().str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

        matched_movies = []
        for liked in liked_movies:
            liked_lower = liked.lower().strip()
            matches = movies[movies["clean_title"].str.contains(liked_lower, case=False, na=False)]
            matched_movies.extend(matches["title"].tolist())

        matched_movies = list(set(matched_movies))
        print(f"Matched movies: {matched_movies}")

        if not matched_movies:
            # fallback random selection
            recs = movies.sample(min(n, len(movies)))[["title", "genres"]].to_dict(orient="records")
        else:
            # Use genre similarity recommendations
            recs = recommend_movies_by_genre(matched_movies, topn=n)
            # If recommendation fails, fallback to random
            if "error" in recs[0]:
                candidates = movies[~movies["title"].isin(matched_movies)]
                recs = candidates.sample(min(n, len(candidates)))[["title", "genres"]].to_dict(orient="records")

        return {"recommendations": recs}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": f"Server Error: {str(e)}"},
            status_code=500
        )


# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
