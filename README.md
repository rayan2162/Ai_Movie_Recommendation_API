# Ai_Movie_Recommendation_API

A content-based movie recommendation API built using FastAPI and scikit-learn. It analyzes movie genres using TF-IDF and cosine similarity to suggest movies based on user preferences. It is a RESTful service that returns personalized movie recommendations based on the genres of movies a user likes. Built with FastAPI, it offers quick and easy integration with frontend applications via HTTP endpoints.

This API uses content-based filtering to recommend movies. It compares the genres of movies liked by a user to all other movies in the dataset using:

- TF-IDF Vectorization: To convert movie genres into vector format.

- Cosine Similarity: To measure how similar one movie is to another based on genre.

When a user submits one or more movie titles they like, the API returns a list of similar movies by genre. If the titles don't match or recommendations can't be generated, the system returns a random fallback list of movies.

## Features

- Recommend similar movies based on genre similarity
- Fallback recommendations using random selection
- Case-insensitive title matching with basic cleaning
- CORS support for frontend (e.g., Laravel, React)
- FastAPI-based (high performance)
- Easily deployable with Uvicorn

## Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt

```

## How It Works

Frontend sends a list of liked movie titles to the API via POST request.

The API:

- Cleans and matches movie titles.
- Finds genre vectors using TF-IDF.
- Calculates similarity with all other movies.
- Returns top N recommendations.
- If no matching titles or recommendations are found:
- The system returns random movies from the dataset as fallback.

## API Endpoints

### `GET /`

- Health check for the API.

**Response :**

```json
{
  "message": "🎬 Movie Recommendation API is running."
}
```

### `GET /movies?limit=10`

Returns the first N movies from the dataset.

**Query Parameters:** limit (optional): Number of movies to return. Default is 10.

**Response :**

```json
[
  {
    "movieId": 1,
    "title": "Toy Story (1995)",
    "genres": "Adventure|Animation|Children|Comedy|Fantasy"
  },
  ...
]
```

### `POST /recommend`

Recommends similar movies based on the input list of liked movies.

**Expected JSON Input:**

```json
{
  "liked_movies": ["Inception", "Titanic"],
  "n": 5
}
```

**Success Response:**

```json
{
  "recommendations": [
    {
      "title": "Interstellar (2014)",
      "genres": "Adventure|Drama|Sci-Fi"
    },
    {
      "title": "The Matrix (1999)",
      "genres": "Action|Sci-Fi"
    },
    ...
  ]
}
```

**Fallback Response (no matches):**

```json
{
  "recommendations": [
    {
      "title": "Finding Nemo (2003)",
      "genres": "Animation|Children|Comedy"
    },
    ...
  ]
}
```

**Error Response :**

```json
{
  "recommendations": [
    {
      "error": "No liked movies provided"
    }
  ]
}
```

## Recommendation Logic

- The genre column in movies.csv is vectorized using TfidfVectorizer.
- Cosine similarity is calculated between all movies.
- Given a list of liked movies, the average similarity scores are computed and sorted.
- Top N similar movies are returned (excluding the input ones).

## File Structure

```txt
ai-movie-recommendation-api/
│
├── main.py           # FastAPI application
├── movies.csv        # Dataset of movies with titles and genres
└── README.md         # Project documentation (this file)
```

---

Made with ❤️ using FastAPI and scikit-learn.
Feel free to fork, star ⭐, or contribute to the project!
