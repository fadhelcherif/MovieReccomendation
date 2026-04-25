import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from recommendation_engine import MovieRecommender

sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

app = FastAPI()

# Load model once
recommender = MovieRecommender()


class PredictRequest(BaseModel):
    favorites: List[str]
    top_n: int = 10


@app.post("/predict")
def predict(request: PredictRequest):
    recs = recommender.get_recommendations_from_favorites(request.favorites, request.top_n)
    if recs.empty:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    # Return as list of dicts
    return {"recommendations": recs[['title', 'genres', 'directors']].to_dict(orient="records")}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(request: PredictRequest):
    return predict(request)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
