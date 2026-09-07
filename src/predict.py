import os
import sys
import pandas as pd
import joblib

# Ensure root folder is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_prediction_pipeline(model_path="models/movie_rating_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline file not found at '{model_path}'. Please run training first.")
    return joblib.load(model_path)

def predict_movie_rating(movie_data, pipeline=None):
    """
    Predicts movie rating given input metadata dictionary.
    """
    if pipeline is None:
        pipeline = load_prediction_pipeline()

    # Convert single dict to DataFrame
    df_input = pd.DataFrame([movie_data])
    
    # Predict rating using pipeline
    predicted_rating = float(pipeline.predict(df_input)[0])
    predicted_rating = round(max(1.0, min(10.0, predicted_rating)), 2)

    # Classify rating quality
    if predicted_rating >= 8.0:
        verdict = "Must-Watch / Blockbuster Quality 🌟"
        category = "Blockbuster"
    elif predicted_rating >= 7.0:
        verdict = "Good / Highly Recommended 👍"
        category = "Good"
    elif predicted_rating >= 5.5:
        verdict = "Average / One-Time Watch 🎬"
        category = "Average"
    else:
        verdict = "Below Average / Flop Potential ⚠️"
        category = "Poor"

    return {
        "predicted_rating": predicted_rating,
        "verdict": verdict,
        "category": category
    }

if __name__ == "__main__":
    # Test sample prediction
    sample_movie = {
        "release_year": 2024,
        "runtime": 148,
        "budget": 160000000.0,
        "revenue": 450000000.0,
        "genres": "Sci-Fi|Action|Adventure",
        "lead_actor": "Leonardo DiCaprio",
        "director": "Christopher Nolan",
        "actor_popularity": 9.2,
        "director_popularity": 9.4
    }

    try:
        pipeline = load_prediction_pipeline()
        res = predict_movie_rating(sample_movie, pipeline)
        print(f"Sample Movie Rating Prediction: {res['predicted_rating']}/10 - Verdict: {res['category']}")
    except FileNotFoundError as e:
        print(e)
