import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Ensure root folder is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.generate_dataset import generate_movie_dataset
from src.preprocessing import MovieFeatureExtractor

def train_and_evaluate_models(data_path="data/movies.csv", model_output_path="models/movie_rating_model.joblib"):
    # 1. Load or Generate Dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at '{data_path}'. Generating a new dataset...")
        generate_movie_dataset(num_samples=2500, output_path=data_path)
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} records.")

    # 2. Split Features and Target
    X = df.drop(columns=['movie_id', 'title', 'rating'])
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Fit Feature Extractor and Scaler on Train Set
    extractor = MovieFeatureExtractor()
    X_train_feat = extractor.fit_transform(X_train, y_train)
    X_test_feat = extractor.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # 4. Define Candidate Regressors
    candidate_models = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42, verbosity=0),
        "LightGBM Regressor": LGBMRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42, verbose=-1)
    }

    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model_obj = None

    print("\n" + "="*60)
    print(f"{'Model':<22} | {'RMSE':<8} | {'MAE':<8} | {'R2 Score':<8}")
    print("="*60)

    for name, model in candidate_models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
        print(f"{name:<22} | {rmse:<8.4f} | {mae:<8.4f} | {r2:<8.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model

    print("="*60)
    print(f"\nBest Performing Model: {best_model_name} (R2: {best_r2:.4f})")

    # 5. Hyperparameter Tuning for Best Model
    print(f"Hyperparameter tuning for {best_model_name}...")
    if best_model_name == "XGBoost Regressor":
        param_dist = {
            'n_estimators': [100, 150, 250],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.07, 0.1],
            'subsample': [0.8, 1.0]
        }
        search = RandomizedSearchCV(XGBRegressor(verbosity=0, random_state=42), param_dist, n_iter=8, cv=3, scoring='r2', random_state=42)
        search.fit(X_train_scaled, y_train)
        best_model_obj = search.best_estimator_
    elif best_model_name == "Random Forest":
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5]
        }
        search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_dist, n_iter=8, cv=3, scoring='r2', random_state=42)
        search.fit(X_train_scaled, y_train)
        best_model_obj = search.best_estimator_

    # Evaluate final tuned model
    final_pred = best_model_obj.predict(X_test_scaled)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    final_mae = mean_absolute_error(y_test, final_pred)
    final_r2 = r2_score(y_test, final_pred)
    print(f"Tuned {best_model_name} Test Results -> RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}, R2: {final_r2:.4f}")

    # 6. Build and Train Final Pipeline on Full Data
    final_pipeline = Pipeline([
        ('preprocessor', MovieFeatureExtractor()),
        ('scaler', StandardScaler()),
        ('regressor', best_model_obj)
    ])

    final_pipeline.fit(X, y)

    # 7. Save Artifacts
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(final_pipeline, model_output_path)
    print(f"\nFinal trained pipeline saved to '{model_output_path}'")

    # Save metrics metadata
    metrics_data = {
        "best_model": best_model_name,
        "test_rmse": round(final_rmse, 4),
        "test_mae": round(final_mae, 4),
        "test_r2": round(final_r2, 4),
        "candidate_results": results
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    return final_pipeline, metrics_data

if __name__ == "__main__":
    train_and_evaluate_models()
