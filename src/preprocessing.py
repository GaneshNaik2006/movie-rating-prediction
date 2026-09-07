import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
    "Music", "Mystery", "Romance", "Sci-Fi", "Thriller", "War"
]

class MovieFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to transform raw movie metadata into numerical features.
    """
    def __init__(self, genres_list=None):
        self.genres_list = genres_list if genres_list is not None else ALL_GENRES
        self.actor_ratings_ = {}
        self.director_ratings_ = {}
        self.global_mean_rating_ = 6.5

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if y is not None:
            df_temp = df.copy()
            df_temp['target_rating'] = y
            self.global_mean_rating_ = float(df_temp['target_rating'].mean())
            
            # Historical mean per actor and director with smoothing (weight=5)
            weight = 5
            
            # Actor stats
            actor_stats = df_temp.groupby('lead_actor')['target_rating'].agg(['count', 'mean'])
            actor_smooth = (actor_stats['count'] * actor_stats['mean'] + weight * self.global_mean_rating_) / (actor_stats['count'] + weight)
            self.actor_ratings_ = actor_smooth.to_dict()

            # Director stats
            dir_stats = df_temp.groupby('director')['target_rating'].agg(['count', 'mean'])
            dir_smooth = (dir_stats['count'] * dir_stats['mean'] + weight * self.global_mean_rating_) / (dir_stats['count'] + weight)
            self.director_ratings_ = dir_smooth.to_dict()

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        
        # Log transformations for skewed financial variables
        log_budget = np.log1p(np.maximum(0, df['budget'].values))
        log_revenue = np.log1p(np.maximum(0, df['revenue'].values))
        
        # ROI metric
        roi = (df['revenue'].values + 1.0) / (df['budget'].values + 1.0)
        log_roi = np.log1p(np.maximum(0, roi))

        # Runtime & Year
        runtime = df['runtime'].values.astype(float)
        release_year = df['release_year'].values.astype(float)
        
        # Actor and Director Popularities
        actor_pop = df['actor_popularity'].values.astype(float)
        dir_pop = df['director_popularity'].values.astype(float)

        # Actor and Director Smooth Mean Target Encoding
        actor_encoded = np.array([self.actor_ratings_.get(act, self.global_mean_rating_) for act in df['lead_actor']])
        dir_encoded = np.array([self.director_ratings_.get(d, self.global_mean_rating_) for d in df['director']])

        # Multi-label Genre One-Hot Encoding
        genre_features = []
        for genres_str in df['genres']:
            g_set = set(str(genres_str).split('|')) if pd.notna(genres_str) else set()
            genre_features.append([1.0 if g in g_set else 0.0 for g in self.genres_list])
        genre_matrix = np.array(genre_features)

        # Stack numerical array
        numerical_features = np.column_stack([
            runtime,
            release_year,
            log_budget,
            log_revenue,
            log_roi,
            actor_pop,
            dir_pop,
            actor_encoded,
            dir_encoded
        ])

        # Combine all features
        X_out = np.hstack([numerical_features, genre_matrix])
        return X_out

    def get_feature_names_out(self, input_features=None):
        num_names = [
            'runtime', 'release_year', 'log_budget', 'log_revenue', 'log_roi',
            'actor_popularity', 'director_popularity', 'actor_encoded_rating', 'director_encoded_rating'
        ]
        genre_names = [f"genre_{g}" for g in self.genres_list]
        return num_names + genre_names
