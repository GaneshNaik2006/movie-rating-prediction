import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure root folder is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import load_prediction_pipeline, predict_movie_rating
from src.preprocessing import ALL_GENRES

st.set_page_config(
    page_title="Movie Rating Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #E50914;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .rating-number {
        font-size: 3.5rem;
        font-weight: 800;
        color: #E50914;
    }
    .verdict-text {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Rating Predictor ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict IMDb / TMDB ratings using Machine Learning Regression algorithms</div>', unsafe_allow_html=True)

# Load Pipeline & Metrics
@st.cache_resource
def get_pipeline():
    model_path = "models/movie_rating_model.joblib"
    if not os.path.exists(model_path):
        return None
    return load_prediction_pipeline(model_path)

@st.cache_data
def get_metrics():
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def get_dataset():
    data_path = "data/movies.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

pipeline = get_pipeline()
metrics = get_metrics()
df_data = get_dataset()

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/movie-projector.png", width=70)
    st.header("📌 Project Details")
    st.write("**Task**: Machine Learning Regression")
    st.write("**Target**: Continuous Vote Rating (1.0 to 10.0)")
    
    if metrics:
        st.divider()
        st.subheader("🏆 Model Performance")
        st.metric(label="Selected Model", value=metrics.get("best_model", "N/A"))
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Test R² Score", value=f"{metrics.get('test_r2', 0):.4f}")
        with col_m2:
            st.metric(label="RMSE", value=f"{metrics.get('test_rmse', 0):.4f}")

# Main Tabs
tab_predict, tab_analytics = st.tabs(["🔮 Predict Rating", "📊 Model & Dataset Analytics"])

with tab_predict:
    if pipeline is None:
        st.error("⚠️ Trained model not found! Please run training first via `python src/train.py`.")
    else:
        st.markdown("### 📝 Enter Movie Metadata")
        
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Movie Title", "Inception: Resurgence")
            release_year = st.slider("Release Year", 1990, 2030, 2024)
            runtime = st.slider("Runtime (minutes)", 60, 240, 148)
            selected_genres = st.multiselect("Genres", ALL_GENRES, default=["Action", "Sci-Fi", "Thriller"])
            
            budget_m = st.number_input("Budget ($ Millions)", min_value=1.0, max_value=400.0, value=150.0, step=5.0)
            revenue_m = st.number_input("Expected Revenue ($ Millions)", min_value=0.0, max_value=2000.0, value=450.0, step=10.0)

        with col2:
            lead_actor = st.selectbox("Lead Actor", [
                "Leonardo DiCaprio", "Meryl Streep", "Tom Hanks", "Denzel Washington",
                "Christian Bale", "Scarlett Johansson", "Robert Downey Jr.", "Brad Pitt",
                "Keanu Reeves", "Tom Cruise", "Emma Stone", "Viola Davis", "Ryan Gosling",
                "Florence Pugh", "Margot Robbie", "Cillian Murphy", "Morgan Freeman",
                "Will Smith", "Dwayne Johnson", "Adam Sandler", "Other / Emerging Actor"
            ])
            actor_pop = st.slider("Lead Actor Popularity Rating (1.0 - 10.0)", 1.0, 10.0, 8.8, step=0.1)

            director = st.selectbox("Director", [
                "Christopher Nolan", "Steven Spielberg", "Martin Scorsese", "Quentin Tarantino",
                "Denis Villeneuve", "Greta Gerwig", "Guillermo del Toro", "James Cameron",
                "David Fincher", "Ridley Scott", "Tim Burton", "Michael Bay", "Zack Snyder",
                "M. Night Shyamalan", "Other / Emerging Director"
            ])
            director_pop = st.slider("Director Popularity Rating (1.0 - 10.0)", 1.0, 10.0, 9.2, step=0.1)

        predict_button = st.button("🚀 Predict Movie Rating", use_container_width=True, type="primary")

        if predict_button:
            genres_str = "|".join(selected_genres) if selected_genres else "Drama"
            input_data = {
                "release_year": release_year,
                "runtime": runtime,
                "budget": budget_m * 1_000_000,
                "revenue": revenue_m * 1_000_000,
                "genres": genres_str,
                "lead_actor": lead_actor,
                "director": director,
                "actor_popularity": actor_pop,
                "director_popularity": director_pop
            }

            res = predict_movie_rating(input_data, pipeline)

            st.divider()
            st.markdown("### 🎯 Rating Prediction Results")
            
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #495057;">Predicted IMDb Rating</div>
                    <div class="rating-number">{res['predicted_rating']} <span style="font-size: 1.8rem; color: #6c757d;">/ 10</span></div>
                    <div class="verdict-text">{res['verdict']}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### 💡 Feature Highlights")
                roi = revenue_m / budget_m if budget_m > 0 else 0
                st.info(f"**Box Office ROI multiplier**: **{roi:.2f}x**")
                st.write(f"- **Star Power (Actor + Director Avg)**: {((actor_pop + director_pop)/2):.1f}/10")
                st.write(f"- **Genres Tagged**: {', '.join(selected_genres) if selected_genres else 'None'}")
                st.write(f"- **Runtime Profile**: {runtime} mins ({'Optimal Length' if 100 <= runtime <= 150 else 'Short/Long'})")

with tab_analytics:
    st.markdown("### 📈 Model Evaluation & Dataset Insights")

    if metrics and "candidate_results" in metrics:
        st.subheader("🤖 Candidate Model Benchmark Comparison")
        res_df = pd.DataFrame(metrics["candidate_results"]).T
        st.dataframe(res_df.style.highlight_max(subset=["R2"], color="#d4edda").highlight_min(subset=["RMSE"], color="#d4edda"), use_container_width=True)

    if df_data is not None:
        st.divider()
        st.subheader("🗂️ Training Dataset Preview")
        st.dataframe(df_data.head(10), use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df_data['rating'], kde=True, color='#E50914', ax=ax, bins=20)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with col_d2:
            st.subheader("Budget vs Rating Scatter")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            sns.scatterplot(data=df_data, x=df_data['budget']/1e6, y='rating', alpha=0.6, color='#1f77b4', ax=ax2)
            ax2.set_xlabel("Budget ($ Millions)")
            ax2.set_ylabel("Rating")
            st.pyplot(fig2)
