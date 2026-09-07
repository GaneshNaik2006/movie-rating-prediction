import os
import random
import numpy as np
import pandas as pd

def generate_movie_dataset(num_samples=2500, output_path="data/movies.csv", seed=42):
    """
    Generates a realistic movie dataset for regression modeling.
    """
    np.random.seed(seed)
    random.seed(seed)

    genres_list = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
        "Music", "Mystery", "Romance", "Sci-Fi", "Thriller", "War"
    ]

    actors = [
        ("Leonardo DiCaprio", 9.2), ("Meryl Streep", 9.1), ("Tom Hanks", 9.0),
        ("Denzel Washington", 8.8), ("Christian Bale", 8.7), ("Scarlett Johansson", 8.5),
        ("Robert Downey Jr.", 8.6), ("Brad Pitt", 8.5), ("Keanu Reeves", 8.2),
        ("Tom Cruise", 8.4), ("Emma Stone", 8.5), ("Viola Davis", 8.7),
        ("Ryan Gosling", 8.3), ("Florence Pugh", 8.2), ("Margot Robbie", 8.4),
        ("Cillian Murphy", 8.8), ("Morgan Freeman", 8.9), ("Samuel L. Jackson", 8.3),
        ("Will Smith", 7.8), ("Dwayne Johnson", 6.8), ("Adam Sandler", 6.2),
        ("Nicolas Cage", 6.5), ("Unknown Actor", 5.0)
    ]

    directors = [
        ("Christopher Nolan", 9.4), ("Steven Spielberg", 9.1), ("Martin Scorsese", 9.2),
        ("Quentin Tarantino", 9.0), ("Denis Villeneuve", 8.9), ("Greta Gerwig", 8.5),
        ("Guillermo del Toro", 8.6), ("James Cameron", 8.8), ("David Fincher", 8.7),
        ("Ridley Scott", 8.0), ("Tim Burton", 7.6), ("Michael Bay", 6.0),
        ("Zack Snyder", 6.4), ("M. Night Shyamalan", 6.3), ("Indie Director", 6.8),
        ("Unknown Director", 5.0)
    ]

    adjectives = ["The Last", "Dark", "Silent", "Eternal", "Golden", "Secret", "Lost", "Frozen", "Broken", "Rising", "Shadow", "Infinite", "Wild"]
    nouns = ["Kingdom", "Journey", "Legacy", "Dream", "Horizon", "Promise", "Chronicles", "Echoes", "Night", "Destiny", "World", "Code", "Storm"]

    data = []

    for i in range(num_samples):
        title = f"{random.choice(adjectives)} {random.choice(nouns)} {random.randint(1, 999)}"
        
        # Release year between 1990 and 2024
        release_year = int(np.random.randint(1990, 2025))
        
        # Runtime in minutes
        runtime = int(np.round(np.random.normal(110, 22)))
        runtime = max(65, min(220, runtime))
        
        # Budget ($1M to $280M) with log-normal distribution
        raw_budget = np.random.lognormal(mean=16.8, sigma=1.2)
        budget = float(np.round(np.clip(raw_budget, 1_000_000, 300_000_000), -4))
        
        # Revenue depends partly on budget and randomness
        revenue_multiplier = np.random.lognormal(mean=0.8, sigma=0.9)
        revenue = float(np.round(budget * revenue_multiplier, -4))

        # Genres (1 to 3 pipe-separated)
        num_genres = random.choices([1, 2, 3], weights=[0.4, 0.45, 0.15])[0]
        selected_genres = random.sample(genres_list, num_genres)
        genres_str = "|".join(selected_genres)
        
        # Lead Actor & Director selection
        actor_name, actor_pop = random.choice(actors)
        director_name, director_pop = random.choice(directors)
        
        # Add slight variation to pop scores per movie
        act_pop_val = max(1.0, min(10.0, actor_pop + np.random.normal(0, 0.3)))
        dir_pop_val = max(1.0, min(10.0, director_pop + np.random.normal(0, 0.3)))

        # Target rating calculation based on features
        base_rating = 6.2
        
        # Director and actor contribution
        dir_effect = (dir_pop_val - 6.0) * 0.35
        act_effect = (act_pop_val - 6.0) * 0.20
        
        # Runtime effect (movies around 100-140 mins get slight boost)
        runtime_effect = 0.3 if 100 <= runtime <= 150 else (-0.3 if runtime < 85 else 0.0)
        
        # Genre effect
        genre_effect = 0.0
        if "Drama" in selected_genres or "Biography" in selected_genres:
            genre_effect += 0.3
        if "Horror" in selected_genres:
            genre_effect -= 0.3
        if "Documentary" in selected_genres:
            genre_effect += 0.4

        # Budget / ROI quality effect
        roi = revenue / (budget + 1e-5)
        roi_effect = 0.25 if roi > 2.5 else (-0.2 if roi < 0.5 else 0.0)

        # Unexplained noise (Gaussian)
        noise = np.random.normal(0, 0.55)
        
        rating = base_rating + dir_effect + act_effect + runtime_effect + genre_effect + roi_effect + noise
        rating = round(float(np.clip(rating, 1.5, 9.8)), 1)

        data.append({
            "movie_id": i + 101,
            "title": title,
            "release_year": release_year,
            "runtime": runtime,
            "budget": budget,
            "revenue": revenue,
            "genres": genres_str,
            "lead_actor": actor_name,
            "director": director_name,
            "actor_popularity": round(act_pop_val, 2),
            "director_popularity": round(dir_pop_val, 2),
            "rating": rating
        })

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} records.")
    return df

if __name__ == "__main__":
    generate_movie_dataset()
