"""
Utility Functions
Helper functions for data processing and recommendations.
"""

import pandas as pd
import pickle
from pathlib import Path

def load_model(path='models/lenskit_model.pkl'):
    """Load the trained LensKit model."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_movies(path='models/movies.csv'):
    """Load the movies dataset."""
    return pd.read_csv(path)

def get_unique_genres(movies_df):
    """Extract all unique genres from the dataset."""
    genres = set()
    for genre_str in movies_df['genres'].dropna():
        if genre_str != '(no genres listed)':
            genres.update(genre_str.split('|'))
    return sorted(genres)

def filter_movies_by_genre(movies_df, genre=None):
    """Filter movies by genre if specified."""
    if genre and genre != "Any":
        mask = movies_df['genres'].str.contains(genre, case=False, na=False)
        return movies_df[mask]
    return movies_df

def create_user_profile(liked_movies, movies_df, user_id=999999):
    """
    Create a synthetic user profile based on liked movies.
    
    Args:
        liked_movies: List of movie titles
        movies_df: DataFrame with movie information
        user_id: Synthetic user ID
    
    Returns:
        DataFrame with user ratings
    """
    liked_ids = movies_df[movies_df['title'].isin(liked_movies)]['movieId'].tolist()
    
    # Assign ratings (5.0 for first, decreasing slightly for others)
    ratings = [5.0 - (i * 0.1) for i in range(len(liked_ids))]
    
    return pd.DataFrame({
        'user': [user_id] * len(liked_ids),
        'item': liked_ids,
        'rating': ratings
    })

def get_movie_details(movie_ids, movies_df):
    """Get detailed information for a list of movie IDs."""
    return movies_df[movies_df['movieId'].isin(movie_ids)][['movieId', 'title', 'genres']]

def format_recommendations(recommendations, movies_df):
    """Format recommendations for display."""
    recs_with_details = recommendations.merge(
        movies_df[['movieId', 'title', 'genres']], 
        left_on='item', 
        right_on='movieId',
        how='left'
    )
    return recs_with_details[['title', 'genres', 'prediction']].rename(columns={
        'prediction': 'Predicted Rating'
    })
