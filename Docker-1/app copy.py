"""
Movie Recommender Streamlit App
User interface for getting personalized movie recommendations.
"""

import streamlit as st
import pandas as pd
from lenskit import Recommender
from lenskit.batch import recommend, predict
from lenskit.algorithms.als import BiasedMF
from pathlib import Path
import utils
import json
from json import JSONDecodeError

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

def normalize_title(title: str) -> str:
    parts = title.rsplit("(", 1)  # separate year if present
    name = parts[0].strip()
    year = "(" + parts[1] if len(parts) > 1 else ""

    for article in ["The", "A", "An"]:
        suffix = f", {article}"
        if name.endswith(suffix):
            name = f"{article} {name[: -len(suffix)]}"

    return f"{name.strip()} {year}".strip()

def read_metrics_txt():
    path = Path("models/metrics.txt")
    if not path.exists():
        return None, None
    try:
        lines = path.read_text().splitlines()
        rmse_val = float(lines[0].split(":")[1].strip())
        mse_val  = float(lines[1].split(":")[1].strip())
        return rmse_val, mse_val
    except Exception:
        return None, None


@st.cache_resource
def load_model_and_data():
    model_path   = 'models/lenskit_model.pkl'
    movies_path  = 'models/movies.csv'

    if not Path(model_path).exists():
        st.error("Model not found! Please run train_model.py first.")
        st.stop()

    model  = utils.load_model(model_path)
    movies = utils.load_movies(movies_path)
    movies['title'] = movies['title'].apply(normalize_title)

    return model, movies



def build_user_profile_from_ratings(liked_with_scores, movies_df, temp_user_id=-99999):
    """liked_with_scores: list[(title, rating)] -> DataFrame[user,item,rating]"""
    title_to_id = dict(zip(movies_df['title'].astype(str), movies_df['movieId']))
    rows = []
    for t, r in liked_with_scores:
        mid = title_to_id.get(str(t))
        if mid is not None:
            rows.append((temp_user_id, mid, float(r)))
    return pd.DataFrame(rows, columns=['user', 'item', 'rating'])

def get_recommendations(model, user_profile, candidate_items, n=5, min_rating=0.5, max_rating=5.0):
    """
    Use saved BiasedMF to score a cold user by passing their ad-hoc ratings.
    `user_profile` must have columns: ['user','item','rating'] for a single user.
    """

    user_id = int(user_profile['user'].iloc[0])

    # Convert to the expected shape: 1-D Series indexed by item id
    # IMPORTANT: index = item ids, values = ratings (float)
    ratings_ser = pd.Series(
        user_profile['rating'].astype('float64').values,
        index=user_profile['item'].astype('int64').values
    )

    # Ensure candidates are plain int64 list/array (not a Series of objects)
    cand = pd.Index(candidate_items).astype('int64').tolist()

    # Predict for the unseen user with provided ratings vector
    preds = model.predict_for_user(user_id, items=cand, ratings=ratings_ser)
   
    # Normalize to DataFrame & clip to bounds
    if isinstance(preds, pd.Series):
        recs = preds.rename('prediction').reset_index().rename(columns={'index': 'item'})
    else:
        recs = preds  # already has ['item','prediction'] or ['item','score']
        if 'prediction' not in recs.columns and 'score' in recs.columns:
            recs = recs.rename(columns={'score': 'prediction'})

    recs['prediction'] = recs['prediction'].clip(min_rating, max_rating)
    return recs.sort_values('prediction', ascending=False).head(n)



def main():
    # ---- Header / hero ----
    st.title("🎬 Movie Recommender System")
    st.caption("Pick five movies you like, rate them, and we’ll predict a personal top five.")
    st.divider()

    # ---- Load model and data ----
    with st.spinner("Loading model and data..."):
        model, movies = load_model_and_data()
        acc_rmse, acc_mse = read_metrics_txt()



    # ---- Prepare training data (only if you still need it elsewhere) ----
    if 'train_data' not in st.session_state:
        ratings = pd.read_csv('data/ratings.csv')
        st.session_state.train_data = ratings.rename(
            columns={'userId': 'user', 'movieId': 'item'}
        )[['user', 'item', 'rating']].copy()

    # ---- Dataset facts & About (side-by-side) ----
    genres = utils.get_unique_genres(movies)

    # assume you've already computed `genres = utils.get_unique_genres(movies)`

    with st.sidebar:
        st.header("Dataset Info")
        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Movies", f"{len(movies):,}")
        with c2:
            st.metric("Available Genres", len(genres))

        # Row 2: metrics
        #m1, m2 = st.columns(2)
        #with m1:
            #st.metric("RMSE", f"{acc_rmse:.2f}" if acc_rmse is not None else "N/A")
        #with m2:
            #st.metric("MSE",  f"{acc_mse:.2f}"  if acc_mse  is not None else "N/A")


    # ---- Preferences section ----
    st.subheader("Tell us about your movie preferences")
    st.caption("Select exactly five titles you’ve enjoyed. You can also add a genre filter on the right.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Search or Select 5 Movies You Like")

        liked_movies = st.multiselect(
            "Search and select movies:",
            options=movies['title'].unique(),
            max_selections=5,
            help="Type to search for movies"
        )
        user_ratings = []
        if liked_movies:
            st.caption("Give each a personal rating (1-5):")
            for title in liked_movies:
                r = st.slider(f"Rating for “{title}”", min_value=1.0, max_value=5.0, step=0.5, value=4.0, key=f"rate_{title}")
                user_ratings.append((title, r))
        
    with col2:
        st.subheader("Optionally, Choose a Genre")
        target_genre = st.selectbox(
            "Genre:",
            options=["Any"] + genres,
            help="Filter recommendations by genre"
        )


    # Recommendation button
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        if len(liked_movies) != 5:
            st.warning(f"⚠️ Please select exactly 5 movies. You've selected {len(liked_movies)}.")
        else:
            with st.spinner("Generating personalized recommendations..."):
                # Build user profile from (title, rating) pairs
                user_profile = build_user_profile_from_ratings(user_ratings, movies, temp_user_id=-99999)
                if user_profile.empty:
                    st.error("Could not map your selections to movie IDs. Check your movies file.")
                    st.stop()

                liked_ids = set(user_profile['item'].tolist())

                # Filter candidate movies by genre (optional)
                filtered_movies = utils.filter_movies_by_genre(movies, target_genre)
                candidate_items = filtered_movies.loc[~filtered_movies['movieId'].isin(liked_ids), 'movieId'].values

                # Get recommendations using saved model + ad-hoc ratings (no retrain)
                recommendations = get_recommendations(
                    model=model,
                    user_profile=user_profile,
                    candidate_items=candidate_items,
                    n=5,
                    min_rating=0.5,  # adjust if your dataset min differs
                    max_rating=5.0
                )

                # Format and display
                formatted_recs = utils.format_recommendations(recommendations, movies)

                st.markdown("---")
                st.header("🌟 Your Personalized Recommendations")
                for idx, row in formatted_recs.iterrows():
                    with st.container():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.subheader(f"{idx + 1}. {row['title']}")
                            st.caption(f"Genres: {row['genres']}")
                        with col_b:
                            st.metric("Predicted Rating", f"{row['Predicted Rating']:.2f}/5.0")
                        st.markdown("---")

    
    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This movie recommender uses collaborative filtering 
            (BiasedMF algorithm from LensKit) to suggest movies 
            based on your preferences.
            
            **How it works:**
            1. Select 5 movies you like
            2. Optionally choose a genre
            3. Get 5 personalized recommendations
            
            The system learns from thousands of user ratings 
            to find movies similar to your taste!
            """
        )
        
if __name__ == "__main__":
    main()