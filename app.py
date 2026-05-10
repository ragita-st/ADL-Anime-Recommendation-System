import streamlit as st
import os                                      # <-- Add this
os.environ['OPENBLAS_NUM_THREADS'] = '1'       # <-- Add this

import implicit
from sklearn.metrics.pairwise import cosine_similarity

# Import from our custom backend modules
from src.data_loader import load_and_clean_data
from src.recommender import get_content_recommendations, get_collab_recommendations

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Anime Hybrid Recommender", layout="wide", page_icon="🎬")

# --- CACHING THE BACKEND ---
# This prevents Streamlit from reloading CSVs every time you click a button
@st.cache_data
def fetch_backend_data():
    return load_and_clean_data()

# This prevents Streamlit from retraining the AI on every click
@st.cache_resource
def train_models(_genre_df, _user_item_matrix):
    genre_sim = cosine_similarity(_genre_df, _genre_df)
    
    model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1, random_state=42)
    model.fit(_user_item_matrix.T.tocsr())
    
    return genre_sim, model

# --- INITIALIZE APP ---
try:
    with st.spinner("Initializing Database and Training Models... (This takes a few seconds)"):
        anime, genre_df, ratings_filtered, user_item_matrix = fetch_backend_data()
        genre_sim, als_model = train_models(genre_df, user_item_matrix)
        
    data_loaded = True
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Please ensure you have placed 'anime.csv' and 'rating.csv' inside the 'data/' folder.")
    data_loaded = False

# --- UI LAYER ---
if data_loaded:
    st.title("✨ Anime Hybrid Recommender System")
    st.markdown("Select a **User ID** and a **Favorite Anime** to see how the two recommendation engines react.")

    # Input Layout
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Show a subset of valid user codes to keep the UI snappy
        valid_users = ratings_filtered['user_code'].unique()
        user_input = st.selectbox("Select User Code (For Collaborative Engine):", valid_users[:500])
        
    with col2:
        # Sort anime alphabetically for easier searching
        anime_list = anime.sort_values('name')['name'].tolist()
        
        # Set a default value to something popular
        default_idx = anime_list.index("Death Note") if "Death Note" in anime_list else 0
        anime_input = st.selectbox("Select Favorite Anime (For Content Engine):", anime_list, index=default_idx)

    # Action Button
    st.write("")
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        st.write("---")
        
        res_col1, res_col2 = st.columns(2)
        
        # ROW 1: Content-Based
        with res_col1:
            st.subheader(f"🍿 Because you liked: *{anime_input}*")
            st.caption("Content-Based Engine (Matches based on Genre tags)")
            
            content_recs = get_content_recommendations(anime_input, anime, genre_sim)
            if content_recs is not None:
                st.dataframe(content_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not find enough genre similarities for this anime.")
                
        # ROW 2: Collaborative
        with res_col2:
            st.subheader(f"🌟 Top Picks For User {user_input}")
            st.caption("Collaborative Engine (Matches based on Crowd Watch History)")
            
            collab_recs = get_collab_recommendations(user_input, als_model, user_item_matrix, ratings_filtered, anime)
            if collab_recs is not None:
                st.dataframe(collab_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("Not enough valid watch history to make a collaborative recommendation.")