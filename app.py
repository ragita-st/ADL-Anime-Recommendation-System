import streamlit as st
import joblib
import scipy.sparse
from src.recommender import get_content_recommendations, get_collab_recommendations

st.set_page_config(page_title="Anime Hybrid Recommender", layout="wide", page_icon="🎬")

@st.cache_resource
def load_artifacts():
    anime = joblib.load('models/anime_cleaned.pkl')
    genre_df = joblib.load('models/genre_df.pkl')
    
    # Load the new lightweight files
    valid_users = joblib.load('models/valid_users.pkl')
    anime_categories = joblib.load('models/anime_categories.pkl')
    user_item_matrix = scipy.sparse.load_npz('models/user_item_matrix.npz')
    als_model = joblib.load('models/als_model.pkl')
    
    return anime, genre_df, valid_users, anime_categories, user_item_matrix, als_model

try:
    anime, genre_df, valid_users, anime_categories, user_item_matrix, als_model = load_artifacts()
    data_loaded = True
except FileNotFoundError:
    st.error("Model artifacts not found!")
    data_loaded = False

if data_loaded:
    st.title("✨ Anime Hybrid Recommender System")
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        # Use the extracted array
        user_input = st.selectbox("Select User Code:", valid_users[:500])
        
    with col2:
        anime_list = anime.sort_values('name')['name'].tolist()
        default_idx = anime_list.index("Death Note") if "Death Note" in anime_list else 0
        anime_input = st.selectbox("Select Favorite Anime:", anime_list, index=default_idx)

    st.write("")
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        st.write("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader(f"🍿 Because you liked: *{anime_input}*")
            content_recs = get_content_recommendations(anime_input, anime, genre_df)
            if content_recs is not None:
                st.dataframe(content_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not find enough genre similarities.")
                
        with res_col2:
            st.subheader(f"🌟 Top Picks For User {user_input}")
            # Pass anime_categories instead of the whole dataframe
            collab_recs = get_collab_recommendations(user_input, als_model, user_item_matrix, anime_categories, anime)
            if collab_recs is not None:
                st.dataframe(collab_recs, use_container_width=True, hide_index=True)
            else:
                st.warning("Not enough valid watch history.")