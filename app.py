import streamlit as st
import joblib
import scipy.sparse
import requests
import time
from src.recommender import get_content_recommendations, get_collab_recommendations

st.set_page_config(page_title="Anime Hybrid Recommender", layout="wide", page_icon="🎬")

# --- API HELPER FUNCTION ---
# We cache this so if the user clicks the same anime again, it loads instantly from memory instead of the internet!
@st.cache_data(show_spinner=False)
def fetch_anime_details(anime_id):
    """Fetches poster image and synopsis from the Jikan API."""
    url = f"https://api.jikan.moe/v4/anime/{anime_id}"
    try:
        # Jikan has a strict rate limit (3 requests/second). We sleep for 0.3s to prevent getting banned.
        time.sleep(0.3) 
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('data', {})
            image_url = data.get('images', {}).get('jpg', {}).get('image_url', None)
            synopsis = data.get('synopsis', 'No synopsis available.')
            return image_url, synopsis
        return None, "Details not found."
    except Exception:
        return None, "Error connecting to database."

# --- UI HELPER FUNCTION ---
def display_recommendation_cards(recommendations_df):
    """Takes a dataframe of recommendations and draws beautiful UI cards."""
    if recommendations_df is None or recommendations_df.empty:
        st.warning("No recommendations found.")
        return

    for _, row in recommendations_df.iterrows():
        # Fetch the image and text from the internet
        img_url, synopsis = fetch_anime_details(row['anime_id'])
        
        # Create a visual card layout: 1/3 of the space for the image, 2/3 for text
        img_col, text_col = st.columns([1, 2.5])
        
        with img_col:
            if img_url:
                # st.image(img_url, use_container_width=True)
                st.image(img_url, width='stretch')
            else:
                st.write("🖼️ Image missing")
                
        with text_col:
            st.markdown(f"#### {row['name']}")
            st.caption(f"⭐ **{row['rating']} / 10** |  🏷️ **{row['genre']}**")
            
            # Truncate the synopsis so it doesn't make the page too long
            short_synopsis = synopsis[:250] + "..." if len(synopsis) > 250 else synopsis
            st.write(short_synopsis)
            
        st.write("---") # Add a divider line between anime

# --- CACHE THE PRE-TRAINED MODELS ---
@st.cache_resource
def load_artifacts():
    anime = joblib.load('models/anime_cleaned.pkl')
    genre_df = joblib.load('models/genre_df.pkl')
    valid_users = joblib.load('models/valid_users.pkl')
    anime_categories = joblib.load('models/anime_categories.pkl')
    user_item_matrix = scipy.sparse.load_npz('models/user_item_matrix.npz')
    als_model = joblib.load('models/als_model.pkl')
    return anime, genre_df, valid_users, anime_categories, user_item_matrix, als_model

try:
    anime, genre_df, valid_users, anime_categories, user_item_matrix, als_model = load_artifacts()
    data_loaded = True
except FileNotFoundError:
    st.error("Model artifacts not found! Run `python train.py` first.")
    data_loaded = False

# --- MAIN APP UI ---
if data_loaded:
    st.title("✨ Anime Hybrid Recommender System")
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.selectbox("Select User Code:", valid_users[:500])
        
    with col2:
        anime_list = anime.sort_values('name')['name'].tolist()
        default_idx = anime_list.index("Death Note") if "Death Note" in anime_list else 0
        anime_input = st.selectbox("Select Favorite Anime:", anime_list, index=default_idx)

    st.write("")
    # if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
    if st.button("🚀 Generate Recommendations", type="primary", width='stretch'):
        st.write("---")
        
        # We wrap the generation in a spinner because downloading 10 images takes a few seconds
        with st.spinner("Pinging MyAnimeList database for images..."):
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader(f"🍿 Because you liked: *{anime_input}*")
                content_recs = get_content_recommendations(anime_input, anime, genre_df)
                display_recommendation_cards(content_recs)
                    
            with res_col2:
                st.subheader(f"🌟 Top Picks For User {user_input}")
                collab_recs = get_collab_recommendations(user_input, als_model, user_item_matrix, anime_categories, anime)
                display_recommendation_cards(collab_recs)