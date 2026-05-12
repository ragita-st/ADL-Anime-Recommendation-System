import streamlit as st
import joblib
import scipy.sparse
import requests
import time
from src.recommender import get_content_recommendations, get_collab_recommendations
from thefuzz import process

st.set_page_config(page_title="Anime Hybrid Recommender", layout="wide", page_icon="🎬")

# --- API HELPER FUNCTION ---
# We cache this so if the user clicks the same anime again, it loads instantly from memory instead of the internet!
@st.cache_data(show_spinner=False)
def fetch_anime_details(anime_id, retries=3):
    """Fetches poster, synopsis, and streaming links with Cloud Rate Limit protection."""
    url = f"https://api.jikan.moe/v4/anime/{anime_id}/full"
    
    for attempt in range(retries):
        try:
            time.sleep(1.0) # 1 full second delay is required for cloud deployments
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json().get('data', {})
                image_url = data.get('images', {}).get('jpg', {}).get('image_url', None)
                synopsis = data.get('synopsis', 'No synopsis available.')
                
                # EXTRACT STREAMING LINKS
                streaming_data = data.get('streaming', [])
                # Create a dictionary of {Platform Name: URL} (e.g., {'Crunchyroll': 'http...'})
                streaming_links = {s['name']: s['url'] for s in streaming_data} if streaming_data else {}
                
                return image_url, synopsis, streaming_links
                
            elif response.status_code == 429:
                time.sleep(2.0) # If banned, wait 2 seconds and try again
                continue
            else:
                break
                
        except requests.exceptions.RequestException:
            time.sleep(1.0)
            continue
            
    return None, "Details currently unavailable due to high traffic.", {}

# --- UI HELPER FUNCTION ---
def display_recommendation_cards(recommendations_df):
    """Takes a dataframe of recommendations and draws beautiful UI cards."""
    if recommendations_df is None or recommendations_df.empty:
        st.warning("No recommendations found.")
        return

    for _, row in recommendations_df.iterrows():
        # Notice we are now unpacking THREE variables!
        img_url, synopsis, streaming_links = fetch_anime_details(row['anime_id'])
        
        img_col, text_col = st.columns([1, 2.5])
        
        with img_col:
            if img_url:
                st.image(img_url, width='stretch')
            else:
                st.image("https://via.placeholder.com/225x320/1E2130/7B61FF.png?text=No+Poster", width='stretch')
                
        with text_col:
            st.markdown(f"### {row['name']}")
            st.metric(label="Community Score", value=f"⭐ {row['rating']}")
            
            genre_html = ""
            for g in str(row['genre']).split(', '):
                genre_html += f"<span style='background-color:#7B61FF; color:white; padding:4px 10px; border-radius:15px; font-size:12px; margin-right:5px; display:inline-block; margin-bottom:5px;'>{g}</span>"
            st.markdown(genre_html, unsafe_allow_html=True)
            
            st.write("") 
            short_synopsis = synopsis[:250] + "..." if len(synopsis) > 250 else synopsis
            st.write(short_synopsis)
            
            # --- NEW: STREAMING BUTTONS ---
            if streaming_links:
                st.write("") # Spacer
                # Limit to the first 3 platforms so the UI doesn't get cluttered
                platforms = list(streaming_links.items())[:3] 
                button_cols = st.columns(len(platforms))
                
                for i, (platform, stream_url) in enumerate(platforms):
                    with button_cols[i]:
                        st.link_button(f"▶ {platform}", stream_url)
            
        st.write("---")

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
        
        # We replace selectbox with a text input!
        raw_search = st.text_input("🔍 Search Anime:", value="Death Note")
        
        # Use TheFuzz to find the closest match mathematically
        best_match, score = process.extractOne(raw_search, anime_list)
        
        if score >= 60:
            # st.success(f"Did you mean: **{best_match}**? *(Match: {score}%)*")
            st.success(f"Did you mean: **{best_match}**?")
            anime_input = best_match # Pass the corrected name to our AI
        else:
            st.error("Could not find a close match. Try typing another title.")
            anime_input = None

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