import streamlit as st
import joblib
import scipy.sparse
import requests
import time
import pandas as pd
from src.recommender import get_content_recommendations, get_collab_recommendations
from thefuzz import process
import plotly.express as px

st.set_page_config(page_title="Anime Hybrid Recommender", layout="wide", page_icon="🎬")

# --- INITIALIZE SESSION STATE (For the Onboarding Flow) ---
if 'onboarded' not in st.session_state:
    st.session_state['onboarded'] = False
if 'user_genres' not in st.session_state:
    st.session_state['user_genres'] = []

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def fetch_anime_details(anime_id, retries=3):
    url = f"https://api.jikan.moe/v4/anime/{anime_id}/full"
    for attempt in range(retries):
        try:
            time.sleep(1.0) 
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json().get('data', {})
                image_url = data.get('images', {}).get('jpg', {}).get('image_url', None)
                synopsis = data.get('synopsis', 'No synopsis available.')
                streaming_data = data.get('streaming', [])
                streaming_links = {s['name']: s['url'] for s in streaming_data} if streaming_data else {}
                return image_url, synopsis, streaming_links
            elif response.status_code == 429:
                time.sleep(2.0)
                continue
            else:
                break
        except requests.exceptions.RequestException:
            time.sleep(1.0)
            continue
    return None, "Details currently unavailable due to high traffic.", {}

def display_recommendation_cards(recommendations_df):
    if recommendations_df is None or recommendations_df.empty:
        st.warning("No recommendations found.")
        return
    for _, row in recommendations_df.iterrows():
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
            if streaming_links:
                st.write("") 
                platforms = list(streaming_links.items())[:3] 
                button_cols = st.columns(len(platforms))
                for i, (platform, stream_url) in enumerate(platforms):
                    with button_cols[i]:
                        st.link_button(f"▶ {platform}", stream_url)
        st.write("---")

def get_starter_pack(genres, anime_df):
    """Generates a Cold-Start list based on user's selected genres."""
    # Filter anime that contain AT LEAST ONE of the user's selected genres
    mask = anime_df['genre'].apply(lambda x: any(g in str(x) for g in genres))
    # Return the Top 5 highest rated matching anime
    return anime_df[mask].sort_values('rating', ascending=False).head(5)

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
    # The columns of genre_df are the actual unique genres from our MultiLabelBinarizer!
    all_unique_genres = list(genre_df.columns) 
except FileNotFoundError:
    st.error("Model artifacts not found! Run `python train.py` first.")
    data_loaded = False

# ==========================================
# 🚀 APP ROUTING (Onboarding vs Main App)
# ==========================================
if data_loaded:
    
    # --- ROUTE 1: THE ONBOARDING SCREEN ---
    if not st.session_state['onboarded']:
        st.title("Welcome to the Anime AI 🎬")
        st.write("To get started, tell us what kind of shows you like!")
        
        st.write("")
        selected_genres = st.multiselect(
            "Select up to 3 of your favorite genres:", 
            options=all_unique_genres,
            max_selections=3
        )
        
        st.write("")
        if st.button("🚀 Enter Dashboard", type="primary"):
            if len(selected_genres) > 0:
                # Save their choices and flag them as onboarded
                st.session_state['user_genres'] = selected_genres
                st.session_state['onboarded'] = True
                st.rerun() # Force Streamlit to refresh the page instantly!
            else:
                st.error("Please select at least one genre to continue.")

    # --- ROUTE 2: THE MAIN DASHBOARD ---
    else:
        st.title("✨ Anime Hybrid Recommender")
        
        # Add a logout button to clear state
        if st.button("🔄 Reset Preferences"):
            st.session_state['onboarded'] = False
            st.session_state['user_genres'] = []
            st.rerun()
            
        st.write("---")
        
        # Build the Tabs!
        tab1, tab2 = st.tabs(["✨ Discover", "📊 Global Analytics"])
        
        # --- TAB 1: DISCOVER ---
        with tab1:
            st.subheader(f"🎁 Your Starter Pack ({', '.join(st.session_state['user_genres'])})")
            st.caption("Based on your onboarding preferences.")
            starter_recs = get_starter_pack(st.session_state['user_genres'], anime)
            
            with st.expander("Click to view your Starter Pack!", expanded=False):
                with st.spinner("Fetching Starter Pack details..."):
                    display_recommendation_cards(starter_recs)
            
            st.write("---")
            st.subheader("🔍 Deep Search Engine")
            
            col1, col2 = st.columns(2)
            with col1:
                user_input = st.selectbox("Select User Code (Collaborative Filter):", valid_users[:500])
            with col2:
                anime_list = anime.sort_values('name')['name'].tolist()
                raw_search = st.text_input("Search Anime (Content Filter):", value="Death Note")
                best_match, score = process.extractOne(raw_search, anime_list)
                
                if score >= 60:
                    st.success(f"Target Acquired: **{best_match}**")
                    anime_input = best_match 
                else:
                    st.error("Could not find a close match.")
                    anime_input = None

            st.write("")
            if st.button("Generate Deep Recommendations", type="primary", width='stretch'):
                with st.spinner("Calculating matrices and fetching MyAnimeList data..."):
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown(f"#### 🍿 Similar to: *{anime_input}*")
                        content_recs = get_content_recommendations(anime_input, anime, genre_df)
                        display_recommendation_cards(content_recs)
                            
                    with res_col2:
                        st.markdown(f"#### 🌟 Top Picks For User {user_input}")
                        collab_recs = get_collab_recommendations(user_input, als_model, user_item_matrix, anime_categories, anime)
                        display_recommendation_cards(collab_recs)

        # --- TAB 2: ANALYTICS ---
        with tab2:
            st.header("📊 Global Anime Landscape")
            st.write("Explore the dataset driving this AI engine.")
            st.write("---")
            
            # --- ROW 1: Top Rated & Most Popular ---
            colA, colB = st.columns(2)
            
            with colA:
                st.subheader("🏆 Highest Rated of All Time")
                top_rated = anime.sort_values('rating', ascending=False).head(10)
                # Create an interactive horizontal bar chart
                fig_rating = px.bar(top_rated, x='rating', y='name', orientation='h', 
                                    color='rating', color_continuous_scale='Purples')
                # Flip the y-axis so the #1 anime is at the top
                fig_rating.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_rating, use_container_width=True)

            with colB:
                st.subheader("🔥 Most Popular (Community Size)")
                # Your Kaggle dataset should have a 'members' column showing community size
                if 'members' in anime.columns:
                    top_pop = anime.sort_values('members', ascending=False).head(10)
                    fig_pop = px.bar(top_pop, x='members', y='name', orientation='h', 
                                     color='members', color_continuous_scale='Purples')
                    fig_pop.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_pop, use_container_width=True)
                else:
                    st.info("Community data not available in this dataset format.")

            st.write("---")
            
            # --- ROW 2: Type Distribution & Top Genres ---
            colC, colD = st.columns(2)
            
            with colC:
                st.subheader("📺 Anime Formats")
                if 'type' in anime.columns:
                    type_counts = anime['type'].value_counts().reset_index()
                    type_counts.columns = ['type', 'count']
                    # Create an interactive Donut Chart
                    fig_type = px.pie(type_counts, values='count', names='type', hole=0.4,
                                      color_discrete_sequence=px.colors.sequential.Purples_r)
                    fig_type.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_type, use_container_width=True)

            with colD:
                st.subheader("🏷️ Most Common Genres")
                # We can do math directly on the genre matrix we saved earlier!
                genre_counts = genre_df.sum().sort_values(ascending=False).head(10).reset_index()
                genre_counts.columns = ['genre', 'count']
                fig_genre = px.bar(genre_counts, x='count', y='genre', orientation='h', 
                                   color='count', color_continuous_scale='Purples')
                fig_genre.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_genre, use_container_width=True)
                
            st.write("---")
            st.subheader("Raw Data Explorer")
            st.dataframe(anime[['name', 'genre', 'type', 'episodes', 'rating']].head(100), use_container_width=True, hide_index=True)