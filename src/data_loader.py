import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import os

def load_and_clean_data(anime_path='data/anime.csv', rating_path='data/rating.csv'):
    """Loads raw CSVs, cleans missing values, and prepares matrices for modeling."""
    
    if not os.path.exists(anime_path) or not os.path.exists(rating_path):
        raise FileNotFoundError(f"Missing data files. Ensure {anime_path} and {rating_path} exist.")

    # 1. Load Data
    anime = pd.read_csv(anime_path)
    ratings = pd.read_csv(rating_path)
    
    # 2. Clean Anime Metadata
    anime['genre'] = anime['genre'].fillna('Unknown')
    anime['type'] = anime['type'].fillna('Unknown')
    anime = anime.dropna(subset=['rating']) # Drop ghost/unreleased anime
    anime['episodes'] = anime['episodes'].replace('Unknown', np.nan)
    
    # 3. Genre Binarization (For Content-Based)
    anime['genre_list'] = anime['genre'].apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(anime['genre_list'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=anime.index)
    
    # 4. Density Filtering (Addressing the Cold-Start Problem)
    ratings_explicit = ratings[ratings['rating'] != -1].copy()
    active_users = ratings_explicit['user_id'].value_counts()[lambda x: x >= 50].index
    popular_anime = ratings_explicit['anime_id'].value_counts()[lambda x: x >= 100].index
    
    ratings_filtered = ratings_explicit[
        (ratings_explicit['user_id'].isin(active_users)) & 
        (ratings_explicit['anime_id'].isin(popular_anime))
    ].copy()
    
    # 5. ID Encoding (Continuous Mapping for ALS)
    ratings_filtered['user_category'] = ratings_filtered['user_id'].astype("category")
    ratings_filtered['anime_category'] = ratings_filtered['anime_id'].astype("category")
    ratings_filtered['user_code'] = ratings_filtered['user_category'].cat.codes
    ratings_filtered['anime_code'] = ratings_filtered['anime_category'].cat.codes
    
    # 6. Sparse User-Item Matrix
    user_item_matrix = csr_matrix((
        ratings_filtered['rating'].astype(float), 
        (ratings_filtered['user_code'], ratings_filtered['anime_code'])
    ))
    
    return anime, genre_df, ratings_filtered, user_item_matrix