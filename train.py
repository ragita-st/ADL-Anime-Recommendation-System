import os
import joblib
import implicit
import scipy.sparse
from src.data_loader import load_and_clean_data

print("1. Loading and cleaning dataset...")
anime, genre_df, ratings_filtered, user_item_matrix = load_and_clean_data()

print("2. Training Collaborative ALS Model...")
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# We ensure it is a CSR matrix here
csr_matrix = user_item_matrix.T.tocsr()
model = implicit.als.AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1, random_state=42)
model.fit(csr_matrix)

print("3. Extracting lightweight mappings and saving...")
os.makedirs('models', exist_ok=True)

# Save the ALS model
joblib.dump(model, 'models/als_model.pkl')

# Save the DataFrames
joblib.dump(anime, 'models/anime_cleaned.pkl')
joblib.dump(genre_df, 'models/genre_df.pkl')

# INSTEAD of saving 234MB of ratings, we extract exactly what the UI needs (Result: ~1MB)
valid_users = ratings_filtered['user_code'].unique()
anime_categories = ratings_filtered['anime_category'].cat.categories

joblib.dump(valid_users, 'models/valid_users.pkl')
joblib.dump(anime_categories, 'models/anime_categories.pkl')

# Compress the sparse matrix natively (Result: Drops from 64MB to ~10MB)
scipy.sparse.save_npz('models/user_item_matrix.npz', csr_matrix.T) # Save the original Item-User orientation

print("✅ Training complete! All artifacts highly compressed and saved.")