import pandas as pd

def get_content_recommendations(anime_title, anime_df, genre_sim_matrix, n=5):
    """Returns anime with similar genres using Cosine Similarity."""
    try:
        # Find index of the target anime
        idx = anime_df[anime_df['name'] == anime_title].index[0]
        
        # Calculate similarity scores
        sim_scores = list(enumerate(genre_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Grab top N (skipping index 0, which is the anime itself)
        top_indices = [i[0] for i in sim_scores[1:n+1]]
        
        # Return clean DataFrame
        return anime_df.iloc[top_indices][['name', 'genre', 'rating']].reset_index(drop=True)
    except IndexError:
        return None

def get_collab_recommendations(user_code, model, user_item_matrix, ratings_filtered, anime_df, n=5):
    """Returns personalized anime using Alternating Least Squares (ALS)."""
    try:
        # Over-predict to account for ghost IDs dropped during cleaning
        ids, scores = model.recommend(user_code, user_item_matrix[user_code], N=n*4)
        
        categories = ratings_filtered['anime_category'].cat.categories
        real_anime_ids = [categories[i] for i in ids if i < len(categories)]
        
        valid_recs = []
        for original_id in real_anime_ids:
            match = anime_df[anime_df['anime_id'] == original_id]
            if not match.empty:
                valid_recs.append(match[['name', 'genre', 'rating']].iloc[0])
            if len(valid_recs) == n:
                break
                
        if not valid_recs:
            return None
            
        return pd.DataFrame(valid_recs).reset_index(drop=True)
    except Exception as e:
        return None