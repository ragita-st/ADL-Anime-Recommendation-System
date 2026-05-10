import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_content_recommendations(anime_title, anime_df, genre_df, n=5):
    # ... (Keep this exactly the same as you had it before) ...
    try:
        idx = anime_df[anime_df['name'] == anime_title].index[0]
        target_vector = genre_df.iloc[idx].values.reshape(1, -1)
        sim_scores = cosine_similarity(target_vector, genre_df)[0]
        sim_scores_list = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores_list[1:n+1]]
        return anime_df.iloc[top_indices][['name', 'genre', 'rating']].reset_index(drop=True)
    except IndexError:
        return None

def get_collab_recommendations(user_code, model, user_item_matrix, anime_categories, anime_df, n=5):
    try:
        ids, scores = model.recommend(user_code, user_item_matrix[user_code], N=n*4)
        
        # We now use the extracted array directly!
        real_anime_ids = [anime_categories[i] for i in ids if i < len(anime_categories)]
        
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